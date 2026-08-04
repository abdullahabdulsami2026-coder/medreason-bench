"""Ollama runner — local open-source LLMs via Ollama's HTTP API.

End-to-end runner for an Ollama daemon (default
``http://localhost:11434``). Useful as an offline / zero-budget
fallback. Default model is ``llama3.1:8b``; any model installed via
``ollama pull <model>`` works.

Tests inject a mock client via the ``client`` constructor arg so unit
tests never touch the daemon.
"""

from __future__ import annotations

import os
import time
from typing import Any

from eval.prompts.confidence import parse_confidence_response
from eval.prompts.mcq import format_mcq
from eval.runners.base import Runner
from eval.schemas import EvalResponse, MCQItem


class OllamaRunner(Runner):
    """Runner for an Ollama daemon serving local open-source LLMs."""

    name: str = "ollama"
    DEFAULT_MODEL: str = "llama3.1:8b"
    DEFAULT_HOST: str = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        host: str | None = None,
        max_retries: int = 4,
        initial_backoff_s: float = 1.0,
        client: Any = None,
    ) -> None:
        """Initialize the Ollama runner.

        Args:
            model: Ollama model tag (must be present on the daemon —
                run ``ollama pull <model>`` first).
            temperature: Sampling temperature (default 0.0).
            top_p: Nucleus-sampling cutoff (default 1.0).
            max_tokens: Cap on response length (mapped to
                ``options.num_predict``).
            host: Override ``OLLAMA_HOST`` from the environment.
                Defaults to ``http://localhost:11434``.
            max_retries: Number of attempts before returning an error
                response.
            initial_backoff_s: Wait between attempt 1 and attempt 2.
                Doubled on each subsequent retry.
            client: An ``ollama.Client`` instance (or test double).
        """
        super().__init__(model, temperature=temperature, top_p=top_p)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.host = host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        if client is not None:
            self._client: Any = client
        else:
            from ollama import Client

            self._client = Client(host=self.host)

    def run(self, item: MCQItem) -> EvalResponse:
        """Send ``item`` through the Ollama daemon.

        See :meth:`eval.runners.base.Runner.run` for the contract.
        """
        prompt = format_mcq(item)
        started = time.perf_counter()
        last_err: str | None = None
        delay = self.initial_backoff_s

        for attempt in range(1, self.max_retries + 1):
            try:
                msg = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                    },
                )
                raw = self._extract_text(msg)
                answer, confidence, rationale = parse_confidence_response(raw)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                return EvalResponse(
                    item_id=item.id,
                    model=self.model,
                    raw_response=raw,
                    parsed_answer=answer,
                    confidence=confidence,
                    rationale=rationale,
                    elapsed_ms=elapsed_ms,
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return EvalResponse(
            item_id=item.id,
            model=self.model,
            raw_response="",
            parsed_answer=None,
            confidence=None,
            rationale=None,
            elapsed_ms=elapsed_ms,
            error=last_err or "unknown error",
        )

    @staticmethod
    def _extract_text(msg: Any) -> str:
        """Extract text from an Ollama ``ChatResponse``.

        Newer versions return a Pydantic-like object with
        ``msg.message.content``; older versions return a dict
        ``{"message": {"content": "..."}}``. The helper handles both.
        """
        message = getattr(msg, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
        if isinstance(msg, dict):
            inner = msg.get("message")
            if isinstance(inner, dict):
                content = inner.get("content")
                if isinstance(content, str):
                    return content
        return ""
