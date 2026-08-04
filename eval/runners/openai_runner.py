"""OpenAI runner — GPT-4o, GPT-4o-mini, and successors.

End-to-end runner for OpenAI's chat-completions API. Sends the
formatted MCQ prompt, parses the structured response, and returns an
:class:`~eval.schemas.EvalResponse`.

This runner is a SCAFFOLD in Phase 4A: built and unit-tested with
mocked clients, but not exercised against the live API in the
zero-budget smoke eval. Wired through ``RUNNER_REGISTRY`` so it is
ready when paid OpenAI access is configured.

Tests inject a mock client via the ``client`` constructor arg so unit
tests never touch the network.
"""

from __future__ import annotations

import os
import time
from typing import Any

from eval.prompts.confidence import parse_confidence_response
from eval.prompts.mcq import format_mcq
from eval.runners.base import Runner
from eval.schemas import EvalResponse, MCQItem


class OpenAIRunner(Runner):
    """Runner for OpenAI chat models (GPT-4o family by default)."""

    name: str = "openai"
    DEFAULT_MODEL: str = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
        max_retries: int = 4,
        initial_backoff_s: float = 1.0,
        client: Any = None,
    ) -> None:
        """Initialize the OpenAI runner.

        Args:
            model: OpenAI model id.
            temperature: Sampling temperature (default 0.0).
            top_p: Nucleus-sampling cutoff (default 1.0).
            max_tokens: Cap on response length.
            api_key: Override ``OPENAI_API_KEY`` from the environment.
            max_retries: Number of attempts before returning an error
                response.
            initial_backoff_s: Wait between attempt 1 and attempt 2.
                Doubled on each subsequent retry.
            client: An ``openai.OpenAI`` instance (or test double).
        """
        super().__init__(model, temperature=temperature, top_p=top_p)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        if client is not None:
            self._client: Any = client
        else:
            from openai import OpenAI

            key = api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if key else OpenAI()

    def run(self, item: MCQItem) -> EvalResponse:
        """Send ``item`` through the OpenAI chat-completions API.

        See :meth:`eval.runners.base.Runner.run` for the contract.
        """
        prompt = format_mcq(item)
        started = time.perf_counter()
        last_err: str | None = None
        delay = self.initial_backoff_s

        for attempt in range(1, self.max_retries + 1):
            try:
                msg = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
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
        """Extract text from an OpenAI chat-completion response."""
        choices = getattr(msg, "choices", None) or []
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
