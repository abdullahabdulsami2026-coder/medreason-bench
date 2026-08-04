"""Anthropic Messages API runner (Claude Opus / Sonnet).

End-to-end runner for the Anthropic ``messages.create`` API. Sends the
formatted MCQ prompt, parses the structured response (ANSWER / CONFIDENCE
/ RATIONALE), and returns an :class:`~eval.schemas.EvalResponse`.

Retries transient failures with exponential backoff (1s → 2s → 4s → 8s by
default). On final failure the runner does **not** raise — it returns an
:class:`EvalResponse` with the ``error`` field populated, per the brief
("DO NOT silently skip items. Log the failure and surface it in the
manifest.").

Tests inject a mock client via the ``client`` constructor arg, so unit
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


class AnthropicRunner(Runner):
    """Runner for Anthropic Claude models via the Messages API."""

    name: str = "anthropic"
    DEFAULT_MODEL: str = "claude-sonnet-4-6"

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
        """Initialize the Anthropic runner.

        Args:
            model: Anthropic model id (default ``claude-sonnet-4-6``).
            temperature: Sampling temperature (default 0.0).
            top_p: Nucleus-sampling cutoff (default 1.0).
            max_tokens: Cap on response length.
            api_key: Override ``ANTHROPIC_API_KEY`` from the environment.
            max_retries: Number of attempts before returning an error
                response. Each attempt doubles the wait.
            initial_backoff_s: Wait between attempt 1 and attempt 2.
                Doubled on each subsequent retry.
            client: An ``anthropic.Anthropic`` instance (or a test
                double). When None, a real client is constructed using
                the api key + environment.
        """
        super().__init__(model, temperature=temperature, top_p=top_p)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        if client is not None:
            self._client: Any = client
        else:
            from anthropic import Anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=key) if key else Anthropic()

    def run(self, item: MCQItem) -> EvalResponse:
        """Send ``item`` through the Anthropic Messages API.

        See :meth:`eval.runners.base.Runner.run` for the contract.
        """
        prompt = format_mcq(item)
        started = time.perf_counter()
        last_err: str | None = None
        delay = self.initial_backoff_s

        for attempt in range(1, self.max_retries + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    messages=[{"role": "user", "content": prompt}],
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
            except Exception as e:  # noqa: BLE001 — runners must never raise
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
        """Concatenate the text from every text-block in a Messages response.

        The Anthropic SDK returns ``msg.content`` as a list of typed content
        blocks. We pull ``text`` from any block that has one and stitch them
        together.
        """
        parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
