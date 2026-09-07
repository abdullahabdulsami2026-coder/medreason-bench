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
        if temperature != 0.0 and top_p != 1.0:
            raise ValueError(
                "Anthropic Claude 4.6+ models accept temperature or top_p, not both; "
                "leave one at its default (temperature=0.0 / top_p=1.0)"
            )
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

    def complete(self, prompt: str) -> str:
        """Send a raw prompt through the Messages API and return the text.

        Retries transient failures with the runner's backoff schedule and
        raises the last error after ``max_retries`` attempts. ``run``
        wraps this for MCQ evaluation; corpus-generation tooling
        (``scripts/generate_drafts.py``) calls it directly.
        """
        # Claude 4.6+ models reject requests that set both temperature and
        # top_p, so top_p is sent only when it deviates from the 1.0 default.
        sampling: dict[str, float] = {"temperature": self.temperature}
        if self.top_p != 1.0:
            sampling = {"top_p": self.top_p}
        last_err: Exception | None = None
        delay = self.initial_backoff_s
        for attempt in range(1, self.max_retries + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    **sampling,
                )
                return self._extract_text(msg)
            except Exception as e:  # noqa: BLE001 — retried, re-raised on final failure
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        assert last_err is not None
        raise last_err

    def run(self, item: MCQItem) -> EvalResponse:
        """Send ``item`` through the Anthropic Messages API.

        See :meth:`eval.runners.base.Runner.run` for the contract.
        """
        started = time.perf_counter()
        try:
            raw = self.complete(format_mcq(item))
        except Exception as e:  # noqa: BLE001 — runners must never raise
            return EvalResponse(
                item_id=item.id,
                model=self.model,
                raw_response="",
                parsed_answer=None,
                confidence=None,
                rationale=None,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        answer, confidence, rationale = parse_confidence_response(raw)
        return EvalResponse(
            item_id=item.id,
            model=self.model,
            raw_response=raw,
            parsed_answer=answer,
            confidence=confidence,
            rationale=rationale,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
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
