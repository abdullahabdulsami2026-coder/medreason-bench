"""Google Gemini runner via the google-genai SDK.

End-to-end runner for the Gemini API. Sends the formatted MCQ prompt,
parses the structured response (ANSWER / CONFIDENCE / RATIONALE), and
returns an :class:`~eval.schemas.EvalResponse`.

Retries transient failures with exponential backoff (1s → 2s → 4s → 8s
by default). On final failure the runner does **not** raise — it
returns an :class:`EvalResponse` with the ``error`` field populated.

Tests inject a mock client via the ``client`` constructor arg so unit
tests never touch the network.

Free tier (Google AI Studio): Gemini 2.5 Pro ~50 requests/day; Gemini
2.5 Flash ~1500/day. The runner does not internally rate-limit; the
caller is responsible for spacing requests.
"""

from __future__ import annotations

import os
import time
from typing import Any

from eval.prompts.confidence import parse_confidence_response
from eval.prompts.mcq import format_mcq
from eval.runners.base import Runner
from eval.schemas import EvalResponse, MCQItem


class GoogleRunner(Runner):
    """Runner for Google Gemini models via the google-genai SDK."""

    name: str = "google"
    DEFAULT_MODEL: str = "gemini-2.5-pro"

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
        """Initialize the Google Gemini runner.

        Args:
            model: Gemini model id (e.g., ``gemini-2.5-pro``,
                ``gemini-2.5-flash``).
            temperature: Sampling temperature (default 0.0).
            top_p: Nucleus-sampling cutoff (default 1.0).
            max_tokens: Cap on response length.
            api_key: Override ``GOOGLE_API_KEY`` from the environment.
            max_retries: Number of attempts before returning an error
                response.
            initial_backoff_s: Wait between attempt 1 and attempt 2.
                Doubled on each subsequent retry.
            client: A ``google.genai.Client`` instance (or test
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
            from google import genai

            key = api_key or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()

    def run(self, item: MCQItem) -> EvalResponse:
        """Send ``item`` through the Gemini API.

        See :meth:`eval.runners.base.Runner.run` for the contract.
        """
        prompt = format_mcq(item)
        started = time.perf_counter()
        last_err: str | None = None
        delay = self.initial_backoff_s

        for attempt in range(1, self.max_retries + 1):
            try:
                config = self._build_config()
                msg = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
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

    def _build_config(self) -> Any:
        """Build a ``GenerateContentConfig`` from the runner's sampling params.

        The config object lets the SDK pass temperature / top_p / max
        tokens to the model. Kept private so tests that inject a mock
        client are not coupled to the SDK's config-object structure.
        """
        try:
            from google.genai import types

            return types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
            )
        except Exception:  # noqa: BLE001 — fall through if SDK shape changes
            return {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_tokens,
            }

    @staticmethod
    def _extract_text(msg: Any) -> str:
        """Extract text from a Gemini ``GenerateContentResponse``.

        The new SDK exposes a ``text`` property; we fall back to walking
        the candidates → content → parts structure if ``text`` is
        absent (e.g., when a mock returns a partial shape).
        """
        text = getattr(msg, "text", None)
        if isinstance(text, str) and text:
            return text
        parts: list[str] = []
        for candidate in getattr(msg, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
