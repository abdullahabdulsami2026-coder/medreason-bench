"""Confidence-elicitation suffix used for calibration metrics.

The suffix instructs the model to emit its answer in a fixed structured form so
the runner can parse three things deterministically:

    ANSWER: <letter>
    CONFIDENCE: <0-100>
    RATIONALE: <2-3 sentences>

Confidence is the model's self-reported probability of being correct, on a
0-100 scale. Brier score and Expected Calibration Error (Phase 1+) consume
this directly.

This suffix is appended verbatim to every MCQ prompt. See
``eval.prompts.mcq.format_mcq``.
"""

from __future__ import annotations

import re

CONFIDENCE_SUFFIX: str = (
    "After your answer, report your confidence as an integer 0-100. "
    "Calibrate honestly: say 50 if unsure, say 95 only if near-certain.\n\n"
    "Format your response EXACTLY as:\n"
    "ANSWER: <letter, or NONE if no option is supported by the vignette>\n"
    "CONFIDENCE: <0-100>\n"
    "RATIONALE: <2-3 sentences>\n"
)

# NONE must be tried before the single-letter branch, otherwise
# "ANSWER: None" would parse as the letter N.
_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(NONE|[A-Z])\b", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*(\d{1,3})", re.IGNORECASE)
_RATIONALE_RE = re.compile(
    r"RATIONALE\s*:\s*(.+?)(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_confidence_response(raw: str) -> tuple[str | None, int | None, str | None]:
    """Parse a model's structured response.

    Args:
        raw: The full text response from the model.

    Returns:
        A tuple ``(answer_letter, confidence_int, rationale)``. Each field
        is ``None`` if the corresponding section was absent or unparseable.
        ``ANSWER: NONE`` (the abstention channel) also yields ``None`` for
        the answer. ``confidence_int`` is clamped to ``[0, 100]``.
    """
    answer_match = _ANSWER_RE.search(raw)
    confidence_match = _CONFIDENCE_RE.search(raw)
    rationale_match = _RATIONALE_RE.search(raw)

    answer: str | None = answer_match.group(1).upper() if answer_match else None
    if answer == "NONE":
        answer = None

    confidence: int | None
    if confidence_match is None:
        confidence = None
    else:
        confidence = max(0, min(100, int(confidence_match.group(1))))

    rationale: str | None = rationale_match.group(1).strip() if rationale_match else None
    return answer, confidence, rationale
