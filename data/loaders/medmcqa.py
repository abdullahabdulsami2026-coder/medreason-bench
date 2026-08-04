"""MedMCQA loader — filters to cardiology / rheumatology / immunology.

Provenance: Pal et al. 2022, "MedMCQA: A Large-scale Multi-Subject
Multi-Choice Dataset for Medical domain Question Answering."
https://medmcqa.github.io/

Source: HuggingFace ``openlifescienceai/medmcqa`` (mirror of the
official dataset). License: MIT.

Each row carries a ``topic_name`` string; we keep rows whose topic_name
or full text matches the shared cardiology/autoimmune keyword filter
(see :mod:`data.filters.keywords`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from data.filters.keywords import matches_subspecialty, specialty_for
from eval.schemas import MCQItem

DATASET_ID: str = "openlifescienceai/medmcqa"
LETTERS: tuple[str, str, str, str] = ("A", "B", "C", "D")


def load_medmcqa(
    split: str = "validation",
    *,
    filter_subspecialty: bool = True,
    limit: int | None = None,
) -> Iterator[MCQItem]:
    """Stream MedMCQA items as :class:`~eval.schemas.MCQItem`.

    Args:
        split: HF split — ``"train"`` (~183k), ``"validation"`` (~4k),
            or ``"test"`` (~6k). Default validation: small enough for
            laptop EDA, large enough for a real eval slice.
        filter_subspecialty: Keep only items mentioning cardiology or
            autoimmune keywords (in topic_name, question, or options).
        limit: Cap on yielded items (post-filter). ``None`` = unlimited.

    Yields:
        :class:`MCQItem` whose ``metadata`` contains ``subject_name``,
        ``topic_name``, ``specialty`` (cardiology/autoimmune/None),
        ``explanation``, and ``choice_type``.
    """
    from datasets import load_dataset  # lazy import keeps test startup fast

    ds: Iterable[dict[str, Any]] = load_dataset(DATASET_ID, split=split)
    yielded = 0
    for row in ds:
        item = _row_to_mcq(row)
        if item is None:
            continue
        if filter_subspecialty:
            haystack = " ".join(
                [
                    item.question,
                    *item.options.values(),
                    str(row.get("topic_name") or ""),
                ]
            )
            if not matches_subspecialty(haystack):
                continue
        yield item
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _row_to_mcq(row: dict[str, Any]) -> MCQItem | None:
    """Convert one MedMCQA row to :class:`MCQItem`; return ``None`` if malformed.

    MedMCQA bundles the clinical scenario into the ``question`` field
    (no separate stem), so we leave ``MCQItem.stem`` empty and put the
    full prompt in ``MCQItem.question``.
    """
    options = {
        "A": row.get("opa") or "",
        "B": row.get("opb") or "",
        "C": row.get("opc") or "",
        "D": row.get("opd") or "",
    }
    if not all(options.values()):
        return None

    cop = row.get("cop")
    # treat unknown/out-of-range answers as adversarial (model should abstain)
    correct: str | None = LETTERS[cop] if isinstance(cop, int) and 0 <= cop < len(LETTERS) else None

    haystack = " ".join(
        [
            str(row.get("question") or ""),
            *options.values(),
            str(row.get("topic_name") or ""),
        ]
    )
    return MCQItem(
        id=f"medmcqa_{row.get('id') or 'unknown'}",
        source="medmcqa",
        stem="",
        question=str(row.get("question") or ""),
        options=options,
        correct=correct,
        metadata={
            "subject_name": row.get("subject_name"),
            "topic_name": row.get("topic_name"),
            "specialty": specialty_for(haystack),
            "explanation": row.get("exp"),
            "choice_type": row.get("choice_type"),
        },
    )
