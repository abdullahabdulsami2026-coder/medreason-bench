"""MedQA-USMLE loader — keyword regex filter for cardio + autoimmune.

Provenance: Jin et al. 2020, "What Disease does this Patient Have? A
Large-scale Open Domain Question Answering Dataset from Medical Exams."
https://github.com/jind11/MedQA

Source: HuggingFace ``bigbio/med_qa`` config
``med_qa_en_4options_bigbio_qa`` — the 4-option English subset in the
BigBio common schema (``id``, ``question``, ``choices``, ``answer``,
``context``). License: MIT.

MedQA has no native subspecialty tags; the loader applies the shared
:mod:`data.filters.keywords` regex to the full prompt + options.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from data.filters.keywords import matches_subspecialty, specialty_for
from eval.schemas import MCQItem

DATASET_ID: str = "bigbio/med_qa"
DATASET_CONFIG: str = "med_qa_en_4options_bigbio_qa"
LETTERS: tuple[str, str, str, str] = ("A", "B", "C", "D")


def load_medqa(
    split: str = "test",
    *,
    filter_subspecialty: bool = True,
    limit: int | None = None,
) -> Iterator[MCQItem]:
    """Stream MedQA-USMLE items as :class:`~eval.schemas.MCQItem`.

    Args:
        split: HF split — ``"train"``, ``"validation"``, or ``"test"``.
            Default ``test`` because MedQA's test split is the standard
            evaluation slice in the literature.
        filter_subspecialty: Keep only items whose text matches the
            cardiology/autoimmune keyword filter.
        limit: Cap on yielded items (post-filter).

    Yields:
        :class:`MCQItem` whose ``metadata`` contains ``specialty``
        (cardiology/autoimmune/None).
    """
    from datasets import load_dataset  # lazy import

    ds: Iterable[dict[str, Any]] = load_dataset(DATASET_ID, name=DATASET_CONFIG, split=split)
    yielded = 0
    for row in ds:
        item = _row_to_mcq(row)
        if item is None:
            continue
        if filter_subspecialty:
            haystack = " ".join([item.stem, item.question, *item.options.values()])
            if not matches_subspecialty(haystack):
                continue
        yield item
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _row_to_mcq(row: dict[str, Any]) -> MCQItem | None:
    """Convert one BigBio-shaped MedQA row to :class:`MCQItem`.

    Returns ``None`` when ``choices`` is empty or has more than 4 entries
    (MedQA-USMLE-4-options uses exactly 4; defensive against config
    mix-ups). The correct option letter is derived by matching
    ``answer[0]`` text against the choices list.
    """
    choices = row.get("choices") or []
    if not isinstance(choices, list) or not 1 <= len(choices) <= len(LETTERS):
        return None

    options = {LETTERS[i]: str(c) for i, c in enumerate(choices)}

    answers = row.get("answer") or []
    correct: str | None = None
    if isinstance(answers, list) and answers:
        ans_text = str(answers[0]).strip()
        for letter, text in options.items():
            if text.strip() == ans_text:
                correct = letter
                break

    stem = str(row.get("context") or "")
    question = str(row.get("question") or "")

    item_id = row.get("id") or row.get("question_id") or row.get("document_id") or "unknown"

    return MCQItem(
        id=f"medqa_{item_id}",
        source="medqa",
        stem=stem,
        question=question,
        options=options,
        correct=correct,
        metadata={
            "specialty": specialty_for(" ".join([stem, question, *options.values()])),
        },
    )
