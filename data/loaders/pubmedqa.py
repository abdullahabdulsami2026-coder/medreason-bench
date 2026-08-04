"""PubMedQA loader — MeSH-term filter for cardio + autoimmune.

Provenance: Jin et al. 2019, "PubMedQA: A Dataset for Biomedical
Research Question Answering."  https://pubmedqa.github.io/

Source: HuggingFace ``qiaojin/PubMedQA`` config ``pqa_labeled``
(1,000 expert-labeled yes/no/maybe items). License: MIT.

PubMedQA items are converted to a 3-option MCQ
(``A=yes, B=no, C=maybe``); the "maybe" option lets the metric capture
abstention/uncertainty without us having to invent adversarial items
for this dataset. The filter checks MeSH terms (``context["meshes"]``)
plus the question text.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from data.filters.keywords import matches_subspecialty, specialty_for
from eval.schemas import MCQItem

DATASET_ID: str = "qiaojin/PubMedQA"
DATASET_CONFIG: str = "pqa_labeled"

OPTIONS: dict[str, str] = {"A": "yes", "B": "no", "C": "maybe"}
DECISION_TO_LETTER: dict[str, str] = {"yes": "A", "no": "B", "maybe": "C"}


def load_pubmedqa(
    split: str = "train",
    *,
    filter_subspecialty: bool = True,
    limit: int | None = None,
) -> Iterator[MCQItem]:
    """Stream PubMedQA items as :class:`~eval.schemas.MCQItem` (yes/no/maybe MCQ).

    Args:
        split: HF split. ``pqa_labeled`` only ships a single ``train``
            split with 1,000 items; we expose ``split`` for other
            configs (``pqa_artificial``, ``pqa_unlabeled``) callers may
            override.
        filter_subspecialty: Keep only items whose MeSH terms or
            question mention cardiology / autoimmune keywords.
        limit: Cap on yielded items (post-filter).

    Yields:
        :class:`MCQItem` with options ``{A: yes, B: no, C: maybe}``,
        and ``metadata`` containing ``meshes`` (list[str]),
        ``specialty``, and the original ``long_answer``.
    """
    from datasets import load_dataset  # lazy import

    ds: Iterable[dict[str, Any]] = load_dataset(DATASET_ID, name=DATASET_CONFIG, split=split)
    yielded = 0
    for row in ds:
        item = _row_to_mcq(row)
        if item is None:
            continue
        if filter_subspecialty:
            meshes = item.metadata.get("meshes") or []
            haystack = " ".join([item.stem, item.question, *map(str, meshes)])
            if not matches_subspecialty(haystack):
                continue
        yield item
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _row_to_mcq(row: dict[str, Any]) -> MCQItem | None:
    """Convert one PubMedQA row to :class:`MCQItem`.

    Returns ``None`` when the row has no question text. Unrecognised
    ``final_decision`` values (anything other than yes/no/maybe) are
    treated as adversarial (``correct=None``).
    """
    question = str(row.get("question") or "").strip()
    if not question:
        return None

    decision = str(row.get("final_decision") or "").lower().strip()
    correct: str | None = DECISION_TO_LETTER.get(decision)

    ctx = row.get("context") or {}
    if isinstance(ctx, dict):
        contexts = ctx.get("contexts") or []
        meshes = ctx.get("meshes") or []
    else:
        contexts = []
        meshes = []

    stem = " ".join(str(c) for c in contexts) if isinstance(contexts, list) else ""
    mesh_list: list[str] = [str(m) for m in meshes] if isinstance(meshes, list) else []

    return MCQItem(
        id=f"pubmedqa_{row.get('pubid') or 'unknown'}",
        source="pubmedqa",
        stem=stem,
        question=question,
        options=dict(OPTIONS),
        correct=correct,
        metadata={
            "meshes": mesh_list,
            "specialty": specialty_for(stem + " " + " ".join(mesh_list)),
            "long_answer": row.get("long_answer"),
        },
    )
