"""Cardiology + autoimmune keyword filter for dataset items.

Datasets without native subspecialty tagging (MedQA, PubMedQA) need a text
filter; datasets that ARE tagged (MedMCQA) use the same filter as a
cross-check against malformed/missing topic labels. A match anywhere in
the stem, question, options, or supplied context (mesh terms,
topic_name) marks the item as in-scope for MedReason-Bench.

The keyword lists trade specificity for recall. Borderline matches and
false positives are inspected in
``notebooks/02_subspecialty_filtering.ipynb`` and pruned per dataset
when the v1.0 filter list is frozen.
"""

from __future__ import annotations

import re

CARDIOLOGY_KEYWORDS: tuple[str, ...] = (
    "cardiac",
    "cardiology",
    "cardiovascular",
    "myocard",
    "coronary",
    "angina",
    "infarct",
    "ischem",
    "atrial fibrillation",
    "atrial flutter",
    "ventricular tachycardia",
    "ventricular fibrillation",
    "arrhythm",
    "valvular",
    "mitral",
    "aortic stenos",
    "aortic regurgit",
    "tricuspid",
    "pulmonic",
    "endocard",
    "pericard",
    "atherosclero",
    "hypertensive heart",
    "stemi",
    "nstemi",
    "echocard",
    "troponin",
    "natriuretic peptide",
    "heart failure",
    "cardiomyopath",
    "pulmonary edema",
    "ace inhibitor",
    "beta-blocker",
    "statin",
)

AUTOIMMUNE_KEYWORDS: tuple[str, ...] = (
    "autoimmune",
    "rheumatoid",
    "rheumatology",
    "lupus",
    "sle ",
    "scleroderma",
    "sjogren",
    "dermatomyositis",
    "polymyositis",
    "vasculit",
    "wegener",
    "churg-strauss",
    "polymyalgia",
    "takayasu",
    "behcet",
    "antiphospholipid",
    "spondyloarthr",
    "ankylos",
    "spondylit",
    "psoriatic arthritis",
    "anti-ccp",
    "anti-ro",
    "anti-la",
    "anti-dsdna",
    "antinuclear antibod",
    "anca",
    "hydroxychloroquine",
    "methotrexate",
    "immunolog",
    "complement deficiency",
    "ssa",
    "ssb",
    "giant cell arteritis",
)

_ALL_KEYWORDS: tuple[str, ...] = (*CARDIOLOGY_KEYWORDS, *AUTOIMMUNE_KEYWORDS)

SUBSPECIALTY_REGEX: re.Pattern[str] = re.compile(
    "|".join(re.escape(k) for k in _ALL_KEYWORDS),
    re.IGNORECASE,
)


def matches_subspecialty(text: str) -> bool:
    """Return True iff ``text`` mentions a cardiology or autoimmune keyword.

    Args:
        text: Free-text to scan (typically the concatenation of an item's
            stem, question, options, and any tagged context).

    Returns:
        True if at least one keyword matches (case-insensitive).
    """
    return bool(SUBSPECIALTY_REGEX.search(text))


def specialty_for(text: str) -> str | None:
    """Return ``"cardiology"``, ``"autoimmune"``, or ``None``.

    Returns the first specialty whose keyword set matches the text. If
    both match, cardiology wins (chosen by enumeration order). Used for
    metadata tagging only — :func:`matches_subspecialty` already handles
    inclusion.
    """
    text_l = text.lower()
    for kw in CARDIOLOGY_KEYWORDS:
        if kw in text_l:
            return "cardiology"
    for kw in AUTOIMMUNE_KEYWORDS:
        if kw in text_l:
            return "autoimmune"
    return None
