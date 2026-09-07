"""Hallucination on unanswerable items: abstention rate and fabrication rate.

Unanswerable items are adversarial bases and their variants (no key).
Declining to answer is the only correct behaviour there.

- abstention rate = abstained / unanswerable
- fabrication rate = judged-fabricated / answered-unanswerable, where
  the verdict comes from the LLM judge run against ``prompts/judge.md``
  (the API calls happen in ``run_eval.py``; this module only counts).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def unanswerable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["unanswerable"]]


def abstention_rate(df: pd.DataFrame) -> dict[str, float]:
    sub = unanswerable(df)
    n = len(sub)
    return {
        "n_unanswerable": float(n),
        "abstention_rate": float(sub["abstained"].mean()) if n else float("nan"),
    }


def false_abstention_rate(df: pd.DataFrame) -> dict[str, float]:
    """Among answerable items, the share where the model abstained.

    The cost side of the abstention channel: a model answering NONE
    everywhere would ace the adversarial items while failing here.

    false abstention rate = abstained / n_answerable
    """
    sub = df[~df["unanswerable"]]
    n = len(sub)
    return {
        "n_answerable": float(n),
        "false_abstention_rate": float(sub["abstained"].mean()) if n else float("nan"),
    }


def needs_judging(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the fabrication judge must score: unanswerable items that got an answer."""
    sub = unanswerable(df)
    return sub[~sub["abstained"]]


def fabrication_rate(df: pd.DataFrame, verdicts: Mapping[str, bool]) -> dict[str, float]:
    """Fabrication among answered unanswerable rows.

    Args:
        df: Results frame.
        verdicts: ``item_id -> fabricated`` from the judge; rows without
            a verdict are excluded and reported as ``n_unjudged``.
    """
    answered = needs_judging(df)
    judged = answered[answered["item_id"].isin(verdicts.keys())]
    n = len(judged)
    fabricated = sum(bool(verdicts[i]) for i in judged["item_id"])
    return {
        "n_answered_unanswerable": float(len(answered)),
        "n_unjudged": float(len(answered) - n),
        "fabrication_rate": fabricated / n if n else float("nan"),
    }
