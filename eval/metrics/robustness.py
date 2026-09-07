"""Robustness across vignette variants: consistency, flip rate, robustness gap.

All functions take a results frame (see ``eval.metrics.__init__``) for a
single model+seed slice or pooled rows; grouping across models is the
caller's job. Variant rows are judged against their own key, so
``correct_override`` on mixed variants is honored automatically.

- consistency = |cases whose base answer == every answer-preserving
  variant answer| / |cases with >= 1 answer-preserving variant|
- flip rate = |answer-preserving variants answered wrong whose base was
  right| / |answer-preserving variants whose base was right|
- robustness gap = base accuracy - mean variant accuracy
"""

from __future__ import annotations

import pandas as pd


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[~df["is_variant"]].set_index("case_id")
    variants = df[df["is_variant"]]
    return base, variants


def consistency_rate(df: pd.DataFrame) -> dict[str, float]:
    """Share of cases answering identically on base + all answer-preserving variants.

    Answer-changing variants are excluded: an ideal model *should* flip
    on them, so identity is not the right expectation there.
    """
    base, variants = _split(df)
    keep = variants[variants["answer_preserving"]]
    n_cases = 0
    n_consistent = 0
    for case_id, sub in keep.groupby("case_id"):
        if case_id not in base.index:
            continue
        n_cases += 1
        answers = set(sub["parsed_answer"].fillna("NONE"))
        base_answer = base.loc[case_id, "parsed_answer"]
        if answers == {base_answer if pd.notna(base_answer) else "NONE"}:
            n_consistent += 1
    return {
        "n_cases": float(n_cases),
        "consistency_rate": n_consistent / n_cases if n_cases else 0.0,
    }


def flip_rate(df: pd.DataFrame) -> dict[str, float]:
    """Among answer-preserving variants whose base item was answered correctly,
    the share answered incorrectly."""
    base, variants = _split(df)
    keep = variants[variants["answer_preserving"]].copy()
    keep["base_correct"] = keep["case_id"].map(base["is_correct"])
    at_risk = keep[keep["base_correct"] == True]  # noqa: E712 — NaN-safe filter
    n = len(at_risk)
    flips = int((~at_risk["is_correct"]).sum())
    return {"n_at_risk": float(n), "flip_rate": flips / n if n else 0.0}


def robustness_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Base accuracy minus variant accuracy, overall and per perturbation type/subtype.

    Rows: one for "all" plus one per perturbation_type and one per
    clinical perturbation_subtype present in the frame.
    """
    base, variants = _split(df)
    base_acc = float(base["is_correct"].mean()) if len(base) else 0.0

    def row(label: str, sub: pd.DataFrame) -> dict[str, float | str]:
        acc = float(sub["is_correct"].mean()) if len(sub) else 0.0
        return {
            "slice": label,
            "n_variants": float(len(sub)),
            "base_accuracy": base_acc,
            "variant_accuracy": acc,
            "gap": base_acc - acc,
        }

    rows = [row("all", variants)]
    for ptype, sub in variants.groupby("perturbation_type"):
        rows.append(row(str(ptype), sub))
    for subtype, sub in variants.dropna(subset=["perturbation_subtype"]).groupby(
        "perturbation_subtype"
    ):
        rows.append(row(f"clinical:{subtype}", sub))
    return pd.DataFrame(rows)
