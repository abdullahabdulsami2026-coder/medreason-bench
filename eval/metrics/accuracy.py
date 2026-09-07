"""Top-1 accuracy on MCQ items.

For non-adversarial items (``correct`` is set), the response is correct iff
``parsed_answer == correct`` (case-insensitive). For adversarial items
(``correct`` is ``None``), the response is correct iff the model abstained
(``parsed_answer is None``); picking any letter on an adversarial item
counts as wrong.

Formula:

    accuracy = n_correct / n

where ``n`` is the number of responses that match a ground-truth item by
``item_id``. Responses without a matching ground-truth item are skipped
and not counted toward ``n``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from eval.schemas import EvalResponse, MCQItem


def top1_accuracy(
    responses: Iterable[EvalResponse],
    ground_truth: Iterable[MCQItem],
) -> dict[str, float]:
    """Compute top-1 accuracy on a parallel iterable of responses + items.

    Args:
        responses: Iterable of model responses; matched to ground truth
            by ``item_id``.
        ground_truth: Iterable of MCQ items with the answer key.

    Returns:
        ``{"n": int, "n_correct": int, "accuracy": float}``. ``accuracy``
        is 0.0 when there are no matched responses.
    """
    by_item: dict[str, MCQItem] = {item.id: item for item in ground_truth}
    n = 0
    n_correct = 0
    for resp in responses:
        truth = by_item.get(resp.item_id)
        if truth is None:
            continue
        n += 1
        if _is_correct(resp, truth):
            n_correct += 1
    accuracy = n_correct / n if n > 0 else 0.0
    return {"n": float(n), "n_correct": float(n_correct), "accuracy": accuracy}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    center = (p + z²/2n) / (1 + z²/n)
    halfwidth = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)

    Returns ``(0.0, 1.0)`` when ``n == 0``.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def accuracy_summary(df: pd.DataFrame) -> dict[str, float]:
    """Accuracy with Wilson 95% CI over a results frame (or any subset)."""
    n = len(df)
    k = int(df["is_correct"].sum())
    lo, hi = wilson_ci(k, n)
    return {"n": float(n), "accuracy": k / n if n else 0.0, "ci_lo": lo, "ci_hi": hi}


def accuracy_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Accuracy + Wilson CI per group (e.g. ["model", "specialty"])."""
    rows = []
    for group, sub in df.groupby(keys, dropna=False):
        key = group if isinstance(group, tuple) else (group,)
        rows.append(dict(zip(keys, key, strict=True)) | accuracy_summary(sub))
    return pd.DataFrame(rows)


def macro_f1(df: pd.DataFrame) -> float:
    """Macro-F1 over answer categories (option letters plus NONE for abstention).

    Classes are taken from the ground truth; per-class F1 = 2PR/(P+R)
    with 0 when a class is never predicted or never present.
    """
    truth = df["correct"].fillna("NONE")
    pred = df["parsed_answer"].fillna("NONE")
    scores = []
    for cls in sorted(truth.unique()):
        tp = int(((truth == cls) & (pred == cls)).sum())
        fp = int(((truth != cls) & (pred == cls)).sum())
        fn = int(((truth == cls) & (pred != cls)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mcnemar_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Exact McNemar test for every model pair on their shared items.

    b = items only model A got right, c = items only model B got right;
    p = exact binomial two-sided test of b against Binomial(b+c, 0.5).
    Seeds are pooled by majority correctness per item before pairing.
    """
    from scipy.stats import binomtest

    per_item = (
        df.groupby(["model", "item_id"])["is_correct"].mean().ge(0.5).rename("right").reset_index()
    )
    models = sorted(per_item["model"].unique())
    rows = []
    for i, a in enumerate(models):
        for b_model in models[i + 1 :]:
            wa = per_item[per_item["model"] == a].set_index("item_id")["right"]
            wb = per_item[per_item["model"] == b_model].set_index("item_id")["right"]
            shared = wa.index.intersection(wb.index)
            b = int((wa[shared] & ~wb[shared]).sum())
            c = int((~wa[shared] & wb[shared]).sum())
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            rows.append(
                {
                    "model_a": a,
                    "model_b": b_model,
                    "n_shared": len(shared),
                    "a_only_right": b,
                    "b_only_right": c,
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(
        rows, columns=["model_a", "model_b", "n_shared", "a_only_right", "b_only_right", "p_value"]
    )


def _is_correct(resp: EvalResponse, truth: MCQItem) -> bool:
    """Decide whether a single response matches the truth.

    Adversarial items (``truth.correct is None``) require the model to
    abstain (``resp.parsed_answer is None``). Otherwise the parsed letter
    must match the ground-truth letter case-insensitively.
    """
    if truth.correct is None:
        return resp.parsed_answer is None
    if resp.parsed_answer is None:
        return False
    return resp.parsed_answer.upper() == truth.correct.upper()
