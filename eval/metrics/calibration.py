"""Calibration: Brier score and 10-bin Expected Calibration Error.

Confidence is the verbalized 0-100 self-report divided by 100 (method
recorded in the run manifest as ``confidence_method``). Both metrics
are computed over *answered* rows with a parseable confidence;
``coverage`` reports how much of the frame that is.

- Brier = mean((confidence - is_correct)²)
- ECE   = Σ_b (n_b / N) · |acc_b - conf_b| over 10 equal-width bins
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["abstained"] & df["confidence"].notna()]


def brier_score(df: pd.DataFrame) -> dict[str, float]:
    scored = _scored(df)
    if scored.empty:
        return {"n": 0.0, "coverage": 0.0, "brier": float("nan")}
    err = (scored["confidence"] - scored["is_correct"].astype(float)) ** 2
    return {
        "n": float(len(scored)),
        "coverage": len(scored) / len(df) if len(df) else 0.0,
        "brier": float(err.mean()),
    }


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bin confidence vs accuracy — input for ECE and the reliability diagram.

    Bins are equal-width on [0, 1]; confidence 1.0 lands in the top bin.
    """
    scored = _scored(df)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(scored["confidence"], edges[1:-1]), 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        sub = scored[idx == b]
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": len(sub),
                "mean_confidence": float(sub["confidence"].mean()) if len(sub) else float("nan"),
                "accuracy": float(sub["is_correct"].mean()) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ece(df: pd.DataFrame) -> dict[str, float]:
    table = reliability_table(df)
    filled = table[table["n"] > 0]
    n_total = int(table["n"].sum())
    if n_total == 0:
        return {"n": 0.0, "ece": float("nan")}
    weighted = (
        filled["n"] / n_total * (filled["accuracy"] - filled["mean_confidence"]).abs()
    ).sum()
    return {"n": float(n_total), "ece": float(weighted)}
