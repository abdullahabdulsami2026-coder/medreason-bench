"""Synthetic-fixture tests proving each metric family computes what it claims."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.metrics import calibration, fairness, hallucination, robustness
from eval.metrics.accuracy import accuracy_summary, macro_f1, mcnemar_pairs, wilson_ci

DEFAULTS: dict[str, Any] = {
    "model": "m",
    "seed": 0,
    "case_id": "c1",
    "is_variant": False,
    "variant_id": "",
    "perturbation_type": "",
    "perturbation_subtype": None,
    "changed_attr": "",
    "answer_preserving": True,
    "specialty": "cardiology",
    "difficulty": "medium",
    "correct": "A",
    "unanswerable": False,
    "parsed_answer": "A",
    "abstained": False,
    "answer_in_options": True,
    "is_correct": True,
    "confidence": np.nan,
    "error": None,
}


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([DEFAULTS | {"item_id": f"i{n}"} | r for n, r in enumerate(rows)])


# ── Accuracy ──────────────────────────────────────────────────────


def test_wilson_ci_known_value() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_accuracy_summary_counts() -> None:
    df = frame([{"is_correct": True}, {"is_correct": True}, {"is_correct": False}])
    out = accuracy_summary(df)
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["ci_lo"] < 2 / 3 < out["ci_hi"]


def test_macro_f1_treats_abstention_as_class() -> None:
    # Truth: A, A, NONE. Pred: A, B, NONE. F1(A)=2/3·2=0.8? P=1,R=.5→2/3; F1(NONE)=1.
    df = frame(
        [
            {"correct": "A", "parsed_answer": "A"},
            {"correct": "A", "parsed_answer": "B", "is_correct": False},
            {"correct": None, "parsed_answer": None, "unanswerable": True, "abstained": True},
        ]
    )
    assert macro_f1(df) == pytest.approx((2 / 3 + 1.0) / 2)


def test_mcnemar_discordant_pairs() -> None:
    rows = []
    for item, (a_right, b_right) in enumerate([(1, 1), (1, 0), (1, 0), (0, 0)]):
        rows.append({"model": "A", "item_id": f"i{item}", "is_correct": bool(a_right)})
        rows.append({"model": "B", "item_id": f"i{item}", "is_correct": bool(b_right)})
    df = pd.DataFrame([DEFAULTS | r for r in rows])
    out = mcnemar_pairs(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["a_only_right"], row["b_only_right"]) == (2, 0)
    assert row["p_value"] == pytest.approx(0.5)


# ── Robustness ────────────────────────────────────────────────────


def robustness_fixture() -> pd.DataFrame:
    return frame(
        [
            {"case_id": "c1", "parsed_answer": "A", "is_correct": True},
            {
                "case_id": "c1",
                "is_variant": True,
                "variant_id": "v1",
                "perturbation_type": "pure_demographic",
                "parsed_answer": "A",
                "is_correct": True,
            },
            {
                "case_id": "c1",
                "is_variant": True,
                "variant_id": "v2",
                "perturbation_type": "clinical_perturbation",
                "perturbation_subtype": "paraphrase",
                "parsed_answer": "B",
                "is_correct": False,
            },
            {"case_id": "c2", "parsed_answer": "B", "is_correct": False},
        ]
    )


def test_consistency_requires_all_preserving_variants_to_match() -> None:
    out = robustness.consistency_rate(robustness_fixture())
    assert out == {"n_cases": 1.0, "consistency_rate": 0.0}


def test_flip_rate_counts_only_correct_bases() -> None:
    out = robustness.flip_rate(robustness_fixture())
    assert out["n_at_risk"] == 2.0
    assert out["flip_rate"] == pytest.approx(0.5)


def test_robustness_gap_slices() -> None:
    gaps = robustness.robustness_gap(robustness_fixture()).set_index("slice")
    assert gaps.loc["all", "gap"] == pytest.approx(0.0)  # base 0.5, variants 0.5
    assert gaps.loc["clinical:paraphrase", "variant_accuracy"] == 0.0


def test_answer_changing_variants_excluded_from_consistency() -> None:
    df = robustness_fixture()
    df.loc[df["variant_id"] == "v2", "answer_preserving"] = False
    out = robustness.consistency_rate(df)
    assert out["consistency_rate"] == 1.0  # only v1 remains and matches


# ── Calibration ───────────────────────────────────────────────────


def calibration_fixture() -> pd.DataFrame:
    return frame(
        [
            {"confidence": 0.95, "is_correct": True},
            {"confidence": 0.95, "is_correct": True},
            {"confidence": 0.95, "is_correct": True},
            {"confidence": 0.55, "is_correct": False, "parsed_answer": "B"},
        ]
    )


def test_brier_known_value() -> None:
    out = calibration.brier_score(calibration_fixture())
    assert out["brier"] == pytest.approx((3 * 0.05**2 + 0.55**2) / 4)
    assert out["coverage"] == 1.0


def test_ece_two_bins() -> None:
    out = calibration.ece(calibration_fixture())
    assert out["ece"] == pytest.approx(0.75 * 0.05 + 0.25 * 0.55)


def test_calibration_excludes_abstentions_and_missing_confidence() -> None:
    df = frame(
        [
            {"confidence": 0.9, "is_correct": True},
            {"abstained": True, "parsed_answer": None, "confidence": 0.9},
            {"confidence": np.nan},
        ]
    )
    assert calibration.brier_score(df)["n"] == 1.0


# ── Hallucination ─────────────────────────────────────────────────


def hallucination_fixture() -> pd.DataFrame:
    return frame(
        [
            {
                "correct": None,
                "unanswerable": True,
                "parsed_answer": None,
                "abstained": True,
                "is_correct": True,
            },
            {
                "case_id": "c2",
                "correct": None,
                "unanswerable": True,
                "parsed_answer": "A",
                "is_correct": False,
            },
            {
                "case_id": "c3",
                "correct": None,
                "unanswerable": True,
                "parsed_answer": "B",
                "is_correct": False,
            },
            {"case_id": "c4"},
        ]
    )


def test_abstention_rate_over_unanswerable_only() -> None:
    out = hallucination.abstention_rate(hallucination_fixture())
    assert out["n_unanswerable"] == 3.0
    assert out["abstention_rate"] == pytest.approx(1 / 3)


def test_false_abstention_rate_over_answerable_only() -> None:
    df = frame(
        [
            {"parsed_answer": None, "abstained": True, "is_correct": False},
            {"case_id": "c2"},
            {"case_id": "c3"},
            {
                "case_id": "c4",
                "correct": None,
                "unanswerable": True,
                "parsed_answer": "A",
                "is_correct": False,
            },
        ]
    )
    out = hallucination.false_abstention_rate(df)
    assert out["n_answerable"] == 3.0
    assert out["false_abstention_rate"] == pytest.approx(1 / 3)


def test_fabrication_rate_uses_verdicts_and_reports_unjudged() -> None:
    df = hallucination_fixture()
    answered = hallucination.needs_judging(df)
    assert set(answered["item_id"]) == {"i1", "i2"}
    out = hallucination.fabrication_rate(df, {"i1": True})
    assert out["fabrication_rate"] == 1.0
    assert out["n_unjudged"] == 1.0


# ── Fairness ──────────────────────────────────────────────────────


def test_within_case_deltas_paired_and_flagged() -> None:
    rows = []
    for i in range(1, 4):  # three sex swaps, each base right / variant wrong
        rows.append({"case_id": f"c{i}", "is_correct": True})
        rows.append(
            {
                "case_id": f"c{i}",
                "is_variant": True,
                "variant_id": "sex",
                "perturbation_type": "pure_demographic",
                "changed_attr": "sex",
                "is_correct": False,
            }
        )
    rows.append({"case_id": "c9", "is_correct": True})
    rows.append(
        {
            "case_id": "c9",
            "is_variant": True,
            "variant_id": "age",
            "perturbation_type": "pure_demographic",
            "changed_attr": "age",
            "is_correct": True,
        }
    )
    df = frame(rows)
    deltas = fairness.within_case_deltas(df).set_index("changed_attr")
    assert deltas.loc["sex", "mean_delta"] == pytest.approx(-1.0)
    assert bool(deltas.loc["sex", "flagged"])  # CI is degenerate at -1, excludes 0
    assert deltas.loc["age", "n_pairs"] == 1
    assert not bool(deltas.loc["age", "flagged"])  # single pair: no CI, never flagged
    assert fairness.max_gap(fairness.within_case_deltas(df)) == pytest.approx(1.0)


def test_fairness_ignores_mixed_and_multi_attr_variants() -> None:
    df = frame(
        [
            {"case_id": "c1", "is_correct": True},
            {
                "case_id": "c1",
                "is_variant": True,
                "perturbation_type": "mixed",
                "changed_attr": "sex",
                "is_correct": False,
            },
            {
                "case_id": "c1",
                "is_variant": True,
                "perturbation_type": "pure_demographic",
                "changed_attr": "age+sex",
                "is_correct": False,
            },
        ]
    )
    deltas = fairness.within_case_deltas(df)
    assert list(deltas["changed_attr"]) == ["age+sex"]
