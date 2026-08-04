"""Tests for ``eval.metrics.accuracy.top1_accuracy``."""

from __future__ import annotations

import pytest

from eval.metrics.accuracy import top1_accuracy
from eval.schemas import EvalResponse, MCQItem


def make_item(item_id: str, correct: str | None) -> MCQItem:
    return MCQItem(
        id=item_id,
        source="test",
        stem="stem",
        question="q",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct=correct,
    )


def make_resp(item_id: str, parsed: str | None) -> EvalResponse:
    return EvalResponse(
        item_id=item_id,
        model="test-model",
        raw_response="x",
        parsed_answer=parsed,
        confidence=None,
        rationale=None,
        elapsed_ms=0,
    )


def test_perfect_score() -> None:
    items = [make_item("1", "A"), make_item("2", "B"), make_item("3", "C")]
    resps = [make_resp("1", "A"), make_resp("2", "B"), make_resp("3", "C")]
    out = top1_accuracy(resps, items)
    assert out["n"] == 3
    assert out["n_correct"] == 3
    assert out["accuracy"] == pytest.approx(1.0)


def test_zero_score() -> None:
    items = [make_item("1", "A"), make_item("2", "B")]
    resps = [make_resp("1", "B"), make_resp("2", "A")]
    out = top1_accuracy(resps, items)
    assert out["n_correct"] == 0
    assert out["accuracy"] == pytest.approx(0.0)


def test_partial_score() -> None:
    items = [make_item(str(i), "A") for i in range(4)]
    resps = [make_resp(str(i), "A" if i < 3 else "B") for i in range(4)]
    out = top1_accuracy(resps, items)
    assert out["n"] == 4
    assert out["n_correct"] == 3
    assert out["accuracy"] == pytest.approx(0.75)


def test_empty_inputs_return_zero_accuracy() -> None:
    out = top1_accuracy([], [])
    assert out == {"n": 0.0, "n_correct": 0.0, "accuracy": 0.0}


def test_responses_without_matching_truth_are_skipped() -> None:
    items = [make_item("1", "A")]
    resps = [make_resp("1", "A"), make_resp("orphan", "A")]
    out = top1_accuracy(resps, items)
    assert out["n"] == 1


def test_case_insensitive_match() -> None:
    items = [make_item("1", "A")]
    resps = [make_resp("1", "a")]
    out = top1_accuracy(resps, items)
    assert out["n_correct"] == 1


def test_unparseable_answer_counts_as_wrong() -> None:
    items = [make_item("1", "A")]
    resps = [make_resp("1", None)]
    out = top1_accuracy(resps, items)
    assert out["n_correct"] == 0


def test_adversarial_correct_when_model_abstains() -> None:
    # Adversarial: truth.correct is None — model is correct iff it abstains.
    items = [make_item("adv", None)]
    resps = [make_resp("adv", None)]
    out = top1_accuracy(resps, items)
    assert out["n_correct"] == 1
    assert out["accuracy"] == pytest.approx(1.0)


def test_adversarial_wrong_when_model_picks() -> None:
    items = [make_item("adv", None)]
    resps = [make_resp("adv", "A")]
    out = top1_accuracy(resps, items)
    assert out["n_correct"] == 0


def test_adversarial_and_normal_mixed() -> None:
    items = [
        make_item("normal_correct", "A"),
        make_item("normal_wrong", "B"),
        make_item("adv_abstained", None),
        make_item("adv_picked", None),
    ]
    resps = [
        make_resp("normal_correct", "A"),  # correct
        make_resp("normal_wrong", "C"),  # wrong
        make_resp("adv_abstained", None),  # correct (abstained on adversarial)
        make_resp("adv_picked", "B"),  # wrong (picked on adversarial)
    ]
    out = top1_accuracy(resps, items)
    assert out["n"] == 4
    assert out["n_correct"] == 2
    assert out["accuracy"] == pytest.approx(0.5)
