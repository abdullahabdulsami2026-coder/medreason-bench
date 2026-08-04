"""Tests for ``eval.pipeline``.

Covers the ``run_eval`` orchestrator + helper functions using a tiny
in-process ``FakeRunner`` so no network calls are issued.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.pipeline import (
    SAMPLE_ITEMS,
    aggregate_prompt_hash,
    get_git_sha,
    main,
    run_eval,
)
from eval.runners.base import Runner
from eval.schemas import EvalResponse, MCQItem, RunMetadata


class FakeRunner(Runner):
    """Returns a canned correct-A response for every item."""

    name: str = "fake"

    def __init__(self) -> None:
        super().__init__("fake-model")

    def run(self, item: MCQItem) -> EvalResponse:
        return EvalResponse(
            item_id=item.id,
            model=self.model,
            raw_response="ANSWER: A\nCONFIDENCE: 80\nRATIONALE: stub.",
            parsed_answer="A",
            confidence=80,
            rationale="stub.",
            elapsed_ms=1,
        )


def test_sample_items_are_5_cardio_plus_5_autoimmune() -> None:
    assert len(SAMPLE_ITEMS) == 10
    cardio = [it for it in SAMPLE_ITEMS if it.metadata.get("specialty") == "cardiology"]
    autoimmune = [it for it in SAMPLE_ITEMS if it.metadata.get("specialty") == "autoimmune"]
    assert len(cardio) == 5
    assert len(autoimmune) == 5
    # all items must have a correct answer letter from the option set
    for it in SAMPLE_ITEMS:
        assert it.correct in it.options


def test_aggregate_prompt_hash_is_stable_and_deterministic() -> None:
    h1 = aggregate_prompt_hash(SAMPLE_ITEMS[:3])
    h2 = aggregate_prompt_hash(SAMPLE_ITEMS[:3])
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_aggregate_prompt_hash_changes_on_different_items() -> None:
    h1 = aggregate_prompt_hash(SAMPLE_ITEMS[:3])
    h2 = aggregate_prompt_hash(SAMPLE_ITEMS[:4])
    assert h1 != h2


def test_get_git_sha_returns_a_string() -> None:
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_run_eval_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    runner = FakeRunner()
    items = SAMPLE_ITEMS[:3]
    jsonl_path, manifest_path, manifest, responses = run_eval(
        runner, items, "unit_test", out_dir=tmp_path
    )

    # files exist
    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert jsonl_path.parent == tmp_path
    assert manifest_path.parent == tmp_path

    # JSONL is parseable and matches the responses we got back
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 3
    parsed = [EvalResponse.model_validate_json(line) for line in lines]
    assert [r.item_id for r in parsed] == [it.id for it in items]
    assert all(r.parsed_answer == "A" for r in parsed)
    assert responses == parsed

    # manifest is valid + contains the right values
    m = RunMetadata.model_validate_json(manifest_path.read_text())
    assert m.model == "fake-model"
    assert m.model_version == "fake-model"
    assert m.dataset == "unit_test"
    assert m.n_items == 3
    assert m.temperature == 0.0
    assert m.top_p == 1.0
    assert m.seed == 42
    assert isinstance(m.prompt_hash, str)
    assert len(m.prompt_hash) == 64
    assert isinstance(m.git_sha, str)
    assert m.run_id  # non-empty
    assert m.total_elapsed_s is not None
    assert m.total_elapsed_s >= 0
    assert m == manifest


def test_run_eval_safe_filename_contains_model_and_dataset(tmp_path: Path) -> None:
    """Slashes in model / dataset should be sanitised in the filename."""
    runner = FakeRunner()
    runner.model = "openai/gpt-4o"  # exercise the slash-replace
    jsonl_path, manifest_path, _, _ = run_eval(
        runner, SAMPLE_ITEMS[:1], "subdir/label", out_dir=tmp_path
    )
    assert "/" not in jsonl_path.stem
    assert "openai_gpt-4o" in jsonl_path.stem
    assert "subdir_label" in jsonl_path.stem


def test_pipeline_smoke_and_dataset_are_mutually_exclusive() -> None:
    """argparse must reject `--smoke --dataset NAME` together (raises SystemExit)."""
    with pytest.raises(SystemExit):
        main(["--smoke", "--dataset", "medmcqa"])
