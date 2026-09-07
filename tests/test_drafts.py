"""Corpus invariants for answer-preserving variants + draft-file validation.

The robustness and fairness metrics assume that ``pure_demographic`` and
``clinical_perturbation`` variants never change the correct answer; these
tests enforce that for the corpus and for everything staged in
``data/vignettes/drafts/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data.vignettes.loader import dataset_version, load_vignettes
from eval.schemas import Demographics, DraftRecord, VignetteVariant

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DRAFTS_DIR: Path = REPO_ROOT / "data" / "vignettes" / "drafts"

# Emptied in v1.1.1: autoimmune_012's variant was reclassified to "mixed",
# resolving the one historical violation. Nothing may join this list.
GRANDFATHERED_OVERRIDES: set[tuple[str, str]] = set()


def all_drafts() -> list[DraftRecord]:
    if not DRAFTS_DIR.is_dir():
        return []
    return [
        DraftRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(DRAFTS_DIR.glob("*/*.json"))
    ]


# ── Corpus invariants ─────────────────────────────────────────────


def test_answer_preserving_variants_have_no_override() -> None:
    violations = []
    for v in load_vignettes():
        for var in v.variants:
            if (
                var.perturbation_type in ("pure_demographic", "clinical_perturbation")
                and var.correct_override is not None
                and (v.id, var.variant_id) not in GRANDFATHERED_OVERRIDES
            ):
                violations.append((v.id, var.variant_id))
    assert violations == []


def test_subtype_only_on_clinical_perturbation() -> None:
    for v in load_vignettes():
        for var in v.variants:
            if var.perturbation_type != "clinical_perturbation":
                assert var.perturbation_subtype is None, (v.id, var.variant_id)


def test_dataset_version_is_semver() -> None:
    parts = dataset_version().split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


# ── Draft files ───────────────────────────────────────────────────


def test_drafts_validate_and_preserve_answers() -> None:
    corpus = {v.id: v for v in load_vignettes()}
    for record in all_drafts():
        if record.status == "promoted":
            # Historical audit record — its content now lives in the corpus,
            # so collision/duplicate checks no longer apply.
            continue
        if record.vignette is not None:
            assert record.category == "adversarial"
            assert record.vignette.correct is None, record.draft_id
            assert record.vignette.rationale, record.draft_id
            assert record.vignette.id not in corpus, f"{record.draft_id} collides with corpus"
            continue
        assert record.variant is not None and record.parent_id is not None
        parent = corpus.get(record.parent_id)
        assert parent is not None, f"{record.draft_id}: parent {record.parent_id} not in corpus"
        var = record.variant
        assert var.correct_override is None, record.draft_id
        assert var.stem_override, record.draft_id
        assert var.rationale_for_variant, record.draft_id
        assert var.variant_id not in {x.variant_id for x in parent.variants}, record.draft_id
        base = parent.demographics.model_dump()
        changed = {
            k
            for k in ("age", "sex", "race", "ethnicity")
            if base[k] != getattr(var.demographics, k)
        }
        if record.category == "clinical_perturbation":
            assert var.perturbation_type == "clinical_perturbation"
            assert var.perturbation_subtype is not None, record.draft_id
            assert changed == set(), (record.draft_id, changed)
        else:
            assert var.perturbation_type == "pure_demographic"
            assert var.perturbation_subtype is None
            assert len(changed) == 1, (record.draft_id, changed)


# ── DraftRecord shape (synthetic) ─────────────────────────────────


def make_variant() -> VignetteVariant:
    return VignetteVariant(
        variant_id="paraphrase_test",
        demographics=Demographics(age=50, sex="female"),
        stem_override="stem",
        perturbation_type="clinical_perturbation",
        perturbation_subtype="paraphrase",
    )


def test_draft_record_requires_exactly_one_payload() -> None:
    with pytest.raises(ValidationError):
        DraftRecord(
            draft_id="x",
            category="clinical_perturbation",
            generator_model="m",
            generated_at="t",
        )


def test_draft_record_variant_requires_parent() -> None:
    with pytest.raises(ValidationError):
        DraftRecord(
            draft_id="x",
            category="clinical_perturbation",
            generator_model="m",
            generated_at="t",
            variant=make_variant(),
        )
    record = DraftRecord(
        draft_id="cardio_001__paraphrase",
        category="clinical_perturbation",
        parent_id="cardio_001",
        generator_model="m",
        generated_at="t",
        variant=make_variant(),
    )
    assert record.reviewed is False and record.status == "pending"


def test_variant_subtype_rejects_unknown_values() -> None:
    payload = make_variant().model_dump()
    payload["perturbation_subtype"] = "typo_subtype"
    with pytest.raises(ValidationError):
        VignetteVariant.model_validate(payload)
    round_trip = VignetteVariant.model_validate(json.loads(make_variant().model_dump_json()))
    assert round_trip.perturbation_subtype == "paraphrase"
