"""Tests for dataset loaders + the keyword filter + vignette converter.

HF ``datasets.load_dataset`` is monkey-patched to return in-memory
fixtures so tests never touch the network.
"""

from __future__ import annotations

from typing import Any

import datasets as hf_datasets
import pytest

from data.filters.keywords import (
    AUTOIMMUNE_KEYWORDS,
    CARDIOLOGY_KEYWORDS,
    matches_subspecialty,
    specialty_for,
)
from data.loaders import medmcqa, medqa, pubmedqa
from data.vignettes.schema import vignette_to_mcq_item, vignette_variant_to_mcq_item
from eval.schemas import Demographics, Vignette, VignetteVariant

# ── Keyword filter ────────────────────────────────────────────────


def test_matches_subspecialty_hits_cardio() -> None:
    assert matches_subspecialty("Acute myocardial infarction with ST elevation")
    assert matches_subspecialty("Patient has aortic stenosis")


def test_matches_subspecialty_hits_autoimmune() -> None:
    assert matches_subspecialty("Diagnosis of rheumatoid arthritis")
    assert matches_subspecialty("Lupus nephritis flare with anti-dsdna")


def test_matches_subspecialty_misses_unrelated() -> None:
    assert not matches_subspecialty("Acute appendicitis with fever and vomiting")
    assert not matches_subspecialty("Bacterial pneumonia, treat with amoxicillin")


def test_specialty_for_returns_correct_label() -> None:
    assert specialty_for("Heart failure with reduced ejection fraction") == "cardiology"
    assert specialty_for("Sjogren syndrome with sicca complex") == "autoimmune"
    assert specialty_for("Acute appendicitis") is None


def test_keyword_lists_are_non_empty() -> None:
    assert len(CARDIOLOGY_KEYWORDS) > 10
    assert len(AUTOIMMUNE_KEYWORDS) > 10


# ── MedMCQA loader ────────────────────────────────────────────────

SAMPLE_MEDMCQA_ROWS: list[dict[str, Any]] = [
    {
        "id": "m1",
        "question": "Most common cause of acute myocardial infarction is?",
        "opa": "Atherosclerosis",
        "opb": "Trauma",
        "opc": "Embolism",
        "opd": "Vasospasm",
        "cop": 0,
        "exp": "Atherosclerotic plaque rupture is the dominant aetiology.",
        "subject_name": "Medicine",
        "topic_name": "Cardiology",
        "choice_type": "single",
    },
    {
        "id": "m2",
        "question": "First-line serology for systemic lupus erythematosus?",
        "opa": "ANA",
        "opb": "Rheumatoid factor",
        "opc": "ESR",
        "opd": "CRP",
        "cop": 0,
        "exp": "ANA is the screening test for SLE.",
        "subject_name": "Medicine",
        "topic_name": "Immunology",
        "choice_type": "single",
    },
    {
        "id": "m3",
        "question": "Most common organism in community-acquired pneumonia?",
        "opa": "Strep pneumoniae",
        "opb": "Staph aureus",
        "opc": "Mycoplasma",
        "opd": "Legionella",
        "cop": 0,
        "exp": "Strep pneumoniae remains #1.",
        "subject_name": "Medicine",
        "topic_name": "Respiratory",
        "choice_type": "single",
    },
    {
        "id": "m4_malformed",
        "question": "Bad row, missing options",
        "opa": "",
        "opb": "B",
        "opc": "C",
        "opd": "D",
        "cop": 1,
        "subject_name": "Medicine",
        "topic_name": "Cardiology",
        "choice_type": "single",
    },
]


def _patch_load(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Replace ``datasets.load_dataset`` with a fixture that returns ``rows``."""

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)


def test_medmcqa_filter_keeps_cardio_and_autoimmune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDMCQA_ROWS)
    items = list(medmcqa.load_medmcqa(filter_subspecialty=True))
    ids = [it.id for it in items]
    assert "medmcqa_m1" in ids  # cardio
    assert "medmcqa_m2" in ids  # autoimmune
    assert "medmcqa_m3" not in ids  # respiratory — filtered out
    assert "medmcqa_m4_malformed" not in ids  # missing option dropped


def test_medmcqa_no_filter_returns_all_well_formed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDMCQA_ROWS)
    items = list(medmcqa.load_medmcqa(filter_subspecialty=False))
    assert len(items) == 3  # malformed dropped


def test_medmcqa_correct_letter_from_cop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDMCQA_ROWS[:2])
    items = list(medmcqa.load_medmcqa(filter_subspecialty=False))
    assert items[0].correct == "A"  # cop=0
    assert items[0].source == "medmcqa"
    assert items[0].metadata["topic_name"] == "Cardiology"
    assert items[0].metadata["specialty"] == "cardiology"


def test_medmcqa_limit_caps_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDMCQA_ROWS)
    items = list(medmcqa.load_medmcqa(filter_subspecialty=False, limit=1))
    assert len(items) == 1


def test_medmcqa_unknown_cop_becomes_adversarial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{**SAMPLE_MEDMCQA_ROWS[0], "id": "weird", "cop": 99}]
    _patch_load(monkeypatch, rows)
    items = list(medmcqa.load_medmcqa(filter_subspecialty=False))
    assert items[0].correct is None


# ── MedQA loader ──────────────────────────────────────────────────

SAMPLE_MEDQA_ROWS: list[dict[str, Any]] = [
    {
        "id": "q1",
        "question": "A 65 year old man has crushing chest pain and ST elevation. Diagnosis?",
        "choices": ["STEMI", "Pneumonia", "GERD", "Pneumothorax"],
        "answer": ["STEMI"],
        "context": "",
    },
    {
        "id": "q2",
        "question": "A 30 year old woman with malar rash and ANA. Most likely?",
        "choices": ["Lupus", "Rosacea", "Eczema", "Psoriasis"],
        "answer": ["Lupus"],
        "context": "",
    },
    {
        "id": "q3",
        "question": "A 40 year old with productive cough and fever. Treatment?",
        "choices": ["Amoxicillin", "Azithromycin", "Ibuprofen", "Paracetamol"],
        "answer": ["Amoxicillin"],
        "context": "",
    },
    {
        "id": "q4_bad",
        "question": "Too many choices",
        "choices": ["A", "B", "C", "D", "E"],
        "answer": ["A"],
        "context": "",
    },
]


def test_medqa_filter_and_correct_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDQA_ROWS)
    items = list(medqa.load_medqa(filter_subspecialty=True))
    ids = {it.id for it in items}
    assert "medqa_q1" in ids
    assert "medqa_q2" in ids
    assert "medqa_q3" not in ids
    assert "medqa_q4_bad" not in ids
    cardio = next(it for it in items if it.id == "medqa_q1")
    assert cardio.correct == "A"
    assert cardio.source == "medqa"
    assert cardio.metadata["specialty"] == "cardiology"


def test_medqa_no_filter_skips_only_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDQA_ROWS)
    items = list(medqa.load_medqa(filter_subspecialty=False))
    assert len(items) == 3  # q4_bad dropped (5 choices > 4)


def test_medqa_limit_caps_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, SAMPLE_MEDQA_ROWS)
    items = list(medqa.load_medqa(filter_subspecialty=False, limit=2))
    assert len(items) == 2


def test_medqa_unknown_answer_text_yields_none_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "q_weird",
            "question": "Cardiac question with mismatched answer",
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer": ["epsilon"],  # not in choices
            "context": "",
        }
    ]
    _patch_load(monkeypatch, rows)
    items = list(medqa.load_medqa(filter_subspecialty=False))
    assert items[0].correct is None


# ── PubMedQA loader ───────────────────────────────────────────────

SAMPLE_PUBMEDQA_ROWS: list[dict[str, Any]] = [
    {
        "pubid": 100,
        "question": "Does aspirin reduce mortality in acute myocardial infarction?",
        "context": {
            "contexts": ["Aspirin trials in AMI patients..."],
            "labels": ["BACKGROUND"],
            "meshes": ["Aspirin", "Myocardial Infarction", "Mortality"],
        },
        "long_answer": "Yes, aspirin reduces mortality...",
        "final_decision": "yes",
    },
    {
        "pubid": 101,
        "question": "Is hydroxychloroquine effective in lupus nephritis?",
        "context": {
            "contexts": ["RCT of hydroxychloroquine in SLE..."],
            "labels": ["METHODS"],
            "meshes": ["Lupus Erythematosus, Systemic", "Hydroxychloroquine"],
        },
        "long_answer": "Maybe, depending on stage...",
        "final_decision": "maybe",
    },
    {
        "pubid": 102,
        "question": "Does metformin improve glycemic control in T2DM?",
        "context": {
            "contexts": ["T2DM cohort treated with metformin..."],
            "labels": ["RESULTS"],
            "meshes": ["Diabetes Mellitus, Type 2", "Metformin"],
        },
        "long_answer": "Yes, ...",
        "final_decision": "no",
    },
    {
        "pubid": 103,
        "question": "",  # malformed: empty question
        "context": {"contexts": [], "labels": [], "meshes": []},
        "long_answer": "",
        "final_decision": "yes",
    },
]


def test_pubmedqa_filter_keeps_in_scope_and_drops_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_PUBMEDQA_ROWS)
    items = list(pubmedqa.load_pubmedqa(filter_subspecialty=True))
    ids = {it.id for it in items}
    assert "pubmedqa_100" in ids  # cardio MI
    assert "pubmedqa_101" in ids  # SLE / hydroxychloroquine
    assert "pubmedqa_102" not in ids  # T2DM filtered out
    assert "pubmedqa_103" not in ids  # empty question dropped


def test_pubmedqa_options_and_correct_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_PUBMEDQA_ROWS[:2])
    items = list(pubmedqa.load_pubmedqa(filter_subspecialty=False))
    yes_item = next(it for it in items if it.id == "pubmedqa_100")
    assert yes_item.options == {"A": "yes", "B": "no", "C": "maybe"}
    assert yes_item.correct == "A"
    assert yes_item.source == "pubmedqa"
    maybe_item = next(it for it in items if it.id == "pubmedqa_101")
    assert maybe_item.correct == "C"


def test_pubmedqa_unknown_decision_is_adversarial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            **SAMPLE_PUBMEDQA_ROWS[0],
            "pubid": 200,
            "final_decision": "unclear",
        }
    ]
    _patch_load(monkeypatch, rows)
    items = list(pubmedqa.load_pubmedqa(filter_subspecialty=False))
    assert items[0].correct is None


def test_pubmedqa_limit_caps_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, SAMPLE_PUBMEDQA_ROWS[:3])
    items = list(pubmedqa.load_pubmedqa(filter_subspecialty=False, limit=1))
    assert len(items) == 1


def test_pubmedqa_meshes_land_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, SAMPLE_PUBMEDQA_ROWS[:1])
    items = list(pubmedqa.load_pubmedqa(filter_subspecialty=False))
    assert "Myocardial Infarction" in items[0].metadata["meshes"]


# ── Vignette converter ────────────────────────────────────────────


def _make_vignette() -> Vignette:
    return Vignette(
        id="cardio_001",
        specialty="cardiology",
        subspecialty="acute_coronary_syndrome",
        stem="A 60-year-old man with crushing chest pain.",
        question="Most likely diagnosis?",
        options={"A": "STEMI", "B": "Pericarditis", "C": "PE", "D": "GERD"},
        correct="A",
        rationale="ST elevation in inferior leads suggests STEMI.",
        difficulty="medium",
        demographics=Demographics(age=60, sex="male"),
        sources=["Harrison's 21e ch. 269"],
    )


def test_vignette_to_mcq_item_shape() -> None:
    v = _make_vignette()
    item = vignette_to_mcq_item(v)
    assert item.id == "vignette_cardio_001"
    assert item.source == "vignette"
    assert item.stem == v.stem
    assert item.question == v.question
    assert item.options == v.options
    assert item.correct == "A"
    assert item.metadata["specialty"] == "cardiology"
    assert item.metadata["subspecialty"] == "acute_coronary_syndrome"
    assert item.metadata["demographics"]["age"] == 60
    assert item.metadata["is_variant"] is False


def test_vignette_variant_with_stem_override() -> None:
    v = _make_vignette()
    variant = VignetteVariant(
        variant_id="elderly_woman",
        demographics=Demographics(age=78, sex="female"),
        stem_override="A 78-year-old woman with crushing chest pain.",
        rationale_for_variant="age + sex perturbation",
    )
    item = vignette_variant_to_mcq_item(v, variant)
    assert item.id == "vignette_cardio_001_elderly_woman"
    assert item.stem == variant.stem_override
    assert item.metadata["is_variant"] is True
    assert item.metadata["parent_vignette_id"] == "cardio_001"
    assert item.metadata["demographics"]["age"] == 78


def test_vignette_variant_falls_back_to_parent_stem() -> None:
    v = _make_vignette()
    variant = VignetteVariant(
        variant_id="younger",
        demographics=Demographics(age=30, sex="male"),
    )
    item = vignette_variant_to_mcq_item(v, variant)
    assert item.stem == v.stem  # parent stem reused
