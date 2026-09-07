"""Pydantic schemas for vignettes, MCQ items, eval responses, and run metadata.

These types are the contract between the loaders, runners, prompts, metrics,
pipeline, and frontend. Every Phase-1+ component speaks Pydantic. JSON dumps
go to ``results/runs/`` and ``results/aggregated/``.

The ``Vignette`` / ``VignetteVariant`` / ``Demographics`` triple is the schema
for the custom hand-written vignettes (Phase 3). ``MCQItem`` is the
runner-facing question shape — agnostic to where it came from (HF dataset
loader or a vignette).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Demographics + variants ───────────────────────────────────────


class Demographics(BaseModel):
    """Patient demographics for a vignette stem.

    Each field is optional so a generic stem can omit anything that isn't
    clinically relevant. Fairness eval (Phase 4) generates ``VignetteVariant``
    perturbations of these fields.
    """

    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["male", "female", "other", "unspecified"] | None = None
    race: str | None = None
    ethnicity: str | None = None


class VignetteVariant(BaseModel):
    """A perturbation of a base vignette stem.

    ``stem_override`` lets a variant rewrite the stem entirely if the
    demographic change requires more than a swap (e.g. pregnancy variant
    on a female-only presentation). When ``stem_override`` is None the
    pipeline substitutes the variant's demographics into the parent
    stem at format time.

    ``perturbation_type`` declares what kind of change this variant
    represents — fairness metrics (Phase 4) treat the three types
    differently, since pure-demographic perturbations should preserve
    the answer while clinical perturbations may not.

    ``correct_override`` lets a variant declare a different correct
    answer than the parent vignette; this is required for ``"mixed"``
    variants whose changes actually move the right answer.
    ``pure_demographic`` and ``clinical_perturbation`` variants must
    preserve the answer (``correct_override`` stays None) — robustness
    and fairness metrics rely on that invariant.

    ``perturbation_subtype`` refines ``clinical_perturbation`` into the
    three answer-preserving robustness probes (distractor insertion,
    history reordering, paraphrase); it is None for other types.
    """

    model_config = ConfigDict(extra="forbid")

    variant_id: str
    demographics: Demographics
    stem_override: str | None = None
    rationale_for_variant: str = ""
    perturbation_type: Literal[
        "pure_demographic",
        "clinical_perturbation",
        "mixed",
    ] = "pure_demographic"
    perturbation_subtype: (
        Literal[
            "distractor_insertion",
            "history_reordering",
            "paraphrase",
        ]
        | None
    ) = None
    correct_override: str | None = None


# ── Vignette + MCQItem ────────────────────────────────────────────


class Vignette(BaseModel):
    """A single hand-written clinical vignette.

    ``correct=None`` marks an adversarial item — there is no correct
    option, and the model is graded on whether it correctly abstains
    (returns no choice) rather than picking.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    specialty: Literal["cardiology", "autoimmune"]
    subspecialty: str
    stem: str
    question: str
    options: dict[str, str]
    correct: str | None
    rationale: str
    difficulty: Literal["easy", "medium", "hard"]
    demographics: Demographics
    variants: list[VignetteVariant] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    license: str = "CC-BY-4.0"


class DraftRecord(BaseModel):
    """A generated corpus-expansion item awaiting human review.

    Drafts live in ``data/vignettes/drafts/`` and never enter the main
    corpus until a reviewer sets ``reviewed=True`` with
    ``status="approved"`` and runs the promote step
    (``scripts/review_drafts.py promote``). Exactly one of ``variant``
    (a new variant for the existing case ``parent_id``) or ``vignette``
    (a new adversarial base case) is set.
    """

    model_config = ConfigDict(extra="forbid")

    draft_id: str
    category: Literal["clinical_perturbation", "pure_demographic", "adversarial"]
    parent_id: str | None = None
    reviewed: bool = False
    status: Literal["pending", "approved", "rejected", "promoted"] = "pending"
    review_note: str = ""
    generator_model: str
    generated_at: str
    variant: VignetteVariant | None = None
    vignette: Vignette | None = None

    @model_validator(mode="after")
    def _payload_shape(self) -> DraftRecord:
        if (self.variant is None) == (self.vignette is None):
            raise ValueError("exactly one of variant/vignette must be set")
        if (self.variant is not None) != (self.parent_id is not None):
            raise ValueError("parent_id is required iff the draft is a variant")
        return self


class MCQItem(BaseModel):
    """Generic multiple-choice item — what runners see.

    Loaders convert their dataset's native shape into ``MCQItem``s; the
    custom ``Vignette`` model emits ``MCQItem``s through a converter as
    well. ``correct=None`` is adversarial.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    stem: str
    question: str
    options: dict[str, str]
    correct: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Runner output + run manifest ──────────────────────────────────


class EvalResponse(BaseModel):
    """A model's output for a single ``MCQItem``.

    ``parsed_answer`` is the letter the model chose (after parsing the
    structured response). ``confidence`` is the model's self-reported
    0-100 calibration estimate. ``error`` is set iff the API call failed
    after all retries; the other fields are best-effort in that case.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    model: str
    raw_response: str
    parsed_answer: str | None
    confidence: int | None = Field(default=None, ge=0, le=100)
    rationale: str | None = None
    elapsed_ms: int = Field(ge=0)
    error: str | None = None


class RunMetadata(BaseModel):
    """Manifest for a complete evaluation run.

    Persisted alongside the JSONL of per-item responses. Every field
    here is required for reproducibility — model id, model version,
    sampling parameters, prompt hash, dataset label, git SHA, and
    timing.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    model: str
    model_version: str
    dataset: str
    dataset_version: str = "unknown"
    prompt_template: str = "mcq"
    confidence_method: str = "verbalized_0_100"
    n_items: int = Field(ge=0)
    temperature: float
    top_p: float
    seed: int | None
    prompt_hash: str
    git_sha: str
    started_at: str
    finished_at: str | None = None
    total_elapsed_s: float | None = None
