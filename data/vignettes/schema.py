"""Re-exports + MCQItem converter for the custom Vignette schema.

The ``Vignette`` / ``VignetteVariant`` / ``Demographics`` Pydantic types
live in :mod:`eval.schemas` so runners and metrics can reference them
without depending on this package. This module re-exports them so
callers thinking of the types as "the vignette JSON shape" can import
locally, and adds the converters that turn vignettes into runner-facing
:class:`~eval.schemas.MCQItem` instances.
"""

from __future__ import annotations

from eval.schemas import Demographics, MCQItem, Vignette, VignetteVariant

__all__ = [
    "Demographics",
    "MCQItem",
    "Vignette",
    "VignetteVariant",
    "vignette_to_mcq_item",
    "vignette_variant_to_mcq_item",
]


def vignette_to_mcq_item(vignette: Vignette) -> MCQItem:
    """Convert a hand-written :class:`Vignette` to runner-facing :class:`MCQItem`.

    Specialty, subspecialty, difficulty, demographics, sources,
    rationale, and license land in ``MCQItem.metadata`` so downstream
    metrics (fairness, reasoning judge) can read them without re-loading
    the vignette JSON.
    """
    return MCQItem(
        id=f"vignette_{vignette.id}",
        source="vignette",
        stem=vignette.stem,
        question=vignette.question,
        options=dict(vignette.options),
        correct=vignette.correct,
        metadata={
            "specialty": vignette.specialty,
            "subspecialty": vignette.subspecialty,
            "difficulty": vignette.difficulty,
            "demographics": vignette.demographics.model_dump(),
            "sources": list(vignette.sources),
            "rationale": vignette.rationale,
            "license": vignette.license,
            "is_variant": False,
            "parent_vignette_id": None,
        },
    )


def vignette_variant_to_mcq_item(vignette: Vignette, variant: VignetteVariant) -> MCQItem:
    """Convert a variant of a vignette to :class:`MCQItem`.

    Uses ``variant.stem_override`` if present; otherwise the parent
    vignette's stem is reused. Demographic substitution happens in the
    Phase-4 fairness pipeline, not here.

    ``variant.correct_override`` (when set) takes precedence over the
    parent's ``correct`` answer — required for ``clinical_perturbation``
    or ``mixed`` variants whose changes shift the right answer.
    """
    stem = variant.stem_override if variant.stem_override is not None else vignette.stem
    correct = variant.correct_override if variant.correct_override is not None else vignette.correct
    return MCQItem(
        id=f"vignette_{vignette.id}_{variant.variant_id}",
        source="vignette",
        stem=stem,
        question=vignette.question,
        options=dict(vignette.options),
        correct=correct,
        metadata={
            "specialty": vignette.specialty,
            "subspecialty": vignette.subspecialty,
            "difficulty": vignette.difficulty,
            "demographics": variant.demographics.model_dump(),
            "sources": list(vignette.sources),
            "rationale": vignette.rationale,
            "license": vignette.license,
            "is_variant": True,
            "parent_vignette_id": vignette.id,
            "variant_rationale": variant.rationale_for_variant,
            "perturbation_type": variant.perturbation_type,
        },
    )
