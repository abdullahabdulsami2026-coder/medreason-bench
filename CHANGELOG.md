# Changelog

Corpus versions live in `data/vignettes/VERSION` and are recorded in every
run manifest as `dataset_version`. Version bumps happen only when reviewed
drafts are promoted (`scripts/review_drafts.py promote --version X.Y.Z`).

## 1.1.1 — 2026-09-05

- autoimmune_050 converted from adversarial to answerable (correct: A, early
  RA) per machine-review finding that its documented negatives disable every
  alternative; its `anti_dsDNA_low_titer_added` mixed variant now carries
  `correct_override: B` (2019 EULAR/ACR score crosses threshold).
- autoimmune_012 variant `very_early_arthralgia_no_synovitis` reclassified
  `pure_demographic` -> `mixed` (clinical change with answer override); test
  grandfather removed.

## 1.1.0 — 2026-09-05

- Promoted 99 reviewed draft(s) into the corpus: 12 adversarial, 30 clinical_perturbation, 57 pure_demographic.
- Previous version: 1.0.0.

## Unreleased

- Corpus-expansion drafts staged under `data/vignettes/drafts/` (not part of
  the corpus until reviewed and promoted): clinical_perturbation variants
  (distractor_insertion / history_reordering / paraphrase, new
  `perturbation_subtype` field), pure_demographic sex-/race-/ethnicity-only
  swap variants, and new adversarial base cases.
- `VignetteVariant` schema gains optional `perturbation_subtype`;
  `RunMetadata` gains `dataset_version`.

## 1.0.0 — 2026-08-30

- Initial corpus: 100 hand-written cases (50 cardiology, 50 autoimmune),
  200 variants (2 per case), 6 adversarial cases. Written by a single author; a planned second-reviewer pass was not carried out.
