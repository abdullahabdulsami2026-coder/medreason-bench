---
# DRAFT — do not upload until corpus review completes and the
# "no results yet" caveats below are resolved.
license: cc-by-4.0
task_categories:
  - question-answering
language:
  - en
tags:
  - medical
  - clinical-reasoning
  - benchmark
  - fairness
  - calibration
  - hallucination
pretty_name: MedReason-Bench
size_categories:
  - n<1K
---

# MedReason-Bench (DRAFT dataset card)

**Status: DRAFT.** This card describes corpus **v1.0.0 only** — the 100
hand-written, single-author cases and their 200 variants.
Machine-generated expansion items exist in the source repository (promoted to the corpus as v1.1.x after LLM prescreen and LLM review only) and are **excluded here**
and must not be uploaded with this dataset.

## Dataset summary

100 hand-written clinical vignettes (50 cardiology, 50 autoimmune
disease) for evaluating LLM clinical reasoning beyond accuracy:
calibration (verbalized confidence), hallucination (adversarial cases
with no supported option, where abstention is correct), and fairness
(within-case demographic variants). Cases were authored for this
benchmark and cannot appear in model training data. Evaluation-only —
do not train on this data.

## Structure

One JSON record per case, matching the Pydantic schema in the source
repo (`eval/schemas.py`): `id`, `specialty`, `subspecialty`, `stem`,
`question`, `options` (A–D), `correct` (letter, or `null` for
adversarial items), `rationale`, `difficulty`, `demographics`
(age/sex/race/ethnicity), `variants` (demographic and clinical
perturbations; `correct_override` when a mixed variant moves the key),
`sources`, `license`.

## Provenance and review

Hand-written by the authors against standard references (each case
cites its sources); single-author authorship; a planned second-reviewer pass was not carried out.
Fictional patients only — no PHI, no EHR derivatives. See the
datasheet (`paper/datasheet.md`) and limitations
(`paper/limitations.md`) in the source repo before drawing
subgroup conclusions.

## Evaluation

The source repository provides the full harness (`run_eval.py`): Wilson
CIs, robustness (consistency/flip/gap), ECE and Brier on verbalized
confidence, abstention and judged fabrication rates, within-case
fairness deltas, and McNemar pairwise tests. **No benchmark results are
published yet**; do not cite numbers from drafts.

## Citation and license

Cases CC-BY-4.0, code MIT. Cite via the repository's `CITATION.cff`
(Abdullah Abdul Sami, Northwestern University). Source: https://github.com/abdullahabdulsami2026-coder/medreason-bench
