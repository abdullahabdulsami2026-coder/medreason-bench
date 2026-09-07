# Datasheet for MedReason-Bench

> Status: scaffold, following Gebru et al., "Datasheets for Datasets"
> (2021). Describes corpus v1.0.0 (100 cases, 200 variants); expansion
> items (v1.1.x) were added after LLM prescreen and LLM review only.

## Motivation

**Why was the dataset created?** To measure whether LLMs reason about
clinical cases or pattern-match: accuracy alone hides miscalibration,
hallucination on unanswerable cases, and demographic sensitivity. No
public benchmark combined cardiology + autoimmune coverage with
within-case demographic variants and adversarial unanswerable items.

**Who created it and under what funding?** Abdullah Abdul Sami
(Northwestern University), unfunded, as an open research artifact.

## Composition

**What do instances represent?** Textbook-style clinical vignettes:
stem, single-best-answer question, four options, key (or `null` for
adversarial items), rationale, difficulty, structured demographics
(age, sex, race, ethnicity — each optional), literature sources, and a
list of variants (see taxonomy in `paper/methods.md`).

**How many instances?** 100 base cases (50 cardiology, 50 autoimmune),
200 variants (2 per case), 6 adversarial bases. Exact per-slice counts
regenerate via `scripts/build_table1.py`.

**Does it contain confidential data?** No. Every case is fictional and
authored for this benchmark; no patient data, no PHI, no MIMIC or EHR
derivatives. Demographics appear only as study variables.

**Are there recommended splits?** No train split exists or is intended;
the corpus is evaluation-only. Training on it would defeat its purpose
(contamination-free by construction) — see Uses.

## Collection process

**How was the data acquired?** Hand-written by a single author against
standard references (Harrison's, society guidelines, subspecialty
texts), each case citing its sources. A planned second-reviewer pass was not carried out, and no answer key was independently adjudicated. No scraping, no crowdworkers, no
patient involvement.

**Expansion process (post-v1.0.0).** Additional variants and
adversarial cases are model-drafted, independently model-pre-screened,
and entered the corpus (v1.1.x) after LLM prescreen and LLM review only, applied under the author's direction; the
pipeline and its audit trail (draft status, prescreen flags, changelog,
version bump) are in the repository.

## Preprocessing / labeling

Labels (correct option, difficulty, perturbation type/subtype) are
author-assigned at writing time. Schema validity is enforced by
Pydantic models (`eval/schemas.py`) and CI tests, including the
invariant that demographic and clinical-perturbation variants never
change the answer key.

## Uses

**Intended.** Evaluating language models' clinical reasoning,
calibration, abstention behaviour, and demographic robustness;
reproducing the paper's tables via `run_eval.py`.

**Discouraged.** Training or fine-tuning (contaminates the benchmark);
clinical decision support (fictional cases, not medical advice);
drawing per-demographic-group conclusions beyond the powered contrasts
documented in `paper/limitations.md`.

## Distribution

Cases are CC-BY-4.0, code MIT, in a public GitHub repository;
HuggingFace and Zenodo deposits are planned (drafts in `release/`).
Citation metadata lives in `CITATION.cff`.

## Maintenance

Maintained by the author. The corpus version (`data/vignettes/VERSION`)
bumps only through the reviewed-promotion pipeline, with changes logged
in `CHANGELOG.md`; issues and contributions go through GitHub
(`CONTRIBUTING.md`).
