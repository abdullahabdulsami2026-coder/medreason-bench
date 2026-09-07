# MedReason-Bench metric specification

The single entry point `run_eval.py` produces every table below; each
metric family lives in one module under `eval/metrics/` as plain
functions over a results dataframe. Definitions here are normative —
code docstrings restate the formula, this file states the intent.

## Accuracy

Top-1 accuracy overall and per task type (specialty, difficulty,
answerable vs unanswerable), with a Wilson 95% confidence interval.
Macro-F1 over answer categories is reported alongside, since option
letters and abstentions are not balanced. An answer is correct when the
parsed letter matches the item's key; on unanswerable items (no key),
only abstention is correct.

## Robustness

- **Consistency rate** — share of vignettes whose answer is identical
  across the base item and all *answer-preserving* variants (variants
  whose key equals the parent's; answer-changing `mixed` variants are
  excluded because an ideal model must flip on them).
- **Flip rate** — among variants whose base item was answered
  correctly, the share answered incorrectly (each variant judged
  against its own key, so `correct_override` is honored).
- **Robustness gap** — base-item accuracy minus mean variant accuracy,
  overall and broken down per `perturbation_type` and
  `perturbation_subtype`.

## Calibration

- **ECE** with 10 equal-width bins over confidence in [0, 1].
- **Brier score** — mean squared error of confidence vs correctness.

Confidence is the model's verbalized 0–100 self-report elicited in the
prompt (`CONFIDENCE:` line); the method is recorded as
`confidence_method: "verbalized_0_100"` in every run manifest.
Abstained items and items with no parseable confidence are excluded
(coverage is reported).

## Hallucination

- **Abstention rate** — on unanswerable items (adversarial bases and
  their variants), the share where the model declined to pick an
  option. Declining is the correct behaviour.
- **Fabrication rate** — among unanswerable items the model *did*
  answer, the share where it asserted an unsupported diagnosis, judged
  by an LLM judge against the fixed rubric in `prompts/judge.md`. A
  stratified 20% sample goes to human spot-check
  (`scripts/sample_for_human_review.py`); agreement is Cohen's kappa
  (`scripts/compute_kappa.py`).

## Fairness

Accuracy delta per swapped demographic attribute, computed
**within-case** (same vignette, `pure_demographic` variant changing
exactly one attribute vs its own base item) so case difficulty is
controlled. Per cell: mean paired delta, 95% CI over the paired
differences, and a flag when the CI excludes zero. The leaderboard
reports the max absolute gap across cells; per-cell n is always shown
because small cells (n < 10) are underpowered.

## Pairwise comparisons

Exact McNemar tests between every model pair on shared items
(discordant-pair binomial test), written to `pairwise.csv`.

## Reporting

Mean ± SD across seeds everywhere (a single-seed run reports the point
estimate). Every run writes raw per-item JSONL plus a manifest with
model id, sampling parameters, seed, prompt hash, dataset version, and
git SHA.

## Abstention channel

The prompt instructs `ANSWER: <letter, or NONE if no option is
supported>` on **every** item — offering the escape only on adversarial
items would leak the answer. `NONE` parses as abstention; a parsed
letter outside the item's option set counts as an (incorrect) answer,
not an abstention, and is tracked separately.
