# Methods

> Status: scaffold. Results sections are placeholders until reviewed
> corpus expansion and multi-model runs complete. No numbers in this
> document are results.

## Dataset construction

MedReason-Bench evaluates clinical reasoning on 100 hand-written
vignettes (50 cardiology, 50 autoimmune disease), authored for this
benchmark so they cannot appear in any model's training data. Each case
carries a stem, a single-best-answer question with four options, a
written rationale, difficulty (easy/medium/hard), structured patient
demographics, literature sources, and a CC-BY-4.0 license. All 100 were written by a single author against standard references; a planned second-reviewer pass was not carried out, and no answer key has been independently adjudicated. A small share
of cases is adversarial: no option is supported by the stem
(`correct = null`), and the graded-correct behaviour is abstention.

The corpus is versioned (`data/vignettes/VERSION`, recorded in every
run manifest as `dataset_version`) and changes only through a gated
pipeline: candidate items are drafted by a generator model
(`scripts/generate_drafts.py`), pre-screened by an independent judge
model with an adversarial-reviewer prompt
(`scripts/prescreen_drafts.py`), then promoted (v1.1.x) after an LLM review pass (decisions logged in
`review_decisions.md`) applied under the author's direction; no
clinician or second human reviewed the additions.

## Variant taxonomy

Every case has variants — re-renderings of the same clinical content:

- **pure_demographic** — exactly one demographic attribute (age, sex,
  race, or ethnicity) changes; all clinical facts, numbers, and
  sentence order are preserved, and the correct answer must not move.
  These power the fairness analysis.
- **clinical_perturbation** — answer-preserving robustness probes,
  subtyped as `distractor_insertion` (one plausible but diagnostically
  irrelevant finding added), `history_reordering` (identical facts,
  different order), and `paraphrase` (rewording with all facts and
  numbers preserved).
- **mixed** — clinically meaningful changes (e.g. an anginal-equivalent
  presentation replacing classic chest pain). These may legitimately
  change the correct answer, declared via `correct_override`; each
  variant is always graded against its own key.

## Metrics

Formal definitions live in `docs/METRICS.md`; in prose:

**Accuracy.** Top-1 accuracy overall and per specialty, difficulty, and
answerable/unanswerable status, with Wilson 95% confidence intervals;
macro-F1 over answer categories (option letters plus abstention)
accompanies it because categories are unbalanced.

**Robustness.** Consistency rate is the share of cases answered
identically across the base item and all answer-preserving variants.
Flip rate is the share of answer-preserving variants answered wrongly
when the base item was answered correctly. The robustness gap is base
accuracy minus variant accuracy, reported per perturbation type and
subtype.

**Calibration.** Confidence is elicited verbally in the prompt as an
integer 0–100 (recorded as `confidence_method: verbalized_0_100`). We
report the Brier score and expected calibration error with 10
equal-width bins, over answered items with parseable confidence, with
coverage stated.

**Hallucination.** On unanswerable items, abstention rate (declining is
correct) and fabrication rate: among unanswerable items the model
answered, the share where it asserted an unsupported diagnosis as
established, per the judge protocol below.

**Fairness.** Accuracy deltas are computed within-case: each
pure_demographic variant is paired with its own base item, so case
difficulty is controlled. Per swapped attribute we report the mean
paired delta with a 95% CI over the paired differences and flag cells
whose CI excludes zero; cell sizes are always reported because small
cells are underpowered.

**Pairwise comparisons.** Exact McNemar tests (binomial test on
discordant pairs) between every model pair on shared items.

All metrics are reported mean ± SD across seeds.

## Models and decoding

The planned model roster spans frontier APIs (Anthropic Claude, OpenAI
GPT, Google Gemini) and open models (via Groq and local Ollama), listed
in the build brief; the final roster is reported with results. Every
model sees byte-identical prompts rendered from the same template, at
temperature 0 with a single sampling parameter (current Anthropic
models reject setting temperature and top_p simultaneously). Runs are
seeded and manifests record model id, sampling parameters, prompt
hash, dataset version, and git SHA. The prompt offers an explicit
abstention channel (`ANSWER: NONE`) on every item — offering it only on
adversarial items would leak the label.

## Judge protocol and inter-rater reliability

Fabrication verdicts come from an LLM judge running the fixed rubric in
`prompts/judge.md` (verdicts FABRICATED / NOT_FABRICATED / BORDERLINE;
only FABRICATED counts toward the rate). The rubric requires an
affirmative assertion of an unsupported conclusion, quoting the
decisive phrase. A stratified 20% sample of verdicts
(`scripts/sample_for_human_review.py`, stratified by model × verdict)
is independently labeled by a human reviewer, and agreement is
summarized with Cohen's kappa (`scripts/compute_kappa.py`). The judge
model, sample size, and kappa are reported with results.

## Results

*Placeholder — populated from `run_eval.py` outputs (table2.csv,
per_case.csv, pairwise.csv, figures/) once evaluation runs on the
reviewed corpus.*
