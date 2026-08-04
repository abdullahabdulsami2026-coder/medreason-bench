# MedReason-Bench

**A benchmark for measuring whether language models actually reason about clinical cases — or just pattern-match.**

Large language models score well on medical exam questions, but a high score can hide two failures that matter in practice: the model may be *confidently wrong* (badly calibrated), and it may be *differently wrong for different patients* (biased by age, sex, or race). MedReason-Bench grades models on the same set of cardiology and autoimmune-disease cases and reports four things instead of one number: accuracy, calibration (does a stated 90% confidence mean 90% correct?), hallucination (does the model refuse when no answer is correct?), and fairness (does the answer change when only the patient's demographics change?). Every model sees identical prompts at temperature 0, so differences between models are attributable to the model rather than to prompt luck. Think of it as an instrument-characterisation study: the point is not the headline reading but the systematic error, the noise floor, and the drift.

> **Status: partially implemented.** The evaluation harness, the 100-case corpus, and accuracy scoring all work end-to-end today. Calibration, fairness, hallucination, and reasoning-quality scoring are stubs — see [What works today](#what-works-today).

---

## Install and run

Requires Python 3.11+ and an API key for at least one model provider.

```bash
git clone https://github.com/abdullahabdulsami2026-coder/medreason-bench.git
cd medreason-bench

python3 -m pip install -e ".[dev]"      # or: uv venv && uv pip install -e ".[dev]"
cp .env.example .env.local              # then add your ANTHROPIC_API_KEY
```

Run the smoke evaluation — 10 hand-written cases against one model, a few cents of API spend:

```bash
python3 -m eval.pipeline --smoke
```

Expected output:

```
Wrote claude-sonnet-4-6__phase1_smoke__20260804T170000Z.jsonl
      claude-sonnet-4-6__phase1_smoke__20260804T170000Z.manifest.json

Ran 10 items in 12.4s (0 errored).
Top-1 accuracy: 9/10 = 90.0%
```

Other useful invocations:

```bash
python3 -m eval.pipeline --dataset vignettes                  # all 100 custom cases
python3 -m eval.pipeline --dataset medmcqa --provider openai  # a public dataset
python3 -m pytest                                             # 69 tests, ~4s, no API calls
```

`--provider` accepts `anthropic`, `openai`, `google`, `groq`, or `ollama`. Ollama runs locally and needs no key.

---

## Inputs and outputs

**Input** — every question, whatever its source, is normalised to one shape:

```jsonc
{
  "id": "cardio_001",
  "source": "vignettes",
  "stem": "A 62-year-old man presents with crushing substernal chest pain...",
  "question": "Which coronary artery is most likely occluded?",
  "options": {"A": "Left anterior descending", "B": "Left circumflex", "C": "Right coronary", "D": "Left main"},
  "correct": "C",          // null marks an adversarial case with no correct answer
  "metadata": {"specialty": "cardiology", "topic": "stemi_localization"}
}
```

Three input sources are wired up: the 100 hand-written cases in `data/vignettes/`, three public medical question sets pulled from HuggingFace (MedMCQA, MedQA-USMLE, PubMedQA), and 10 built-in smoke cases.

**Output** — two files per run, written to `results/runs/`:

1. `<model>__<dataset>__<timestamp>.jsonl` — one line per question:

```jsonc
{
  "item_id": "cardio_001",
  "model": "claude-sonnet-4-6",
  "parsed_answer": "C",      // null = the model abstained
  "confidence": 85,          // model's self-reported 0-100
  "rationale": "Inferior ST elevation in II, III, aVF localises to...",
  "elapsed_ms": 1240,
  "error": null
}
```

2. `<...>.manifest.json` — the reproducibility record: model ID, temperature, seed, git commit SHA, a SHA-256 hash over all rendered prompts, and wall-clock timing. A change to a prompt template produces a different hash even when the questions are unchanged, so silently-drifted runs are detectable.

---

## The case corpus

100 hand-written clinical cases, 50 cardiology and 50 autoimmune, licensed CC-BY-4.0:

| Property | Count | Why it's there |
| --- | ---: | --- |
| Cases | 100 | Written for this benchmark, so they cannot be in any model's training data |
| Fairness variants | 200 | Each case is re-rendered with age/sex/race perturbed. A well-behaved model gives the *same* answer; the gap between variants is the fairness metric |
| Adversarial cases | 6 | No option is correct. Credit is given only for abstaining — this measures hallucination directly |

Every case carries a written rationale and literature sources, so a disputed grading decision can be adjudicated rather than argued.

## What works today

| Component | State |
| --- | --- |
| Evaluation harness + CLI | Working |
| Runners: Anthropic, OpenAI, Google, Groq, Ollama | Working |
| Dataset loaders: MedMCQA, MedQA, PubMedQA, vignettes | Working |
| Accuracy (incl. abstention scoring on adversarial cases) | Working |
| 100-case corpus with variants | Complete |
| Test suite | 69 tests passing |
| Calibration, fairness, hallucination, reasoning-judge metrics | **Stubs** — the corpus supports them; the scoring code is not written |
| Leaderboard aggregation + web frontend | **Stubs** |
| Together.ai runner | **Stub** |

There are no published results yet: `results/` and `paper/figures/` are empty, so this README shows no leaderboard rather than a placeholder one.

---

## Adding a model

1. Add a file in `eval/runners/` subclassing `eval.runners.base.Runner`.
2. Implement `run(item) -> EvalResponse`. It must never raise — on failure, return a response with `error` set.
3. Register it in `RUNNER_REGISTRY` in `eval/pipeline.py`.
4. Add a test in `tests/test_runners.py` with a mocked SDK client.

## License

Code MIT (`LICENSE`). Clinical cases CC-BY-4.0. Built on MedMCQA, MedQA, and PubMedQA.

Authors: **Abdullah Abdul Sami** (Northwestern University) · **Hansraj Hitesh** (Indiana Wesleyan University). Citation metadata in [`CITATION.cff`](CITATION.cff).
