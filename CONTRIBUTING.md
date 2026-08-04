# Contributing to MedReason-Bench

Thanks for considering a contribution. This project is a public, peer-review-quality benchmark — code style, reproducibility, and clinical accuracy all matter.

## Workflow

1. Fork or branch (`feature/<short-name>`, `fix/<short-name>`).
2. Write code with type hints and docstrings.
3. Add tests under `tests/`. Coverage target on `eval/` is **≥80 %**.
4. Run pre-commit locally: `pre-commit run --all-files`.
5. Open a PR against `main`. CI (lint, type-check, tests) must pass.
6. Use [Conventional Commits](https://www.conventionalcommits.org/) — e.g.
   `feat(eval): add anthropic runner`, `fix(metrics): ECE binning off-by-one`.

## Adding a model runner

1. Create `eval/runners/<provider>_runner.py`. Subclass `eval.runners.base.Runner`.
2. Implement `run(item: MCQItem) -> EvalResponse`. Use the SDK's recommended retry strategy; never silently swallow failures.
3. Register the runner in `eval/pipeline.py`'s dispatch table.
4. Add a unit test in `tests/test_runners.py` that mocks the SDK and verifies the response is parsed into a valid `EvalResponse`.

## Adding a metric

1. Create `eval/metrics/<metric>.py`. Top of file: docstring with the formula and the literature reference.
2. Function signature: `metric(responses: list[EvalResponse], ground_truth: list[MCQItem]) -> dict`.
3. Add a test in `tests/test_metrics.py` with at least two cases, including an edge case (empty input, single item).

## Adding a vignette

1. Place the JSON in `data/vignettes/<specialty>/<id>.json`. The schema is in `data/vignettes/schema.py`.
2. Cite source(s) — textbook, guideline, peer-reviewed paper. No PHI.
3. Two-reviewer rule: a second author must review and sign off in `data/vignettes/REVIEW_LOG.md`.

## Reporting issues

Please file issues on the upstream repo: https://github.com/abdullahabdulsami2026-coder/medreason-bench/issues

Include: dataset / model / commit SHA / Python version / minimal reproduction.

## License

By contributing, you agree your contributions will be licensed under the MIT License (code) or CC-BY-4.0 (vignettes), matching the rest of the project.
