# Vignette Review Log

Two-reviewer pass for every hand-written vignette in
`data/vignettes/cardiology/` and `data/vignettes/autoimmune/`. The
brief (Phase 3, Day 7–10) requires Abdullah and Hansraj to both sign
off before a vignette is considered v1.0-eligible.

## What goes in a row

- **vignette_id** — matches `Vignette.id` (e.g. `cardio_001`,
  `autoimmune_023`). One row per vignette. Variants are reviewed as
  part of the parent vignette.
- **written_by** — initials (AAS / HH).
- **reviewed_by** — initials of the second reviewer (must differ
  from `written_by`).
- **reviewed_at** — ISO date `YYYY-MM-DD`.
- **issues** — concise notes (clinical accuracy, plausible
  distractors, demographic balance, source quality, adversarial
  marking). Empty if none.
- **resolved** — `yes` once issues are addressed; `n/a` if no issues.

## Log

| vignette_id | written_by | reviewed_by | reviewed_at | issues | resolved |
| ----------- | ---------- | ----------- | ----------- | ------ | -------- |
|             |            |             |             |        |          |
