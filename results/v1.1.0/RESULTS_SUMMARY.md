# Results summary

## Reproducibility (methods section)

- Models: claude-haiku-4-5
- Seeds: 1 (0..0); temperature 0.0
- Prompt template: `mcq`; aggregate prompt hash `26c85aa0de7fa420b02ec067051c2cd51c6640906371829c4544eda1f2d0cbb0`
- Confidence: verbalized 0–100; judge model claude-sonnet-4-6
- Corpus: v1.1.1 (399 items: 112 base + 287 variants)
- Code commit: `ca9491116219353462b666d342a0cca1de616f95`

## Headline

- **claude-haiku-4-5**: accuracy 86.2% [82.5, 89.3], consistency 95.9%, ECE 0.081, Brier 0.092, abstention 0.0% / false-abstention 0.0%, fabrication 79.3%

## Pairwise McNemar

Only one model in this run — no pairs.

## Cells with n < 5 (do not interpret)

- demographic_cell `age+ethnicity`: n=1
- demographic_cell `age+ethnicity+sex`: n=1
- demographic_cell `age+race`: n=1
- demographic_cell `age+race+sex`: n=1
