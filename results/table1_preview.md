# Table 1 — Dataset descriptives

Corpus version: 1.1.1. Counts regenerate via `python3 scripts/build_table1.py`.

| | Corpus |
|---|---|
| Cases | 112 |
| Adversarial cases (correct = null) | 17 |
| Variants | 287 |

## Cases by specialty x difficulty

| specialty, difficulty | Corpus |
|---|---|
| autoimmune, easy | 5 |
| autoimmune, hard | 17 |
| autoimmune, medium | 33 |
| cardiology, easy | 9 |
| cardiology, hard | 19 |
| cardiology, medium | 29 |

## Variants by perturbation type

| perturbation_type | Corpus |
|---|---|
| clinical_perturbation | 30 |
| mixed | 106 |
| pure_demographic | 151 |

## Clinical-perturbation subtypes

| perturbation_subtype | Corpus |
|---|---|
| distractor_insertion | 10 |
| history_reordering | 10 |
| paraphrase | 10 |

## Pure-demographic swap cells (changed attributes)

Fairness metrics compare accuracy within-case across these cells; single-attribute cells (sex, race, ethnicity) are the primary contrasts.

| changed attribute(s) | Corpus |
|---|---|
| (stem-only) | 1 |
| age | 31 |
| age+ethnicity | 1 |
| age+ethnicity+sex | 1 |
| age+race | 1 |
| age+race+sex | 1 |
| age+sex | 55 |
| ethnicity | 20 |
| race | 20 |
| sex | 20 |
