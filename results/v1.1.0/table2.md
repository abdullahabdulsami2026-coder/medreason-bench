# Table 2 — Accuracy by variant type and demographic cell

n = distinct items in the slice; accuracy is mean ± SD across seeds. Cells with n < 5 are underpowered — see RESULTS_SUMMARY.md.

| model | slice type | slice | n | accuracy |
|---|---|---|---|---|
| claude-haiku-4-5 | variant_type | base | 112 | 83.0% |
| claude-haiku-4-5 | variant_type | clinical_perturbation | 30 | 100.0% |
| claude-haiku-4-5 | variant_type | mixed | 105 | 74.3% |
| claude-haiku-4-5 | variant_type | pure_demographic | 152 | 94.1% |
| claude-haiku-4-5 | clinical_subtype | distractor_insertion | 10 | 100.0% |
| claude-haiku-4-5 | clinical_subtype | history_reordering | 10 | 100.0% |
| claude-haiku-4-5 | clinical_subtype | paraphrase | 10 | 100.0% |
| claude-haiku-4-5 | demographic_cell | age | 32 | 84.4% |
| claude-haiku-4-5 | demographic_cell | age+ethnicity | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+race | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+sex | 55 | 94.5% |
| claude-haiku-4-5 | demographic_cell | ethnicity | 20 | 100.0% |
| claude-haiku-4-5 | demographic_cell | race | 20 | 100.0% |
| claude-haiku-4-5 | demographic_cell | sex | 20 | 100.0% |