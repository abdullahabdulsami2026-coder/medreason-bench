# Table 2 — Accuracy by variant type and demographic cell

n = distinct items in the slice; accuracy is mean ± SD across seeds. Cells with n < 5 are underpowered — see RESULTS_SUMMARY.md.

| model | slice type | slice | n | accuracy |
|---|---|---|---|---|
| claude-haiku-4-5 | variant_type | base | 112 | 83.9% |
| claude-haiku-4-5 | variant_type | clinical_perturbation | 30 | 100.0% |
| claude-haiku-4-5 | variant_type | mixed | 106 | 74.5% |
| claude-haiku-4-5 | variant_type | pure_demographic | 151 | 94.7% |
| claude-haiku-4-5 | clinical_subtype | distractor_insertion | 10 | 100.0% |
| claude-haiku-4-5 | clinical_subtype | history_reordering | 10 | 100.0% |
| claude-haiku-4-5 | clinical_subtype | paraphrase | 10 | 100.0% |
| claude-haiku-4-5 | demographic_cell | age | 31 | 87.1% |
| claude-haiku-4-5 | demographic_cell | age+ethnicity | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+race | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| claude-haiku-4-5 | demographic_cell | age+sex | 55 | 94.5% |
| claude-haiku-4-5 | demographic_cell | ethnicity | 20 | 100.0% |
| claude-haiku-4-5 | demographic_cell | race | 20 | 100.0% |
| claude-haiku-4-5 | demographic_cell | sex | 20 | 100.0% |
| gemma2:9b | variant_type | base | 112 | 78.6% |
| gemma2:9b | variant_type | clinical_perturbation | 30 | 86.7% |
| gemma2:9b | variant_type | mixed | 106 | 67.0% |
| gemma2:9b | variant_type | pure_demographic | 151 | 88.1% |
| gemma2:9b | clinical_subtype | distractor_insertion | 10 | 100.0% |
| gemma2:9b | clinical_subtype | history_reordering | 10 | 90.0% |
| gemma2:9b | clinical_subtype | paraphrase | 10 | 70.0% |
| gemma2:9b | demographic_cell | age | 31 | 80.6% |
| gemma2:9b | demographic_cell | age+ethnicity | 1 ⚠ | 0.0% |
| gemma2:9b | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| gemma2:9b | demographic_cell | age+race | 1 ⚠ | 100.0% |
| gemma2:9b | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| gemma2:9b | demographic_cell | age+sex | 55 | 92.7% |
| gemma2:9b | demographic_cell | ethnicity | 20 | 80.0% |
| gemma2:9b | demographic_cell | race | 20 | 95.0% |
| gemma2:9b | demographic_cell | sex | 20 | 95.0% |
| llama3.1:8b | variant_type | base | 112 | 83.0% |
| llama3.1:8b | variant_type | clinical_perturbation | 30 | 96.7% |
| llama3.1:8b | variant_type | mixed | 106 | 54.7% |
| llama3.1:8b | variant_type | pure_demographic | 151 | 95.4% |
| llama3.1:8b | clinical_subtype | distractor_insertion | 10 | 100.0% |
| llama3.1:8b | clinical_subtype | history_reordering | 10 | 90.0% |
| llama3.1:8b | clinical_subtype | paraphrase | 10 | 100.0% |
| llama3.1:8b | demographic_cell | age | 31 | 90.3% |
| llama3.1:8b | demographic_cell | age+ethnicity | 1 ⚠ | 100.0% |
| llama3.1:8b | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| llama3.1:8b | demographic_cell | age+race | 1 ⚠ | 100.0% |
| llama3.1:8b | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| llama3.1:8b | demographic_cell | age+sex | 55 | 94.5% |
| llama3.1:8b | demographic_cell | ethnicity | 20 | 100.0% |
| llama3.1:8b | demographic_cell | race | 20 | 100.0% |
| llama3.1:8b | demographic_cell | sex | 20 | 100.0% |
| mistral:7b | variant_type | base | 112 | 83.0% |
| mistral:7b | variant_type | clinical_perturbation | 30 | 86.7% |
| mistral:7b | variant_type | mixed | 106 | 65.1% |
| mistral:7b | variant_type | pure_demographic | 151 | 90.7% |
| mistral:7b | clinical_subtype | distractor_insertion | 10 | 90.0% |
| mistral:7b | clinical_subtype | history_reordering | 10 | 100.0% |
| mistral:7b | clinical_subtype | paraphrase | 10 | 70.0% |
| mistral:7b | demographic_cell | age | 31 | 87.1% |
| mistral:7b | demographic_cell | age+ethnicity | 1 ⚠ | 0.0% |
| mistral:7b | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| mistral:7b | demographic_cell | age+race | 1 ⚠ | 100.0% |
| mistral:7b | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| mistral:7b | demographic_cell | age+sex | 55 | 92.7% |
| mistral:7b | demographic_cell | ethnicity | 20 | 90.0% |
| mistral:7b | demographic_cell | race | 20 | 95.0% |
| mistral:7b | demographic_cell | sex | 20 | 95.0% |
| phi4 | variant_type | base | 112 | 83.9% |
| phi4 | variant_type | clinical_perturbation | 30 | 96.7% |
| phi4 | variant_type | mixed | 106 | 72.6% |
| phi4 | variant_type | pure_demographic | 151 | 94.0% |
| phi4 | clinical_subtype | distractor_insertion | 10 | 100.0% |
| phi4 | clinical_subtype | history_reordering | 10 | 100.0% |
| phi4 | clinical_subtype | paraphrase | 10 | 90.0% |
| phi4 | demographic_cell | age | 31 | 87.1% |
| phi4 | demographic_cell | age+ethnicity | 1 ⚠ | 0.0% |
| phi4 | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| phi4 | demographic_cell | age+race | 1 ⚠ | 100.0% |
| phi4 | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| phi4 | demographic_cell | age+sex | 55 | 94.5% |
| phi4 | demographic_cell | ethnicity | 20 | 100.0% |
| phi4 | demographic_cell | race | 20 | 100.0% |
| phi4 | demographic_cell | sex | 20 | 100.0% |
| qwen2.5:7b | variant_type | base | 112 | 79.5% |
| qwen2.5:7b | variant_type | clinical_perturbation | 30 | 86.7% |
| qwen2.5:7b | variant_type | mixed | 106 | 64.2% |
| qwen2.5:7b | variant_type | pure_demographic | 151 | 89.4% |
| qwen2.5:7b | clinical_subtype | distractor_insertion | 10 | 100.0% |
| qwen2.5:7b | clinical_subtype | history_reordering | 10 | 90.0% |
| qwen2.5:7b | clinical_subtype | paraphrase | 10 | 70.0% |
| qwen2.5:7b | demographic_cell | age | 31 | 87.1% |
| qwen2.5:7b | demographic_cell | age+ethnicity | 1 ⚠ | 100.0% |
| qwen2.5:7b | demographic_cell | age+ethnicity+sex | 1 ⚠ | 100.0% |
| qwen2.5:7b | demographic_cell | age+race | 1 ⚠ | 100.0% |
| qwen2.5:7b | demographic_cell | age+race+sex | 1 ⚠ | 100.0% |
| qwen2.5:7b | demographic_cell | age+sex | 55 | 92.7% |
| qwen2.5:7b | demographic_cell | ethnicity | 20 | 90.0% |
| qwen2.5:7b | demographic_cell | race | 20 | 90.0% |
| qwen2.5:7b | demographic_cell | sex | 20 | 85.0% |