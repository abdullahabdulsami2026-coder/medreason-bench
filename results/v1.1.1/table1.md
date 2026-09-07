# Table 1 — Model metrics

Corpus v1.1.1, mean ± SD across seeds; Wilson 95% CI on pooled accuracy.

| model | accuracy | 95% CI | consistency | ECE | Brier | abstention (unanswerable) | false-abstention (answerable) |
|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 86.7% | [83.0, 89.7] | 95.9% | 0.076 | 0.090 | 0.0% | 0.0% |
| phi4 | 85.7% | [81.9, 88.8] | 93.9% | 0.064 | 0.108 | 0.0% | 0.0% |
| mistral:7b | 81.5% | [77.3, 85.0] | 95.9% | 0.134 | 0.165 | 15.4% | 0.0% |
| llama3.1:8b | 81.2% | [77.1, 84.7] | 96.9% | 0.074 | 0.147 | 0.0% | 0.0% |
| gemma2:9b | 79.7% | [75.5, 83.4] | 94.9% | 0.110 | 0.166 | 0.0% | 0.0% |
| qwen2.5:7b | 79.7% | [75.5, 83.4] | 95.9% | 0.130 | 0.171 | 0.0% | 0.0% |

Full metric set (macro-F1, flip rate, robustness gap, fabrication, max fairness gap) in table1.csv.