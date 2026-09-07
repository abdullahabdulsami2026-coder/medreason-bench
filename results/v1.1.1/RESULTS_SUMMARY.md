# Results summary

## Reproducibility (methods section)

- Models: claude-haiku-4-5, ollama/llama3.1:8b, ollama/qwen2.5:7b, ollama/mistral:7b, ollama/phi4, ollama/gemma2:9b
- Seeds: 1 (0..0); temperature 0.0
- Prompt template: `mcq`; aggregate prompt hash `26c85aa0de7fa420b02ec067051c2cd51c6640906371829c4544eda1f2d0cbb0`
- Confidence: verbalized 0–100; judge model claude-sonnet-4-6
- Corpus: v1.1.1 (399 items: 112 base + 287 variants)
- Code commit: `ca9491116219353462b666d342a0cca1de616f95`

### Model versions

- claude-haiku-4-5: Anthropic API alias (version pinned by provider at run time)
- ollama/llama3.1:8b: digest 46e0c10c039e (ollama 0.33.3, Metal/Apple M1 Max)
- ollama/qwen2.5:7b: digest 845dbda0ea48 (ollama 0.33.3, Metal/Apple M1 Max)
- ollama/mistral:7b: digest 6577803aa9a0 (ollama 0.33.3, Metal/Apple M1 Max)
- ollama/phi4:latest: digest ac896e5b8b34 (ollama 0.33.3, Metal/Apple M1 Max)
- ollama/gemma2:9b: digest ff02c3702f32 (ollama 0.33.3, Metal/Apple M1 Max)

## Headline

- **claude-haiku-4-5**: accuracy 86.7% [83.0, 89.7], consistency 95.9%, ECE 0.076, Brier 0.090, abstention 0.0% / false-abstention 0.0%, fabrication 76.9%
- **phi4**: accuracy 85.7% [81.9, 88.8], consistency 93.9%, ECE 0.064, Brier 0.108, abstention 0.0% / false-abstention 0.0%, fabrication 80.8%
- **mistral:7b**: accuracy 81.5% [77.3, 85.0], consistency 95.9%, ECE 0.134, Brier 0.165, abstention 15.4% / false-abstention 0.0%, fabrication 90.9%
- **llama3.1:8b**: accuracy 81.2% [77.1, 84.7], consistency 96.9%, ECE 0.074, Brier 0.147, abstention 0.0% / false-abstention 0.0%, fabrication 88.5%
- **gemma2:9b**: accuracy 79.7% [75.5, 83.4], consistency 94.9%, ECE 0.110, Brier 0.166, abstention 0.0% / false-abstention 0.0%, fabrication 88.5%
- **qwen2.5:7b**: accuracy 79.7% [75.5, 83.4], consistency 95.9%, ECE 0.130, Brier 0.171, abstention 0.0% / false-abstention 0.0%, fabrication 92.3%

## Pairwise McNemar

- claude-haiku-4-5 vs gemma2:9b: discordant 34/6 on 399 shared items, p=0.0000 (significant at p<0.05)
- claude-haiku-4-5 vs llama3.1:8b: discordant 26/4 on 399 shared items, p=0.0001 (significant at p<0.05)
- claude-haiku-4-5 vs mistral:7b: discordant 33/12 on 399 shared items, p=0.0025 (significant at p<0.05)
- claude-haiku-4-5 vs phi4: discordant 9/5 on 399 shared items, p=0.4240 (not significant)
- claude-haiku-4-5 vs qwen2.5:7b: discordant 36/8 on 399 shared items, p=0.0000 (significant at p<0.05)
- gemma2:9b vs llama3.1:8b: discordant 18/24 on 399 shared items, p=0.4408 (not significant)
- gemma2:9b vs mistral:7b: discordant 15/22 on 399 shared items, p=0.3240 (not significant)
- gemma2:9b vs phi4: discordant 6/30 on 399 shared items, p=0.0001 (significant at p<0.05)
- gemma2:9b vs qwen2.5:7b: discordant 13/13 on 399 shared items, p=1.0000 (not significant)
- llama3.1:8b vs mistral:7b: discordant 25/26 on 399 shared items, p=1.0000 (not significant)
- llama3.1:8b vs phi4: discordant 6/24 on 399 shared items, p=0.0014 (significant at p<0.05)
- llama3.1:8b vs qwen2.5:7b: discordant 23/17 on 399 shared items, p=0.4296 (not significant)
- mistral:7b vs phi4: discordant 13/30 on 399 shared items, p=0.0137 (significant at p<0.05)
- mistral:7b vs qwen2.5:7b: discordant 25/18 on 399 shared items, p=0.3604 (not significant)
- phi4 vs qwen2.5:7b: discordant 34/10 on 399 shared items, p=0.0004 (significant at p<0.05)

## Cells with n < 5 (do not interpret)

- demographic_cell `age+ethnicity`: n=1
- demographic_cell `age+ethnicity+sex`: n=1
- demographic_cell `age+race`: n=1
- demographic_cell `age+race+sex`: n=1
