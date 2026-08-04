# Paper

LaTeX scaffold for the MedReason-Bench preprint. Builds with `latexmk -pdf main.tex`.

Figures auto-export from `notebooks/03_results_analysis.ipynb` and
`notebooks/04_fairness_deep_dive.ipynb` into `paper/figures/`.

Tables auto-generate from `results/aggregated/leaderboard.json` via
`scripts/build_paper_tables.py` (added in Phase 6).
