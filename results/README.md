# Results

Evaluation output lands here. Both subdirectories are empty in version control by
design — no runs are committed.

| Path | Contents |
| --- | --- |
| `runs/` | Raw per-run output from `scripts/run_full_eval.sh` (or `python3 -m eval.pipeline`): one `.jsonl` of per-item responses plus a `.manifest.json` reproducibility record per run. |
| `aggregated/` | Leaderboard tables built from `runs/` by `scripts/build_leaderboard.py`. |

Populated by `run_full_eval`; regenerate rather than commit. No results have been
produced yet — see the project status in the top-level [README](../README.md).
