"""Pull a stratified 20% sample of fabrication-judge verdicts for human spot-check.

Reads ``<results-dir>/judge_verdicts.jsonl``, stratifies by
(model, verdict) so rare verdicts are represented, and writes
``<results-dir>/human_review_sample.csv`` with an empty
``human_verdict`` column to fill in (FABRICATED / NOT_FABRICATED /
BORDERLINE). Agreement is then scored by ``scripts/compute_kappa.py``.

Usage:
    python3 scripts/sample_for_human_review.py --results-dir results/baseline_haiku
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    results_dir = REPO_ROOT / args.results_dir
    verdicts_path = results_dir / "judge_verdicts.jsonl"
    if not verdicts_path.exists():
        print(f"No {verdicts_path.relative_to(REPO_ROOT)} — run run_eval.py first.")
        return 1
    rows = [json.loads(line) for line in verdicts_path.read_text().splitlines() if line]
    df = pd.DataFrame(rows)

    rng = random.Random(args.seed)
    picked = []
    for _, stratum in df.groupby(["model", "verdict"]):
        k = max(1, math.ceil(len(stratum) * args.fraction))
        picked.extend(rng.sample(list(stratum.index), k))
    sample = df.loc[sorted(picked)].copy()
    sample["human_verdict"] = ""
    sample["human_note"] = ""

    out = results_dir / "human_review_sample.csv"
    sample.to_csv(out, index=False)
    print(
        f"Wrote {len(sample)}/{len(df)} verdicts to {out.relative_to(REPO_ROOT)} "
        f"(stratified by model x verdict, seed {args.seed}).\n"
        "Fill human_verdict, then: python3 scripts/compute_kappa.py "
        f"--reviewed {out.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
