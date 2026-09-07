"""Cohen's kappa between the fabrication judge and the human spot-check.

kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
p_e the chance agreement from the two raters' marginal distributions.
Rows with an empty ``human_verdict`` are skipped and counted.

Usage:
    python3 scripts/compute_kappa.py --reviewed results/baseline_haiku/human_review_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in ca) / n**2
    return (p_o - p_e) / (1 - p_e) if p_e != 1 else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviewed", required=True)
    args = parser.parse_args(argv)

    df = pd.read_csv(REPO_ROOT / args.reviewed).fillna({"human_verdict": ""})
    done = df[df["human_verdict"].str.strip() != ""]
    if done.empty:
        print("No rows have human_verdict filled in yet.")
        return 1
    judge = done["verdict"].str.strip().str.upper().tolist()
    human = done["human_verdict"].str.strip().str.upper().tolist()
    kappa = cohens_kappa(judge, human)
    agree = sum(x == y for x, y in zip(judge, human, strict=True))
    print(f"{len(done)}/{len(df)} rows reviewed; raw agreement {agree}/{len(done)}")
    print(f"Cohen's kappa: {kappa:.3f}")
    disagreements = done[
        done["verdict"].str.upper() != done["human_verdict"].str.strip().str.upper()
    ]
    for _, r in disagreements.iterrows():
        print(
            f"  disagree {r['model']}/{r['item_id']}: judge={r['verdict']} "
            f"human={r['human_verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
