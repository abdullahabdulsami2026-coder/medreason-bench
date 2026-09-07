"""Build results/table1.md — dataset descriptives for the vignette corpus.

Reports case counts by specialty x difficulty, adversarial counts,
variant counts by perturbation_type and perturbation_subtype, and the
per-cell demographic-swap counts (which attributes a pure_demographic
variant changes) that the fairness metrics stratify on.

Usage:
    python3 scripts/build_table1.py [--include-drafts] [--out results/table1.md]

``--include-drafts`` adds a second set of columns previewing the corpus
as it would look if all pending + approved drafts were promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.vignettes.loader import dataset_version, load_vignettes  # noqa: E402
from eval.schemas import DraftRecord, Vignette, VignetteVariant  # noqa: E402

DRAFTS_DIR = REPO_ROOT / "data" / "vignettes" / "drafts"


def load_draft_payloads() -> tuple[list[tuple[str, VignetteVariant]], list[Vignette]]:
    """(parent_id, variant) pairs and new vignettes from non-rejected drafts."""
    variants: list[tuple[str, VignetteVariant]] = []
    vignettes: list[Vignette] = []
    for path in sorted(DRAFTS_DIR.glob("*/*.json")):
        record = DraftRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.status == "rejected":
            continue
        if record.variant is not None and record.parent_id is not None:
            variants.append((record.parent_id, record.variant))
        elif record.vignette is not None:
            vignettes.append(record.vignette)
    return variants, vignettes


class Descriptives:
    """Counters for one corpus snapshot (with or without drafts)."""

    def __init__(self, cases: list[Vignette], extra: list[tuple[str, VignetteVariant]]) -> None:
        by_id = {v.id: v for v in cases}
        pairs = [(v, var) for v in cases for var in v.variants]
        pairs += [(by_id[pid], var) for pid, var in extra if pid in by_id]

        self.n_cases = len(cases)
        self.n_variants = len(pairs)
        self.cases_by_bucket = Counter((v.specialty, v.difficulty) for v in cases)
        self.n_adversarial = sum(1 for v in cases if v.correct is None)
        self.variant_types = Counter(var.perturbation_type for _, var in pairs)
        self.subtypes = Counter(
            var.perturbation_subtype for _, var in pairs if var.perturbation_subtype is not None
        )
        self.demo_cells: Counter[str] = Counter()
        for v, var in pairs:
            if var.perturbation_type != "pure_demographic":
                continue
            base = v.demographics.model_dump()
            changed = sorted(
                k
                for k in ("age", "sex", "race", "ethnicity")
                if base[k] != getattr(var.demographics, k)
            )
            self.demo_cells["+".join(changed) if changed else "(stem-only)"] += 1


def rows(keys: list[str], *snaps: Descriptives, get: str) -> list[str]:
    out = []
    for key in keys:
        counts = " | ".join(str(getattr(s, get).get(key, 0)) for s in snaps)
        out.append(f"| {key} | {counts} |")
    return out


def build(include_drafts: bool) -> str:
    cases = load_vignettes()
    snaps = [Descriptives(cases, [])]
    header = "Corpus"
    if include_drafts:
        extra_variants, extra_vignettes = load_draft_payloads()
        snaps.append(Descriptives(cases + extra_vignettes, extra_variants))
        header = "Corpus | + drafts"

    lines = [
        "# Table 1 — Dataset descriptives",
        "",
        f"Corpus version: {dataset_version()}. "
        "Counts regenerate via `python3 scripts/build_table1.py`.",
        "",
        f"| | {header} |",
        f"|---|{'---|' * len(snaps)}",
        f"| Cases | {' | '.join(str(s.n_cases) for s in snaps)} |",
        f"| Adversarial cases (correct = null) | "
        f"{' | '.join(str(s.n_adversarial) for s in snaps)} |",
        f"| Variants | {' | '.join(str(s.n_variants) for s in snaps)} |",
        "",
        "## Cases by specialty x difficulty",
        "",
        f"| specialty, difficulty | {header} |",
        f"|---|{'---|' * len(snaps)}",
    ]
    bucket_keys = sorted({k for s in snaps for k in s.cases_by_bucket})
    for sp, diff in bucket_keys:
        counts = " | ".join(str(s.cases_by_bucket.get((sp, diff), 0)) for s in snaps)
        lines.append(f"| {sp}, {diff} | {counts} |")

    lines += [
        "",
        "## Variants by perturbation type",
        "",
        f"| perturbation_type | {header} |",
        f"|---|{'---|' * len(snaps)}",
        *rows(
            sorted({k for s in snaps for k in s.variant_types}),
            *snaps,
            get="variant_types",
        ),
        "",
        "## Clinical-perturbation subtypes",
        "",
        f"| perturbation_subtype | {header} |",
        f"|---|{'---|' * len(snaps)}",
        *rows(
            sorted({k for s in snaps for k in s.subtypes})
            or ["distractor_insertion", "history_reordering", "paraphrase"],
            *snaps,
            get="subtypes",
        ),
        "",
        "## Pure-demographic swap cells (changed attributes)",
        "",
        "Fairness metrics compare accuracy within-case across these cells; "
        "single-attribute cells (sex, race, ethnicity) are the primary contrasts.",
        "",
        f"| changed attribute(s) | {header} |",
        f"|---|{'---|' * len(snaps)}",
        *rows(
            sorted({k for s in snaps for k in s.demo_cells}),
            *snaps,
            get="demo_cells",
        ),
        "",
    ]
    return "\n".join(lines)


def corpus_stats() -> dict[str, object]:
    """Machine-readable corpus descriptives, incl. demographic-cell sizes."""
    d = Descriptives(load_vignettes(), [])
    return {
        "dataset_version": dataset_version(),
        "n_cases": d.n_cases,
        "n_variants": d.n_variants,
        "n_adversarial_cases": d.n_adversarial,
        "cases_by_specialty_difficulty": {
            f"{sp}/{diff}": n for (sp, diff), n in sorted(d.cases_by_bucket.items())
        },
        "variants_by_perturbation_type": dict(sorted(d.variant_types.items())),
        "clinical_perturbation_subtypes": dict(sorted(d.subtypes.items())),
        "demographic_swap_cells": dict(sorted(d.demo_cells.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-drafts", action="store_true")
    parser.add_argument("--out", default="results/table1.md")
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Also write machine-readable corpus stats to this path.",
    )
    args = parser.parse_args(argv)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.include_drafts), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    if args.stats_json:
        stats_path = REPO_ROOT / args.stats_json
        stats_path.write_text(json.dumps(corpus_stats(), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {stats_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
