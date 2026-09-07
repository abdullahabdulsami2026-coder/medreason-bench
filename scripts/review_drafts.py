"""Review corpus-expansion drafts and promote approved ones into the corpus.

Nothing enters ``data/vignettes/{cardiology,autoimmune}/`` until a human
approves it here. Three subcommands:

    python3 scripts/review_drafts.py list
        Draft counts by category and status.

    python3 scripts/review_drafts.py review [--category NAME] [--all]
        Interactive review of pending drafts (--all revisits everything).
        Variants show the original and variant stems side by side plus a
        word-level change summary. Per draft: [a]pprove / [e]dit (opens
        $EDITOR, revalidates) / [r]eject / [s]kip / [q]uit.

    python3 scripts/review_drafts.py promote --version X.Y.Z
        Merge approved drafts into the corpus: variants are appended to
        their parent case file, adversarial vignettes become new case
        files. Bumps data/vignettes/VERSION, prepends a CHANGELOG.md
        entry, marks the drafts promoted, and revalidates the corpus.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.vignettes.loader import VERSION_FILE, dataset_version, load_vignettes  # noqa: E402
from eval.schemas import DraftRecord, Vignette  # noqa: E402

DRAFTS_DIR = REPO_ROOT / "data" / "vignettes" / "drafts"
PRESCREEN_CSV = DRAFTS_DIR / "prescreen.csv"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CATEGORIES = ("clinical_perturbation", "pure_demographic", "adversarial")
SPECIALTY_DIR = {"cardiology": "cardiology", "autoimmune": "autoimmune"}


def prescreen_flags() -> dict[str, tuple[str, str]]:
    """draft_id -> (flag, reason) from scripts/prescreen_drafts.py, if it ran."""
    if not PRESCREEN_CSV.exists():
        return {}
    with PRESCREEN_CSV.open() as fp:
        return {row["draft_id"]: (row["flag"], row["reason"]) for row in csv.DictReader(fp)}


def draft_files(category: str | None = None) -> list[Path]:
    cats = (category,) if category else CATEGORIES
    out: list[Path] = []
    for cat in cats:
        cat_dir = DRAFTS_DIR / cat
        if cat_dir.is_dir():
            out.extend(sorted(cat_dir.glob("*.json")))
    return out


def read_draft(path: Path) -> DraftRecord:
    return DraftRecord.model_validate_json(path.read_text(encoding="utf-8"))


def write_draft(path: Path, record: DraftRecord) -> None:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


# ── Display helpers ───────────────────────────────────────────────


def side_by_side(left: str, right: str, left_title: str, right_title: str) -> str:
    width = min(shutil.get_terminal_size((120, 40)).columns, 160)
    col = (width - 3) // 2
    lw = textwrap.wrap(left, col) or [""]
    rw = textwrap.wrap(right, col) or [""]
    lines = [f"{left_title:<{col}} | {right_title}", f"{'-' * col}-+-{'-' * col}"]
    for i in range(max(len(lw), len(rw))):
        lline = lw[i] if i < len(lw) else ""
        rline = rw[i] if i < len(rw) else ""
        lines.append(f"{lline:<{col}} | {rline}")
    return "\n".join(lines)


def word_changes(original: str, revised: str, limit: int = 12) -> list[str]:
    """Human-scannable word-level diff: '- removed' / '+ added' chunks."""
    matcher = difflib.SequenceMatcher(a=original.split(), b=revised.split())
    changes = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        removed = " ".join(matcher.a[a0:a1])
        added = " ".join(matcher.b[b0:b1])
        if removed:
            changes.append(f"  - {removed[:160]}")
        if added:
            changes.append(f"  + {added[:160]}")
    if len(changes) > limit:
        changes = changes[:limit] + [f"  ... {len(changes) - limit} more change chunks"]
    return changes


def show_variant(record: DraftRecord, parent: Vignette) -> None:
    variant = record.variant
    assert variant is not None
    subtype = f" / {variant.perturbation_subtype}" if variant.perturbation_subtype else ""
    print(f"\n=== {record.draft_id}  [{variant.perturbation_type}{subtype}] ===")
    print(
        f"parent: {parent.id} ({parent.specialty}, {parent.difficulty})  "
        f"correct: {parent.correct}  model: {record.generator_model}"
    )
    base_demo = parent.demographics.model_dump()
    var_demo = variant.demographics.model_dump()
    demo_delta = {
        k: f"{base_demo[k]!r} -> {var_demo[k]!r}" for k in base_demo if base_demo[k] != var_demo[k]
    }
    print(f"demographics: {demo_delta or 'unchanged'}")
    print(f"question: {parent.question}\n")
    print(
        side_by_side(
            parent.stem, variant.stem_override or parent.stem, "ORIGINAL STEM", "VARIANT STEM"
        )
    )
    print("\nword-level changes:")
    for line in word_changes(parent.stem, variant.stem_override or parent.stem):
        print(line)
    print(f"\nrationale_for_variant: {variant.rationale_for_variant}")


def show_vignette(record: DraftRecord) -> None:
    v = record.vignette
    assert v is not None
    print(f"\n=== {record.draft_id}  [adversarial — correct is null] ===")
    print(f"{v.specialty} / {v.subspecialty} / {v.difficulty}  model: {record.generator_model}")
    print(f"demographics: {v.demographics.model_dump()}\n")
    print(textwrap.fill(v.stem, 100))
    print(f"\nQ: {v.question}")
    for letter, text in sorted(v.options.items()):
        print(f"  {letter}. {text}")
    print(f"\nrationale (why abstention is correct):\n{textwrap.fill(v.rationale, 100)}")
    print("\nsources (verify these are real):")
    for s in v.sources:
        print(f"  - {s}")


# ── Subcommands ───────────────────────────────────────────────────


def cmd_list() -> int:
    counts: Counter[tuple[str, str]] = Counter()
    for path in draft_files():
        record = read_draft(path)
        counts[(record.category, record.status)] += 1
    if not counts:
        print("No drafts found. Generate some with scripts/generate_drafts.py.")
        return 0
    print(f"{'category':<24} {'status':<10} count")
    for (cat, status), n in sorted(counts.items()):
        print(f"{cat:<24} {status:<10} {n}")
    print(f"\ncorpus version: {dataset_version()}")
    return 0


def cmd_review(category: str | None, include_reviewed: bool) -> int:
    corpus = {v.id: v for v in load_vignettes()}
    flags = prescreen_flags()
    pending = []
    for path in draft_files(category):
        record = read_draft(path)
        if include_reviewed or record.status == "pending":
            pending.append((path, record))
    # Pre-screen-flagged drafts first, so the risky ones get fresh eyes.
    pending.sort(key=lambda pr: flags.get(pr[1].draft_id, ("zz",))[0] != "review")
    if not pending:
        print("Nothing to review.")
        return 0
    n_flagged = sum(1 for _, r in pending if flags.get(r.draft_id, ("",))[0] == "review")
    print(f"{len(pending)} draft(s) to review ({n_flagged} pre-screen flagged, shown first).")

    for i, (path, record) in enumerate(pending, 1):
        while True:
            if record.variant is not None:
                assert record.parent_id is not None
                show_variant(record, corpus[record.parent_id])
            else:
                show_vignette(record)
            flag = flags.get(record.draft_id)
            if flag is not None:
                marker = "⚠ FLAGGED" if flag[0] == "review" else "pre-screen ok"
                print(f"\npre-screen: {marker}" + (f" — {flag[1]}" if flag[0] == "review" else ""))
            print(f"\n[{i}/{len(pending)}] current status: {record.status}")
            choice = input("[a]pprove / [e]dit / [r]eject / [s]kip / [q]uit > ").strip().lower()
            if choice == "a":
                record.reviewed = True
                record.status = "approved"
                write_draft(path, record)
                break
            if choice == "r":
                record.reviewed = True
                record.status = "rejected"
                record.review_note = input("reject note (optional) > ").strip()
                write_draft(path, record)
                break
            if choice == "e":
                editor = os.environ.get("EDITOR", "vi")
                subprocess.run([editor, str(path)], check=False)
                try:
                    record = read_draft(path)
                    print("re-validated OK — review the edited draft:")
                except Exception as e:  # noqa: BLE001 — show the error, re-edit
                    print(f"\nINVALID after edit: {e}\nFix it or the draft stays pending.")
                continue
            if choice == "s":
                break
            if choice == "q":
                return 0
            print("unrecognized choice")
    return 0


def merge_variant(record: DraftRecord) -> Path:
    """Append the draft's variant to its parent case file (raw JSON, order kept)."""
    assert record.variant is not None and record.parent_id is not None
    parent_path = next(
        p
        for sp in SPECIALTY_DIR.values()
        for p in (REPO_ROOT / "data" / "vignettes" / sp).glob("*.json")
        if not p.name.startswith("_")
        and json.loads(p.read_text(encoding="utf-8"))["id"] == record.parent_id
    )
    payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if any(v["variant_id"] == record.variant.variant_id for v in payload["variants"]):
        raise ValueError(f"{record.parent_id} already has variant {record.variant.variant_id}")
    variant_dict = record.variant.model_dump()
    if variant_dict.get("perturbation_subtype") is None:
        variant_dict.pop("perturbation_subtype", None)  # match existing file style
    payload["variants"].append(variant_dict)
    Vignette.model_validate(payload)
    parent_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return parent_path


def merge_vignette(record: DraftRecord) -> Path:
    assert record.vignette is not None
    v = record.vignette
    out = REPO_ROOT / "data" / "vignettes" / SPECIALTY_DIR[v.specialty] / f"{v.id}.json"
    if out.exists():
        raise ValueError(f"{out.name} already exists")
    out.write_text(
        json.dumps(v.model_dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def cmd_promote(version: str) -> int:
    approved: list[tuple[Path, DraftRecord]] = []
    for path in draft_files():
        record = read_draft(path)
        if record.status == "approved":
            if not record.reviewed:
                print(f"refusing {record.draft_id}: approved but reviewed is false")
                return 1
            approved.append((path, record))
    if not approved:
        print("No approved drafts to promote.")
        return 0

    touched: list[str] = []
    by_category: Counter[str] = Counter()
    for path, record in approved:
        target = merge_variant(record) if record.variant is not None else merge_vignette(record)
        touched.append(str(target.relative_to(REPO_ROOT)))
        by_category[record.category] += 1
        record.status = "promoted"
        write_draft(path, record)

    load_vignettes()  # full-corpus validation; raises on any schema violation

    old_version = dataset_version()
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_table1.py"),
            "--out",
            "results/table1_preview.md",
            "--stats-json",
            "data/vignettes/corpus_stats.json",
        ],
        check=True,
    )
    today = dt.date.today().isoformat()
    summary = ", ".join(f"{n} {cat}" for cat, n in sorted(by_category.items()))
    entry = (
        f"## {version} — {today}\n\n"
        f"- Promoted {len(approved)} reviewed draft(s) into the corpus: {summary}.\n"
        f"- Previous version: {old_version}.\n\n"
    )
    existing = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else "# Changelog\n\n"
    head, _, tail = existing.partition("\n## ")
    CHANGELOG.write_text(head + "\n" + entry + ("## " + tail if tail else ""), encoding="utf-8")

    print(f"Promoted {len(approved)} draft(s) ({summary}).")
    print(f"Corpus version: {old_version} -> {version}. Files touched:")
    for t in sorted(set(touched)):
        print(f"  {t}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list")
    review = sub.add_parser("review")
    review.add_argument("--category", choices=list(CATEGORIES), default=None)
    review.add_argument("--all", action="store_true", help="Revisit already-reviewed drafts too.")
    promote = sub.add_parser("promote")
    promote.add_argument("--version", required=True, help="New corpus version, e.g. 1.1.0.")
    args = parser.parse_args(argv)

    if args.cmd == "review":
        return cmd_review(args.category, args.all)
    if args.cmd == "promote":
        return cmd_promote(args.version)
    return cmd_list()


if __name__ == "__main__":
    sys.exit(main())
