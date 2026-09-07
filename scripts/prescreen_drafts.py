"""Second-model pre-screen of corpus-expansion drafts.

An independent judge (default claude-sonnet-4-6, prompted as an
adversarial reviewer — a different framing from the generator) answers,
per draft:

1. Is the stated correct answer actually supported by the stem?
2. For answer-preserving variants: did anything change between base and
   variant that could plausibly change the answer?
3. For adversarial cases: is there an option a competent clinician
   would still defensibly pick from the stem?

Each draft gets ``ok`` or ``review`` plus a one-line reason in
``data/vignettes/drafts/prescreen.csv``; ``review_drafts.py`` shows the
flag and puts flagged drafts first. Already-screened drafts are
skipped, so the script is resumable.

Usage:
    python3 scripts/prescreen_drafts.py [--model claude-sonnet-4-6]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.vignettes.loader import load_vignettes  # noqa: E402
from eval.runners.anthropic_runner import AnthropicRunner  # noqa: E402
from eval.schemas import DraftRecord, Vignette  # noqa: E402

DRAFTS_DIR = REPO_ROOT / "data" / "vignettes" / "drafts"
PRESCREEN_CSV = DRAFTS_DIR / "prescreen.csv"
FIELDS = ["draft_id", "category", "flag", "reason", "judge_model", "judged_at"]

VARIANT_PROMPT = """\
You are an adversarial reviewer for a clinical-reasoning benchmark. A generator
produced a VARIANT of an existing vignette; the benchmark requires that the variant
preserves the original correct answer exactly. Your job is to find problems, not to
be agreeable.

Original stem:
{base_stem}

Variant stem ({variant_kind}):
{variant_stem}

Question: {question}

Options:
{options_block}

Stated correct answer: {correct}. Original rationale: {rationale}

Answer these:
1. Is {correct} still unambiguously the best answer given the VARIANT stem?
2. Did the variant add, remove, or alter anything that could plausibly move a
   competent clinician to a different option (new finding, lost finding, changed
   number, demographic change with real clinical import)?

Reply EXACTLY as:
FLAG: <OK if the answer is preserved and unambiguous, REVIEW otherwise>
REASON: <one line; if REVIEW, name the specific problem>
"""

ADVERSARIAL_PROMPT = """\
You are an adversarial reviewer for a clinical-reasoning benchmark. A generator
produced a vignette claimed to be UNANSWERABLE: none of the options is supposed to
be supported by the stem, and the graded-correct behaviour is abstention. Your job
is to find problems, not to be agreeable.

Stem:
{stem}

Question: {question}

Options:
{options_block}

Generator's rationale for unanswerability: {rationale}

Every multiple-choice item admits a least-bad forced guess; that alone does NOT
fail the item (the benchmark scores abstention as correct precisely because the
stem does not license any option). Judge against this standard instead:

1. Does the stem POSITIVELY ESTABLISH any single option — could a clinician cite
   specific findings present in the stem that license that option over the others,
   without asserting facts not in evidence? If yes, name the option and the
   findings; the item fails.
2. Is the stem internally consistent and clinically plausible?

Reply EXACTLY as:
FLAG: <OK if no option is established by stem findings and the stem is consistent,
REVIEW otherwise>
REASON: <one line; if REVIEW, name the established option + findings, or the
inconsistency>
"""


def build_prompt(record: DraftRecord, corpus: dict[str, Vignette]) -> str:
    if record.vignette is not None:
        v = record.vignette
        return ADVERSARIAL_PROMPT.format(
            stem=v.stem,
            question=v.question,
            options_block="\n".join(f"{k}. {t}" for k, t in sorted(v.options.items())),
            rationale=v.rationale,
        )
    assert record.variant is not None and record.parent_id is not None
    parent = corpus[record.parent_id]
    variant = record.variant
    kind = variant.perturbation_type + (
        f" / {variant.perturbation_subtype}" if variant.perturbation_subtype else ""
    )
    return VARIANT_PROMPT.format(
        base_stem=parent.stem,
        variant_stem=variant.stem_override or parent.stem,
        variant_kind=kind,
        question=parent.question,
        options_block="\n".join(f"{k}. {t}" for k, t in sorted(parent.options.items())),
        correct=parent.correct,
        rationale=parent.rationale,
    )


def parse_flag(text: str) -> tuple[str, str]:
    flag = "review"  # unparseable judge output lands in the review pile
    reason = text.strip().replace("\n", " ")[:300]
    for line in text.splitlines():
        upper = line.upper().replace("**", "").strip()
        if upper.startswith("FLAG:"):
            flag = "ok" if "OK" in upper.split(":", 1)[1] else "review"
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return flag, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=REPO_ROOT / ".env.local", override=False)
        load_dotenv(override=False)
    except ImportError:
        pass

    corpus = {v.id: v for v in load_vignettes()}
    done: set[str] = set()
    if PRESCREEN_CSV.exists():
        with PRESCREEN_CSV.open() as fp:
            done = {row["draft_id"] for row in csv.DictReader(fp)}

    records = [
        DraftRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(DRAFTS_DIR.glob("*/*.json"))
    ]
    todo = [r for r in records if r.draft_id not in done and r.status != "rejected"]
    print(f"{len(todo)} draft(s) to pre-screen ({len(done)} already done)")
    if not todo:
        return 0

    judge = AnthropicRunner(model=args.model, max_tokens=512)
    new_file = not PRESCREEN_CSV.exists()
    n_flagged = 0
    with PRESCREEN_CSV.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for record in todo:
            try:
                text = judge.complete(build_prompt(record, corpus))
            except Exception as e:  # noqa: BLE001 — leave the draft unscreened, keep going
                print(f"  judge failed on {record.draft_id}: {type(e).__name__}: {e}")
                continue
            flag, reason = parse_flag(text)
            n_flagged += flag == "review"
            writer.writerow(
                {
                    "draft_id": record.draft_id,
                    "category": record.category,
                    "flag": flag,
                    "reason": reason,
                    "judge_model": args.model,
                    "judged_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                }
            )
            fp.flush()
            print(f"  {record.draft_id}: {flag}" + (f" — {reason}" if flag == "review" else ""))
    print(
        f"\nDone. {n_flagged} flagged for review-first. See {PRESCREEN_CSV.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
