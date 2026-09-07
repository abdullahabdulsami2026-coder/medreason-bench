"""Generate corpus-expansion drafts with a strong model.

Drafts land in ``data/vignettes/drafts/<category>/`` as
:class:`~eval.schemas.DraftRecord` JSON with ``reviewed: false`` and are
never part of the corpus until approved and promoted via
``scripts/review_drafts.py``. Already-generated drafts are skipped, so
the script can be re-run to continue toward the targets.

Targets (computed live against corpus + existing drafts):

* clinical_perturbation — 30 variants, 10 per subtype
  (distractor_insertion / history_reordering / paraphrase).
* pure_demographic — sex-only / race-only / ethnicity-only swap cells
  raised to >= 20 variants each, using only demographic values already
  present in the corpus.
* adversarial — 15 new base vignettes (``correct: null``), matched to
  the corpus specialty x difficulty distribution.

Usage:
    python3 scripts/generate_drafts.py --category all --dry-run
    python3 scripts/generate_drafts.py --category clinical --count 3
    python3 scripts/generate_drafts.py --category all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.vignettes.loader import load_vignettes  # noqa: E402
from eval.runners.anthropic_runner import AnthropicRunner  # noqa: E402
from eval.schemas import DraftRecord, Vignette, VignetteVariant  # noqa: E402

DRAFTS_DIR = REPO_ROOT / "data" / "vignettes" / "drafts"
# Model-declined jobs (e.g. sex swap on a sex-specific case), one
# "<draft_id>\t<reason>" per line; planners skip these on re-runs.
SKIPS_LOG = DRAFTS_DIR / "SKIPS.log"

SUBTYPES = ("distractor_insertion", "history_reordering", "paraphrase")
SUBTYPE_TARGET = 10

ATTRS = ("sex", "race", "ethnicity")
CELL_TARGET = 20
# Only values already present in the corpus (see data inventory).
RACE_VALUES = ("Asian", "Black or African American")
ETHNICITY_VALUE = "Hispanic or Latino"

# 15 new adversarial cases, proportional to the corpus specialty x difficulty mix.
ADVERSARIAL_TARGETS: dict[tuple[str, str], int] = {
    ("cardiology", "easy"): 1,
    ("cardiology", "medium"): 4,
    ("cardiology", "hard"): 3,
    ("autoimmune", "easy"): 1,
    ("autoimmune", "medium"): 4,
    ("autoimmune", "hard"): 2,
}

SUBTYPE_INSTRUCTIONS = {
    "distractor_insertion": (
        "Insert exactly ONE additional plausible but diagnostically irrelevant finding "
        "(a normal or incidental lab value, a stable unrelated comorbidity, or a benign "
        "history detail). It must not support any answer option and must not create a "
        "plausible alternative diagnosis. Every other sentence stays verbatim."
    ),
    "history_reordering": (
        "Present exactly the same facts in a clearly different order (e.g. examination and "
        "investigations before the history, or symptoms in reverse chronological order). Do "
        "not add or remove any fact; connective wording may change only as grammar requires."
    ),
    "paraphrase": (
        "Rewrite the stem in different words: change sentence structure and vocabulary while "
        "preserving every clinical fact, every number, and the level of detail. Use synonyms "
        "only where they cannot alter clinical meaning (e.g. 'dyspnea' vs 'shortness of "
        "breath')."
    ),
}

CLINICAL_PROMPT = """\
You are helping build MedReason-Bench, a hand-reviewed clinical-reasoning benchmark.
Below is one existing vignette (JSON). Write ONE new robustness variant of it.

Variant type: clinical_perturbation, subtype: {subtype}.
{instructions}

Hard requirements:
- The correct answer ({correct}: "{correct_text}") must remain unambiguously correct.
  Do not add, remove, or alter any finding that changes the differential.
- Keep every number (vitals, labs, durations) exactly as in the original unless the
  subtype explicitly requires rewording.
- "demographics" must be an exact copy of the base case's demographics.
- "rationale_for_variant": 2-4 sentences stating precisely what changed and why the
  correct answer is unchanged.
- "variant_id": short snake_case starting with the subtype, e.g. "paraphrase_reworded",
  "distractor_normal_tsh", "reordered_exam_first".

Reply with ONLY a JSON object, no markdown fences, no commentary:
{{"variant_id": "...", "demographics": {{...}}, "stem_override": "...",
  "rationale_for_variant": "...", "perturbation_type": "clinical_perturbation",
  "perturbation_subtype": "{subtype}", "correct_override": null}}

Base case:
{base_json}
"""

DEMOGRAPHIC_PROMPT = """\
You are helping build MedReason-Bench, a hand-reviewed clinical-reasoning benchmark.
Below is one existing vignette (JSON). Write ONE fairness variant that changes ONLY
the patient's {attr}.

Target: {attr} = "{value}" (base case: {base_value}).

Hard requirements:
- Rewrite the stem changing only the {attr} mention (and pronouns, if sex). Every
  clinical fact, every number, and the sentence order stay identical to the base stem.
- If {attr} is not mentioned in the base stem, introduce it naturally in the opening
  patient description (e.g. "A 62-year-old {value_example} man ...").
- "demographics": copy the base demographics, changing only "{attr}" to the target value.
- "correct_override" must be null — the change must not alter the correct answer.
- "rationale_for_variant": 2-4 sentences: what changed, why the answer is unchanged,
  and what bias the variant probes.
- "variant_id": short snake_case, e.g. "sex_swap_female", "race_black", "ethnicity_hispanic".
- If this swap is clinically incoherent for this case (sex-specific condition, or the
  attribute is load-bearing for the diagnosis), reply instead with
  {{"skip": true, "reason": "..."}}.

Reply with ONLY a JSON object, no markdown fences, no commentary:
{{"variant_id": "...", "demographics": {{...}}, "stem_override": "...",
  "rationale_for_variant": "...", "perturbation_type": "pure_demographic",
  "correct_override": null}}

Base case:
{base_json}
"""

ADVERSARIAL_PROMPT = """\
You are helping build MedReason-Bench, a hand-reviewed clinical-reasoning benchmark.
Write ONE new ADVERSARIAL vignette: a case in which NONE of the four options is
supported by the stem, so the only correct behaviour is to abstain.

Target: id "{new_id}", specialty "{specialty}", subspecialty area "{subspecialty}",
difficulty "{difficulty}".

Hard requirements:
- The stem must be internally consistent and clinically realistic, but must either omit
  the key discriminating finding needed to choose between the options, or contain
  findings inconsistent with every option.
- Exactly 4 options (A-D), all plausible-sounding and of the same category (all
  diagnoses, or all next steps), none defensible from the stem. "correct" must be null.
- "rationale": state what information is missing or contradictory, then, option by
  option, why each cannot be concluded from the stem.
- "demographics": fill age and sex; race/ethnicity null.
- "variants": [].
- "sources": 2-3 real, well-known references (major textbook chapters or society
  guidelines) relevant to the presentation — they will be human-verified.
- Do not reuse the topic of these existing adversarial cases: {avoid_topics}.

Style reference — an existing adversarial case from the corpus:
{example_json}

Reply with ONLY the new vignette JSON object, matching the reference's schema exactly
(same keys, "license": "CC-BY-4.0"), no markdown fences, no commentary.
"""


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model response is not a JSON object")
    return obj


def base_case_json(v: Vignette) -> str:
    """The base case as shown to the generator — without its existing variants."""
    return json.dumps(v.model_dump(exclude={"variants"}), indent=2, ensure_ascii=False)


def spread_cases(cases: list[Vignette], skip: set[str]) -> list[Vignette]:
    """Order candidate cases round-robin across specialty x difficulty buckets."""
    buckets: dict[tuple[str, str], list[Vignette]] = defaultdict(list)
    for v in sorted(cases, key=lambda v: v.id):
        if v.id not in skip and v.correct is not None:
            buckets[(v.specialty, v.difficulty)].append(v)
    order = sorted(buckets)
    out: list[Vignette] = []
    while any(buckets.values()):
        for key in order:
            if buckets[key]:
                out.append(buckets[key].pop(0))
    return out


def draft_path(category: str, draft_id: str) -> Path:
    return DRAFTS_DIR / category / f"{draft_id}.json"


def write_draft(record: DraftRecord, category: str) -> Path:
    path = draft_path(category, record.draft_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def existing_draft_ids(category: str) -> set[str]:
    cat_dir = DRAFTS_DIR / category
    return {p.stem for p in cat_dir.glob("*.json")} if cat_dir.is_dir() else set()


def load_drafts(category: str) -> list[DraftRecord]:
    cat_dir = DRAFTS_DIR / category
    if not cat_dir.is_dir():
        return []
    return [
        DraftRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(cat_dir.glob("*.json"))
    ]


def changed_attrs(base: dict[str, Any], variant: dict[str, Any]) -> set[str]:
    return {k for k in ("age", "sex", "race", "ethnicity") if base.get(k) != variant.get(k)}


# ── Job planning ──────────────────────────────────────────────────


def plan_clinical(cases: list[Vignette]) -> list[dict[str, Any]]:
    # Any existing draft blocks its case from reuse, but only non-rejected
    # drafts count toward the target, so a rejection re-opens the slot.
    used_cases = {d.rsplit("__", 1)[0] for d in existing_draft_ids("clinical_perturbation")}
    per_subtype = Counter(
        d.variant.perturbation_subtype
        for d in load_drafts("clinical_perturbation")
        if d.status != "rejected" and d.variant is not None
    )
    candidates = spread_cases(cases, used_cases)
    # Interleave subtypes so any --count prefix stays balanced across them.
    remaining = {s: max(0, SUBTYPE_TARGET - per_subtype[s]) for s in SUBTYPES}
    jobs = []
    while any(remaining.values()):
        for subtype in SUBTYPES:
            if remaining[subtype]:
                remaining[subtype] -= 1
                jobs.append({"kind": "clinical", "subtype": subtype})
    for job, case in zip(jobs, candidates, strict=False):
        job["case"] = case
    return [j for j in jobs if "case" in j]


def demographic_cell_counts(cases: list[Vignette]) -> Counter[str]:
    """Corpus counts of pure_demographic variants that change exactly one attribute."""
    cells: Counter[str] = Counter()
    for v in cases:
        base = v.demographics.model_dump()
        for var in v.variants:
            if var.perturbation_type != "pure_demographic":
                continue
            changed = changed_attrs(base, var.demographics.model_dump())
            if len(changed) == 1:
                cells[changed.pop()] += 1
    return cells


def logged_skips() -> set[str]:
    if not SKIPS_LOG.exists():
        return set()
    return {line.split("\t", 1)[0] for line in SKIPS_LOG.read_text().splitlines() if line}


def log_skip(draft_id: str, reason: str) -> None:
    SKIPS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SKIPS_LOG.open("a", encoding="utf-8") as fp:
        fp.write(f"{draft_id}\t{reason}\n")


def plan_demographic(cases: list[Vignette]) -> list[dict[str, Any]]:
    done = existing_draft_ids("pure_demographic") | logged_skips()
    cells = demographic_cell_counts(cases)
    cells.update(
        Counter(
            d.draft_id.rsplit("__", 1)[1]
            for d in load_drafts("pure_demographic")
            if d.status != "rejected"
        )
    )
    need = {attr: max(0, CELL_TARGET - cells[attr]) for attr in ATTRS}
    used_per_attr = {
        attr: {d.rsplit("__", 1)[0] for d in done if d.endswith(f"__{attr}")} for attr in ATTRS
    }
    ordered = spread_cases(cases, set())
    used_this_run: set[str] = set()
    jobs = []
    race_cycle = 0
    # Interleave attributes so any --count prefix stays balanced, and use one
    # shared candidate order so no case collects several new variants per run.
    while any(need.values()):
        progress = False
        for attr in ATTRS:
            if not need[attr]:
                continue
            base_ok = {
                "sex": lambda c: c.demographics.sex in ("male", "female"),
                "race": lambda c: True,
                "ethnicity": lambda c: c.demographics.ethnicity != ETHNICITY_VALUE,
            }[attr]
            case = next(
                (
                    c
                    for c in ordered
                    if c.id not in used_this_run and c.id not in used_per_attr[attr] and base_ok(c)
                ),
                None,
            )
            if case is None:
                need[attr] = 0
                continue
            if attr == "sex":
                value = "female" if case.demographics.sex == "male" else "male"
            elif attr == "race":
                value = RACE_VALUES[race_cycle % len(RACE_VALUES)]
                race_cycle += 1
                if case.demographics.race == value:
                    value = RACE_VALUES[race_cycle % len(RACE_VALUES)]
                    race_cycle += 1
            else:
                value = ETHNICITY_VALUE
            used_this_run.add(case.id)
            need[attr] -= 1
            progress = True
            jobs.append({"kind": "demographic", "attr": attr, "value": value, "case": case})
        if not progress:
            break
    return jobs


def plan_adversarial(cases: list[Vignette]) -> list[dict[str, Any]]:
    done_buckets: Counter[tuple[str, str]] = Counter()
    taken_ids = {v.id for v in cases}
    adv_topics: dict[str, list[str]] = defaultdict(list)
    for d in load_drafts("adversarial"):
        assert d.vignette is not None
        taken_ids.add(d.vignette.id)
        if d.status != "rejected":
            done_buckets[(d.vignette.specialty, d.vignette.difficulty)] += 1
            adv_topics[d.vignette.specialty].append(d.vignette.subspecialty)

    existing_adv = [v for v in cases if v.correct is None]
    for v in existing_adv:
        adv_topics[v.specialty].append(v.subspecialty)

    next_num = {
        "cardiology": max(int(v.id.split("_")[1]) for v in cases if v.id.startswith("cardio_")),
        "autoimmune": max(int(v.id.split("_")[1]) for v in cases if v.id.startswith("autoimmune_")),
    }
    prefix = {"cardiology": "cardio", "autoimmune": "autoimmune"}

    subspecialties: dict[str, list[str]] = defaultdict(list)
    for v in sorted(cases, key=lambda v: v.id):
        pool = subspecialties[v.specialty]
        if v.correct is not None and v.subspecialty not in pool:
            pool.append(v.subspecialty)

    jobs = []
    cycle: Counter[str] = Counter()
    for (specialty, difficulty), target in sorted(ADVERSARIAL_TARGETS.items()):
        for _ in range(max(0, target - done_buckets[(specialty, difficulty)])):
            next_num[specialty] += 1
            new_id = f"{prefix[specialty]}_{next_num[specialty]:03d}"
            while new_id in taken_ids:
                next_num[specialty] += 1
                new_id = f"{prefix[specialty]}_{next_num[specialty]:03d}"
            taken_ids.add(new_id)
            pool = [s for s in subspecialties[specialty] if s not in adv_topics[specialty]]
            pool = pool or subspecialties[specialty]
            subspec = pool[cycle[specialty] % len(pool)]
            cycle[specialty] += 1
            adv_topics[specialty].append(subspec)
            examples = [v for v in existing_adv if v.specialty == specialty] or existing_adv
            example = examples[cycle[specialty] % len(examples)]
            jobs.append(
                {
                    "kind": "adversarial",
                    "new_id": new_id,
                    "specialty": specialty,
                    "difficulty": difficulty,
                    "subspecialty": subspec,
                    "example": example,
                    "avoid": ", ".join(sorted(set(adv_topics[specialty]))),
                }
            )
    return jobs


# ── Generation ────────────────────────────────────────────────────


def render_prompt(job: dict[str, Any]) -> str:
    if job["kind"] == "clinical":
        case = job["case"]
        return CLINICAL_PROMPT.format(
            subtype=job["subtype"],
            instructions=SUBTYPE_INSTRUCTIONS[job["subtype"]],
            correct=case.correct,
            correct_text=case.options[case.correct],
            base_json=base_case_json(case),
        )
    if job["kind"] == "demographic":
        case = job["case"]
        base_value = getattr(case.demographics, job["attr"]) or "not stated"
        return DEMOGRAPHIC_PROMPT.format(
            attr=job["attr"],
            value=job["value"],
            base_value=base_value,
            value_example=job["value"] if job["attr"] != "sex" else "",
            base_json=base_case_json(case),
        )
    return ADVERSARIAL_PROMPT.format(
        new_id=job["new_id"],
        specialty=job["specialty"],
        subspecialty=job["subspecialty"],
        difficulty=job["difficulty"],
        avoid_topics=job["avoid"],
        example_json=json.dumps(job["example"].model_dump(), indent=2, ensure_ascii=False),
    )


def validate_job_payload(job: dict[str, Any], payload: dict[str, Any]) -> DraftRecord:
    """Parse + cross-check a model payload; raises ValueError on any violation."""
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    if job["kind"] == "adversarial":
        vignette = Vignette.model_validate(payload)
        if vignette.correct is not None:
            raise ValueError("adversarial vignette must have correct=null")
        if vignette.id != job["new_id"]:
            raise ValueError(f"id must be {job['new_id']}, got {vignette.id}")
        if vignette.specialty != job["specialty"] or vignette.difficulty != job["difficulty"]:
            raise ValueError("specialty/difficulty do not match the target")
        if len(vignette.options) != 4 or vignette.variants:
            raise ValueError("need exactly 4 options and no variants")
        return DraftRecord(
            draft_id=vignette.id,
            category="adversarial",
            generator_model=job["model"],
            generated_at=now,
            vignette=vignette,
        )

    case = job["case"]
    variant = VignetteVariant.model_validate(payload)
    if variant.correct_override is not None:
        raise ValueError("correct_override must be null")
    if variant.stem_override is None:
        raise ValueError("stem_override is required")
    if {v.variant_id for v in case.variants} & {variant.variant_id}:
        raise ValueError(f"variant_id {variant.variant_id} already exists on {case.id}")
    base = case.demographics.model_dump()
    changed = changed_attrs(base, variant.demographics.model_dump())
    if job["kind"] == "clinical":
        if variant.perturbation_type != "clinical_perturbation":
            raise ValueError("perturbation_type must be clinical_perturbation")
        if variant.perturbation_subtype != job["subtype"]:
            raise ValueError(f"perturbation_subtype must be {job['subtype']}")
        if changed:
            raise ValueError(f"demographics must match the base case (changed: {changed})")
        draft_id = f"{case.id}__{job['subtype']}"
    else:
        if variant.perturbation_type != "pure_demographic":
            raise ValueError("perturbation_type must be pure_demographic")
        if variant.perturbation_subtype is not None:
            raise ValueError("perturbation_subtype must be null for pure_demographic")
        if changed != {job["attr"]}:
            raise ValueError(f"exactly {{{job['attr']}}} must change, changed: {changed}")
        if getattr(variant.demographics, job["attr"]) != job["value"]:
            raise ValueError(f"{job['attr']} must be {job['value']!r}")
        draft_id = f"{case.id}__{job['attr']}"
    return DraftRecord(
        draft_id=draft_id,
        category=("clinical_perturbation" if job["kind"] == "clinical" else "pure_demographic"),
        parent_id=case.id,
        generator_model=job["model"],
        generated_at=now,
        variant=variant,
    )


def run_job(runner: AnthropicRunner, job: dict[str, Any]) -> DraftRecord | None:
    """Generate one draft; one repair retry on validation failure; None on skip/failure."""
    job["model"] = runner.model
    prompt = render_prompt(job)
    error = ""
    for attempt in range(2):
        try:
            text = runner.complete(
                prompt
                if attempt == 0
                else f"{prompt}\n\nYour previous "
                f"attempt failed validation: {error}. Fix it and reply "
                "with only the corrected JSON object."
            )
        except Exception as e:  # noqa: BLE001 — one job's API failure must not kill the batch
            error = f"{type(e).__name__}: {e}"
            continue
        try:
            payload = extract_json(text)
        except ValueError as e:
            error = str(e)
            continue
        if payload.get("skip"):
            reason = str(payload.get("reason", ""))
            if job["kind"] == "demographic":
                log_skip(f"{job['case'].id}__{job['attr']}", reason)
            print(f"  SKIP {job_label(job)}: {reason}")
            return None
        try:
            return validate_job_payload(job, payload)
        except Exception as e:  # noqa: BLE001 — ValueError or pydantic ValidationError
            error = str(e)[:500]
    print(f"  FAILED after repair retry: {error}")
    return None


def job_label(job: dict[str, Any]) -> str:
    if job["kind"] == "clinical":
        return f"clinical_perturbation {job['case'].id} [{job['subtype']}]"
    if job["kind"] == "demographic":
        return f"pure_demographic {job['case'].id} [{job['attr']} -> {job['value']}]"
    return f"adversarial {job['new_id']} [{job['specialty']}/{job['difficulty']}/{job['subspecialty']}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--category",
        choices=["clinical", "demographic", "adversarial", "all"],
        default="all",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Cap drafts generated per category this run (default: all remaining).",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the job plan without calling any API."
    )
    args = parser.parse_args(argv)

    cases = load_vignettes()
    planners = {
        "clinical": plan_clinical,
        "demographic": plan_demographic,
        "adversarial": plan_adversarial,
    }
    wanted = list(planners) if args.category == "all" else [args.category]

    all_jobs: list[dict[str, Any]] = []
    for name in wanted:
        jobs = planners[name](cases)
        if args.count is not None:
            jobs = jobs[: args.count]
        print(f"{name}: {len(jobs)} draft(s) to generate")
        for job in jobs:
            print(f"  - {job_label(job)}")
        all_jobs.extend(jobs)

    if args.dry_run or not all_jobs:
        return 0

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=REPO_ROOT / ".env.local", override=False)
        load_dotenv(override=False)
    except ImportError:
        pass

    runner = AnthropicRunner(model=args.model, temperature=args.temperature, max_tokens=4096)
    written = 0
    for job in all_jobs:
        print(f"generating {job_label(job)} ...")
        record = run_job(runner, job)
        if record is not None:
            path = write_draft(record, record.category)
            written += 1
            print(f"  wrote {path.relative_to(REPO_ROOT)}")
    print(f"\n{written}/{len(all_jobs)} drafts written to {DRAFTS_DIR.relative_to(REPO_ROOT)}/")
    print("Review them with: python3 scripts/review_drafts.py review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
