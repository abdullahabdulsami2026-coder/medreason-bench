"""Single reproducible entry point for the paper's results tables.

Runs every requested model x seed over the full vignette corpus (base
cases + variants), writes raw per-item JSONL + manifests, then
aggregates every metric family (see docs/METRICS.md) into the paper
tables and figures:

    <out>/raw/<model>_<seed>.jsonl (+ .manifest.json)
    <out>/judge_verdicts.jsonl          # fabrication-judge cache
    <out>/table1.csv, table1.md         # per-model metrics (acc/CI, consistency,
                                        #   ECE, Brier, abstention, false-abstention)
    <out>/table2.csv, table2.md         # accuracy by variant type + demographic cell
    <out>/per_case.csv                  # per model x case results
    <out>/pairwise.csv                  # McNemar between models
    <out>/RESULTS_SUMMARY.md            # headline + methods block + n<5 cells
    <out>/figures/, corpus_stats.json

Runs are resumable: an existing raw file with the expected item count is
reused instead of re-calling the API, so aggregation can be re-run free.

Usage:
    python3 run_eval.py --models claude-haiku-4-5 --seeds 1 --out-dir results/baseline_haiku
    python3 run_eval.py --models claude-sonnet-4-6,gpt-4o --seeds 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.vignettes.loader import dataset_version, load_vignettes
from data.vignettes.schema import vignette_to_mcq_item, vignette_variant_to_mcq_item
from eval.metrics import accuracy as acc
from eval.metrics import calibration, fairness, hallucination, robustness
from eval.pipeline import RUNNER_REGISTRY, aggregate_prompt_hash, get_git_sha
from eval.runners.anthropic_runner import AnthropicRunner
from eval.runners.base import Runner
from eval.schemas import EvalResponse, MCQItem, RunMetadata, Vignette

REPO_ROOT = Path(__file__).resolve().parent
JUDGE_TEMPLATE = REPO_ROOT / "prompts" / "judge.md"

PROVIDER_PREFIXES = [("claude", "anthropic"), ("gpt", "openai"), ("gemini", "google")]

# Seconds between calls, tuned to free-tier rate limits; override with
# --min-interval. Paid Anthropic tiers need no throttle.
PROVIDER_MIN_INTERVAL = {"google": 6.0, "groq": 2.5}

# $ per MTok (input, output). Providers not listed are treated as free
# (Gemini free tier, Groq free tier, local Ollama).
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """ "provider/model" -> explicit pair; bare names fall back to prefix inference.

    Ollama tags contain colons ("llama3.1:8b"), Groq ids are bare
    ("llama-3.3-70b-versatile"), so explicit "groq/..." or "ollama/..."
    is the unambiguous form for open models.
    """
    if "/" in spec:
        provider, model = spec.split("/", 1)
        if provider not in RUNNER_REGISTRY:
            raise SystemExit(f"unknown provider {provider!r} in {spec!r}")
        return provider, model
    model = spec
    provider = next((prov for pre, prov in PROVIDER_PREFIXES if model.startswith(pre)), "ollama")
    return provider, model


def is_rate_limit(error: str) -> bool:
    lowered = error.lower()
    return "429" in error or any(
        s in lowered for s in ("rate limit", "rate_limit", "quota", "resource_exhausted")
    )


# ── Item + frame construction ─────────────────────────────────────


def build_items(limit: int | None = None) -> tuple[list[MCQItem], dict[str, dict[str, Any]]]:
    """All base + variant MCQItems, plus per-item metadata for the results frame."""
    cases = load_vignettes()
    if limit is not None:
        cases = sorted(cases, key=lambda v: v.id)[:limit]
    items: list[MCQItem] = []
    meta: dict[str, dict[str, Any]] = {}
    for case in cases:
        base = vignette_to_mcq_item(case)
        items.append(base)
        meta[base.id] = _meta(case, base, None)
        for variant in case.variants:
            it = vignette_variant_to_mcq_item(case, variant)
            items.append(it)
            meta[it.id] = _meta(case, it, variant)
    return items, meta


def _meta(case: Vignette, item: MCQItem, variant: Any) -> dict[str, Any]:
    changed = ""
    preserving = True
    if variant is not None:
        base_demo = case.demographics.model_dump()
        diff = sorted(
            k
            for k in ("age", "sex", "race", "ethnicity")
            if base_demo[k] != getattr(variant.demographics, k)
        )
        changed = diff[0] if len(diff) == 1 else "+".join(diff)
        preserving = item.correct == case.correct
    return {
        "case_id": case.id,
        "is_variant": variant is not None,
        "variant_id": variant.variant_id if variant is not None else "",
        "perturbation_type": variant.perturbation_type if variant is not None else "",
        "perturbation_subtype": variant.perturbation_subtype if variant is not None else None,
        "changed_attr": changed,
        "answer_preserving": preserving,
        "specialty": case.specialty,
        "difficulty": case.difficulty,
        "correct": item.correct,
        "options": set(item.options),
    }


def build_frame(
    rows: list[tuple[str, int, EvalResponse]], meta: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    records = []
    for model, seed, resp in rows:
        m = meta[resp.item_id]
        answer = resp.parsed_answer
        in_options = answer is not None and answer in m["options"]
        abstained = answer is None and resp.error is None
        correct = m["correct"]
        is_correct = abstained if correct is None else (answer == correct)
        records.append(
            {
                "model": model,
                "seed": seed,
                "item_id": resp.item_id,
                "case_id": m["case_id"],
                "is_variant": m["is_variant"],
                "variant_id": m["variant_id"],
                "perturbation_type": m["perturbation_type"],
                "perturbation_subtype": m["perturbation_subtype"],
                "changed_attr": m["changed_attr"],
                "answer_preserving": m["answer_preserving"],
                "specialty": m["specialty"],
                "difficulty": m["difficulty"],
                "correct": correct,
                "unanswerable": correct is None,
                "parsed_answer": answer,
                "abstained": abstained,
                "answer_in_options": in_options,
                "is_correct": bool(is_correct),
                "confidence": resp.confidence / 100 if resp.confidence is not None else np.nan,
                "error": resp.error,
            }
        )
    return pd.DataFrame(records)


# ── Model runs (resumable) ────────────────────────────────────────


def load_checkpoint(jsonl: Path, valid_ids: set[str]) -> dict[str, EvalResponse]:
    """Non-error responses already on disk, keyed by item id.

    Errored rows are dropped so a resume retries them; rows for items no
    longer in the corpus are ignored.
    """
    if not jsonl.exists():
        return {}
    kept: dict[str, EvalResponse] = {}
    for line in jsonl.read_text().splitlines():
        if not line:
            continue
        resp = EvalResponse.model_validate_json(line)
        if resp.error is None and resp.item_id in valid_ids:
            kept[resp.item_id] = resp
    return kept


def pending_items(
    items: list[MCQItem], jsonl: Path, reuse_raw: Path | None, model: str, seed: int
) -> tuple[dict[str, EvalResponse], list[MCQItem]]:
    """(responses already available, items still needing an API call)."""
    valid_ids = {i.id for i in items}
    have = load_checkpoint(jsonl, valid_ids)
    if reuse_raw is not None:
        old = reuse_raw / f"{model.replace('/', '_')}_{seed}.jsonl"
        for item_id, resp in load_checkpoint(old, valid_ids).items():
            have.setdefault(item_id, resp)
    return have, [i for i in items if i.id not in have]


def run_model_seed(
    runner: Runner,
    items: list[MCQItem],
    seed: int,
    raw_dir: Path,
    args: argparse.Namespace,
    min_interval: float = 0.0,
    reuse_raw: Path | None = None,
) -> list[EvalResponse] | None:
    """Run (or resume) one model x seed; returns None when paused on a rate cap.

    The JSONL is an append-mode checkpoint: reused and fresh responses
    are flushed per item, so a free-tier daily cap just pauses the run —
    re-invoking run_eval.py resumes from the last completed item. Three
    consecutive rate-limit errors trigger the pause (backoff inside the
    runner has already been exhausted by then). Rate-limited rows are
    never written to the checkpoint.
    """
    safe_model = runner.model.replace("/", "_")
    jsonl = raw_dir / f"{safe_model}_{seed}.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    have, todo = pending_items(items, jsonl, reuse_raw, runner.model, seed)
    if not todo:
        print(f"  {jsonl.name}: complete ({len(have)} items, no calls needed)")
    else:
        print(f"  {jsonl.name}: {len(have)} done, {len(todo)} to call")

    started_at = dt.datetime.now(dt.UTC).isoformat()
    t0 = time.perf_counter()
    consecutive_limits = 0
    with jsonl.open("w") as fp:
        for resp in have.values():  # rewrite checkpoint: reused + prior rows first
            fp.write(resp.model_dump_json() + "\n")
        fp.flush()
        for i, item in enumerate(todo, 1):
            while True:
                resp = runner.run(item)
                if resp.error is None or not is_rate_limit(resp.error):
                    break
                consecutive_limits += 1
                if consecutive_limits >= 3:
                    print(
                        f"  {runner.model} seed {seed}: rate/quota cap after "
                        f"{len(have)} items — paused, re-run to resume.\n"
                        f"    last error: {resp.error[:200]}"
                    )
                    return None
                time.sleep(30 * consecutive_limits)
            consecutive_limits = 0
            have[item.id] = resp
            fp.write(resp.model_dump_json() + "\n")
            fp.flush()
            if i % 25 == 0:
                print(f"  {runner.model} seed {seed}: {i}/{len(todo)} new calls")
            if min_interval:
                time.sleep(min_interval)

    if len(have) < len(items):  # non-rate-limit errors left gaps; keep rows, report
        missing = len(items) - len(have)
        print(f"  {runner.model} seed {seed}: {missing} item(s) still missing — re-run to retry")
        return None
    responses = [have[i.id] for i in items]
    manifest = RunMetadata(
        run_id=str(uuid.uuid4()),
        model=runner.model,
        model_version=runner.model,
        dataset="vignettes+variants",
        dataset_version=dataset_version(),
        prompt_template=args.prompt_template,
        n_items=len(items),
        temperature=runner.temperature,
        top_p=runner.top_p,
        seed=seed,
        prompt_hash=aggregate_prompt_hash(items),
        git_sha=get_git_sha(),
        started_at=started_at,
        finished_at=dt.datetime.now(dt.UTC).isoformat(),
        total_elapsed_s=time.perf_counter() - t0,
    )
    jsonl.with_suffix(".manifest.json").write_text(manifest.model_dump_json(indent=2))
    return responses


# ── Fabrication judge ─────────────────────────────────────────────


def judge_fabrications(
    frame: pd.DataFrame,
    items_by_id: dict[str, MCQItem],
    responses_by_key: dict[tuple[str, int, str], EvalResponse],
    out_dir: Path,
    judge_model: str,
) -> dict[tuple[str, int, str], str]:
    """Score answered-unanswerable rows with the fixed rubric; cache verdicts."""
    cache_path = out_dir / "judge_verdicts.jsonl"
    verdicts: dict[tuple[str, int, str], str] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            verdicts[(row["model"], row["seed"], row["item_id"])] = row["verdict"]

    todo = [
        (r["model"], r["seed"], r["item_id"])
        for _, r in hallucination.needs_judging(frame).iterrows()
        if (r["model"], r["seed"], r["item_id"]) not in verdicts
    ]
    if not todo:
        return verdicts

    template = JUDGE_TEMPLATE.read_text(encoding="utf-8")
    judge = AnthropicRunner(model=judge_model, max_tokens=512)
    print(f"judging {len(todo)} answered unanswerable item(s) with {judge_model}")
    with cache_path.open("a") as fp:
        for model, seed, item_id in todo:
            item = items_by_id[item_id]
            resp = responses_by_key[(model, seed, item_id)]
            prompt = template.format(
                stem=item.stem,
                question=item.question,
                options_block="\n".join(f"{k}. {v}" for k, v in sorted(item.options.items())),
                answer=resp.parsed_answer,
                confidence=resp.confidence if resp.confidence is not None else "not stated",
                rationale=resp.rationale or resp.raw_response,
            )
            try:
                text = judge.complete(prompt)
            except Exception as e:  # noqa: BLE001 — a judge failure must not sink the run
                print(f"  judge failed on {item_id}: {type(e).__name__}: {e}")
                continue
            verdict = "BORDERLINE"
            for candidate in ("NOT_FABRICATED", "FABRICATED", "BORDERLINE"):
                if f"VERDICT: {candidate}" in text.upper().replace("**", ""):
                    verdict = candidate
                    break
            reason = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in text.splitlines()
                    if line.upper().startswith("REASON")
                ),
                "",
            )
            verdicts[(model, seed, item_id)] = verdict
            fp.write(
                json.dumps(
                    {
                        "model": model,
                        "seed": seed,
                        "item_id": item_id,
                        "verdict": verdict,
                        "reason": reason,
                        "judge_model": judge_model,
                    }
                )
                + "\n"
            )
    return verdicts


# ── Aggregation ───────────────────────────────────────────────────


def seed_metrics(sub: pd.DataFrame, verdicts: dict[tuple[str, int, str], str]) -> dict[str, float]:
    model = sub["model"].iloc[0]
    seed = int(sub["seed"].iloc[0])
    slice_verdicts = {
        item_id: verdict == "FABRICATED"
        for (m, s, item_id), verdict in verdicts.items()
        if m == model and s == seed
    }
    return {
        "accuracy": acc.accuracy_summary(sub)["accuracy"],
        "macro_f1": acc.macro_f1(sub),
        "consistency": robustness.consistency_rate(sub)["consistency_rate"],
        "flip_rate": robustness.flip_rate(sub)["flip_rate"],
        "robustness_gap": float(
            robustness.robustness_gap(sub).query("slice == 'all'")["gap"].iloc[0]
        ),
        "ece": calibration.ece(sub)["ece"],
        "brier": calibration.brier_score(sub)["brier"],
        "abstention_rate": hallucination.abstention_rate(sub)["abstention_rate"],
        "false_abstention_rate": hallucination.false_abstention_rate(sub)["false_abstention_rate"],
        "fabrication_rate": hallucination.fabrication_rate(sub, slice_verdicts)["fabrication_rate"],
        "max_fairness_gap": fairness.max_gap(fairness.within_case_deltas(sub)),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0


def _fmt(r: pd.Series, key: str, pct: bool = True) -> str:
    scale, suffix, prec = (100, "%", 1) if pct else (1, "", 3)
    s = f"{r[key] * scale:.{prec}f}{suffix}"
    if r["n_seeds"] > 1:
        s += f" ± {r[f'{key}_sd'] * scale:.{prec}f}"
    return s


def build_table1(
    frame: pd.DataFrame, verdicts: dict[tuple[str, int, str], str]
) -> tuple[pd.DataFrame, str]:
    """Table 1 — per-model metrics, mean ± SD across seeds.

    The CSV carries every metric family; the markdown shows the paper's
    columns (accuracy, consistency, ECE, Brier, abstention on
    unanswerable, false-abstention on answerable).
    """
    rows = []
    for model, model_frame in frame.groupby("model"):
        per_seed = [seed_metrics(s, verdicts) for _, s in model_frame.groupby("seed")]
        pooled = acc.accuracy_summary(model_frame)
        row: dict[str, Any] = {"model": model, "n_seeds": len(per_seed)}
        for key in per_seed[0]:
            mean, sd = mean_sd([m[key] for m in per_seed])
            row[key] = mean
            row[f"{key}_sd"] = sd
        row["acc_ci_lo"], row["acc_ci_hi"] = pooled["ci_lo"], pooled["ci_hi"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    lines = [
        "# Table 1 — Model metrics",
        "",
        f"Corpus v{dataset_version()}, mean ± SD across seeds; Wilson 95% CI on pooled accuracy.",
        "",
        "| model | accuracy | 95% CI | consistency | ECE | Brier "
        "| abstention (unanswerable) | false-abstention (answerable) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {_fmt(r, 'accuracy')} "
            f"| [{r['acc_ci_lo'] * 100:.1f}, {r['acc_ci_hi'] * 100:.1f}] "
            f"| {_fmt(r, 'consistency')} "
            f"| {_fmt(r, 'ece', pct=False)} | {_fmt(r, 'brier', pct=False)} "
            f"| {_fmt(r, 'abstention_rate')} | {_fmt(r, 'false_abstention_rate')} |"
        )
    lines += [
        "",
        "Full metric set (macro-F1, flip rate, robustness gap, fabrication, "
        "max fairness gap) in table1.csv.",
    ]
    return table, "\n".join(lines)


def _slice_specs(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """(slice_type, slice_name, row mask) for table 2's breakdown."""
    specs: list[tuple[str, str, pd.Series]] = [("variant_type", "base", ~frame["is_variant"])]
    for ptype in sorted(frame.loc[frame["is_variant"], "perturbation_type"].unique()):
        specs.append(("variant_type", str(ptype), frame["perturbation_type"] == ptype))
    for subtype in sorted(frame["perturbation_subtype"].dropna().unique()):
        specs.append(("clinical_subtype", str(subtype), frame["perturbation_subtype"] == subtype))
    demo = frame["is_variant"] & (frame["perturbation_type"] == "pure_demographic")
    for cell in sorted(frame.loc[demo & (frame["changed_attr"] != ""), "changed_attr"].unique()):
        specs.append(("demographic_cell", str(cell), demo & (frame["changed_attr"] == cell)))
    return specs


def build_table2(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Table 2 — accuracy by variant type and demographic cell, with item n."""
    rows = []
    for model, model_frame in frame.groupby("model"):
        for slice_type, name, mask in _slice_specs(frame):
            sub = model_frame[mask.reindex(model_frame.index, fill_value=False)]
            if sub.empty:
                continue
            per_seed = [float(s["is_correct"].mean()) for _, s in sub.groupby("seed")]
            mean, sd = mean_sd(per_seed)
            rows.append(
                {
                    "model": model,
                    "slice_type": slice_type,
                    "slice": name,
                    "n_items": sub["item_id"].nunique(),
                    "accuracy": mean,
                    "accuracy_sd": sd,
                    "n_seeds": len(per_seed),
                }
            )
    table = pd.DataFrame(rows)

    lines = [
        "# Table 2 — Accuracy by variant type and demographic cell",
        "",
        "n = distinct items in the slice; accuracy is mean ± SD across seeds. "
        "Cells with n < 5 are underpowered — see RESULTS_SUMMARY.md.",
        "",
        "| model | slice type | slice | n | accuracy |",
        "|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        acc_s = f"{r['accuracy'] * 100:.1f}%"
        if r["n_seeds"] > 1:
            acc_s += f" ± {r['accuracy_sd'] * 100:.1f}"
        flag = " ⚠" if r["n_items"] < 5 else ""
        lines.append(
            f"| {r['model']} | {r['slice_type']} | {r['slice']} | {r['n_items']}{flag} | {acc_s} |"
        )
    return table, "\n".join(lines)


def build_summary(
    frame: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    pairwise: pd.DataFrame,
    run_info: dict[str, Any],
) -> str:
    """RESULTS_SUMMARY.md — headline numbers, methods block, low-n cells."""
    low_n = (
        table2[table2["slice_type"] != "variant_type"]
        .groupby(["slice_type", "slice"])["n_items"]
        .first()
        .reset_index()
    )
    low_n = low_n[low_n["n_items"] < 5]

    lines = [
        "# Results summary",
        "",
        "## Reproducibility (methods section)",
        "",
        f"- Models: {', '.join(run_info['models'])}",
        f"- Seeds: {run_info['seeds']} (0..{run_info['seeds'] - 1}); temperature "
        f"{run_info['temperature']}",
        f"- Prompt template: `{run_info['prompt_template']}`; aggregate prompt hash "
        f"`{run_info['prompt_hash']}`",
        f"- Confidence: verbalized 0–100; judge model {run_info['judge_model']}",
        f"- Corpus: v{run_info['dataset_version']} ({run_info['n_items']} items: "
        f"{run_info['n_base']} base + {run_info['n_variants']} variants)",
        f"- Code commit: `{run_info['git_sha']}`",
        "",
        "## Headline",
        "",
    ]
    for _, r in table1.iterrows():
        lines.append(
            f"- **{r['model']}**: accuracy {_fmt(r, 'accuracy')} "
            f"[{r['acc_ci_lo'] * 100:.1f}, {r['acc_ci_hi'] * 100:.1f}], "
            f"consistency {_fmt(r, 'consistency')}, ECE {_fmt(r, 'ece', pct=False)}, "
            f"Brier {_fmt(r, 'brier', pct=False)}, "
            f"abstention {_fmt(r, 'abstention_rate')} / false-abstention "
            f"{_fmt(r, 'false_abstention_rate')}, fabrication {_fmt(r, 'fabrication_rate')}"
        )

    lines += ["", "## Pairwise McNemar", ""]
    if pairwise.empty:
        lines.append("Only one model in this run — no pairs.")
    else:
        for _, r in pairwise.iterrows():
            verdict = "significant at p<0.05" if r["p_value"] < 0.05 else "not significant"
            lines.append(
                f"- {r['model_a']} vs {r['model_b']}: discordant "
                f"{int(r['a_only_right'])}/{int(r['b_only_right'])} on {int(r['n_shared'])} "
                f"shared items, p={r['p_value']:.4f} ({verdict})"
            )

    lines += ["", "## Cells with n < 5 (do not interpret)", ""]
    if low_n.empty:
        lines.append("None — every breakdown cell has n ≥ 5.")
    else:
        for _, r in low_n.iterrows():
            lines.append(f"- {r['slice_type']} `{r['slice']}`: n={int(r['n_items'])}")
    lines.append("")
    return "\n".join(lines)


def build_per_case(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, case_id), sub in frame.groupby(["model", "case_id"]):
        base = sub[~sub["is_variant"]]
        variants = sub[sub["is_variant"]]
        rows.append(
            {
                "model": model,
                "case_id": case_id,
                "specialty": sub["specialty"].iloc[0],
                "difficulty": sub["difficulty"].iloc[0],
                "unanswerable": bool(base["unanswerable"].iloc[0]) if len(base) else False,
                "base_accuracy": float(base["is_correct"].mean()) if len(base) else np.nan,
                "base_confidence": float(base["confidence"].mean()) if len(base) else np.nan,
                "n_variants": variants["item_id"].nunique(),
                "variant_accuracy": float(variants["is_correct"].mean())
                if len(variants)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_figures(frame: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for model, sub in frame.groupby("model"):
        table = calibration.reliability_table(sub)
        filled = table[table["n"] > 0]
        ax.plot(filled["mean_confidence"], filled["accuracy"], "o-", label=str(model))
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (10 equal-width bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    gap_frames = []
    for model, sub in frame.groupby("model"):
        g = robustness.robustness_gap(sub)
        g["model"] = model
        gap_frames.append(g)
    gaps = pd.concat(gap_frames)
    slices = list(gaps["slice"].unique())
    width = 0.8 / gaps["model"].nunique()
    for i, (model, sub) in enumerate(gaps.groupby("model")):
        sub = sub.set_index("slice").reindex(slices)
        ax.bar(
            [j + i * width for j in range(len(slices))],
            sub["gap"],
            width,
            label=str(model),
        )
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(slices))])
    ax.set_xticklabels(slices, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("base acc − variant acc")
    ax.set_title("Robustness gap by perturbation slice")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "robustness_gap.png", dpi=150)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        required=True,
        help='Comma-separated model ids; "provider/model" pins the provider '
        "(e.g. groq/llama-3.3-70b-versatile, ollama/llama3.1:8b).",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-template", default="mcq", choices=["mcq"])
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Cap on base cases.")
    parser.add_argument(
        "--reuse-raw",
        default=None,
        help="Directory of prior raw JSONLs; matching non-error responses for "
        "still-existing items are reused instead of re-called.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=None,
        help="Seconds between calls (default: per-provider free-tier setting).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Abort before any call if the estimated API cost exceeds this (USD).",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=REPO_ROOT / ".env.local", override=False)
        load_dotenv(override=False)
    except ImportError:
        pass

    out_dir = REPO_ROOT / args.out_dir
    items, meta = build_items(args.limit)
    items_by_id = {i.id: i for i in items}
    print(f"{len(items)} items ({sum(1 for m in meta.values() if not m['is_variant'])} base cases)")

    specs = [parse_model_spec(s.strip()) for s in args.models.split(",")]
    reuse_raw = (REPO_ROOT / args.reuse_raw) if args.reuse_raw else None
    raw_dir = out_dir / "raw"

    # Pre-run cost gate: estimate paid-API spend from the calls actually
    # needed (checkpoints and reused raw subtracted), abort over --max-cost.
    from eval.prompts.mcq import format_mcq

    est_rows = []
    total_cost = 0.0
    for provider, model in specs:
        n_calls = 0
        in_tokens = 0.0
        for seed in range(args.seeds):
            jsonl = raw_dir / f"{model.replace('/', '_')}_{seed}.jsonl"
            _, todo = pending_items(items, jsonl, reuse_raw, model, seed)
            n_calls += len(todo)
            in_tokens += sum(len(format_mcq(i)) for i in todo) / 4
        in_price, out_price = PRICES.get(model, (0.0, 0.0))
        cost = (in_tokens * in_price + n_calls * 250 * out_price) / 1e6
        total_cost += cost
        est_rows.append((provider, model, n_calls, cost))
    n_unanswerable = sum(1 for m in meta.values() if m["correct"] is None)
    judge_in, judge_out = PRICES.get(args.judge_model, (0.0, 0.0))
    judge_cost = (
        n_unanswerable * len(specs) * args.seeds * (1500 * judge_in + 150 * judge_out) / 1e6
    )
    total_cost += judge_cost
    print("\nCost estimate (paid APIs only; free-tier/local models at $0):")
    for provider, model, n_calls, cost in est_rows:
        print(f"  {provider}/{model}: {n_calls} call(s) ≈ ${cost:.2f}")
    print(f"  fabrication judge ({args.judge_model}): worst case ≈ ${judge_cost:.2f}")
    print(f"  TOTAL ≈ ${total_cost:.2f} (cap ${args.max_cost:.2f})")
    if total_cost > args.max_cost:
        print("Estimated cost exceeds --max-cost; aborting before any API call.")
        return 1

    rows: list[tuple[str, int, EvalResponse]] = []
    responses_by_key: dict[tuple[str, int, str], EvalResponse] = {}
    paused: list[str] = []
    for provider, model in specs:
        runner_cls = RUNNER_REGISTRY[provider]
        interval = (
            args.min_interval
            if args.min_interval is not None
            else PROVIDER_MIN_INTERVAL.get(provider, 0.0)
        )
        for seed in range(args.seeds):
            runner = runner_cls(model=model, temperature=args.temperature)
            print(f"running {provider}/{model} seed {seed}")
            responses = run_model_seed(
                runner,
                items,
                seed,
                raw_dir,
                args,
                min_interval=interval,
                reuse_raw=reuse_raw,
            )
            if responses is None:
                paused.append(f"{model} seed {seed}")
                continue
            for resp in responses:
                rows.append((model, seed, resp))
                responses_by_key[(model, seed, resp.item_id)] = resp

    if paused:
        print(
            f"\nPAUSED (rate/quota cap or gaps): {', '.join(paused)} — "
            "re-run the same command to resume; checkpoints are kept."
        )
    if not rows:
        print("No completed model runs to aggregate yet.")
        return 1

    frame = build_frame(rows, meta)
    n_errors = int(frame["error"].notna().sum())
    if n_errors:
        print(f"WARNING: {n_errors} item(s) errored; they count as incorrect")

    verdicts = judge_fabrications(frame, items_by_id, responses_by_key, out_dir, args.judge_model)

    table1, table1_md = build_table1(frame, verdicts)
    table1.to_csv(out_dir / "table1.csv", index=False)
    (out_dir / "table1.md").write_text(table1_md, encoding="utf-8")
    table2, table2_md = build_table2(frame)
    table2.to_csv(out_dir / "table2.csv", index=False)
    (out_dir / "table2.md").write_text(table2_md, encoding="utf-8")
    build_per_case(frame).to_csv(out_dir / "per_case.csv", index=False)
    pairwise = acc.mcnemar_pairs(frame)
    pairwise.to_csv(out_dir / "pairwise.csv", index=False)
    write_figures(frame, out_dir / "figures")

    run_info = {
        "models": [m.strip() for m in args.models.split(",")],
        "seeds": args.seeds,
        "temperature": args.temperature,
        "prompt_template": args.prompt_template,
        "prompt_hash": aggregate_prompt_hash(items),
        "judge_model": args.judge_model,
        "dataset_version": dataset_version(),
        "n_items": len(items),
        "n_base": sum(1 for m in meta.values() if not m["is_variant"]),
        "n_variants": sum(1 for m in meta.values() if m["is_variant"]),
        "git_sha": get_git_sha(),
    }
    summary = build_summary(frame, table1, table2, pairwise, run_info)
    (out_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
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
    shutil.copy(REPO_ROOT / "data" / "vignettes" / "corpus_stats.json", out_dir)
    print(
        f"\nWrote table1/table2 (csv+md), per_case.csv, pairwise.csv, RESULTS_SUMMARY.md, "
        f"figures/, corpus_stats.json to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
