"""Eval pipeline.

Runs ``provider × dataset`` end-to-end; writes
``results/runs/<model>__<dataset>__<timestamp>.jsonl`` plus a sibling
``.manifest.json`` with run metadata.

The CLI exposes two run modes — ``--smoke`` for the 10 hand-coded
sample items (5 cardio + 5 autoimmune), and ``--dataset NAME`` for any
dataset registered in :data:`DATASET_REGISTRY` (Phase 2: medmcqa,
medqa, pubmedqa).

Usage:
    python -m eval.pipeline --smoke
    python -m eval.pipeline --smoke --model claude-sonnet-4-6
    python -m eval.pipeline --dataset medmcqa --provider anthropic
"""

# NB: tests must NEVER instantiate AnthropicRunner without a mock
# client= — see tests/test_runners.py for the pattern.

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from data.loaders.medmcqa import load_medmcqa
from data.loaders.medqa import load_medqa
from data.loaders.pubmedqa import load_pubmedqa
from data.vignettes.loader import load_vignettes
from data.vignettes.schema import vignette_to_mcq_item
from eval.metrics.accuracy import top1_accuracy
from eval.prompts.mcq import prompt_hash
from eval.runners.anthropic_runner import AnthropicRunner
from eval.runners.base import Runner
from eval.runners.google_runner import GoogleRunner
from eval.runners.groq_runner import GroqRunner
from eval.runners.ollama_runner import OllamaRunner
from eval.runners.openai_runner import OpenAIRunner
from eval.schemas import EvalResponse, MCQItem, RunMetadata

# Provider → Runner class.
RUNNER_REGISTRY: dict[str, type[Runner]] = {
    "anthropic": AnthropicRunner,
    "google": GoogleRunner,
    "groq": GroqRunner,
    "openai": OpenAIRunner,
    "ollama": OllamaRunner,
}

# Dataset name → callable that materialises a list of MCQItems.
# Default cap of 100 items per dataset is a guardrail against accidental
# full-corpus runs; Phase 4 lifts the cap behind an explicit flag.
_DATASET_LIMIT: int = 100
DATASET_REGISTRY: dict[str, Callable[[], list[MCQItem]]] = {
    "medmcqa": lambda: list(load_medmcqa(limit=_DATASET_LIMIT)),
    "medqa": lambda: list(load_medqa(limit=_DATASET_LIMIT)),
    "pubmedqa": lambda: list(load_pubmedqa(limit=_DATASET_LIMIT)),
    # Phase 3 scaffold: yields 0 items until the 100 hand-written
    # vignettes land. _TEMPLATE.json files are skipped by load_vignettes.
    "vignettes": lambda: [vignette_to_mcq_item(v) for v in load_vignettes()],
}


# ── 10 hand-coded sample items (5 cardiology + 5 autoimmune) ──────
# Phase 2 wires real HF datasets; until then these drive the smoke run.

SAMPLE_ITEMS: list[MCQItem] = [
    MCQItem(
        id="smoke_cardio_001",
        source="phase1_smoke",
        stem=(
            "A 62-year-old man presents to the emergency department with crushing "
            "substernal chest pain that began 1 hour ago. He is diaphoretic and "
            "anxious. His ECG shows 3 mm ST-segment elevation in leads II, III, "
            "and aVF, with reciprocal ST depression in leads I and aVL."
        ),
        question="Which coronary artery is most likely occluded?",
        options={
            "A": "Left anterior descending artery",
            "B": "Left circumflex artery",
            "C": "Right coronary artery",
            "D": "Left main coronary artery",
        },
        correct="C",
        metadata={"specialty": "cardiology", "topic": "stemi_localization"},
    ),
    MCQItem(
        id="smoke_cardio_002",
        source="phase1_smoke",
        stem=(
            "A 70-year-old woman with a long history of poorly controlled "
            "hypertension reports progressive dyspnea on exertion, orthopnea, and "
            "bilateral leg swelling over 2 months. Examination shows jugular "
            "venous distention to the angle of the jaw, an S3 gallop, and "
            "bibasilar crackles."
        ),
        question="Which initial diagnostic workup is most appropriate?",
        options={
            "A": "CT pulmonary angiogram",
            "B": "B-type natriuretic peptide and transthoracic echocardiogram",
            "C": "Exercise treadmill stress test",
            "D": "24-hour ambulatory ECG (Holter) monitor",
        },
        correct="B",
        metadata={"specialty": "cardiology", "topic": "heart_failure_workup"},
    ),
    MCQItem(
        id="smoke_cardio_003",
        source="phase1_smoke",
        stem=(
            "A 75-year-old hemodynamically stable man presents with palpitations "
            "and a heart rate of 130 bpm. ECG confirms atrial fibrillation with "
            "rapid ventricular response. He has hypertension and chronic kidney "
            "disease but no heart failure or pre-excitation syndrome."
        ),
        question=("Which medication is the most appropriate initial choice for rate control?"),
        options={
            "A": "Metoprolol",
            "B": "Amiodarone",
            "C": "Digoxin",
            "D": "Adenosine",
        },
        correct="A",
        metadata={"specialty": "cardiology", "topic": "afib_rate_control"},
    ),
    MCQItem(
        id="smoke_cardio_004",
        source="phase1_smoke",
        stem=(
            "A 78-year-old woman presents with exertional dyspnea and one episode "
            "of syncope. Auscultation reveals a harsh systolic ejection murmur "
            "loudest at the right upper sternal border that radiates to the "
            "carotids, with a delayed and diminished carotid upstroke "
            "(pulsus parvus et tardus)."
        ),
        question="Which valvular abnormality is most likely?",
        options={
            "A": "Mitral regurgitation",
            "B": "Aortic stenosis",
            "C": "Pulmonic stenosis",
            "D": "Tricuspid regurgitation",
        },
        correct="B",
        metadata={"specialty": "cardiology", "topic": "aortic_stenosis"},
    ),
    MCQItem(
        id="smoke_cardio_005",
        source="phase1_smoke",
        stem=(
            "A 55-year-old woman with no prior cardiovascular events has an "
            "LDL-C of 195 mg/dL, blood pressure 132/82 mm Hg, no diabetes, and "
            "is a non-smoker. Her 10-year ASCVD risk is calculated at 6%."
        ),
        question=(
            "Which lipid-lowering recommendation aligns with current ACC/AHA "
            "primary-prevention guidance?"
        ),
        options={
            "A": "No statin indicated",
            "B": "Low-intensity statin",
            "C": "Moderate- to high-intensity statin (LDL-C ≥ 190 mg/dL alone "
            "is a class I indication)",
            "D": "Defer statin therapy until age 65",
        },
        correct="C",
        metadata={"specialty": "cardiology", "topic": "statin_primary_prevention"},
    ),
    MCQItem(
        id="smoke_autoimmune_001",
        source="phase1_smoke",
        stem=(
            "A 27-year-old woman presents with a malar rash sparing the "
            "nasolabial folds, painless oral ulcers, and arthralgias of the "
            "hands. ANA is positive at a titer of 1:320 with a homogeneous "
            "pattern."
        ),
        question=(
            "Which additional autoantibody is most specific for systemic lupus erythematosus?"
        ),
        options={
            "A": "Anti-Ro/SSA",
            "B": "Anti-double-stranded DNA (anti-dsDNA)",
            "C": "Rheumatoid factor",
            "D": "Anti-centromere",
        },
        correct="B",
        metadata={"specialty": "autoimmune", "topic": "sle_serology"},
    ),
    MCQItem(
        id="smoke_autoimmune_002",
        source="phase1_smoke",
        stem=(
            "A 45-year-old woman reports 6 weeks of morning stiffness lasting "
            "approximately 90 minutes, with symmetric pain and swelling in "
            "both wrists, the metacarpophalangeal (MCP) joints, and proximal "
            "interphalangeal (PIP) joints."
        ),
        question=("Which joint pattern is most consistent with rheumatoid arthritis?"),
        options={
            "A": "Asymmetric large-joint oligoarthritis",
            "B": "Symmetric small-joint polyarthritis with prolonged morning stiffness",
            "C": "Distal interphalangeal (DIP) joint involvement only",
            "D": "Episodic monoarthritis of the great toe",
        },
        correct="B",
        metadata={"specialty": "autoimmune", "topic": "ra_joint_pattern"},
    ),
    MCQItem(
        id="smoke_autoimmune_003",
        source="phase1_smoke",
        stem=(
            "A 58-year-old woman reports 6 months of persistent dry eyes and "
            "dry mouth requiring frequent sips of water. Anti-Ro/SSA and "
            "anti-La/SSB antibodies are positive. Schirmer test is abnormal."
        ),
        question=("Which clinical pattern is the hallmark of primary Sjögren syndrome?"),
        options={
            "A": "Recurrent oral and genital ulcers",
            "B": "Xerostomia and keratoconjunctivitis sicca (the sicca complex)",
            "C": "Symmetric polyarthritis with rheumatoid-factor positivity",
            "D": "Photosensitive malar rash",
        },
        correct="B",
        metadata={"specialty": "autoimmune", "topic": "sjogren_sicca"},
    ),
    MCQItem(
        id="smoke_autoimmune_004",
        source="phase1_smoke",
        stem=(
            "A 50-year-old woman with newly diagnosed rheumatoid arthritis is "
            "starting weekly low-dose methotrexate."
        ),
        question=(
            "Which co-medication is routinely recommended to reduce "
            "methotrexate-related adverse effects (e.g., stomatitis, "
            "transaminitis)?"
        ),
        options={
            "A": "Folic acid",
            "B": "Vitamin B12",
            "C": "Vitamin D",
            "D": "Pyridoxine (vitamin B6) only",
        },
        correct="A",
        metadata={"specialty": "autoimmune", "topic": "methotrexate_folate"},
    ),
    MCQItem(
        id="smoke_autoimmune_005",
        source="phase1_smoke",
        stem=(
            "A 50-year-old man presents with several months of chronic sinusitis, "
            "hemoptysis, and rising creatinine. Urinalysis shows red-cell casts. "
            "Cytoplasmic anti-neutrophil cytoplasmic antibodies (cANCA) "
            "targeting proteinase-3 (PR3) are positive."
        ),
        question="Which diagnosis is most likely?",
        options={
            "A": "Microscopic polyangiitis",
            "B": "Eosinophilic granulomatosis with polyangiitis (Churg-Strauss)",
            "C": "Granulomatosis with polyangiitis (Wegener's)",
            "D": "IgA vasculitis",
        },
        correct="C",
        metadata={"specialty": "autoimmune", "topic": "anca_vasculitis"},
    ),
]


# ── Pipeline helpers ──────────────────────────────────────────────


def get_git_sha() -> str:
    """Return the current commit SHA, or ``"unknown"`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def aggregate_prompt_hash(items: list[MCQItem]) -> str:
    """SHA-256 over the rendered prompts for every item, in order.

    Used in the run manifest so a prompt-template change shows up as a
    different hash even when the items themselves are unchanged.
    """
    h = hashlib.sha256()
    for item in items:
        h.update(prompt_hash(item).encode("utf-8"))
    return h.hexdigest()


def run_eval(
    runner: Runner,
    items: list[MCQItem],
    dataset_label: str,
    *,
    out_dir: Path = Path("results/runs"),
    seed: int | None = 42,
) -> tuple[Path, Path, RunMetadata, list[EvalResponse]]:
    """Run all items through the runner; persist responses + manifest.

    Args:
        runner: The runner to dispatch each item through.
        items: Ground-truth MCQ items.
        dataset_label: Short label used in output filenames.
        out_dir: Where the JSONL + manifest land.
        seed: Recorded in the manifest. Sampling determinism is enforced
            by ``temperature=0`` on the runner — most provider SDKs do
            not yet expose a ``seed`` parameter.

    Returns:
        ``(jsonl_path, manifest_path, manifest, responses)`` — the
        responses list mirrors what was written to disk and is handy
        for downstream metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_model = runner.model.replace("/", "_")
    safe_dataset = dataset_label.replace("/", "_")
    base_name = f"{safe_model}__{safe_dataset}__{timestamp}"
    jsonl_path = out_dir / f"{base_name}.jsonl"
    manifest_path = out_dir / f"{base_name}.manifest.json"

    started_at = dt.datetime.now(dt.UTC).isoformat()
    t0 = time.perf_counter()

    responses: list[EvalResponse] = []
    with jsonl_path.open("w") as fp:
        for item in items:
            resp = runner.run(item)
            responses.append(resp)
            fp.write(resp.model_dump_json() + "\n")

    finished_at = dt.datetime.now(dt.UTC).isoformat()
    elapsed_s = time.perf_counter() - t0

    manifest = RunMetadata(
        run_id=str(uuid.uuid4()),
        model=runner.model,
        model_version=runner.model,  # SDK exposes no separate version string
        dataset=dataset_label,
        n_items=len(items),
        temperature=runner.temperature,
        top_p=runner.top_p,
        seed=seed,
        prompt_hash=aggregate_prompt_hash(items),
        git_sha=get_git_sha(),
        started_at=started_at,
        finished_at=finished_at,
        total_elapsed_s=elapsed_s,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    return jsonl_path, manifest_path, manifest, responses


def main(argv: list[str] | None = None) -> int:
    """Pipeline CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="eval.pipeline",
        description="Run a model over MCQ items and write JSONL + manifest.",
    )
    # --smoke and --dataset are mutually exclusive; exactly one is required.
    # argparse raises SystemExit (via parser.error) if violated.
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help="Run on the 10 hand-coded sample items (5 cardio + 5 autoimmune).",
    )
    mode_group.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="HF dataset to evaluate on (Phase 2). Mutually exclusive with --smoke.",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=list(RUNNER_REGISTRY.keys()),
        help="Provider whose runner to use. More land in Phase 4.",
    )
    parser.add_argument(
        "--model",
        default=AnthropicRunner.DEFAULT_MODEL,
        help=(
            "Model id (must match the chosen --provider; default: "
            f"{AnthropicRunner.DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results/runs",
        help="Directory for JSONL + manifest output (default: results/runs).",
    )
    args = parser.parse_args(argv)

    # Load credentials from .env.local (preferred, gitignored) then .env
    # as fallback. Silent if dotenv is not installed.
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=".env.local", override=False)
        load_dotenv(override=False)
    except ImportError:
        pass

    items: list[MCQItem]
    dataset_label: str
    if args.dataset:
        print(f"Loading dataset: {args.dataset}...", file=sys.stderr)
        items = DATASET_REGISTRY[args.dataset]()
        dataset_label = args.dataset
    else:
        items = SAMPLE_ITEMS
        dataset_label = "phase1_smoke"

    runner_cls = RUNNER_REGISTRY[args.provider]
    runner = runner_cls(model=args.model)
    jsonl_path, manifest_path, manifest, responses = run_eval(
        runner, items, dataset_label, out_dir=Path(args.results_dir)
    )

    score = top1_accuracy(responses, items)
    n_errors = sum(1 for r in responses if r.error is not None)

    print(f"\nWrote {jsonl_path.name}")
    print(f"      {manifest_path.name}")
    elapsed = manifest.total_elapsed_s or 0.0
    print(f"\nRan {len(items)} items in {elapsed:.1f}s ({n_errors} errored).")
    print(f"Top-1 accuracy: {int(score['n_correct'])}/{int(score['n'])} = {score['accuracy']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
