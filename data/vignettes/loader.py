"""Filesystem loader for hand-written clinical vignettes.

Walks ``data/vignettes/<specialty>/*.json``, skips files whose name
starts with an underscore (templates / scratch files), and parses each
remaining file into a :class:`~eval.schemas.Vignette` instance.

The loader is the single entry-point that ``eval.pipeline`` (via the
``vignettes`` dataset registration) and the fairness eval (Phase 4)
both call to materialise the vignettes corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from eval.schemas import Vignette

VIGNETTES_DIR: Path = Path(__file__).parent
VERSION_FILE: Path = VIGNETTES_DIR / "VERSION"

Specialty = Literal["cardiology", "autoimmune"]
_ALL_SPECIALTIES: tuple[str, ...] = ("cardiology", "autoimmune")


def dataset_version() -> str:
    """Corpus version from ``data/vignettes/VERSION``, recorded in run manifests.

    Bumped by ``scripts/review_drafts.py promote`` whenever approved
    drafts are merged into the corpus.
    """
    return VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "unknown"


def load_vignettes(
    specialty: Specialty | None = None,
    *,
    base_dir: Path | None = None,
) -> list[Vignette]:
    """Load every vignette JSON file under ``base_dir`` (default: this dir).

    Args:
        specialty: When given, only walk that specialty's subdirectory.
            ``None`` (the default) walks both cardiology and autoimmune.
        base_dir: Override the vignettes root. Defaults to the directory
            containing this module so production callers don't need to
            think about paths; tests use ``tmp_path`` to isolate
            fixtures.

    Returns:
        A list of :class:`Vignette` instances, sorted by file path
        within each specialty for deterministic ordering. Files whose
        name starts with ``_`` (e.g. ``_TEMPLATE.json``, ``_DRAFT_*``)
        are skipped.

    Raises:
        pydantic.ValidationError: If a JSON file does not match the
            :class:`Vignette` schema. Loader does not try/except —
            schema violations should fail loudly so reviewers see them.
    """
    root = base_dir if base_dir is not None else VIGNETTES_DIR
    specialties: Iterable[str] = (specialty,) if specialty is not None else _ALL_SPECIALTIES

    out: list[Vignette] = []
    for sp in specialties:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for path in sorted(sp_dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.append(Vignette.model_validate(payload))
    return out
