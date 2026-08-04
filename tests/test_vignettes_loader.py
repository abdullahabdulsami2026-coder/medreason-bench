"""Tests for the hand-written vignette loader + the two _TEMPLATE files.

The Phase-3 scaffold ships only ``_TEMPLATE.json`` files in each
specialty directory; the loader skips files starting with ``_``, so
:func:`load_vignettes` returns ``[]`` until real vignettes land.

Two additional tests load the templates directly (bypassing the
underscore-skip rule) and validate them against the
:class:`Vignette` schema — the templates are the contract for every
vignette that follows.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.vignettes.loader import load_vignettes
from eval.schemas import Vignette

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CARDIO_TEMPLATE: Path = REPO_ROOT / "data/vignettes/cardiology/_TEMPLATE.json"
AUTOIMMUNE_TEMPLATE: Path = REPO_ROOT / "data/vignettes/autoimmune/_TEMPLATE.json"


# ── Loader: should skip _-prefixed files ─────────────────────────


def test_load_vignettes_skips_underscore_files_in_isolation(tmp_path: Path) -> None:
    """In an isolated dir containing only underscore-prefixed files, the
    loader yields []. Decoupled from the real ``data/vignettes`` dir so
    this test stays valid as real vignettes are added."""
    (tmp_path / "cardiology").mkdir()
    (tmp_path / "autoimmune").mkdir()
    template_payload = json.loads(CARDIO_TEMPLATE.read_text())
    autoimmune_payload = json.loads(AUTOIMMUNE_TEMPLATE.read_text())
    (tmp_path / "cardiology" / "_TEMPLATE.json").write_text(json.dumps(template_payload))
    (tmp_path / "cardiology" / "_DRAFT_scratch.json").write_text(json.dumps(template_payload))
    (tmp_path / "autoimmune" / "_TEMPLATE.json").write_text(json.dumps(autoimmune_payload))

    assert load_vignettes(base_dir=tmp_path) == []
    assert load_vignettes("cardiology", base_dir=tmp_path) == []
    assert load_vignettes("autoimmune", base_dir=tmp_path) == []


def test_load_vignettes_real_directory_validates() -> None:
    """All non-underscore JSON files in ``data/vignettes/`` validate against
    the schema and have unique IDs. Coverage grows as new vignettes land
    without the test needing to be updated."""
    items = load_vignettes()
    ids = [v.id for v in items]
    assert len(ids) == len(set(ids)), "Vignette IDs must be unique"
    for v in items:
        assert v.specialty in {"cardiology", "autoimmune"}
        assert v.correct is None or v.correct in v.options
        for variant in v.variants:
            override = variant.correct_override
            assert override is None or override in v.options


def test_load_vignettes_picks_up_real_files(tmp_path: Path) -> None:
    """In an isolated dir with one real (non-underscore) vignette + one
    template, the loader yields exactly the real vignette."""
    cardio_dir = tmp_path / "cardiology"
    cardio_dir.mkdir()
    template_payload = json.loads(CARDIO_TEMPLATE.read_text())
    real_payload = {**template_payload, "id": "cardio_real_001"}
    (cardio_dir / "_TEMPLATE.json").write_text(json.dumps(template_payload))
    (cardio_dir / "cardio_real_001.json").write_text(json.dumps(real_payload))

    items = load_vignettes("cardiology", base_dir=tmp_path)
    assert len(items) == 1
    assert items[0].id == "cardio_real_001"


# ── Templates: must be valid Vignette objects ─────────────────────


def _load_template(path: Path) -> Vignette:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Vignette.model_validate(payload)


def test_cardiology_template_validates_as_vignette() -> None:
    v = _load_template(CARDIO_TEMPLATE)
    assert v.specialty == "cardiology"
    assert v.correct in v.options
    assert len(v.variants) >= 2
    assert v.difficulty in {"easy", "medium", "hard"}
    assert len(v.sources) >= 1
    # variants must each have a valid demographics block
    for variant in v.variants:
        assert variant.variant_id  # non-empty
        assert variant.demographics is not None


def test_autoimmune_template_validates_as_vignette() -> None:
    v = _load_template(AUTOIMMUNE_TEMPLATE)
    assert v.specialty == "autoimmune"
    assert v.correct in v.options
    assert len(v.variants) >= 2
    assert v.difficulty in {"easy", "medium", "hard"}
    assert len(v.sources) >= 1
    for variant in v.variants:
        assert variant.variant_id
        assert variant.demographics is not None
