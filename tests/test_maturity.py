"""Guard for ``tvbo/plot/maturity.yaml``: every capability names a known level, stage and status, and the evidence behind every claim still exists, so the maturity wheel cannot claim what tvbo no longer ships."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import typer
import yaml

from tvbo.cli import app

ROOT = Path(__file__).resolve().parents[1]
TABLE = yaml.safe_load((ROOT / "tvbo/plot/maturity.yaml").read_text())
STAGES = set(yaml.safe_load((ROOT / "tvbo/plot/lifecycle.yaml").read_text())["stages"])
CAPS = TABLE["capabilities"]


def _cli_exists(words: str) -> bool:
    cmd = typer.main.get_command(app)
    for word in words.split():
        cmd = getattr(cmd, "commands", {}).get(word)
        if cmd is None:
            return False
    return True


def _python_exists(dotted: str) -> bool:
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def _schema_exists(cls: str) -> bool:
    return any(cls in (yaml.safe_load(p.read_text()) or {}).get("classes", {}) for p in (ROOT / "schema").glob("*.yaml"))


CHECKS = {
    "cli": _cli_exists,
    "python": _python_exists,
    "skill": lambda name: (ROOT / "tvbo/skills/canonical" / name / "SKILL.md").is_file(),
    "file": lambda rel: (ROOT / rel).exists(),
    "schema": _schema_exists,
}


@pytest.mark.parametrize("cap", CAPS, ids=lambda c: f"{c['level']}-{c['stage']}-{c['practice']}")
def test_capability_is_well_formed(cap):
    assert cap["level"] in TABLE["levels"]
    assert cap["stage"] in STAGES
    assert cap["status"] in TABLE["statuses"]
    assert bool(cap.get("evidence")) == (cap["status"] != "not yet")
    assert bool(cap.get("missing")) == (cap["status"] != "ships")


@pytest.mark.parametrize(
    "kind,ref", [(k, v) for c in CAPS for e in c.get("evidence", []) for k, v in e.items()], ids=lambda x: str(x)
)
def test_evidence_exists(kind, ref):
    assert CHECKS[kind](ref), f"{kind} evidence {ref!r} no longer exists"


def test_banded_wheel_draws_every_capability():
    from tvbo.plot import lifecycle

    svg = lifecycle.wheel("tvbo:themes/dark", bands=list(TABLE["levels"]), current="scalable")
    for status in TABLE["statuses"]:
        assert svg.count(f'status-{status.replace(" ", "-")}"') == sum(c["status"] == status for c in CAPS)
    assert all(f"L{i} {lv['name']}" in svg for i, lv in enumerate(TABLE["levels"].values(), start=1))


def test_banded_wheel_refuses_unknown_names():
    from tvbo.plot import lifecycle

    with pytest.raises(ValueError, match="not maturity levels"):
        lifecycle.wheel(bands=["initial", "heroic"])
    with pytest.raises(ValueError, match="current must be one of the bands"):
        lifecycle.wheel(bands=["initial"], current="optimizing")


def test_a_wheel_lights_only_the_named_stages_of_its_current_band():
    """``lit`` keeps the named stages of the current band in full tone and dims its other stages, and needs a current band."""
    from tvbo.plot.lifecycle import _hue, _theme, wheel

    bands = ["defined", "scalable", "optimizing"]
    svg = wheel(
        "tvbo:themes/default",
        bands=bands,
        current="scalable",
        lit=["generate", "run"],
        cells=[
            {"stage": s, "band": b, "status": "ships"}
            for b in bands
            for s in ("explore", "specify", "generate", "run", "report", "share")
        ],
    )
    colours = _theme("tvbo:themes/default")
    for stage, full in (("generate", True), ("run", True), ("specify", False)):
        cell = re.search(rf'class="bcell band-scalable stage-{stage}[^"]*" fill="([^"]+)"', svg).group(1)
        assert (cell == _hue(colours, stage)) is full
    with pytest.raises(ValueError, match="lit needs current"):
        wheel(bands=bands, lit=["run"])


def test_a_locator_lights_only_its_level_and_stage():
    """The corner mark keeps the cells of its level and stage in full tone and dims the rest, so exactly one stage-by-band cell group stands out, and names the level in its centre and nothing else."""
    from tvbo.plot.lifecycle import _hue, _theme, locator

    svg = locator("tvbo:themes/default", bands=["defined", "scalable", "optimizing"], level="scalable", stage="run")
    run_hue = _hue(_theme("tvbo:themes/default"), "run")
    lit = re.findall(r'class="bcell band-(\w+) stage-(\w+)[^"]*" (?:fill|style)="[^"]*' + run_hue, svg)
    assert lit and set(lit) == {("scalable", "run")}
    assert re.findall(r"<text[^>]*>([^<]*)</text>", svg) == ["L4"]
    assert "<text" not in locator("tvbo:themes/default", bands=["defined", "scalable", "optimizing"], stage="run")


def test_a_wheel_whose_cells_share_one_status_draws_no_key():
    """One solid cell per stage and band says nothing a key could tell apart, so the key is left out."""
    from tvbo.plot.lifecycle import stages, wheel

    bands = ["defined", "scalable", "optimizing"]
    solid = [{"stage": s, "band": b, "status": "ships"} for b in bands for s in stages()]
    assert 'class="key' not in wheel("tvbo:themes/default", bands=bands, cells=solid)
    assert 'class="key' in wheel("tvbo:themes/default", bands=bands)
