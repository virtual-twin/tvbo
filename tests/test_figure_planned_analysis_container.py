"""A figure generated before an analysis it reads has run names the container that analysis writes, so its frozen script draws the analysis once it is there.

A workflow kit runs experiments and never analyses, so a figure bound to an analysis (`used: {analysis: <name>}`) whose container is absent when the kit is frozen reads it only once it is staged where the analysis writes it. The figure's render rule lists that path as an input and the frozen `plot.py` opens it: a panel with a `placeholder` draws the label until the container is there and the data after, without the script being generated again, and a panel without one fails naming the path.

Self-contained: one study directory in `tmp_path`, the frozen script run with this interpreter as a kit's render rule runs it, and one container written between two runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot
from tvbo.adapters import figure_workflow as fw
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path


@pytest.fixture
def planned(tmp_path) -> str:
    """Where analysis `trace` writes its container in a study at `tmp_path` whose results directory holds nothing, resolved as a figure resolves it."""
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    return str(analysis_container_path(results, "trace").resolve())


def _figure(placeholder=None):
    """One cartesian panel drawing analysis `trace`'s output `v` over `t`."""
    layer = P.Layer(used=P.DataRef(analysis="trace", output="v"), encoding=P.Encoding(x="t", y="v"))
    panel = P.Panel(panel_key="a", kind="cartesian", layers=[layer], placeholder=placeholder)
    return P.Figure(name="fig", layout="a", panels={"a": panel})


def _freeze(root: Path, figure) -> Path:
    """Write the figure's script into the study at *root*, as a kit freezes it, drawing to ``fig.png`` beside it."""
    script = root / "plot.py"
    script.write_text(bsplot.render_code(figure, root, str(root / "fig.png")), encoding="utf-8")
    return script


def _run(script: Path) -> subprocess.CompletedProcess:
    """Run a frozen script with this interpreter, as a kit's render rule runs it."""
    return subprocess.run([sys.executable, str(script)], cwd=script.parent, capture_output=True, text=True)


def test_the_render_rule_waits_for_the_container_an_absent_analysis_writes(tmp_path, planned):
    """The binding was dropped from the rule's inputs, so the kit named the file nowhere (G214); nothing in the kit writes it, so the figure stays out of the default target."""
    (context,) = fw.figure_contexts([_figure()], base_dir=tmp_path)
    assert context["inputs"] == [{"value": planned, "raw": False}]
    assert context["kit_satisfiable"] is False


def test_the_frozen_script_reads_the_container_an_absent_analysis_writes(tmp_path, planned):
    """The script held an empty path in its place (G214)."""
    assert planned in bsplot.render_code(_figure(), tmp_path, "fig.png")


def test_a_placeholder_panel_draws_the_analysis_once_its_container_is_written(tmp_path, planned):
    """The placeholder stayed for good: the script held an empty path, so writing the container where the analysis writes it changed nothing (G214)."""
    script, out = _freeze(tmp_path, _figure(placeholder="needs analysis trace")), tmp_path / "fig.png"
    before = _run(script)
    assert before.returncode == 0, before.stderr
    assert "panel a: placeholder (FileNotFoundError" in before.stdout and out.is_file()
    out.unlink()
    xr.Dataset({"v": ("t", [0.0, 1.0, 0.5])}, coords={"t": [0.0, 0.5, 1.0]}).to_netcdf(planned, engine="h5netcdf")
    after = _run(script)
    assert after.returncode == 0, after.stderr
    assert "panel a: placeholder" not in after.stdout and out.is_file()


def test_a_panel_without_a_placeholder_names_the_container_it_reads(tmp_path, planned):
    """The script failed on the empty path without naming the file it needed (G214)."""
    done = _run(_freeze(tmp_path, _figure()))
    assert done.returncode != 0
    assert "FileNotFoundError" in done.stderr and planned in done.stderr
