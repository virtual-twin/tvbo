"""Tests for TVBO's declarative figure codegen.

Covers the two figure adapters:

* ``tvbo.adapters.bsplot`` — resolves a declarative :class:`Figure` into a
  codegen context (``build_context``), emits a self-contained ``plot.py``
  (``render_code``), and emits + execs it under matplotlib Agg (``render``),
  plus the presentation-only ``TRANSFORMS`` and the ``_container_path`` PROV
  resolver.
* ``tvbo.adapters.figure_workflow`` — lowers a figure's PROV ``used`` edges into
  a Snakemake render rule (``emit_figure_rules``).

Figure objects are constructed inline. Tests that need experiment result
containers point at the Taher2019 replication study and skip when it (or a
specific container) is absent, so the suite is robust on a fresh checkout.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot
from tvbo.adapters import figure_workflow as fw

# --------------------------------------------------------------------------- fixtures / gating

TAHER_BASE = Path(
    "/Users/leonmartin_bih/projects/TVB-O/tvbo-manuscript/"
    "use-cases/replication_studies/Taher2019"
)
EXP3_IRI = "tvbo:exp/Taher2019/exp-3"
MISSING_IRI = "tvbo:exp/Taher2019/exp-999"  # deliberately non-existent -> placeholder

# Resolve once so every data-backed test shares the same gate.
_EXP3_CONTAINER = bsplot._container_path(EXP3_IRI, TAHER_BASE) if TAHER_BASE.is_dir() else ""

requires_exp3 = pytest.mark.skipif(
    not _EXP3_CONTAINER,
    reason=f"Taher2019 exp-3 container not present under {TAHER_BASE}",
)


def _cartesian_figure(iri=EXP3_IRI, output="delta_omega", **fig_kw):
    """A minimal one-panel cartesian line Figure over *iri*/*output*."""
    return P.Figure(
        name=fig_kw.pop("name", "fig"),
        layout=fig_kw.pop("layout", "a"),
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="cartesian",
                layers=[
                    P.Layer(
                        used=P.DataRef(iri=iri, output=output),
                        encoding=P.Encoding(x="KuramotoInertia.K", y=output),
                    )
                ],
            )
        },
        **fig_kw,
    )


# --------------------------------------------------------------------------- public surface

def test_public_surface():
    """The documented adapter surface is importable and well-typed."""
    assert set(bsplot.TRANSFORMS) == {"up_branch", "down_branch", "order_by_branch"}
    assert all(callable(fn) for fn in bsplot.TRANSFORMS.values())
    assert isinstance(bsplot.CUSTOM_PANELS, dict)
    assert callable(bsplot._style_kwargs)
    assert callable(bsplot._annotations)


# --------------------------------------------------------------------------- _container_path

@requires_exp3
def test_container_path_resolves_existing():
    """A real experiment IRI resolves to an existing, non-network ``.h5`` container."""
    path = bsplot._container_path(EXP3_IRI, TAHER_BASE)
    assert path
    p = Path(path)
    assert p.is_file()
    assert p.suffix == ".h5"
    assert "network" not in p.name  # the *_network.h5 sidecar is skipped


def test_container_path_unresolved_returns_empty():
    """An unknown experiment IRI or ``None`` resolves to the empty string, not an error."""
    assert bsplot._container_path(MISSING_IRI, TAHER_BASE) == ""
    assert bsplot._container_path(None, TAHER_BASE) == ""
    assert bsplot._container_path("", TAHER_BASE) == ""
    # Even a plausible-looking IRI under a base with no output/ tree stays empty.
    assert bsplot._container_path(EXP3_IRI, Path("/nonexistent/base/dir")) == ""


# --------------------------------------------------------------------------- transforms

def _hysteresis_da():
    """Synthetic up-then-down hysteresis scan: coord rises to 4.0 then falls to 0.0."""
    up = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    down = np.array([3.0, 2.0, 1.0, 0.0])
    coord = np.concatenate([up, down])
    return xr.DataArray(np.arange(coord.size, dtype=float), dims=["K"], coords={"K": coord})


def test_up_down_branch_split_at_argmax():
    """``up_branch``/``down_branch`` split the scan at the sweep reversal (argmax of coord)."""
    da = _hysteresis_da()
    nup = int(np.argmax(da["K"].values)) + 1  # reversal is inclusive on both halves

    ub = bsplot.up_branch(da)
    db = bsplot.down_branch(da)

    # up half is 0..reversal; down half is reversal..end; together they re-cover the scan.
    assert ub["K"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert db["K"].values.tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]
    assert ub.sizes["K"] == nup
    assert db.sizes["K"] == da.sizes["K"] - nup + 1  # reversal point shared by both halves
    # The shared reversal point is the peak of the coordinate.
    assert float(ub["K"].values[-1]) == float(db["K"].values[0]) == float(da["K"].values.max())


def test_up_down_branch_without_coord():
    """With no coordinate on the dim the transforms fall back to the positional index."""
    da = xr.DataArray(np.arange(6.0), dims=["K"])  # monotone index 0..5, argmax at the end
    ub = bsplot.up_branch(da)
    db = bsplot.down_branch(da)
    assert ub.sizes["K"] == 6  # whole thing is the up-sweep
    assert db.sizes["K"] == 1  # only the reversal point remains


def test_order_by_branch_sorts_and_noop():
    """``order_by_branch`` sorts by ``branch_point`` when present, else is a no-op."""
    scrambled = xr.DataArray(
        [3.0, 1.0, 2.0], dims=["branch_point"], coords={"branch_point": [3, 1, 2]}
    )
    ordered = bsplot.order_by_branch(scrambled)
    assert ordered["branch_point"].values.tolist() == [1, 2, 3]
    assert ordered.values.tolist() == [1.0, 2.0, 3.0]

    # No branch_point dim/coord -> returned unchanged.
    plain = xr.DataArray([9.0, 8.0], dims=["K"], coords={"K": [0, 1]})
    out = bsplot.order_by_branch(plain)
    assert out.dims == plain.dims
    assert out.values.tolist() == plain.values.tolist()


# --------------------------------------------------------------------------- build_context

@requires_exp3
def test_build_context_resolves_everything():
    """A mixed cartesian/heatmap/image + guarded figure resolves into the full context."""
    figure = P.Figure(
        name="mixed",
        layout="ab/cd",
        width=180,
        height=120,
        font_size=8,
        panel_number_format="({})",
        panel_number_loc="upper right",
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="cartesian",
                layers=[
                    P.Layer(
                        used=P.DataRef(iri=EXP3_IRI, output="delta_omega"),
                        encoding=P.Encoding(x="KuramotoInertia.K", y="delta_omega"),
                        style=P.Style(
                            color="red",
                            opts={"linestyle": P.Argument(name="linestyle", value="--")},
                        ),
                    )
                ],
                annotations=[
                    P.Annotation(text="corner", loc="upper left"),
                    P.Annotation(text="xy", x=0.2, y=0.3),
                ],
            ),
            "b": P.Panel(
                panel_key="b",
                kind="heatmap",
                layers=[
                    P.Layer(
                        used=P.DataRef(iri=EXP3_IRI, output="omega_profile"),
                        encoding=P.Encoding(x="KuramotoInertia.K", y="mode"),
                    )
                ],
            ),
            "c": P.Panel(panel_key="c", kind="image", path="/tmp/panel_c.png"),
            "d": P.Panel(
                panel_key="d",
                kind="cartesian",
                placeholder="no data",
                layers=[
                    P.Layer(
                        used=P.DataRef(iri=MISSING_IRI, output="delta_omega"),
                        encoding=P.Encoding(x="KuramotoInertia.K", y="delta_omega"),
                    )
                ],
            ),
        },
    )
    ctx = bsplot.build_context(figure, TAHER_BASE, "out.png")

    # top-level sizing / formatting contract (subplots kwargs resolved in Python)
    assert ctx["name"] == "mixed"
    assert ctx["subplots_kwargs"]["layout"] == "ab\ncd"  # '/' -> mosaic row split
    assert ctx["subplots_kwargs"]["figsize"] == (180 / 25.4, 120 / 25.4)  # mm -> inches
    assert ctx["font_size"] == 8
    assert ctx["dpi"] == 200  # default
    assert ctx["auto_format"] is True
    assert ctx["panel_numbers"] is True
    # panel_number_loc "upper right" -> resolved into each panel's placement kwargs
    assert ctx["panels"][0]["number_kwargs"]["ha"] == "right"

    panels = {p["key"]: p for p in ctx["panels"]}
    assert set(panels) == {"a", "b", "c", "d"}

    # (a) cartesian line: container resolved, mark defaulted, style/opts merged, axopts defaulted
    a = panels["a"]
    assert a["kind"] == "cartesian"
    assert a["letter"] == "(a)"  # panel_number_format applied
    la = a["layers"][0]
    assert Path(la["container"]).is_file()
    assert la["mark"] == "line"
    assert la["style"] == {"color": "red", "linestyle": "--"}
    assert a["axopts"]["xlabel"] == "KuramotoInertia.K"
    assert a["axopts"]["ylabel"] == "delta_omega"
    # annotations -> [{text, x, y}] in axes-fraction coords
    assert a["annotations"] == [
        {"text": "corner", "x": 0.03, "y": 0.95},
        {"text": "xy", "x": 0.2, "y": 0.3},
    ]
    assert a["placeholder"] is None

    # (b) heatmap: mark defaults to 'heatmap' from the panel kind
    b = panels["b"]
    assert b["letter"] == "(b)"
    assert b["layers"][0]["mark"] == "heatmap"
    assert Path(b["layers"][0]["container"]).is_file()

    # (c) image: carries its path, no layers/axopts to resolve
    c = panels["c"]
    assert c["kind"] == "image"
    assert c["path"] == "/tmp/panel_c.png"
    assert c["letter"] == "(c)"

    # (d) guarded panel over a missing container: placeholder text surfaced, container empty
    d = panels["d"]
    assert d["placeholder"] == "no data"
    assert d["letter"] == "(d)"
    assert d["layers"][0]["container"] == ""


def test_build_context_default_letter_format():
    """With no ``panel_number_format`` the letter is the bare index letter."""
    ctx = bsplot.build_context(_cartesian_figure(), TAHER_BASE, "out.png")
    assert ctx["panels"][0]["letter"] == "a"


# --------------------------------------------------------------------------- render_code validity

def _emit(**fig_kw) -> str:
    return bsplot.render_code(_cartesian_figure(**fig_kw), TAHER_BASE, "out.png")


def test_render_code_is_valid_python():
    """The emitted plot.py parses and compiles, and contains the core scaffold."""
    code = _emit()
    ast.parse(code)          # syntactically valid
    compile(code, "<figure>", "exec")  # and compiles
    assert "bsplot.figure.subplots" in code
    assert "savefig" in code
    assert "def main():" in code


def test_render_code_font_size_emitted():
    """``font.size`` is set iff a physical ``font_size`` is declared."""
    assert "font.size" in _emit(font_size=9)
    assert "font.size" not in _emit()


def test_render_code_panel_numbers_toggle():
    """A ``bsplot.add_panel_number`` call is emitted iff panel numbering is on."""
    assert "_bpanels.add_panel_number(axd[" in _emit(panel_numbers=True)
    assert "_bpanels.add_panel_number(axd[" not in _emit(panel_numbers=False)


def test_render_code_auto_format_toggle():
    """``bsplot.style.format_fig`` appears iff auto_format is not disabled."""
    assert "bsplot.style.format_fig" in _emit(auto_format=True)
    assert "bsplot.style.format_fig" in _emit()  # default-on
    assert "bsplot.style.format_fig" not in _emit(auto_format=False)


# --------------------------------------------------------------------------- render round-trip

@requires_exp3
def test_render_writes_png(tmp_path):
    """A cartesian figure over the real exp-3 container renders to a non-empty PNG."""
    out = tmp_path / "delta_omega.png"
    fig = bsplot.render(_cartesian_figure(name="rt"), TAHER_BASE, str(out))
    assert out.is_file()
    assert out.stat().st_size > 0
    assert fig is not None  # render returns the matplotlib Figure


@requires_exp3
def test_render_guarded_missing_container_does_not_raise(tmp_path):
    """A guarded panel whose container is missing renders via the placeholder path."""
    out = tmp_path / "guarded.png"
    figure = P.Figure(
        name="guarded",
        layout="a",
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="cartesian",
                placeholder="no data",
                layers=[
                    P.Layer(
                        used=P.DataRef(iri=MISSING_IRI, output="delta_omega"),
                        encoding=P.Encoding(x="KuramotoInertia.K", y="delta_omega"),
                    )
                ],
            )
        },
    )
    # Must not raise despite the missing container — the try/except placeholder catches it.
    bsplot.render(figure, TAHER_BASE, str(out))
    assert out.is_file()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------- figure_workflow

def _rule_overrides():
    """A WorkflowConfig whose slurm block seeds figure render resources."""
    return P.WorkflowConfig(
        slurm=P.WorkflowEngineConfig(
            cpus_per_task=4, mem="8G", time="02:00:00", partition="gpu"
        )
    )


def test_emit_figure_rules_resources():
    """Rule text carries a ``rule fig_...`` and a ``resources:`` block from the overrides."""
    figure = _cartesian_figure(name="taher_fig1", workflow_overrides=_rule_overrides())
    text = fw.emit_figure_rules([figure], base_dir=TAHER_BASE)

    assert "rule fig_taher_fig1:" in text
    assert "resources:" in text
    # 8G -> 8000 MB, 02:00:00 -> 120 min, cpus_per_task passed through, partition surfaced.
    assert "cpus_per_task=4" in text
    assert "mem_mb=8000" in text
    assert "runtime=120" in text
    assert "slurm_partition='gpu'" in text
    assert "output:" in text
    assert "figures/taher_fig1.png" in text


@requires_exp3
def test_emit_figure_rules_input_is_resolved_container():
    """A layer's PROV ``used`` container becomes the render rule's ``input:``."""
    figure = _cartesian_figure(name="taher_fig1")
    text = fw.emit_figure_rules([figure], base_dir=TAHER_BASE)
    assert "input:" in text
    assert _EXP3_CONTAINER in text  # the exact resolved container path is the dependency


def test_emit_figure_rules_no_input_when_unresolved():
    """A figure whose container cannot resolve emits a rule but no ``input:`` block."""
    figure = _cartesian_figure(name="orphan", iri=MISSING_IRI)
    text = fw.emit_figure_rules([figure], base_dir=TAHER_BASE)
    assert "rule fig_orphan:" in text
    # No resolvable container -> the rule declares no inputs (can't depend on a missing file).
    assert "input:" not in text
