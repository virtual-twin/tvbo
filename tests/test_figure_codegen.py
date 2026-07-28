"""Tests for TVBO's declarative figure codegen.

Covers the two figure adapters:

* ``tvbo.adapters.bsplot`` — resolves a declarative :class:`Figure` into a
  codegen context (``build_context``), emits a self-contained ``plot.py``
  (``render_code``), and emits + execs it under matplotlib Agg (``render``),
  plus the register/lookup API a study extends the engine through and the
  ``_container_path`` PROV resolver.
* ``tvbo.adapters.figure_workflow`` — lowers a figure's PROV ``used`` edges into
  a Snakemake render rule (``emit_figure_rules``).

Figure objects are constructed inline. Tests that need experiment result
containers point at the Taher2019 replication study and skip when it (or a
specific container) is absent, so the suite is robust on a fresh checkout.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
    """The documented adapter surface is importable and well-typed. Core ships no built-in
    transforms/panels — the registries are filled by studies through the register API."""
    assert isinstance(bsplot.TRANSFORMS, dict)
    assert isinstance(bsplot.CUSTOM_PANELS, dict)
    assert callable(bsplot.register_transform)
    assert callable(bsplot.register_panel)
    assert callable(bsplot.load_layer)
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


def test_container_path_resolves_results_dir_container(tmp_path):
    """A figure layer may bind a derived-figure container a replication study writes with
    ``ExperimentResult.save`` under ``output/results/<name>/result.h5`` (the ``results_io``
    convention), not only a ``tvbo run`` experiment under ``output/nc/<exp>/``. The last IRI
    segment names the results subdirectory."""
    bsplot._container_path.cache_clear()  # the resolver is lru_cached
    result = tmp_path / "output" / "results" / "fig3" / "result.h5"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"")  # existence is all the resolver checks
    assert bsplot._container_path("tvbo:result/Deco2014/fig3", tmp_path) == str(result.resolve())
    # An unknown name under the same base stays empty (no results/ subdir for it).
    assert bsplot._container_path("tvbo:result/Deco2014/figX", tmp_path) == ""


def test_container_path_resolves_flat_nc_container(tmp_path):
    """A whole-study ``tvbo run`` writes BIDS-style result files FLAT inside ``output/nc/``
    (``exp-<id>_desc-<model>_result.h5``), not in a per-experiment ``output/nc/<exp>/``
    subdirectory. The resolver must find that layout (regression: a figure's ``used`` IRI
    silently resolved to "" for it, so custom panels fell back to placeholder/degenerate
    data — the Cortes2013 Fig-2/3 bug)."""
    bsplot._container_path.cache_clear()  # the resolver is lru_cached
    nc = tmp_path / "output" / "nc"
    nc.mkdir(parents=True)
    result = nc / "exp-1_desc-TsodyksMarkram_result.h5"
    result.write_bytes(b"")  # existence is all the resolver checks
    (nc / "exp-1_desc-TsodyksMarkram_network.h5").write_bytes(b"")  # sidecar must be skipped
    got = bsplot._container_path("tvbo:exp/Cortes2013/exp-1", tmp_path)
    assert got == str(result.resolve())
    assert "network" not in Path(got).name


def test_container_path_flat_nc_no_exp_prefix_collision(tmp_path):
    """The flat-in-nc glob is boundary-anchored (``exp-1_*``) so a request for ``exp-1`` never
    grabs ``exp-10``'s container."""
    bsplot._container_path.cache_clear()
    nc = tmp_path / "output" / "nc"
    nc.mkdir(parents=True)
    (nc / "exp-10_desc-Model_result.h5").write_bytes(b"")   # only exp-10 exists so far
    assert bsplot._container_path("tvbo:exp/S/exp-1", tmp_path) == ""  # must NOT match exp-10
    (nc / "exp-1_desc-Model_result.h5").write_bytes(b"")
    bsplot._container_path.cache_clear()  # the "" above is memoised for these exact args
    assert bsplot._container_path("tvbo:exp/S/exp-1", tmp_path).endswith(
        "exp-1_desc-Model_result.h5")


def test_container_path_output_root_no_exp_prefix_collision(tmp_path):
    """The flat OUTPUT-ROOT glob (output/<exp>_*result.h5) is boundary-anchored too, so exp-1
    never binds exp-10's container there either."""
    bsplot._container_path.cache_clear()
    out = tmp_path / "output"
    out.mkdir(parents=True)
    (out / "exp-10_desc-Model_result.h5").write_bytes(b"")  # only exp-10 at the output root
    assert bsplot._container_path("tvbo:exp/S/exp-1", tmp_path) == ""  # must NOT match exp-10
    (out / "exp-1_desc-Model_result.h5").write_bytes(b"")
    bsplot._container_path.cache_clear()
    assert bsplot._container_path("tvbo:exp/S/exp-1", tmp_path).endswith(
        "exp-1_desc-Model_result.h5")


def test_container_path_digit_bearing_non_experiment_iri_does_not_misbind(tmp_path):
    """A non-experiment IRI whose last segment merely CONTAINS digits (a connectome/dataset
    ref like ``rec-avgMatrix_atlas-HCPMMP1``) must not be read as ``exp-1`` and grab exp-1's
    container. Only a strict ``exp-N`` / ``expN`` / bare-``N`` key yields exp candidates."""
    bsplot._container_path.cache_clear()
    nc = tmp_path / "output" / "nc"
    nc.mkdir(parents=True)
    (nc / "exp-1_desc-Model_result.h5").write_bytes(b"")   # exp-1 exists...
    # ...but a connectome IRI ending in HCPMMP1 must NOT resolve to it.
    assert bsplot._container_path("tvbo:dataset/rec-avgMatrix_atlas-HCPMMP1", tmp_path) == ""
    bsplot._container_path.cache_clear()
    # A genuine experiment ref still resolves.
    assert bsplot._container_path("tvbo:exp/S/exp-1", tmp_path).endswith("exp-1_desc-Model_result.h5")


def test_used_ref_prefers_in_study_experiment_id(tmp_path):
    """A figure layer's ``used`` may bind an in-study experiment by id (``experiment: 2``) rather
    than a raw ``iri`` — it resolves to that experiment's container with no hardcoded study key.
    An explicit ``iri`` still wins when both are given."""
    bsplot._container_path.cache_clear()
    nc = tmp_path / "output" / "nc"
    nc.mkdir(parents=True)
    (nc / "exp-2_desc-Model_result.h5").write_bytes(b"")
    assert bsplot._used_ref(P.DataRef(experiment="2")) == "exp-2"
    assert bsplot._container_path(bsplot._used_ref(P.DataRef(experiment="2")), tmp_path).endswith(
        "exp-2_desc-Model_result.h5")
    assert bsplot._used_ref(P.DataRef(iri="tvbo:exp/S/exp-5", experiment="2")) == "tvbo:exp/S/exp-5"
    assert bsplot._used_ref(P.DataRef()) is None


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


def test_build_context_image_path_resolved_against_base_dir(tmp_path):
    """An ``image`` panel's path is spec-relative, so the spec stays portable; the emitted
    plot.py runs from an arbitrary cwd, so it must be resolved against base_dir (an
    absolute path is passed through). Same contract as a study .mplstyle."""
    figure = P.Figure(
        name="img",
        layout="ab",
        panels={
            "a": P.Panel(panel_key="a", kind="image", path="original_study/img/fig5.png"),
            "b": P.Panel(panel_key="b", kind="image", path="/abs/elsewhere.png"),
        },
    )
    panels = {p["key"]: p for p in bsplot.build_context(figure, tmp_path, "out.png")["panels"]}
    assert panels["a"]["path"] == str(tmp_path / "original_study" / "img" / "fig5.png")
    assert panels["b"]["path"] == "/abs/elsewhere.png"


def test_build_context_default_letter_format():
    """With no ``panel_number_format`` the letter is the bare index letter."""
    ctx = bsplot.build_context(_cartesian_figure(), TAHER_BASE, "out.png")
    assert ctx["panels"][0]["letter"] == "a"


def test_build_context_dataclass_flavor_kind_mark_loc():
    """Production loads figures as the *dataclass* datamodel — both ``tvbo figure
    render`` (``schema.Figure``) and ``SimulationStudy`` (extends
    ``schema.SimulationStudy``) — whose enum-valued slots are NOT ``==`` a bare
    string. The rest of this module builds pydantic objects (where the enum is a
    ``str``), so the adapter must normalise ``kind``/``mark``/``loc`` for the
    dataclass flavour or custom/image/heatmap dispatch and corner placement break
    only in production.
    """
    from tvbo.datamodel import schema as D

    fig = D.Figure(
        name="dc", layout="ab/cd", panel_number_loc="upper right",
        panels={
            "a": D.Panel(panel_key="a", kind="custom", render="custom_panel"),
            "b": D.Panel(panel_key="b", kind="image", path="/tmp/x.png"),
            "c": D.Panel(
                panel_key="c", kind="cartesian", number_loc="lower left",
                layers=[D.Layer(used=D.DataRef(iri="tvbo:exp/x", output="v"),
                                mark="scatter", encoding=D.Encoding(x="K", y="v"))],
            ),
            "d": D.Panel(
                panel_key="d", kind="heatmap",
                layers=[D.Layer(used=D.DataRef(iri="tvbo:exp/x", output="m"),
                                encoding=D.Encoding(x="K", y="n"))],
            ),
        },
    )
    ctx = bsplot.build_context(fig, TAHER_BASE, "out.png")
    by = {p["key"]: p for p in ctx["panels"]}

    # kind resolves to a plain string -> custom dispatch + image/heatmap branching
    assert by["a"]["kind"] == "custom" and by["a"]["ctx"] is not None
    assert by["b"]["kind"] == "image"
    assert by["c"]["layers"][0]["mark"] == "scatter"    # explicit mark survives, not defaulted to line
    assert by["d"]["layers"][0]["mark"] == "heatmap"    # implied by the heatmap panel kind
    # Corner enum -> corner placement kwargs (figure default, and per-panel override)
    assert by["a"]["number_kwargs"]["ha"] == "right"
    assert (by["c"]["number_kwargs"]["ha"], by["c"]["number_kwargs"]["va"]) == ("left", "bottom")

    # the emitted plot.py parses and dispatches each kind
    code = bsplot.render_code(fig, TAHER_BASE, "out.png")
    ast.parse(code)
    assert "_registered(_CP," in code   # custom-panel callable dispatch
    assert "pcolormesh" in code     # heatmap
    assert "imread" in code         # image


def test_register_panel_and_transform():
    """The register_* decorators populate the shared registries, so a study's
    code_source module adds its own custom panels/transforms; the emitted plot.py
    looks them up by name in these dicts.
    """
    @bsplot.register_panel("_test_panel")
    def _panel(fig, ax, ctx):
        return "drawn"

    @bsplot.register_transform("_test_tf")
    def _tf(da):
        return da

    try:
        assert bsplot.CUSTOM_PANELS["_test_panel"] is _panel
        assert bsplot.TRANSFORMS["_test_tf"] is _tf
        assert _panel(None, None, None) == "drawn"      # decorator returns the fn unchanged
    finally:                                            # keep the module-level registries clean for other tests
        bsplot.CUSTOM_PANELS.pop("_test_panel", None)
        bsplot.TRANSFORMS.pop("_test_tf", None)


def test_unregistered_name_error_points_at_code_modules():
    """Core ships no panels/transforms, so "declared a name but never registered it" (a
    missing or unimportable code_modules entry) is THE common failure. It must name the
    culprit and the fix rather than surfacing a bare KeyError from a generated file."""
    fig = P.Figure(
        name="oops", layout="a",
        panels={"a": P.Panel(panel_key="a", kind="custom", render="never_registered")},
    )
    code = bsplot.render_code(fig, TAHER_BASE, "out.png")
    with pytest.raises(KeyError) as exc:
        exec(compile(code, "plot.py", "exec"), {"__name__": "__main__"})
    msg = str(exc.value)
    assert "never_registered" in msg and "code_modules" in msg

    # The adapter's own load_layer path carries the same guidance for a transform.
    with pytest.raises(KeyError, match="code_modules"):
        bsplot.load_layer({"container": "x.h5", "output": "y", "transform": "nope"})


def test_render_code_emits_study_code_module_imports():
    """A figure declaring code_modules emits an import for each, so a study's
    code_source panels/transforms register when plot.py runs (the study load
    puts code/ on the path; importing the module fires the @register_* decorators).
    A figure with none emits no such import.
    """
    fig = P.Figure(
        name="withcode", layout="a", code_modules=["taher2019_figures", "myextra"],
        panels={"a": P.Panel(panel_key="a", kind="custom", render="custom_panel")},
    )
    code = bsplot.render_code(fig, TAHER_BASE, "out.png")
    ast.parse(code)
    assert "import taher2019_figures" in code
    assert "import myextra" in code
    # no code_modules -> no study-import block
    assert "registers this study" not in bsplot.render_code(_cartesian_figure(), TAHER_BASE, "out.png")


def test_study_code_module_roundtrip(tmp_path, monkeypatch):
    """End-to-end proof of the register API + code_modules + emit-wiring together:
    a study module registers a panel core tvbo never knew about; the emitted
    plot.py imports the module (firing the decorator), so the panel dispatches
    and the figure draws.
    """
    import sys
    import textwrap

    (tmp_path / "study_demo_panels.py").write_text(textwrap.dedent('''
        from tvbo.adapters import bsplot

        @bsplot.register_panel("demo_roundtrip_panel")
        def demo(fig, ax, ctx):
            ax.text(0.5, 0.5, "study", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))          # a study load would put its code/ dir here
    assert "demo_roundtrip_panel" not in bsplot.CUSTOM_PANELS    # core tvbo does not know it

    fig = P.Figure(
        name="rt", layout="a", code_modules=["study_demo_panels"],
        panels={"a": P.Panel(panel_key="a", kind="custom", render="demo_roundtrip_panel")},
    )
    out = tmp_path / "rt.png"
    try:
        bsplot.render(fig, str(tmp_path), str(out))     # emitted plot.py imports the module...
        assert "demo_roundtrip_panel" in bsplot.CUSTOM_PANELS   # ...which registered the panel...
        assert out.is_file() and out.stat().st_size > 0         # ...and it drew.
    finally:
        bsplot.CUSTOM_PANELS.pop("demo_roundtrip_panel", None)
        sys.modules.pop("study_demo_panels", None)


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


def test_render_code_trim_margins_toggle():
    """Trimming re-crops to content, so the saved aspect can drift from width/height;
    ``trim_margins: false`` is the opt-out that preserves the declared proportions.
    Default (unset) trims."""
    assert "'bbox_inches': 'tight'" in _emit()
    assert "'bbox_inches': 'tight'" in _emit(trim_margins=True)
    assert "bbox_inches" not in _emit(trim_margins=False)


def test_render_code_panel_numbers_toggle():
    """A ``bsplot.add_panel_number`` call is emitted iff panel numbering is on."""
    assert "_bpanels.add_panel_number(axd[" in _emit(panel_numbers=True)
    assert "_bpanels.add_panel_number(axd[" not in _emit(panel_numbers=False)


def test_render_code_auto_format_toggle():
    """``bsplot.style.format_fig`` appears iff auto_format is not disabled."""
    assert "bsplot.style.format_fig" in _emit(auto_format=True)
    assert "bsplot.style.format_fig" in _emit()  # default-on
    assert "bsplot.style.format_fig" not in _emit(auto_format=False)


# --------------------------------------------------------------------------- placeholder / axes / matrix

def _placeholder_figure(**panel_kw):
    """A one-panel figure whose panel carries a placeholder and nothing else to draw."""
    return P.Figure(
        name="ph", layout="a",
        panels={"a": P.Panel(panel_key="a", kind="cartesian",
                             placeholder="needs exp-2 (resting BOLD)", **panel_kw)},
    )


def test_placeholder_only_panel_draws_the_label():
    """A panel with a placeholder and NO layers draws the label, not an empty 0-1 axes.

    The guarded draw cannot catch this case: with nothing bound, the panel body raises
    nothing and the honest placeholder would silently never appear."""
    ctx = bsplot.build_context(_placeholder_figure(), TAHER_BASE, "out.png")
    assert ctx["panels"][0]["placeholder_only"] is True

    code = bsplot.render_code(_placeholder_figure(), TAHER_BASE, "out.png")
    ast.parse(code)
    body = code.split("def _panel_a(")[1].split("def ")[0]
    assert "_placeholder(ax, 'needs exp-2 (resting BOLD)')" in body


def test_placeholder_with_data_stays_a_guarded_fallback():
    """A placeholder panel that DOES bind data keeps the try/except fallback — the label
    is the honest stand-in for a missing container, not the panel itself."""
    figure = _cartesian_figure(iri=MISSING_IRI)
    figure.panels["a"].placeholder = "no data"
    ctx = bsplot.build_context(figure, TAHER_BASE, "out.png")
    assert ctx["panels"][0]["placeholder_only"] is False

    code = bsplot.render_code(figure, TAHER_BASE, "out.png")
    assert "try:" in code and "_placeholder(axd['a'], 'no data')" in code


def test_axvline_and_axhline_accept_scalar_or_list():
    """Reference lines are declarative axis directives: the paper's dashed verticals at
    N = 10/100/200 are one ``axvline`` list, not three hand-drawn calls."""
    figure = _cartesian_figure()
    figure.panels["a"].opts = {
        "axvline": P.Argument(name="axvline", value=[10, 100, 200]),
        "axhline": P.Argument(name="axhline", value=0.0),
    }
    ctx = bsplot.build_context(figure, TAHER_BASE, "out.png")
    assert ctx["panels"][0]["axopts"]["axvline"] == [10, 100, 200]
    assert ctx["panels"][0]["axopts"]["axhline"] == 0.0
    code = bsplot.render_code(figure, TAHER_BASE, "out.png")
    ast.parse(code)
    assert "'axvline': [10, 100, 200]" in code


def _matrix_figure():
    """A heatmap panel composed of two triangles: data below, model above the diagonal."""
    return P.Figure(
        name="fc", layout="a",
        panels={"a": P.Panel(
            panel_key="a", kind="heatmap",
            opts={"invert_y": P.Argument(name="invert_y", value=True),
                  "aspect": P.Argument(name="aspect", value="equal")},
            layers=[
                P.Layer(used=P.DataRef(iri=EXP3_IRI, output="fc_data"), triangle="lower",
                        encoding=P.Encoding(x="region_j", y="region_i"),
                        style=P.Style(colormap="seismic",
                                      opts={"vmin": P.Argument(name="vmin", value=-1.0),
                                            "vmax": P.Argument(name="vmax", value=1.0)})),
                P.Layer(used=P.DataRef(iri=EXP3_IRI, output="fc_model"), triangle="upper",
                        encoding=P.Encoding(x="region_j", y="region_i"),
                        style=P.Style(colormap="seismic")),
            ])},
    )


def test_split_triangle_matrix_layers():
    """Two layers compose ONE matrix: each masks its half, they share a single colourbar,
    and the panel reads with the matrix convention (row 0 at top)."""
    ctx = bsplot.build_context(_matrix_figure(), TAHER_BASE, "out.png")
    panel = ctx["panels"][0]
    assert [l["triangle"] for l in panel["layers"]] == ["lower", "upper"]
    assert panel["colorbar"] is True
    assert panel["axopts"]["invert_y"] is True and panel["axopts"]["aspect"] == "equal"
    # A heatmap layer's Style resolves to mesh kwargs (colormap + limits), not line kwargs.
    assert panel["layers"][0]["style"] == {"cmap": "seismic", "vmin": -1.0, "vmax": 1.0}

    code = bsplot.render_code(_matrix_figure(), TAHER_BASE, "out.png")
    ast.parse(code)
    assert code.count("_triangle(_C, 'lower')") == 1
    assert code.count("_triangle(_C, 'upper')") == 1
    assert code.count("fig.colorbar(") == 1          # one scale for the composed matrix


def test_triangle_masks_the_other_half():
    """The emitted ``_triangle`` helper keeps exactly one half and NaNs the rest."""
    ns: dict = {}
    exec(compile(bsplot.render_code(_matrix_figure(), TAHER_BASE, "out.png"),
                 "<figure>", "exec"), ns)
    import numpy as np

    m = np.arange(9.0).reshape(3, 3)
    upper, lower = ns["_triangle"](m, "upper"), ns["_triangle"](m, "lower")
    assert np.isnan(np.diag(upper)).all() and np.isnan(np.diag(lower)).all()
    assert upper[0, 2] == m[0, 2] and np.isnan(upper[2, 0])
    assert lower[2, 0] == m[2, 0] and np.isnan(lower[0, 2])
    with pytest.raises(ValueError, match="square"):
        ns["_triangle"](np.zeros((2, 3)), "upper")


def test_colorbar_suppressed_by_opt():
    """`colorbar: false` drops the scale where the paper prints none."""
    figure = _matrix_figure()
    figure.panels["a"].opts["colorbar"] = P.Argument(name="colorbar", value=False)
    ctx = bsplot.build_context(figure, TAHER_BASE, "out.png")
    assert ctx["panels"][0]["colorbar"] is False
    assert "fig.colorbar(" not in bsplot.render_code(figure, TAHER_BASE, "out.png")


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
    # 8G -> 8192 MiB (Slurm sizes are binary), 02:00:00 -> 120 min, cpus_per_task
    # passed through, partition surfaced.
    assert "cpus_per_task=4" in text
    assert "mem_mb=8192" in text
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
