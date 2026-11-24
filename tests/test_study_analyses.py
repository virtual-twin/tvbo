"""A study's declarative ``analyses:`` — schedule, execution, persistence, binding.

An analysis is the study-scope form of ``Analysis``: a ``FunctionCall`` whose result the study keeps, so a figure can bind a quantity no simulation produced. These cover the contract a recipe author relies on — what runs when, what lands on disk, and what a ``used: {analysis: …}`` resolves to — plus the errors that must be loud rather than silent (duplicate names, unknown references, cycles, an unrenderable backend).
"""

import sys
import types

import numpy as np
import pytest
import xarray as xr

from tvbo.data import analysis_io, dataref
from tvbo.datamodel.schema import Analysis
from tvbo.utils.study_layout import study_path


@pytest.fixture
def calls(monkeypatch):
    """A throwaway module of analysis callables, importable by bare name."""
    mod = types.ModuleType("_test_analysis_calls")
    mod.record = []

    def spectrum(n=4, scale=1.0):
        mod.record.append("spectrum")
        return {"power": xr.DataArray(np.arange(n) * scale, dims=["mode"], coords={"mode": np.arange(1, n + 1)})}

    def total(power=None):
        mod.record.append("total")
        return {"sum": float(np.asarray(power).sum())}

    def bare(n=3):
        return np.ones(n)

    def square(n=2):
        return np.eye(n)

    class Scaled:
        def __init__(self, factor=1.0):
            self.factor = factor

        def __call__(self, power=None):
            return {"scaled": np.asarray(power) * self.factor}

    mod.spectrum, mod.total, mod.bare, mod.square, mod.Scaled = spectrum, total, bare, square, Scaled
    monkeypatch.setitem(sys.modules, "_test_analysis_calls", mod)
    return mod


def _analysis(name, fn, args=None, **kw):
    return Analysis(name=name, callable={"name": fn, "module": "_test_analysis_calls"}, arguments=args or {}, **kw)


# ------------------------------------------------------------------ persistence


def test_a_declared_analysis_writes_a_container_a_figure_can_bind(tmp_path, calls):
    """The result lands where every consumer looks, keyed so ``output:`` addresses it."""
    path = analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 5}}), tmp_path)

    assert path == tmp_path / "ana-spec_result.h5"
    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        assert "observation__power" in ds.data_vars
        # `output: power` must address it through the shared suffix rule.
        assert dataref.match_output(ds.data_vars, "power") == "observation__power"
    finally:
        ds.close()


def test_labelled_dims_and_coordinates_survive_the_round_trip(tmp_path, calls):
    """A grammar panel names dims in its ``encoding``, so the writer must not flatten them."""
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)

    ds = xr.open_dataset(tmp_path / "ana-spec_result.h5", engine="h5netcdf")
    try:
        da = ds["observation__power"]
        assert da.dims == ("mode",)
        assert list(da.coords["mode"].values) == [1, 2, 3, 4]
    finally:
        ds.close()


def test_a_bare_array_return_is_keyed_by_the_analysis_name(tmp_path, calls):
    analysis_io.run_analysis(_analysis("ones", "bare", {"n": {"value": 3}}), tmp_path)

    ds = xr.open_dataset(tmp_path / "ana-ones_result.h5", engine="h5netcdf")
    try:
        assert list(ds.data_vars) == ["observation__ones"]
    finally:
        ds.close()


def test_a_bare_array_takes_the_dims_its_analysis_declares(tmp_path, calls):
    """The in-process renderer binds a declared ``dims:`` as the tvboptim one does, rather than naming the axes ``<key>_d<i>``."""
    analysis_io.run_analysis(_analysis("eye", "square", {"n": {"value": 2}}, dims=["node", "node_j"]), tmp_path)

    ds = xr.open_dataset(tmp_path / "ana-eye_result.h5", engine="h5netcdf")
    try:
        assert ds["observation__eye"].dims == ("node", "node_j")
    finally:
        ds.close()


def test_a_bare_array_that_contradicts_its_declared_dims_raises(tmp_path, calls):
    with pytest.raises(ValueError, match=r"declares dims \['node'\] but its output 'eye' has rank 2"):
        analysis_io.run_analysis(_analysis("eye", "square", {"n": {"value": 2}}, dims=["node"]), tmp_path)


def test_a_transient_analysis_feeds_its_reader_and_leaves_nothing_under_the_study(tmp_path, calls):
    """``persist: false``: the reader resolves the table from outside the study, and the study holds only the reduction."""
    table = _analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)
    reduction = _analysis("total", "total", {"power": {"used": {"analysis": "table", "output": "power"}}})

    written = analysis_io.run_analyses([table, reduction], tmp_path)

    assert not (tmp_path / "ana-table_result.h5").exists()
    assert not (tmp_path / "ana-table_result.yaml").exists()
    assert not written[0].is_relative_to(tmp_path)
    ds = xr.open_dataset(tmp_path / "ana-total_result.h5", engine="h5netcdf")
    try:
        assert float(ds["observation__sum"]) == 10.0
    finally:
        ds.close()
    assert calls.record == ["spectrum", "total"]


def test_a_transient_analysis_is_neither_missing_nor_stale(tmp_path, calls):
    """The manifest's staleness pass skips it: there is no container to be missing, and its readers re-run it."""
    from tvbo.data.study_manifest import _stale_or_missing_analyses

    table = _analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)
    reduction = _analysis("total", "total", {"power": {"used": {"analysis": "table", "output": "power"}}})
    analysis_io.run_analyses([table, reduction], tmp_path)

    inv = types.SimpleNamespace(analyses=[table, reduction], datasets=None)
    assert _stale_or_missing_analyses(inv, tmp_path, None) == []


def test_a_transient_analysis_is_scoped_to_its_study_and_refused_to_a_figure(tmp_path, calls):
    """The registry is keyed by study: another study's lookup of the same name does not see it, and a container reader (a figure, a report, a kit) is refused it by name while the in-process ``used:`` resolver reads it."""
    study_a, study_b = tmp_path / "a", tmp_path / "b"
    table = _analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)
    analysis_io.run_analyses([table], study_a)

    with pytest.raises(FileNotFoundError, match="persist: false"):
        dataref.locate_analysis_container(study_a, "table")
    assert dataref.locate_analysis_container(study_a, "table", transient=True).is_file()
    with pytest.raises(FileNotFoundError, match="tvbo run"):
        dataref.locate_analysis_container(study_b, "table")


def test_a_recomputed_transient_analysis_marks_no_other_consumer_stale(tmp_path, calls, caplog):
    """Re-deriving one consumer re-runs the transient table it reads; the table's other consumers keep valid numbers and are not reported stale."""
    import logging

    from tvbo.run.study import warn_stale_analyses

    table = _analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)
    first = _analysis("first", "total", {"power": {"used": {"analysis": "table", "output": "power"}}})
    second = _analysis("second", "total", {"power": {"used": {"analysis": "table", "output": "power"}}})
    analysis_io.run_analyses([table, first, second], tmp_path)

    with caplog.at_level(logging.WARNING):
        warn_stale_analyses([table, first, second], str(tmp_path / "study.yaml"), tmp_path, recomputed={"table", "first"})
    assert "stale" not in caplog.text and "second" not in caplog.text

    persisted = _analysis("table", "spectrum", {"n": {"value": 5}})
    analysis_io.run_analyses([persisted], tmp_path)
    with caplog.at_level(logging.WARNING):
        warn_stale_analyses([persisted, first, second], str(tmp_path / "study.yaml"), tmp_path, recomputed={"table", "first"})
    assert "second" in caplog.text


def test_two_studies_keep_their_own_transient_container_of_one_name(tmp_path, calls):
    """Two studies in one process, each with a transient analysis named alike, write two files, and each study reads its own."""
    study_a, study_b = tmp_path / "a", tmp_path / "b"
    analysis_io.run_analyses([_analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)], study_a)
    analysis_io.run_analyses([_analysis("table", "spectrum", {"n": {"value": 3}}, persist=False)], study_b)

    path_a = dataref.locate_analysis_container(study_a, "table", transient=True)
    path_b = dataref.locate_analysis_container(study_b, "table", transient=True)
    assert path_a != path_b
    for path, n in ((path_a, 5), (path_b, 3)):
        ds = xr.open_dataset(path, engine="h5netcdf")
        try:
            assert ds["observation__power"].size == n
        finally:
            ds.close()


def test_a_persisted_write_replaces_an_earlier_transient_one(tmp_path, calls):
    """An analysis written transient and then persisted under one study is read from the study afterwards, by figures as by in-run readers."""
    analysis_io.run_analyses([_analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)], tmp_path)
    analysis_io.run_analyses([_analysis("table", "spectrum", {"n": {"value": 4}})], tmp_path)

    kept = tmp_path / "ana-table_result.h5"
    assert dataref.locate_analysis_container(tmp_path, "table") == kept
    assert dataref.locate_analysis_container(tmp_path, "table", transient=True) == kept


def test_a_report_number_is_refused_a_transient_analysis(tmp_path, calls):
    """A report binds kept results: a reference to a transient analysis is refused by name rather than read from a file deleted at exit."""
    analysis_io.run_analyses([_analysis("table", "spectrum", {"n": {"value": 5}}, persist=False)], tmp_path)
    ref = types.SimpleNamespace(analysis="table", output="power", experiment=None, iri=None, sel=None, reconcile=None)

    assert dataref.resolve_dataref(ref, results_root=str(tmp_path)).size == 5
    with pytest.raises(FileNotFoundError, match="persist: false"):
        dataref.resolve_dataref(ref, results_root=str(tmp_path), transient=False)


def test_a_persisted_analysis_is_the_default(tmp_path, calls):
    assert analysis_io.is_persisted(_analysis("spec", "spectrum", {"n": {"value": 4}}))
    assert analysis_io.is_persisted({"name": "spec", "callable": {"name": "spectrum", "module": "_test_analysis_calls"}})
    assert not analysis_io.is_persisted({"name": "spec", "persist": False})


def test_the_sidecar_records_what_was_invoked_and_what_it_used(tmp_path, calls):
    """Provenance is the point of declaring the analysis, so it must be on disk."""
    import yaml

    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 5}}), tmp_path)
    record = yaml.safe_load((tmp_path / "ana-spec_result.yaml").read_text())

    assert record["callable"] == {"module": "_test_analysis_calls", "name": "spectrum"}
    assert record["arguments"]["n"] == {"value": 5}
    assert record["outputs"] == ["power"]


# ------------------------------------------------------------------ referencing


def test_an_analysis_reads_another_analysis_by_name(tmp_path, calls):
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)
    analysis_io.run_analysis(_analysis("agg", "total", {"power": {"used": {"analysis": "spec", "output": "power"}}}), tmp_path)

    ds = xr.open_dataset(tmp_path / "ana-agg_result.h5", engine="h5netcdf")
    try:
        assert float(ds["observation__sum"]) == pytest.approx(0 + 1 + 2 + 3)
    finally:
        ds.close()


def test_a_sourced_argument_arrives_labelled_not_bare(tmp_path, calls):
    """Sourced data stays an xarray, so the callable selects by name, never by position."""
    seen = {}
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)
    calls.total = lambda power=None: seen.setdefault("power", power) and {"sum": 0.0}
    kwargs = analysis_io._kwargs_of(
        _analysis("agg", "total", {"power": {"used": {"analysis": "spec", "output": "power"}}}), tmp_path
    )

    assert isinstance(kwargs["power"], xr.DataArray)
    assert kwargs["power"].dims == ("mode",)


def test_the_container_name_is_a_legal_bids_filename(tmp_path, calls):
    """A name with an underscore in it is not a BIDS entity value, so the writer must not emit one.

    The writer and ``tvbo validate study`` read the same rule from :func:`tvbo.adapters.bids.entity_value`; when this path spelled the name itself, an analysis called ``calcium_c10`` wrote a container the validator rejected.
    """
    from tvbo.cli.validate import _entity_problems

    path = analysis_io.run_analysis(_analysis("calcium_c10", "spectrum", {"n": {"value": 3}}), tmp_path)

    assert path == tmp_path / "ana-calciumc10_result.h5"
    assert _entity_problems(path.name) == []


def test_the_reader_finds_what_the_writer_named(tmp_path, calls):
    """The declared name is what a figure or a report asks for; sanitizing is the path's business."""
    analysis_io.run_analysis(_analysis("calcium_c10", "spectrum", {"n": {"value": 3}}), tmp_path)

    assert dataref.locate_analysis_container(tmp_path, "calcium_c10").is_file()


def test_two_names_that_collapse_to_one_container_are_refused():
    """Otherwise the second analysis silently replaces the first."""
    with pytest.raises(ValueError, match="same container"):
        analysis_io.run_analyses([_analysis("calcium_c10", "spectrum"), _analysis("calciumc10", "bare")])


def test_locate_analysis_container_names_the_fix_when_it_has_not_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="tvbo run"):
        dataref.locate_analysis_container(tmp_path, "never_ran")


def test_a_figure_layer_binds_an_analysis_by_name(tmp_path, calls):
    """``used: {analysis: …}`` resolves to the same container the run wrote."""
    from tvbo.adapters import bsplot
    from tvbo.datamodel.schema import DataRef

    study_dir = tmp_path / "study"
    results = study_path("results", root=study_dir)
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), results)

    bsplot._container_path.cache_clear()
    key = bsplot._used_ref(DataRef(analysis="spec", output="power"))
    assert bsplot._container_path(key, study_dir) == str((results / "ana-spec_result.h5").resolve())


# ----------------------------------------------------------- reconcile by label


def _per_node(labels):
    """``n<k> -> k`` on the node order *labels*, or ``0, 1, 2`` on an unlabelled node axis for ``None``."""
    values = [float(label[1:]) for label in labels] if labels else [0.0, 1.0, 2.0]
    return xr.DataArray(values, dims=["node"], coords={"node": labels} if labels else {})


def _fc(labels, *, both_axes=True):
    """``(n<i>, n<j>) -> 10 i + j`` on the node order *labels*, its ``node_j`` axis unlabelled unless *both_axes*."""
    k = np.array([float(label[1:]) for label in labels])
    coords = {"node_i": labels, **({"node_j": labels} if both_axes else {})}
    return xr.DataArray(10 * k[:, None] + k[None, :], dims=["node_i", "node_j"], coords=coords)


_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _stored(tmp_path, calls, **arrays):
    """Run one analysis per keyword, each writing the array it names as output ``v``."""
    for name, da in arrays.items():
        setattr(calls, name, lambda da=da: {"v": da})
        analysis_io.run_analysis(_analysis(name, name), tmp_path)


def _by_label(name):
    return {"used": {"analysis": name, "output": "v", "reconcile": "by_label"}}


def _both_by_label():
    return {"sim": _by_label("sim"), "emp": _by_label("emp")}


def _pair(tmp_path, calls, arguments, returns=None, **kw):
    """Run analysis ``pair`` on *arguments* and return what its callable was handed; the callable returns ``returns(**arguments)``, else a scalar, into ``ana-pair_result.h5``."""
    seen = {}

    def pair(**given):
        seen.update(given)
        return returns(**given) if returns else {"r": 0.0}

    calls.pair = pair
    analysis_io.run_analysis(_analysis("pair", "pair", arguments, **kw), tmp_path)
    return seen


@pytest.mark.parametrize("declared", [("sim", "emp"), ("emp", "sim")], ids=["sim-first", "emp-first"])
def test_by_label_arguments_are_laid_out_on_one_node_order(tmp_path, calls, declared):
    """Two per-node arrays stored on different node orders arrive on one order, node for node, whichever argument is declared first; before, they arrived in their sources' orders and a positional callable paired n2 with n0 (G207)."""
    _stored(tmp_path, calls, sim=_per_node(["n2", "n0", "n1"]), emp=_per_node(["n1", "n2", "n0"]))

    seen = _pair(tmp_path, calls, {key: _by_label(key) for key in declared})

    for key in ("sim", "emp"):
        assert list(seen[key].coords["node"].values) == ["n0", "n1", "n2"]
        np.testing.assert_array_equal(seen[key].values, [0.0, 1.0, 2.0])


def test_by_label_arguments_that_share_a_node_order_keep_it(tmp_path, calls):
    _stored(tmp_path, calls, sim=_per_node(["n2", "n0", "n1"]), emp=_per_node(["n2", "n0", "n1"]))

    seen = _pair(tmp_path, calls, _both_by_label())

    assert list(seen["sim"].coords["node"].values) == list(seen["emp"].coords["node"].values) == ["n2", "n0", "n1"]


@pytest.mark.parametrize(
    ("sim", "emp", "arguments", "reason"),
    [
        (
            _per_node(["n2", "n0", "n1"]),
            _per_node(["n1", "n0"]),
            _both_by_label(),
            r"no entry along 'node' is labelled \['n2'\]",
        ),
        (
            _per_node(["n2", "n0", "n1"]),
            _per_node(["n1", "n2", "n0"]),
            {"sim": {"used": {"analysis": "sim", "output": "v"}}, "emp": _by_label("emp")},
            "no other argument of it declares it",
        ),
        (
            _per_node(["n2", "n0", "n1"]),
            _per_node(None),
            _both_by_label(),
            "it carries no axis labelled with names",
        ),
        (
            xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=["band"], coords={"band": _BANDS}),
            xr.DataArray(np.ones((5, 68)), dims=["band", "node"], coords={"band": _BANDS}),
            _both_by_label(),
            "its axis 'node' carries no coordinate",
        ),
        (
            _fc(["n2", "n0", "n1"]),
            _fc(["n1", "n2", "n0"], both_axes=False),
            _both_by_label(),
            "its axis 'node_j' carries no coordinate",
        ),
    ],
    ids=[
        "a-label-the-other-lacks",
        "the-only-by-label-argument",
        "no-labelled-axis",
        "an-unlabelled-node-axis-beside-a-labelled-band",
        "a-half-labelled-matrix",
    ],
)
def test_a_by_label_argument_it_cannot_match_is_refused_by_name(tmp_path, calls, sim, emp, arguments, reason):
    """Refused naming the analysis and the argument, where it was handed over in its source's order or, on an unlabelled axis, paired by position (G207)."""
    _stored(tmp_path, calls, sim=sim, emp=emp)

    with pytest.raises(ValueError, match=f"analysis 'pair', argument 'emp': .*{reason}"):
        _pair(tmp_path, calls, arguments)


@pytest.mark.parametrize(
    "mask",
    [{"used": {"analysis": "mask", "output": "v"}}, {"value": [1.0, 0.0, 1.0]}],
    ids=["a-mask-from-the-same-source", "a-per-node-literal"],
)
def test_an_argument_beside_by_label_arguments_that_move_is_refused_unless_it_declares_by_label(tmp_path, calls, mask):
    """A mask stored on `sim`'s node order stayed in it while the two `by_label` FCs moved to their sorted labels, and the callable paired it with them by position (G207)."""
    _stored(tmp_path, calls, sim=_fc(["n2", "n0", "n1"]), emp=_fc(["n1", "n2", "n0"]), mask=_per_node(["n2", "n0", "n1"]))

    with pytest.raises(ValueError, match=r"analysis 'pair', argument 'mask': .*without declaring `reconcile: by_label`"):
        _pair(tmp_path, calls, {**_both_by_label(), "mask": mask})


def test_a_bare_return_axis_named_like_the_node_axis_takes_the_node_order(tmp_path, calls):
    """A bare per-node array was stored on the sorted node order its `by_label` arguments moved onto, with no labels to say which node each entry is (G207)."""
    _stored(tmp_path, calls, sim=_per_node(["n2", "n0", "n1"]), emp=_per_node(["n1", "n2", "n0"]))

    _pair(tmp_path, calls, _both_by_label(), returns=lambda sim, emp: np.asarray(sim), dims=["node"])

    ds = xr.open_dataset(tmp_path / "ana-pair_result.h5", engine="h5netcdf")
    try:
        assert list(ds["observation__pair"]["node"].values) == ["n0", "n1", "n2"]
        np.testing.assert_array_equal(ds["observation__pair"].values, [0.0, 1.0, 2.0])
    finally:
        ds.close()


@pytest.mark.parametrize(
    ("returns", "dims", "axis"),
    [(lambda sim, emp: np.asarray(sim), ["band"], "band"), (lambda sim, emp: {"v": np.asarray(sim)}, None, "v_d0")],
    ids=["a-declared-axis-under-another-name", "an-undeclared-axis"],
)
def test_a_bare_return_axis_of_the_node_count_under_another_name_is_refused_by_name(tmp_path, calls, returns, dims, axis):
    """Keyed by name, never by length, so an axis that only shares the node count is never labelled with nodes (G207)."""
    _stored(tmp_path, calls, sim=_per_node(["n2", "n0", "n1"]), emp=_per_node(["n1", "n2", "n0"]))

    with pytest.raises(ValueError, match=f"analysis 'pair': its `reconcile: by_label` .* axis '{axis}' of that length"):
        _pair(tmp_path, calls, _both_by_label(), returns=returns, dims=dims)


def test_a_bare_return_axis_named_like_the_node_axis_of_another_length_is_refused_by_name(tmp_path, calls):
    """A `node` axis of two entries beside a node order of three was stored unlabelled, as if it held nodes (G207)."""
    _stored(tmp_path, calls, sim=_per_node(["n2", "n0", "n1"]), emp=_per_node(["n1", "n2", "n0"]))

    with pytest.raises(ValueError, match=r"analysis 'pair', output 'pair': its axis 'node' .* has 2 entries .* has 3 labels"):
        _pair(tmp_path, calls, _both_by_label(), returns=lambda sim, emp: np.ones(2), dims=["node"])


# --------------------------------------------------------------------- schedule


def test_an_analysis_reading_an_experiment_is_deferred_past_the_runs():
    """Ordering follows the ``used:`` edges, including transitively."""
    first = _analysis("basis", "spectrum")
    after = _analysis("fc", "total", {"x": {"used": {"experiment": "2", "output": "bold"}}})
    downstream = _analysis("summary", "total", {"x": {"used": {"analysis": "fc", "output": "sum"}}})

    before, post = analysis_io.schedule([downstream, after, first])

    assert [a.name for a in before] == ["basis"]
    assert [a.name for a in post] == ["fc", "summary"]


def test_analyses_are_ordered_by_their_dependencies_not_declaration():
    consumer = _analysis("b", "total", {"x": {"used": {"analysis": "a", "output": "power"}}})
    producer = _analysis("a", "spectrum")

    before, _ = analysis_io.schedule([consumer, producer])

    assert [a.name for a in before] == ["a", "b"]


def test_duplicate_analysis_names_are_refused():
    with pytest.raises(ValueError, match="overwrite"):
        analysis_io.schedule([_analysis("same", "spectrum"), _analysis("same", "bare")])


def test_an_unknown_analysis_reference_is_refused():
    with pytest.raises(KeyError, match="does not declare"):
        analysis_io.schedule([_analysis("b", "total", {"x": {"used": {"analysis": "ghost", "output": "p"}}})])


def test_a_dependency_cycle_is_refused():
    a = _analysis("a", "total", {"x": {"used": {"analysis": "b", "output": "p"}}})
    b = _analysis("b", "total", {"x": {"used": {"analysis": "a", "output": "p"}}})

    with pytest.raises(ValueError, match="circular"):
        analysis_io.schedule([a, b])


# --------------------------------------------------------------------- renderer


def test_the_declared_backend_selects_the_renderer(tmp_path, calls):
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 3}}, execution={"backend": "inprocess"}), tmp_path)

    assert (tmp_path / "ana-spec_result.h5").is_file()


def test_a_backend_with_no_analysis_renderer_raises_rather_than_substituting(tmp_path, calls):
    """An analysis asking for an engine we cannot render must not be run by another."""
    with pytest.raises(NotImplementedError, match="no analysis renderer"):
        analysis_io.run_analysis(_analysis("spec", "spectrum", execution={"backend": "julia"}), tmp_path)


def test_a_class_call_analysis_is_instantiated_then_invoked(tmp_path, calls):
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)
    scaled = Analysis(
        name="scaled",
        class_call={
            "name": "Scaled",
            "module": "_test_analysis_calls",
            "constructor_args": [{"name": "factor", "value": 10.0}],
        },
        arguments={"power": {"used": {"analysis": "spec", "output": "power"}}},
    )
    analysis_io.run_analysis(scaled, tmp_path)

    ds = xr.open_dataset(tmp_path / "ana-scaled_result.h5", engine="h5netcdf")
    try:
        assert list(ds["observation__scaled"].values) == [0.0, 10.0, 20.0, 30.0]
    finally:
        ds.close()


def test_an_observation_pipeline_form_is_refused_with_the_reason(tmp_path):
    """`equation:`/`function:` describe a step inside a solve, not a call over containers."""
    with pytest.raises(ValueError, match="observation\n? *pipeline|observation pipeline"):
        analysis_io.run_analysis(Analysis(name="e", equation={"rhs": "x + 1"}), tmp_path)


def test_value_and_used_on_one_argument_are_refused(tmp_path, calls):
    bad = _analysis("spec", "spectrum", {"n": {"value": 5, "used": {"analysis": "other", "output": "p"}}})
    with pytest.raises(ValueError, match="mutually exclusive"):
        analysis_io.run_analysis(bad, tmp_path)


# ------------------------------------------------------- the solve form is intact


def test_the_solve_form_of_analysis_still_validates():
    """One class covers both; declaring an in-solve analysis must be unaffected."""
    a = Analysis(type="lyapunov", target="loss", wrt=["G"])

    assert (a.type, a.target, list(a.wrt)) == ("lyapunov", "loss", ["G"])
    assert a.name is None and a.callable is None


# ------------------------------------------------------- invalidation after a partial run


def test_dependents_of_an_experiment_are_found_transitively(calls):
    """A partial run leaves these holding the PREVIOUS run's numbers, and nothing raises.

    The second-order rows are the point: a hand-written invalidation list reliably keeps the landscape built from the reduction built from the run that just changed.
    """
    analyses = [
        _analysis("reduction", "spectrum", {"n": {"used": {"experiment": 1, "output": "w"}}}),
        _analysis("landscape", "total", {"power": {"used": {"analysis": "reduction"}}}),
        _analysis("correlation", "total", {"power": {"used": {"analysis": "landscape"}}}),
        _analysis("unrelated", "spectrum", {"n": {"used": {"experiment": 7, "output": "w"}}}),
    ]

    assert analysis_io.dependents_of(analyses, experiments=["1"]) == ["reduction", "landscape", "correlation"]
    assert analysis_io.dependents_of(analyses, experiments=["7"]) == ["unrelated"]
    assert analysis_io.dependents_of(analyses, experiments=["9"]) == []


def test_dependents_of_an_analysis_are_found_transitively(calls):
    """`--analysis` re-derives one container; everything reading it is now the stale side.

    Same walk as for an experiment, seeded on the derivation instead — which is what makes editing a callable safe to do piecemeal, since no cache invalidates on a code change.
    """
    analyses = [
        _analysis("reduction", "spectrum", {"n": {"used": {"experiment": 1, "output": "w"}}}),
        _analysis("landscape", "total", {"power": {"used": {"analysis": "reduction"}}}),
        _analysis("correlation", "total", {"power": {"used": {"analysis": "landscape"}}}),
        _analysis("unrelated", "spectrum", {"n": {"used": {"experiment": 7, "output": "w"}}}),
    ]

    assert analysis_io.dependents_of(analyses, changed_analyses=["reduction"]) == ["reduction", "landscape", "correlation"]
    assert analysis_io.dependents_of(analyses, changed_analyses=["landscape"]) == ["landscape", "correlation"]
    assert analysis_io.dependents_of(analyses, changed_analyses=["correlation"]) == ["correlation"]
    assert analysis_io.dependents_of(analyses, changed_analyses=["unrelated"]) == ["unrelated"]


def test_naming_an_analysis_pulls_in_only_the_inputs_that_are_missing(calls):
    """`--analysis` re-derives what was named; it is not a back-door whole-study run.

    An input that already has a container is left alone — recomputing it is what the caller chose not to do. One that has never been produced is pulled in, transitively, because it cannot be read.
    """
    from tvbo.data.analysis_io import analysis_closure

    analyses = [
        _analysis("basis", "spectrum", {"n": {"used": {"experiment": 1, "output": "w"}}}),
        _analysis("curves", "total", {"power": {"used": {"analysis": "basis"}}}),
        _analysis("summary", "total", {"power": {"used": {"analysis": "curves"}}}),
    ]

    assert analysis_closure(analyses, ["summary"], exists=lambda n: True) == {"summary"}
    assert analysis_closure(analyses, ["summary"], exists=lambda n: False) == {"summary", "curves", "basis"}
    # `curves` exists, so it is readable and the walk stops there — `basis` is never reached.
    assert analysis_closure(analyses, ["summary"], exists=lambda n: n == "curves") == {"summary"}
    assert analysis_closure(analyses, ["summary"], exists=lambda n: n == "basis") == {"summary", "curves"}


@pytest.mark.parametrize("written", ["exp-1", "exp1", "1", 1])
def test_every_spelling_of_an_experiment_id_is_the_same_edge(calls, written):
    """A recipe may write `exp-3` where the runtime knows the experiment as `3`.

    Matching only the literal string makes the whole staleness warning report nothing, and an empty stale set is indistinguishable from a clean one.
    """
    analyses = [_analysis("reduction", "spectrum", {"n": {"used": {"experiment": written, "output": "w"}}})]

    assert analysis_io.dependents_of(analyses, experiments=["1"]) == ["reduction"]


def test_an_iri_reference_counts_as_a_dependency(calls):
    """A layer may name its source by IRI rather than by id; both are the same edge."""
    analyses = [_analysis("from_iri", "spectrum", {"n": {"used": {"iri": "tvbo:exp/Study/exp-3", "output": "w"}}})]

    assert analysis_io.dependents_of(analyses, experiments=["3"]) == ["from_iri"]


def test_an_experiment_known_only_by_key_still_matches_a_numeric_used_edge(calls):
    """Both sides of the staleness walk must normalise, or neither side ever intersects.

    `dependencies` emits the recipe spelling AND the bare id; the wanted set comes from `experiment_ids`. An experiment carrying only `key: exp-1` yields nothing numeric unless that side normalises too — and an empty stale set reads exactly like a clean one.
    """
    from types import SimpleNamespace

    from tvbo.run.study import experiment_ids

    exp = SimpleNamespace(key="exp-1")
    analyses = [_analysis("reduction", "spectrum", {"n": {"used": {"experiment": "exp-1", "output": "w"}}})]

    wanted = experiment_ids(exp)

    assert "1" in wanted and "exp-1" in wanted
    assert analysis_io.dependents_of(analyses, experiments=wanted) == ["reduction"]
