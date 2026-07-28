"""A study's declarative ``analyses:`` — schedule, execution, persistence, binding.

An analysis is the study-scope form of ``Analysis``: a ``FunctionCall`` whose result the
study keeps, so a figure can bind a quantity no simulation produced. These cover the
contract a recipe author relies on — what runs when, what lands on disk, and what a
``used: {analysis: …}`` resolves to — plus the errors that must be loud rather than
silent (duplicate names, unknown references, cycles, an unrenderable backend).
"""
import sys
import types

import numpy as np
import pytest
import xarray as xr

from tvbo.data import analysis_io, dataref
from tvbo.datamodel.schema import Analysis


@pytest.fixture
def calls(monkeypatch):
    """A throwaway module of analysis callables, importable by bare name."""
    mod = types.ModuleType("_test_analysis_calls")
    mod.record = []

    def spectrum(n=4, scale=1.0):
        mod.record.append("spectrum")
        return {"power": xr.DataArray(np.arange(n) * scale, dims=["mode"],
                                      coords={"mode": np.arange(1, n + 1)})}

    def total(power=None):
        mod.record.append("total")
        return {"sum": float(np.asarray(power).sum())}

    def bare(n=3):
        return np.ones(n)

    class Scaled:
        def __init__(self, factor=1.0):
            self.factor = factor

        def __call__(self, power=None):
            return {"scaled": np.asarray(power) * self.factor}

    mod.spectrum, mod.total, mod.bare, mod.Scaled = spectrum, total, bare, Scaled
    monkeypatch.setitem(sys.modules, "_test_analysis_calls", mod)
    return mod


def _analysis(name, fn, args=None, **kw):
    return Analysis(name=name, callable={"name": fn, "module": "_test_analysis_calls"},
                    arguments=args or {}, **kw)


# ------------------------------------------------------------------ persistence


def test_a_declared_analysis_writes_a_container_a_figure_can_bind(tmp_path, calls):
    """The result lands where every consumer looks, keyed so ``output:`` addresses it."""
    path = analysis_io.run_analysis(
        _analysis("spec", "spectrum", {"n": {"value": 5}}), tmp_path)

    assert path == tmp_path / "results" / "spec" / "result.h5"
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

    ds = xr.open_dataset(tmp_path / "results" / "spec" / "result.h5", engine="h5netcdf")
    try:
        da = ds["observation__power"]
        assert da.dims == ("mode",)
        assert list(da.coords["mode"].values) == [1, 2, 3, 4]
    finally:
        ds.close()


def test_a_bare_array_return_is_keyed_by_the_analysis_name(tmp_path, calls):
    analysis_io.run_analysis(_analysis("ones", "bare", {"n": {"value": 3}}), tmp_path)

    ds = xr.open_dataset(tmp_path / "results" / "ones" / "result.h5", engine="h5netcdf")
    try:
        assert list(ds.data_vars) == ["observation__ones"]
    finally:
        ds.close()


def test_the_sidecar_records_what_was_invoked_and_what_it_used(tmp_path, calls):
    """Provenance is the point of declaring the analysis, so it must be on disk."""
    import yaml

    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 5}}), tmp_path)
    record = yaml.safe_load((tmp_path / "results" / "spec" / "result.yaml").read_text())

    assert record["callable"] == {"module": "_test_analysis_calls", "name": "spectrum"}
    assert record["arguments"]["n"] == {"value": 5}
    assert record["outputs"] == ["power"]


# ------------------------------------------------------------------ referencing


def test_an_analysis_reads_another_analysis_by_name(tmp_path, calls):
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)
    analysis_io.run_analysis(
        _analysis("agg", "total", {"power": {"used": {"analysis": "spec", "output": "power"}}}),
        tmp_path)

    ds = xr.open_dataset(tmp_path / "results" / "agg" / "result.h5", engine="h5netcdf")
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
        _analysis("agg", "total", {"power": {"used": {"analysis": "spec", "output": "power"}}}),
        tmp_path)

    assert isinstance(kwargs["power"], xr.DataArray)
    assert kwargs["power"].dims == ("mode",)


def test_locate_analysis_container_names_the_fix_when_it_has_not_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="tvbo run"):
        dataref.locate_analysis_container(tmp_path, "never_ran")


def test_a_figure_layer_binds_an_analysis_by_name(tmp_path, calls):
    """``used: {analysis: …}`` resolves to the same container the run wrote."""
    from tvbo.adapters import bsplot
    from tvbo.datamodel.schema import DataRef

    study_dir = tmp_path / "study"
    analysis_io.run_analysis(
        _analysis("spec", "spectrum", {"n": {"value": 4}}), study_dir / "output")

    bsplot._container_path.cache_clear()
    key = bsplot._used_ref(DataRef(analysis="spec", output="power"))
    assert bsplot._container_path(key, study_dir) == str(
        (study_dir / "output" / "results" / "spec" / "result.h5").resolve())


# --------------------------------------------------------------------- schedule


def test_an_analysis_reading_an_experiment_is_deferred_past_the_runs():
    """Ordering follows the ``used:`` edges, including transitively."""
    first = _analysis("basis", "spectrum")
    after = _analysis("fc", "total", {"x": {"used": {"experiment": "2", "output": "bold"}}})
    downstream = _analysis("summary", "total",
                           {"x": {"used": {"analysis": "fc", "output": "sum"}}})

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
        analysis_io.schedule(
            [_analysis("b", "total", {"x": {"used": {"analysis": "ghost", "output": "p"}}})])


def test_a_dependency_cycle_is_refused():
    a = _analysis("a", "total", {"x": {"used": {"analysis": "b", "output": "p"}}})
    b = _analysis("b", "total", {"x": {"used": {"analysis": "a", "output": "p"}}})

    with pytest.raises(ValueError, match="circular"):
        analysis_io.schedule([a, b])


# --------------------------------------------------------------------- renderer


def test_the_declared_backend_selects_the_renderer(tmp_path, calls):
    analysis_io.run_analysis(
        _analysis("spec", "spectrum", {"n": {"value": 3}},
                  execution={"backend": "inprocess"}), tmp_path)

    assert (tmp_path / "results" / "spec" / "result.h5").is_file()


def test_a_backend_with_no_analysis_renderer_raises_rather_than_substituting(tmp_path, calls):
    """An analysis asking for an engine we cannot render must not be run by another."""
    with pytest.raises(NotImplementedError, match="no analysis renderer"):
        analysis_io.run_analysis(
            _analysis("spec", "spectrum", execution={"backend": "julia"}), tmp_path)


def test_a_class_call_analysis_is_instantiated_then_invoked(tmp_path, calls):
    analysis_io.run_analysis(_analysis("spec", "spectrum", {"n": {"value": 4}}), tmp_path)
    scaled = Analysis(
        name="scaled",
        class_call={"name": "Scaled", "module": "_test_analysis_calls",
                    "constructor_args": [{"name": "factor", "value": 10.0}]},
        arguments={"power": {"used": {"analysis": "spec", "output": "power"}}},
    )
    analysis_io.run_analysis(scaled, tmp_path)

    ds = xr.open_dataset(tmp_path / "results" / "scaled" / "result.h5", engine="h5netcdf")
    try:
        assert list(ds["observation__scaled"].values) == [0.0, 10.0, 20.0, 30.0]
    finally:
        ds.close()


def test_an_observation_pipeline_form_is_refused_with_the_reason(tmp_path):
    """`equation:`/`function:` describe a step inside a solve, not a call over containers."""
    with pytest.raises(ValueError, match="observation\n? *pipeline|observation pipeline"):
        analysis_io.run_analysis(
            Analysis(name="e", equation={"rhs": "x + 1"}), tmp_path)


def test_value_and_used_on_one_argument_are_refused(tmp_path, calls):
    bad = _analysis("spec", "spectrum",
                    {"n": {"value": 5, "used": {"analysis": "other", "output": "p"}}})
    with pytest.raises(ValueError, match="mutually exclusive"):
        analysis_io.run_analysis(bad, tmp_path)


# ------------------------------------------------------- the solve form is intact


def test_the_solve_form_of_analysis_still_validates():
    """One class covers both; declaring an in-solve analysis must be unaffected."""
    a = Analysis(type="lyapunov", target="loss", wrt=["G"])

    assert (a.type, a.target, list(a.wrt)) == ("lyapunov", "loss", ["G"])
    assert a.name is None and a.callable is None
