"""Tests for the declarative ``equation:`` analysis form and its tvboptim renderer.

Locks in the contract that makes an analysis metadata-native rather than a pointer at
arbitrary ``code/`` Python: the expression lowers to JAX, ``apply_on_dimension`` becomes
a vmap over that axis, ``aggregate`` reduces a named one, and the output's axis names are
DERIVED from that declaration and cross-checked against the result — never read off its
shape. An expression that reshapes must say so with ``dims:``.

The sharding tests read the device count from ``conftest`` rather than requesting their
own. Appending a second ``--xla_force_host_platform_device_count`` here used to override
it, but only when this module was imported before JAX — so the suite saw 4 devices alone
and 8 alongside other tests, and an assertion that held at 4 failed at 8.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tvbo.data.analysis_io import RENDERERS, render_tvboptim


class _Agg:
    def __init__(self, over, type_="mean"):
        self.over, self.type = over, type_


class _Eq:
    def __init__(self, rhs):
        self.rhs = rhs


class _Analysis:
    """Minimal stand-in carrying only the slots the renderer reads."""

    def __init__(self, name="a", rhs=None, apply_on_dimension=None, aggregate=None,
                 dims=None, callable_=None, empty_equation=False, execution=None):
        self.name = name
        self.equation = _Eq(rhs) if rhs is not None or empty_equation else None
        self.apply_on_dimension = apply_on_dimension
        self.aggregate = aggregate
        self.dims = dims
        self.callable = callable_
        self.class_call = None
        self.function = None
        self.execution = execution


def _da(values, dims, coords=None):
    return xr.DataArray(np.asarray(values, dtype=float), dims=dims, coords=coords or {})


def test_registered_under_tvboptim():
    assert RENDERERS["tvboptim"] is render_tvboptim


def test_elementwise_expression_over_two_labelled_inputs():
    a = _da([[1.0, 2.0], [3.0, 4.0]], ["subject", "mode"],
            {"subject": [10, 11], "mode": [1, 2]})
    b = _da([1.0, 1.0], ["mode"], {"mode": [1, 2]})
    out = render_tvboptim(
        _Analysis(rhs="a - b", apply_on_dimension="subject"), {"a": a, "b": b})
    assert out.dims == ("subject", "mode")
    np.testing.assert_allclose(out.values, [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(out.coords["subject"].values, [10, 11])


def test_unmapped_argument_is_broadcast_not_indexed():
    """An argument without the mapped axis must reach every element whole."""
    a = _da([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["subject", "mode"])
    b = _da([1.0, 2.0, 3.0], ["mode"])
    out = render_tvboptim(
        _Analysis(rhs="a * b", apply_on_dimension="subject"), {"a": a, "b": b})
    np.testing.assert_allclose(out.values, [[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])


def test_aggregate_reduces_the_named_axis():
    a = _da([[1.0, 3.0], [5.0, 9.0]], ["subject", "mode"], {"subject": [1, 2]})
    out = render_tvboptim(
        _Analysis(rhs="a", apply_on_dimension="subject", aggregate=_Agg("mode")),
        {"a": a})
    assert out.dims == ("subject",)
    np.testing.assert_allclose(out.values, [2.0, 7.0])


def test_aggregate_sum_and_none():
    a = _da([[1.0, 3.0]], ["subject", "mode"])
    total = render_tvboptim(
        _Analysis(rhs="a", apply_on_dimension="subject", aggregate=_Agg("mode", "sum")),
        {"a": a})
    np.testing.assert_allclose(total.values, [4.0])
    kept = render_tvboptim(
        _Analysis(rhs="a", apply_on_dimension="subject", aggregate=_Agg("mode", "none")),
        {"a": a})
    assert kept.dims == ("subject", "mode")


def test_scalar_argument_is_not_treated_as_an_axis():
    a = _da([[1.0, 2.0]], ["subject", "mode"])
    out = render_tvboptim(
        _Analysis(rhs="a * k", apply_on_dimension="subject"), {"a": a, "k": 3.0})
    assert out.dims == ("subject", "mode")
    np.testing.assert_allclose(out.values, [[3.0, 6.0]])


def test_result_without_mapping_keeps_input_axes():
    a = _da([1.0, 2.0, 3.0], ["mode"], {"mode": [1, 2, 3]})
    out = render_tvboptim(_Analysis(rhs="a ** 2"), {"a": a})
    assert out.dims == ("mode",)
    np.testing.assert_allclose(out.values, [1.0, 4.0, 9.0])


# ------------------------------------------------ arguments are keyed, never positional


def test_two_inputs_carrying_the_same_axes_in_different_orders_agree():
    """The containers are keyed, so (node, time) and (time, node) are the same data."""
    values = np.arange(6.0).reshape(2, 3)
    a = _da(values, ["node", "time"])
    b = _da(values.T, ["time", "node"])
    out = render_tvboptim(_Analysis(rhs="a - b"), {"a": a, "b": b})
    assert out.dims == ("node", "time")
    np.testing.assert_allclose(out.values, np.zeros((2, 3)))


def test_a_square_result_is_not_silently_transposed():
    """Equal axis lengths are exactly when a positional pairing goes unnoticed."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = _da(values, ["node", "time"])
    b = _da(values.T, ["time", "node"])
    out = render_tvboptim(_Analysis(rhs="a * b"), {"a": a, "b": b})
    np.testing.assert_allclose(out.values, values * values)


def test_an_axis_an_argument_lacks_becomes_a_length_one_axis_in_place():
    """Right-alignment would land a per-node vector on the time axis of a (node, time)."""
    a = _da(np.ones((3, 2)), ["node", "time"])
    b = _da([1.0, 2.0, 3.0], ["node"])
    out = render_tvboptim(_Analysis(rhs="a * b"), {"a": a, "b": b})
    assert out.dims == ("node", "time")
    np.testing.assert_allclose(out.values, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


def test_ordering_holds_under_a_mapped_axis_too():
    a = _da(np.arange(12.0).reshape(2, 2, 3), ["subject", "node", "time"])
    b = _da(np.arange(12.0).reshape(2, 2, 3).transpose(0, 2, 1), ["subject", "time", "node"])
    out = render_tvboptim(
        _Analysis(rhs="a - b", apply_on_dimension="subject"), {"a": a, "b": b})
    assert out.dims == ("subject", "node", "time")
    np.testing.assert_allclose(out.values, np.zeros((2, 2, 3)))


# ------------------------------------------------ axes are declared, never inferred


def test_reducing_expression_without_dims_is_refused():
    """A rank the declaration does not imply must fail loudly, not be named from shape."""
    a = _da([[1.0, 2.0], [3.0, 4.0]], ["subject", "mode"])
    with pytest.raises(ValueError, match="declared, never read off a shape"):
        render_tvboptim(
            _Analysis(rhs="sum(a)", apply_on_dimension="subject"), {"a": a})


def test_explicit_dims_are_honoured_for_a_reducing_expression():
    a = _da([[1.0, 2.0], [3.0, 4.0]], ["subject", "mode"], {"subject": [7, 8]})
    out = render_tvboptim(
        _Analysis(rhs="sum(a)", apply_on_dimension="subject", dims=["subject"]),
        {"a": a})
    assert out.dims == ("subject",)
    np.testing.assert_allclose(out.values, [3.0, 7.0])
    np.testing.assert_array_equal(out.coords["subject"].values, [7, 8])


def test_mapped_axis_absent_from_every_input_is_refused():
    a = _da([1.0, 2.0], ["mode"])
    with pytest.raises(ValueError, match="names an axis no argument carries"):
        render_tvboptim(_Analysis(rhs="a", apply_on_dimension="subject"), {"a": a})


def test_aggregate_over_an_axis_the_expression_lacks_is_refused():
    a = _da([[1.0, 2.0]], ["subject", "mode"])
    with pytest.raises(ValueError, match="does not produce"):
        render_tvboptim(
            _Analysis(rhs="a", apply_on_dimension="subject", aggregate=_Agg("time")),
            {"a": a})


def test_callable_analysis_is_refused_rather_than_traced():
    with pytest.raises(ValueError, match="will not trace"):
        render_tvboptim(_Analysis(callable_=object()), {})


def test_equation_without_rhs_is_refused():
    with pytest.raises(ValueError, match="needs an `rhs`"):
        render_tvboptim(_Analysis(empty_equation=True), {})


# ------------------------------------------------------------ vmap is real parallelism


def test_mapped_and_serial_results_agree():
    """vmap over the declared axis must not change the answer."""
    rng = np.random.default_rng(0)
    a = _da(rng.normal(size=(6, 4)), ["subject", "mode"])
    b = _da(rng.normal(size=(4,)), ["mode"])
    mapped = render_tvboptim(
        _Analysis(rhs="(a - b) ** 2", apply_on_dimension="subject",
                  aggregate=_Agg("mode")), {"a": a, "b": b})
    serial = ((a.values - b.values) ** 2).mean(axis=1)
    np.testing.assert_allclose(mapped.values, serial, rtol=1e-6)


# ---------------------------------------------------- the declared dispatch, end to end


class _Execution:
    def __init__(self, backend=None, n_workers=1, batch_size=None, accelerator="auto"):
        self.backend = backend
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.accelerator = accelerator


class _Arg:
    def __init__(self, name, value=None, used=None):
        self.name, self.value, self.used = name, value, used


class _Used:
    """A `used:` DataRef naming another analysis's container, as a recipe writes it."""

    def __init__(self, analysis, output):
        self.analysis, self.output = analysis, output
        self.experiment = self.iri = self.sel = self.reconcile = self.transform = None


def _seed_container(tmp_path, name, array):
    """Persist *array* as an analysis container so a `used:` ref can resolve it."""
    import xarray as xr

    from tvbo.data.analysis_io import container_path

    path = container_path(name, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({f"observation__{name}": array}).to_netcdf(path, engine="h5netcdf")
    return path


def test_execution_backend_routes_through_the_registry_and_writes_a_container(tmp_path):
    """`execution.backend: tvboptim` must reach this renderer and persist labelled axes."""
    import xarray as xr

    from tvbo.data.analysis_io import container_path, run_analysis

    _seed_container(tmp_path, "raw", _da([[1.0, 2.0], [3.0, 4.0]], ["subject", "mode"]))

    analysis = _Analysis(name="scored", rhs="a * 2", apply_on_dimension="subject")
    analysis.execution = _Execution("tvboptim")
    analysis.arguments = [_Arg("a", used=_Used("raw", "raw"))]

    path = run_analysis(analysis, tmp_path)
    assert path == container_path("scored", tmp_path)

    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        got = ds["observation__scored"]
        assert got.dims == ("subject", "mode")
        np.testing.assert_allclose(got.values, [[2.0, 4.0], [6.0, 8.0]])
    finally:
        ds.close()


def test_unknown_backend_still_raises_rather_than_falling_back(tmp_path):
    from tvbo.data.analysis_io import run_analysis

    _seed_container(tmp_path, "raw", _da([1.0], ["mode"]))
    analysis = _Analysis(name="x", rhs="a")
    analysis.execution = _Execution("definitely-not-a-backend")
    analysis.arguments = [_Arg("a", used=_Used("raw", "raw"))]
    with pytest.raises(NotImplementedError, match="no analysis renderer"):
        run_analysis(analysis, tmp_path)


# ------------------------------------------- sharding the declared axis across devices


def _device_count():
    import jax

    return jax.local_device_count()


COHORT_SIZE = 7
"""Subjects in :func:`_cohort` — deliberately not a multiple of any device count."""


def _shard_count(workers):
    """Shards the renderer will use: ``min(workers, devices, items)``.

    The clamp is three-way, so the shard count is NOT the device count whenever the
    cohort is smaller than the machine — 8 devices and 7 subjects shard 7 ways. Asserting
    against the device count instead passed only on hosts with at most 7 usable devices.
    """
    return min(workers, _device_count(), COHORT_SIZE)


def _cohort(n=COHORT_SIZE, m=3, seed=0):
    """A ragged cohort — 7 subjects, never evenly divisible — so padding and trimming
    are always exercised."""
    rng = np.random.default_rng(seed)
    a = _da(rng.normal(size=(n, m)), ["subject", "mode"], {"subject": np.arange(n)})
    b = _da(rng.normal(size=(m,)), ["mode"])
    return {"a": a, "b": b}


def _rendered(execution=None, **extra):
    return render_tvboptim(
        _Analysis(rhs="(a - b) ** 2", apply_on_dimension="subject",
                  execution=execution, **extra),
        _cohort(),
    )


@pytest.mark.parametrize("execution", [
    pytest.param(_Execution(batch_size=3), id="chunked-one-device"),
    pytest.param(_Execution(n_workers=4), id="sharded-auto-width"),
    pytest.param(_Execution(n_workers=4, batch_size=2), id="sharded-explicit-width"),
    pytest.param(_Execution(n_workers=99), id="more-shards-than-devices"),
])
def test_sharded_result_is_identical_to_the_single_device_map(execution):
    """Spreading the axis must change only where the work runs, never the numbers."""
    if execution.n_workers > 1 and _device_count() < 2:
        pytest.skip("multi-device sharding needs >1 local device")
    np.testing.assert_array_equal(_rendered(execution).values, _rendered().values)


def test_sharded_output_keeps_declared_axes_and_coords():
    """The shard reshape is undone before the result is labelled, padding trimmed."""
    if _device_count() < 2:
        pytest.skip("multi-device sharding needs >1 local device")
    out = _rendered(_Execution(n_workers=4))
    assert out.dims == ("subject", "mode")
    assert out.shape == (7, 3)
    np.testing.assert_array_equal(out.coords["subject"].values, np.arange(7))


def test_aggregate_reduces_after_the_shards_are_trimmed():
    """A reduction must not see the pad lanes: it runs on the reassembled result."""
    if _device_count() < 2:
        pytest.skip("multi-device sharding needs >1 local device")
    kwargs = _cohort()
    sharded = render_tvboptim(
        _Analysis(rhs="(a - b) ** 2", apply_on_dimension="subject",
                  aggregate=_Agg("mode"), execution=_Execution(n_workers=4)), kwargs)
    serial = ((kwargs["a"].values - kwargs["b"].values) ** 2).mean(axis=1)
    assert sharded.dims == ("subject",)
    np.testing.assert_allclose(sharded.values, serial, rtol=1e-6)


def test_more_shards_than_devices_is_logged_not_silently_clamped(caplog):
    """An unmeetable n_workers must be reported with the width actually used."""
    if _device_count() < 2:
        pytest.skip("multi-device sharding needs >1 local device")
    with caplog.at_level("WARNING", logger="tvbo.run"):
        _rendered(_Execution(n_workers=99))
    assert "n_workers=99" in caplog.text
    assert f"axis {_shard_count(99)} ways" in caplog.text


def test_workers_without_a_mapped_axis_warns_and_still_computes(caplog):
    a = _da([1.0, 2.0, 3.0], ["mode"])
    with caplog.at_level("WARNING", logger="tvbo.run"):
        out = render_tvboptim(
            _Analysis(rhs="a ** 2", execution=_Execution(n_workers=4)), {"a": a})
    np.testing.assert_allclose(out.values, [1.0, 4.0, 9.0])
    assert "no `apply_on_dimension:`" in caplog.text


def test_parallelism_is_declared_never_inferred(monkeypatch):
    """With no `execution:`, the renderer must not reach for the device machinery."""
    import tvboptim.execution as tvboptim_execution

    def _fail(*args, **kwargs):
        raise AssertionError("undeclared analysis must not use ParallelExecution")

    monkeypatch.setattr(tvboptim_execution, "ParallelExecution", _fail)
    assert _rendered().shape == (7, 3)
    assert _rendered(_Execution(backend="tvboptim")).shape == (7, 3)


def test_device_plan_honours_declared_width_and_clamps_to_the_shard():
    from tvbo.data.analysis_io import _device_plan

    analysis = _Analysis(apply_on_dimension="subject", execution=_Execution(batch_size=3))
    assert _device_plan(analysis, 12, None) == (1, 3)

    wide = _Analysis(apply_on_dimension="subject", execution=_Execution(batch_size=999))
    assert _device_plan(wide, 5, None) == (1, 5)


def test_device_plan_shards_no_finer_than_the_item_count():
    from tvbo.data.analysis_io import _device_plan

    analysis = _Analysis(apply_on_dimension="subject", execution=_Execution(n_workers=4))
    n_pmap, _ = _device_plan(analysis, 2, None)
    assert n_pmap == min(2, _device_count())


def test_auto_width_is_bounded_by_the_per_lane_memory_budget(monkeypatch):
    """`batch_size` unset resolves through the same budget an exploration's auto uses."""
    from tvbo.data import analysis_io
    from tvbo.templates.tvboptim import callbacks

    monkeypatch.setattr(callbacks, "auto_nvmap_budget_bytes", lambda: 1024)
    analysis = _Analysis(apply_on_dimension="subject", execution=_Execution(n_workers=1))
    _, n_vmap = analysis_io._device_plan(analysis, 64, 512)
    assert n_vmap == max(1, 1024 // (512 * callbacks.shared_ram_device_count()))


# ------------------------------------------------------- accelerator is declared too


@pytest.mark.parametrize("accelerator,expected", [
    ("auto", None), ("cpu", "cpu"), ("gpu", "cuda"), ("tpu", "tpu"),
    ("GPU", "cuda"), (None, None), ("", None),
])
def test_accelerator_maps_to_the_jax_platform_name(accelerator, expected):
    from tvbo.templates.tvboptim.utils import jax_platform

    assert jax_platform(accelerator) == expected


def test_accelerator_that_cannot_be_applied_is_reported(caplog):
    """JAX is already up in this process, so a foreign platform must warn, not pass."""
    import jax  # noqa: F401  (ensures the already-initialised branch)

    with caplog.at_level("WARNING", logger="tvbo.run"):
        render_tvboptim(
            _Analysis(rhs="a * 2", execution=_Execution(accelerator="tpu")),
            {"a": _da([1.0], ["mode"])})
    assert "cannot be applied" in caplog.text and "JAX_PLATFORMS=tpu" in caplog.text
