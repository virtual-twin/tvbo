"""Unit tests for the shared cross-experiment DataRef resolver (``tvbo/data/dataref.py``).

Light — synthetic xarray datasets written to a temp HDF5, no JAX, no big grids. Covers
every resolution path the design enumerates: intra-study ``experiment`` id, ``iri``
(trailing number and filesystem path), the ``source_experiment`` fallback, the local
no-WHERE guard, ``sel`` nearest on an indexed *and* a non-index coordinate, the
``output`` ``__``-suffix matcher, and ``by_label`` reconcile (identity + permuted).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tvbo.data import dataref as dr


def _ref(**kw):
    """A minimal DataRef stand-in (SimpleNamespace) matching the resolver's duck-typing."""
    kw.setdefault("experiment", None)
    kw.setdefault("iri", None)
    kw.setdefault("output", None)
    kw.setdefault("sel", None)
    kw.setdefault("reconcile", None)
    return SimpleNamespace(**kw)


def _sel(**pairs):
    """``DataRef.sel`` as a list of Argument stand-ins (name = dim, value = coord)."""
    return [SimpleNamespace(name=k, value=v) for k, v in pairs.items()]


@pytest.fixture
def sweep_h5(tmp_path):
    """A swept container: dims (branch_point, node); K a non-index coord along branch_point."""
    K = np.array([700.0, 900.0, 1100.0, 1300.0, 1500.0])
    xi = np.arange(5)[:, None] + np.array([0.0, 0.1, 0.2])[None, :]   # row r -> r + {0,.1,.2}
    ds = xr.Dataset(
        {"observation__lyapunov_xi": (("branch_point", "node"), xi)},
        coords={"KuramotoInertia.K": ("branch_point", K),
                "node": ["A", "B", "C"]},
    )
    p = tmp_path / "sub" / "study_exp-32_result.h5"
    p.parent.mkdir(parents=True)
    ds.to_netcdf(p, engine="h5netcdf")
    return tmp_path, p


@pytest.fixture
def vec_h5(tmp_path):
    """A per-node vector with string node labels, for reconcile tests."""
    ds = xr.Dataset({"g": (("node",), np.array([10.0, 20.0, 30.0]))},
                    coords={"node": ["A", "B", "C"]})
    p = tmp_path / "out" / "nc" / "exp5" / "study_exp-5_result.h5"
    p.parent.mkdir(parents=True)
    ds.to_netcdf(p, engine="h5netcdf")
    return tmp_path, p


# --------------------------------------------------------------------------- WHERE

def test_experiment_id_from_iri():
    assert dr.experiment_id("tvbo:exp/Study/exp-32") == "32"
    assert dr.experiment_id("exp32") == "32"
    assert dr.experiment_id("32") == "32"
    assert dr.experiment_id("curated-result") is None
    assert dr.experiment_id(None) is None
    # A curated/dataset iri whose last segment merely contains digits is NOT an experiment.
    assert dr.experiment_id("tvbo:net/rec-avgMatrix_atlas-HCPMMP1") is None
    assert dr.experiment_id("rec-avgMatrix_atlas-HCPMMP1") is None
    assert dr.experiment_id("exp-32_desc-Foo") is None


def test_locate_container_by_experiment(sweep_h5):
    root, path = sweep_h5
    got = dr.locate_container(_ref(experiment="32"), results_root=root)
    assert got == path


def test_locate_container_by_experiment_exp_dash_spelling(sweep_h5):
    """The ``exp-N`` spelling (as written in a recipe's ``used: {experiment: exp-32}``)
    resolves at resolve time, matching the planner — it must not raise on ``int('exp-32')``.
    Covers both the direct ``experiment`` path and the from_experiment fallback path."""
    root, path = sweep_h5
    assert dr.locate_container(_ref(experiment="exp-32"), results_root=root) == path
    assert dr.locate_container(_ref(), results_root=root, fallback_experiment="exp-32") == path


def test_locate_container_by_iri_number(sweep_h5):
    root, path = sweep_h5
    got = dr.locate_container(_ref(iri="tvbo:exp/Study/exp-32"), results_root=root)
    assert got == path


def test_locate_container_by_iri_path(sweep_h5):
    _, path = sweep_h5
    got = dr.locate_container(_ref(iri=str(path)), results_root=None)
    assert got == path


def test_locate_container_fallback(sweep_h5):
    root, path = sweep_h5
    got = dr.locate_container(_ref(), results_root=root, fallback_experiment=32)
    assert got == path


def test_locate_container_local_raises():
    with pytest.raises(ValueError):
        dr.locate_container(_ref(), results_root=None)


def test_locate_container_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dr.locate_container(_ref(experiment="99"), results_root=tmp_path)


def test_is_local_ref():
    assert dr.is_local_ref(_ref(output="observations.solitary"))
    assert not dr.is_local_ref(_ref(experiment="32"))
    assert not dr.is_local_ref(_ref(iri="tvbo:exp/S/exp-1"))


def test_skip_network_sidecar(tmp_path):
    (tmp_path / "study_exp-7_network.h5").write_bytes(b"")
    xr.Dataset({"g": (("node",), [1.0])}, coords={"node": ["A"]}).to_netcdf(
        tmp_path / "study_exp-7_result.h5", engine="h5netcdf")
    got = dr.locate_container(_ref(experiment="7"), results_root=tmp_path)
    assert got.name == "study_exp-7_result.h5"


# --------------------------------------------------------------------------- WHICH

def test_match_output_exact():
    assert dr.match_output(["theta_final", "g"], "g") == "g"


def test_match_output_suffix():
    assert dr.match_output(["observation__lyapunov_xi", "theta"], "lyapunov_xi") \
        == "observation__lyapunov_xi"
    assert dr.match_output(["estimate__wLRE"], "wLRE") == "estimate__wLRE"


def test_match_output_missing():
    with pytest.raises(KeyError):
        dr.match_output(["a", "b"], "nope")


# --------------------------------------------------------------------------- SLICE

def test_select_nearest_on_non_index_coord(sweep_h5):
    _, path = sweep_h5
    da = xr.open_dataset(path, engine="h5netcdf")["observation__lyapunov_xi"]
    out = dr.select_labeled(da, {"KuramotoInertia.K": 1307})     # nearest -> K=1300 (row 3)
    np.testing.assert_allclose(out.values, [3.0, 3.1, 3.2])


def test_select_nearest_on_index_dim():
    da = xr.DataArray([10.0, 20.0, 30.0], dims=["k"], coords={"k": [1.0, 2.0, 3.0]})
    assert float(dr.select_labeled(da, {"k": 2.2})) == 20.0


def test_select_exact_label():
    da = xr.DataArray([10.0, 20.0], dims=["node"], coords={"node": ["A", "B"]})
    assert float(dr.select_labeled(da, {"node": "B"})) == 20.0


def test_select_numeric_list_on_non_index_coord():
    # A numeric list on a non-dimension coordinate uses nearest per element (not exact isin,
    # which would silently miss on a continuous sweep).
    da = xr.DataArray(np.arange(5.0), dims=["point"],
                      coords={"K": ("point", [700.0, 900.0, 1100.0, 1300.0, 1500.0])})
    out = dr.select_labeled(da, {"K": [817, 1307]})     # nearest -> 900 (idx1), 1300 (idx3)
    np.testing.assert_allclose(out.values, [1.0, 3.0])


def test_select_empty_is_identity():
    da = xr.DataArray([1.0, 2.0], dims=["x"])
    assert dr.select_labeled(da, None) is da


def test_select_unknown_key_raises():
    da = xr.DataArray([1.0], dims=["x"], coords={"x": [0.0]})
    with pytest.raises(KeyError):
        dr.select_labeled(da, {"y": 1.0})


# --------------------------------------------------------------------------- RECONCILE

def test_reconcile_identity():
    da = xr.DataArray([10.0, 20.0, 30.0], dims=["node"], coords={"node": ["A", "B", "C"]})
    out = dr.reconcile_by_label(da, {"A": "A", "B": "B", "C": "C"}, ["A", "B", "C"])
    np.testing.assert_allclose(out.values, [10.0, 20.0, 30.0])


def test_reconcile_permuted_and_aliased():
    da = xr.DataArray([10.0, 20.0, 30.0], dims=["node"], coords={"node": ["A", "B", "C"]})
    amap = {"A": "A", "B": "B", "C": "C", "ALIAS_C": "C"}
    out = dr.reconcile_by_label(da, amap, ["C", "A", "B"])
    np.testing.assert_allclose(out.values, [30.0, 10.0, 20.0])


def test_reconcile_matrix_both_axes():
    m = np.arange(9.0).reshape(3, 3)
    da = xr.DataArray(m, dims=["node_i", "node_j"],
                      coords={"node_i": ["A", "B", "C"], "node_j": ["A", "B", "C"]})
    out = dr.reconcile_by_label(da, {k: k for k in "ABC"}, ["C", "B", "A"])
    # rows and cols both reversed -> element [i,j] becomes original [2-i, 2-j]
    np.testing.assert_allclose(out.values, m[::-1, ::-1])


def test_reconcile_leaves_unlabelled_axis():
    da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])       # no coords -> untouched
    out = dr.reconcile_by_label(da, {}, ["Z"])
    np.testing.assert_allclose(out.values, [1.0, 2.0, 3.0])


def test_reconcile_skips_non_node_string_axis():
    # A labelled NON-node axis ('pop') must be left alone; only the node axis is reordered.
    da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=["pop", "node"],
                      coords={"pop": ["E", "I"], "node": ["A", "B", "C"]})
    out = dr.reconcile_by_label(da, {k: k for k in "ABC"}, ["C", "A", "B"])
    assert list(out.coords["pop"].values) == ["E", "I"]          # untouched (no label overlap)
    assert list(out.coords["node"].values) == ["C", "A", "B"]    # reconciled
    np.testing.assert_allclose(out.sel(node="A").values, [0.0, 3.0])


# --------------------------------------------------------------------------- sel/mode helpers

def test_sel_dict_and_reconcile_mode():
    ref = _ref(sel=_sel(**{"KuramotoInertia.K": 817}), reconcile="by_label")
    assert dr.sel_dict(ref) == {"KuramotoInertia.K": 817}
    assert dr.reconcile_mode(ref) == "by_label"
    assert dr.reconcile_mode(_ref()) == "none"


def test_sel_dict_reads_the_keyed_dict_spelling():
    """A study writes ``sel: {variable: phi}`` — a NAME-KEYED collection, which is what the
    loader hands back. Reading only the list spelling silently dropped the selection, so a
    sourced argument arrived unsliced (whole trajectory instead of one state variable)."""
    sel = {"variable": SimpleNamespace(name="variable", value="phi"),
           "time": SimpleNamespace(name="time", value=[0.006, 0.016])}
    assert dr.sel_dict(_ref(sel=sel)) == {"variable": "phi", "time": [0.006, 0.016]}


# --------------------------------------------------------------------------- full pipeline

def test_resolve_dataref_end_to_end(sweep_h5):
    root, _ = sweep_h5
    ref = _ref(experiment="32", output="lyapunov_xi", sel=_sel(**{"KuramotoInertia.K": 1307}))
    out = dr.resolve_dataref(ref, results_root=root)
    assert list(out.coords["node"].values) == ["A", "B", "C"]
    np.testing.assert_allclose(out.values, [3.0, 3.1, 3.2])


def test_resolve_dataref_with_reconcile(vec_h5):
    root, _ = vec_h5
    ref = _ref(experiment="5", output="g", reconcile="by_label")
    out = dr.resolve_dataref(ref, results_root=root,
                             alias_map={k: k for k in "ABC"}, model_labels=["C", "A", "B"])
    np.testing.assert_allclose(out.values, [30.0, 10.0, 20.0])
    assert list(out.coords["node"].values) == ["C", "A", "B"]


# --------------------------------------------------------------------------- transform

def test_apply_transform_and_none():
    from tvbo.adapters import bsplot

    @bsplot.register_transform("_xsrc_test_last_point")
    def _last(da):
        return da.isel({da.dims[0]: -1})

    da = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=["point", "node"])
    np.testing.assert_allclose(dr.apply_transform(da, "_xsrc_test_last_point").values, [4.0, 5.0])
    assert dr.apply_transform(da, None) is da


def test_resolve_dataref_with_transform(sweep_h5):
    root, _ = sweep_h5
    from tvbo.adapters import bsplot

    @bsplot.register_transform("_xsrc_test_endpoint")
    def _endpoint(da):
        return da.isel({da.dims[0]: -1})       # last branch point (the operating-point pattern)

    ref = _ref(experiment="32", output="lyapunov_xi", transform="_xsrc_test_endpoint")
    out = dr.resolve_dataref(ref, results_root=root)
    np.testing.assert_allclose(out.values, [4.0, 4.1, 4.2])   # row 4 (last of 5 points)
