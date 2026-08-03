"""ExperimentResult.save() records an algorithm's tuned FREE parameters as
``estimate__<param>`` on LABELLED node axes (``node`` for vectors, ``node_i``+
``node_j`` for per-edge matrices), so a ``from_experiment`` warm-start reloads them as
a prior location and reconciles by label with the existing `.sel` path.

Source-side, additive, container-layer only (no codegen). Filtered to free params, so it
never shadows a ``<sv>_final`` state observation.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.data.types import ExperimentResult, _free_param_names


def _param(name, free):
    return SimpleNamespace(name=name, free=free)


def _fake_source(labels):
    """A stand-in experiment: J_i (dynamics) + wLRE/wFFI (coupling) free; I_o/G fixed."""
    return SimpleNamespace(
        network=SimpleNamespace(node_labels=labels),
        dynamics=SimpleNamespace(parameters={
            "J_i": _param("J_i", True),
            "I_o": _param("I_o", False),
        }),
        coupling=SimpleNamespace(parameters={
            "wLRE": _param("wLRE", True),
            "wFFI": _param("wFFI", True),
            "G": _param("G", False),
        }),
    )


class _JaxParam:
    """Mimics a tvboptim Parameter leaf exposing ``__jax_array__`` (the real state
    wraps tuned params this way, not as bare ndarrays)."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def __jax_array__(self):
        return self._arr


def _algo_state(n):
    return SimpleNamespace(
        coupling=SimpleNamespace(EIBLinearCoupling=SimpleNamespace(
            wLRE=_JaxParam(np.full((n, n), 0.7)),
            wFFI=_JaxParam(np.full((n, n), 0.3)),
            G=_JaxParam(np.array(2.0)),        # fixed — not recorded
        )),
        dynamics=SimpleNamespace(
            J_i=_JaxParam(np.arange(n, dtype=float)),
            S_e=_JaxParam(np.full(n, 0.204)),  # state variable — not recorded
        ),
    )


def test_free_param_names_collects_dynamics_and_coupling():
    src = _fake_source([f"n{i}" for i in range(3)])
    assert _free_param_names(src) == {"J_i", "wLRE", "wFFI"}
    assert _free_param_names(None) == set()


def test_save_records_labelled_estimate_for_free_params_only(tmp_path):
    xr = pytest.importorskip("xarray")
    n = 4
    labels = [f"R{i}" for i in range(n)]
    res = ExperimentResult(
        algorithms={"fic_eib": SimpleNamespace(state=_algo_state(n))},
        source=_fake_source(labels),
    )
    written = res.save(str(tmp_path), compress=False, record_only=False)
    h5 = [p for p in written if p.endswith(".h5")]
    assert h5, f"expected an .h5 result, got {written}"

    ds = xr.open_dataset(h5[0], engine="h5netcdf")
    try:
        names = set(ds.data_vars)
        assert {"estimate__wLRE", "estimate__wFFI", "estimate__J_i"} <= names
        # fixed param (G) and state variable (S_e) excluded
        assert "estimate__G" not in names
        assert "estimate__S_e" not in names

        # matrix: labelled on BOTH node axes, labels preserved in order
        wl = ds["estimate__wLRE"]
        assert wl.dims == ("node_i", "node_j")
        assert list(wl["node_i"].values) == labels
        assert list(wl["node_j"].values) == labels
        assert wl.shape == (n, n)

        # vector: labelled on its single node axis
        ji = ds["estimate__J_i"]
        assert ji.dims == ("node",)
        assert list(ji["node"].values) == labels
        np.testing.assert_allclose(ji.values, np.arange(n))
    finally:
        ds.close()


def test_save_records_bare_jax_array_params(tmp_path):
    """The real tvboptim state stores tuned params as BARE jax arrays (``ArrayImpl``),
    not wrapped in a ``__jax_array__`` object. A jax array carries an empty ``__dict__``,
    so a container-vs-leaf guard that only excludes numpy arrays mis-recurses into it and
    silently drops every per-node array param (J_i / wLRE / wFFI). Guard the leaf path
    for the real representation, not just the ``_JaxParam`` mock."""
    xr = pytest.importorskip("xarray")
    jnp = pytest.importorskip("jax.numpy")
    n = 4
    labels = [f"R{i}" for i in range(n)]
    state = SimpleNamespace(
        dynamics=SimpleNamespace(J_i=jnp.arange(n, dtype=float)),
        coupling=SimpleNamespace(EIBLinearCoupling=SimpleNamespace(
            wLRE=jnp.full((n, n), 0.7), wFFI=jnp.full((n, n), 0.3))),
    )
    res = ExperimentResult(algorithms={"fic_eib": SimpleNamespace(state=state)},
                           source=_fake_source(labels))
    written = res.save(str(tmp_path), compress=False, record_only=False)
    ds = xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf")
    try:
        names = set(ds.data_vars)
        assert {"estimate__J_i", "estimate__wLRE", "estimate__wFFI"} <= names
        assert ds["estimate__J_i"].dims == ("node",)
        assert list(ds["estimate__J_i"]["node"].values) == labels
        assert ds["estimate__wLRE"].dims == ("node_i", "node_j")
    finally:
        ds.close()


def test_estimate_uses_resolved_not_placeholder_labels(tmp_path):
    """A bids: source can carry placeholder node_labels (region_<i>) until hydrated; the
    estimate must be recorded with the RESOLVED labels (what the consumer reconciles
    against), else the warm-start `.sel` can't align."""
    xr = pytest.importorskip("xarray")

    class _Src:
        network = SimpleNamespace(node_labels=["region_0", "region_1"])   # placeholders
        dynamics = SimpleNamespace(parameters={"J_i": _param("J_i", True)})
        coupling = None

        def _resolve_model_node_labels(self):
            return ["L_V1", "R_V1"]   # hydrated real labels

    state = SimpleNamespace(dynamics=SimpleNamespace(J_i=_JaxParam([1.0, 2.0])),
                            coupling=SimpleNamespace())
    res = ExperimentResult(algorithms={"a": SimpleNamespace(state=state)}, source=_Src())
    written = res.save(str(tmp_path), compress=False, record_only=False)
    ds = xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf")
    try:
        assert [str(x) for x in ds["estimate__J_i"]["node"].values] == ["L_V1", "R_V1"]
    finally:
        ds.close()


def test_unlabelled_source_falls_back_to_generic(tmp_path):
    """No source node_labels → generic dims (still recorded, just no reconcile coords)."""
    xr = pytest.importorskip("xarray")
    n = 3
    res = ExperimentResult(
        algorithms={"fic_eib": SimpleNamespace(state=_algo_state(n))},
        source=_fake_source(None),
    )
    written = res.save(str(tmp_path), compress=False, record_only=False)
    h5 = [p for p in written if p.endswith(".h5")]
    assert h5
    ds = xr.open_dataset(h5[0], engine="h5netcdf")
    try:
        assert "estimate__J_i" in set(ds.data_vars)
        assert "node" not in ds["estimate__J_i"].coords   # no label coords
    finally:
        ds.close()
