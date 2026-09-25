"""A per-subject cohort is read per subject by a per-subject run, and as a whole cohort by everything else.

The fan-out writes one `sub-<id>_exp-<N>_…_result.h5` shard per subject into one directory. A per-subject experiment that sources another per-subject experiment (a forward run reading the same subject's fit) must get its own subject's shard; before this, `locate_exp_container` returned the first shard for every subject, so subject B's run silently used subject A's parameters. A study analysis, which has no subject of its own, gets the whole cohort stacked on a `subject` axis instead of one subject posing as the cohort.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tvbo.data import dataref as dr


def _ref(**kw):
    """A minimal DataRef stand-in matching the resolver's duck-typing."""
    for key in ("experiment", "iri", "output", "sel", "reconcile", "analysis", "transform"):
        kw.setdefault(key, None)
    return SimpleNamespace(**kw)


@pytest.fixture
def cohort(tmp_path):
    """Three subjects' shards of experiment 30, each holding a per-node estimate that identifies its subject."""
    for k, sid in enumerate(("01", "02", "03")):
        ds = xr.Dataset({"estimate__J_i": (("node",), np.full(4, float(k + 1)))}, coords={"node": list("ABCD")})
        ds.to_netcdf(tmp_path / f"sub-{sid}_exp-30_desc-Model_result.h5", engine="h5netcdf")
    return tmp_path


def test_a_subject_reads_its_own_shard(cohort):
    assert dr.locate_exp_container(cohort, 30, subject="02").name == "sub-02_exp-30_desc-Model_result.h5"
    assert dr.locate_exp_container(cohort, 30, subject="sub-03").name == "sub-03_exp-30_desc-Model_result.h5"


def test_a_subject_without_a_shard_is_refused(cohort):
    """Falling back to another subject's shard is the plausible wrong answer this exists to prevent."""
    with pytest.raises(FileNotFoundError, match="no shard for subject '99'"):
        dr.locate_exp_container(cohort, 30, subject="99")


def test_a_group_source_serves_every_subject(tmp_path):
    """A per-subject fit warm-starts from ONE group fit, which carries no subject entity."""
    (tmp_path / "exp-34_desc-Model_result.h5").write_bytes(b"")
    assert dr.locate_exp_container(tmp_path, 34, subject="02").name == "exp-34_desc-Model_result.h5"


def test_resolve_dataref_selects_the_subject(cohort):
    da = dr.resolve_dataref(_ref(experiment=30, output="J_i"), results_root=cohort, subject="03")
    assert da.dims == ("node",)
    np.testing.assert_array_equal(da.values, 3.0)


def test_without_a_subject_the_cohort_is_stacked(cohort):
    """An analysis reading a per-subject experiment sees every subject, keyed by id, not the first shard."""
    da = dr.resolve_dataref(_ref(experiment=30, output="J_i"), results_root=cohort)
    assert da.dims == ("subject", "node")
    assert list(da.subject.values) == ["01", "02", "03"]
    np.testing.assert_array_equal(da.sel(subject="02").values, 2.0)


def test_a_single_run_is_not_stacked(tmp_path):
    xr.Dataset({"g": (("node",), np.arange(3.0))}).to_netcdf(tmp_path / "exp-5_desc-M_result.h5", engine="h5netcdf")
    da = dr.resolve_dataref(_ref(experiment=5, output="g"), results_root=tmp_path)
    assert da.dims == ("node",)


def test_cohort_shards_is_none_for_a_single_run(tmp_path, cohort):
    (tmp_path / "exp-5_desc-M_result.h5").write_bytes(b"")
    assert dr.cohort_shards(tmp_path, 5) is None
    assert list(dr.cohort_shards(cohort, 30)) == ["01", "02", "03"]


def test_source_producers_reads_the_declared_algorithm_order(tmp_path):
    """A forward run that declares no algorithm reads which one the source finished with from the source's own sidecar."""
    h5 = tmp_path / "exp-34_desc-M_result.h5"
    h5.write_bytes(b"")
    h5.with_suffix(".yaml").write_text("algorithms:\n  fic: {name: fic}\n  fic_eib: {name: fic_eib}\n", encoding="utf-8")
    assert dr.source_producers(h5) == ["fic_eib", "fic"]
    assert dr.source_producers(tmp_path / "missing.h5") == []
