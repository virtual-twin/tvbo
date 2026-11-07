"""The Lead-DBS recount rewrites tracked network companions in place, so what it leaves alone is what is held here.

``scripts/rebuild_leaddbs_cohort_networks.py`` replaces the weight and length matrices of HDF5 companions the database tracks. Everything else in a companion stays, an edge group it does not recount included, and it replaces the matrices of every companion or of none: a run that stops after some networks were written leaves a database that is part old counts and part new, with nothing recording which is which.

Each companion here is a toy written into ``tmp_path`` and the count is a stand-in, so neither MRtrix nor the database is touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tvbo.data.matrix_io import write_matrix

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rebuild_leaddbs_cohort_networks.py"
COHORTS = ("MghUscHcp32", "PPMI85")
RECOUNTED = ("edges/weight/", "edges/length/")


def _symmetric(seed: int) -> np.ndarray:
    """A symmetric 4 x 4 matrix with a zero diagonal and one unconnected pair."""
    upper = np.triu(np.random.default_rng(seed).uniform(1.0, 90.0, (4, 4)), 1)
    upper[0, 3] = 0.0
    return upper + upper.T


def _write_companion(path: Path, seed: int) -> None:
    """A companion as the script meets one: sparse weights and dense lengths at 32 bits, each with attributes, beside a node table and a sidecar."""
    with h5py.File(path, "w") as f:
        f.attrs["tvbo_class"] = "tvbo:Network"
        f.create_dataset("nodes/parent_index", data=np.arange(4, dtype=np.int32))
        for name, fmt, salt in (("weight", "csr", 0), ("length", "dense", 100)):
            group = f.create_group(f"edges/{name}")
            group.attrs["directed"] = False
            write_matrix(group, _symmetric(seed + salt), fmt=fmt, dtype=np.float32)
        f["edges/length"].attrs["unit"] = "mm"
    path.with_suffix(".yaml").write_text("label: toy\n", encoding="utf-8")


def _add_an_edge_group_the_script_does_not_recount(path: Path) -> None:
    with h5py.File(path, "a") as f:
        write_matrix(f.create_group("edges/fc"), _symmetric(200), fmt="dense")
        f["edges/fc"].attrs["unit"] = "1"


def _contents(path: Path) -> dict:
    """Every group and dataset of a companion by name: its attributes, then a dataset's precision and values."""
    held = {}

    def visit(name, obj):
        attrs = {key: np.asarray(value).tolist() for key, value in obj.attrs.items()}
        is_dataset = isinstance(obj, h5py.Dataset)
        held[name + ("" if is_dataset else "/")] = (attrs, *((str(obj.dtype), obj[()].tolist()) if is_dataset else ()))

    with h5py.File(path, "r") as f:
        held["/"] = (dict(f.attrs),)
        f.visititems(visit)
    return held


@pytest.fixture
def networks(tmp_path, monkeypatch):
    """The script with two toy companions as its database and a stand-in count, as ``(module, companions, recounts)``."""
    spec = importlib.util.spec_from_file_location("rebuild_leaddbs_cohort_networks", _SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    companions = [tmp_path / f"tpl-Toy_cohort-{c}_rec-{c}_atlas-Toy_desc-SC_relmat.h5" for c in COHORTS]
    for seed, path in enumerate(companions):
        _write_companion(path, seed)
    recounts = {f"{c}.tck": {"weight": _symmetric(10 + i), "length": _symmetric(20 + i)} for i, c in enumerate(COHORTS)}
    monkeypatch.setattr(script, "NETWORKS", tmp_path)
    monkeypatch.setattr(script, "VOLUMES", {"tpl-Toy_{c}_atlas-Toy": Path("toy_dseg.nii.gz")})
    monkeypatch.setattr(script, "ensure_mrtrix", lambda: None)
    monkeypatch.setattr(script, "count", lambda tck, volume: recounts[tck])
    monkeypatch.setattr(sys, "argv", ["rebuild", "--mgh", "MghUscHcp32.tck", "--ppmi", "PPMI85.tck"])
    return script, companions, recounts


def test_a_recount_replaces_the_weights_and_lengths_and_leaves_the_rest_of_a_companion(networks, tmp_path):
    script, companions, recounts = networks
    for path in companions:
        _add_an_edge_group_the_script_does_not_recount(path)
    before = [_contents(path) for path in companions]

    script.main()

    for path, was, new in zip(companions, before, recounts.values(), strict=True):
        now = _contents(path)
        assert {name: held[:2] for name, held in now.items()} == {name: held[:2] for name, held in was.items()}
        kept = {name: held for name, held in now.items() if not name.startswith(RECOUNTED)}
        assert kept == {name: held for name, held in was.items() if not name.startswith(RECOUNTED)}
        for layer, matrix in script.stored(path).items():
            np.testing.assert_array_equal(matrix, new[layer].astype(np.float32))
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "error", [pytest.param(OSError, id="its-write-fails"), pytest.param(SystemExit, id="it-does-not-read-back")]
)
def test_a_network_that_cannot_be_rewritten_leaves_every_companion_as_it_was(networks, tmp_path, monkeypatch, error):
    """The second network fails after the first was recounted and written out: neither companion changes and no partial file stays."""
    script, companions, _ = networks
    write = script.write_matrix

    def faulty(group, matrix, **kwargs):
        if "PPMI85" not in group.file.filename:
            return write(group, matrix, **kwargs)
        if error is OSError:
            raise OSError(28, "No space left on device")
        return write(group, matrix + 1.0, **kwargs)

    monkeypatch.setattr(script, "write_matrix", faulty)
    before = {path.name: path.read_bytes() for path in companions}

    with pytest.raises(error):
        script.main()

    changed = [name for name, held in before.items() if (tmp_path / name).read_bytes() != held]
    assert (changed, [p.name for p in tmp_path.glob("*.part")]) == ([], [])
