"""Golden corpus for simulation output — the guard rail that freezes the numbers.

The generated-code corpus (``test_codegen_golden_corpus``) freezes what codegen *emits*.
That is necessary but not sufficient: a change can leave every emitted character identical
and still move the numbers — a different noise draw, a reordered reduction, a solver that
evaluates auxiliaries at a different point in the step. Equally, emitted code can change
for a purely cosmetic reason and leave the numbers untouched. The two corpora fail on
disjoint classes of regression, so both are needed.

Each case is a small, fully declarative ``SimulationExperiment`` under ``specs/``, chosen
to cover one axis the others cannot:

===========================  ==================================================
``pendulum_euler``           first-order integrator arithmetic
``pendulum_heun``            second-order integrator arithmetic
``pendulum_noise_seeded``    stochastic draw + seed threading
``conditional_piecewise``    ``Piecewise`` through codegen into a running solver
``network_ring_kuramoto``    coupling sum, weight matrix, per-node state axis
``modes_reducedset_hmr``     the ``mode_dot`` / ``mode_sum`` array primitives
===========================  ==================================================

Runs are reproducible: every spec produces byte-identical output across processes and
across ``PYTHONHASHSEED`` values, seeded noise included. Comparison is nevertheless made
with a tolerance rather than bit-for-bit, because floating-point results are not portable
across architectures and CI does not run on the machine that produced the reference.

:data:`DEFAULT_TOLERANCE` is tight enough that any change to the arithmetic is caught and
loose enough to survive a different CPU or BLAS; :data:`TOLERANCES` overrides it per spec
so a chaotic system can state its own without loosening the gate for everything else.

Both live here rather than in the spec files. A spec is a ``SimulationExperiment`` and
rejects keys outside the schema, so a ``golden_tolerance:`` entry would make the spec fail
to load; and reading it back would mean parsing the file a second time outside TVBO's YAML
dialect, which breaks on ``!include``.

The frozen output lives in ``expected/`` rather than ``output/`` because ``.gitignore``
carries a bare ``output`` rule — under that name the references would be silently
untracked and this suite would pass on CI while asserting nothing.

See ``tests/golden.py`` for the regeneration and reconciliation semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .golden import GoldenCorpus

CORPUS_ROOT = Path(__file__).parent / "reference_data" / "numerical"
SPECS = CORPUS_ROOT / "specs"
SPEC_PATHS = sorted(p for ext in ("*.yaml", "*.yml") for p in SPECS.glob(ext))

DEFAULT_TOLERANCE = {"rtol": 1e-9, "atol": 1e-12}
TOLERANCES: dict[str, dict[str, float]] = {}
"""Per-spec overrides of :data:`DEFAULT_TOLERANCE`, keyed by spec stem.

Empty, and that is the intended state: every spec here is deterministic enough to hold the
default. An entry belongs here only when a case is genuinely chaotic — a Lyapunov-positive
system whose trajectory separates faster than the default allows — and it should say which,
so a loosened bound stays attached to the reason for it rather than spreading to the corpus.
"""


def _run(spec: Path):
    """Run a spec and return its integration output as a labelled DataArray."""
    from tvbo.classes.experiment import SimulationExperiment

    return SimulationExperiment.from_file(str(spec)).run().integration.data


def _capture(spec: Path) -> dict:
    """Values, dimension names, coordinate labels and the time base.

    Time is pinned as numbers rather than labels: a regression in transient handling, in the
    ms-versus-s unit, or an off-by-one in the sample stamps can leave every state value and
    the sample count untouched and would otherwise pass.
    """
    data = _run(spec)
    return {
        "values": np.asarray(data.values, dtype=float),
        "time": np.asarray(data.coords["time"].values, dtype=float),
        "dims": [str(d) for d in data.dims],
        "labels": {
            str(name): [str(v) for v in np.atleast_1d(coord.values)]
            for name, coord in data.coords.items()
            if str(name) != "time"
        },
    }


def _write(path: Path, produced: dict) -> None:
    np.savez_compressed(
        path,
        values=produced["values"],
        time=produced["time"],
        dims=np.array(produced["dims"]),
        **{f"coord__{k}": np.array(v) for k, v in produced["labels"].items()},
    )


def _read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as ref:
        return {
            "values": ref["values"],
            "time": ref["time"],
            "dims": [str(d) for d in ref["dims"]],
            "labels": {k[len("coord__"):]: [str(v) for v in ref[k]] for k in ref.files
                       if k.startswith("coord__")},
        }


def _compare(produced: dict, expected: dict, tol: dict) -> str | None:
    """Structure first, then labels, then values — each reported in its own terms."""
    if produced["dims"] != expected["dims"]:
        return f"  dimension names changed — {expected['dims']} → {produced['dims']}"
    if set(produced["labels"]) != set(expected["labels"]):
        return (
            f"  coordinate set changed — {sorted(expected['labels'])} → "
            f"{sorted(produced['labels'])}"
        )
    for key, want in expected["labels"].items():
        if produced["labels"][key] != want:
            return f"  '{key}' coordinate labels changed — {want} → {produced['labels'][key]}"
    if produced["time"].shape != expected["time"].shape:
        return f"  sample count changed — {expected['time'].shape} → {produced['time'].shape}"
    if not np.allclose(produced["time"], expected["time"], **tol):
        first = int(np.argmax(np.abs(produced["time"] - expected["time"])))
        return (
            f"  time base changed — sample {first} was {expected['time'][first]!r}, "
            f"now {produced['time'][first]!r}"
        )
    values, reference = produced["values"], expected["values"]
    if values.shape != reference.shape:
        return f"  output shape changed — {reference.shape} → {values.shape}"
    if np.allclose(values, reference, **tol):
        return None
    delta = np.abs(values - reference)
    idx = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return (
        f"  beyond rtol={tol['rtol']:g} atol={tol['atol']:g}\n"
        f"  max abs difference {delta.max():.3e} at index {idx}\n"
        f"  reference {reference[idx]!r}  →  now {values[idx]!r}\n"
        f"  mean abs difference {delta.mean():.3e} over {delta.size} samples"
    )


def _corpus_for(stem: str) -> GoldenCorpus:
    """The corpus as seen by one spec, carrying that spec's tolerance."""
    tol = {**DEFAULT_TOLERANCE, **TOLERANCES.get(stem, {})}
    return GoldenCorpus(
        CORPUS_ROOT / "expected",
        ".npz",
        write=_write,
        read=_read,
        compare=lambda produced, expected: _compare(produced, expected, tol),
    )


CORPUS = _corpus_for("")


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("spec", SPEC_PATHS, ids=[p.stem for p in SPEC_PATHS])
def test_simulation_output_matches_golden(spec: Path, regenerate: bool):
    """Running a curated spec reproduces its frozen output, values, labels and time base.

    Runs twice and asserts the two agree bit for bit before comparing. Reproducibility is
    what makes a frozen trajectory meaningful — without it a failure could not be told from
    run-to-run jitter — so it is asserted here rather than behind a marker CI never enables.
    """
    pytest.importorskip("tvboptim")
    produced = _capture(spec)
    assert produced["values"].tobytes() == _capture(spec)["values"].tobytes(), (
        f"{spec.stem} does not run reproducibly"
    )
    _corpus_for(spec.stem).check(
        spec.stem, produced, regenerate=regenerate, what="simulation output"
    )


@pytest.mark.backend_core
def test_corpus_covers_every_spec(regenerate: bool):
    """Every spec has a frozen output, and none outlives its spec.

    Pure filesystem arithmetic, so it needs no simulation backend and is marked for the core
    shard rather than the tvboptim one — it is the signal that a spec was added without its
    reference, and that mistake should surface on every pull request, not only where the
    backend happens to be installed.
    """
    CORPUS.reconcile((p.stem for p in SPEC_PATHS), regenerate=regenerate, what="specs")
