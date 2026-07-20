"""Eager materialisation goes through the printer tables, not a second numpy evaluator.

This is what makes one primitive definition per backend real rather than aspirational:
`materialize` renders each step with the numpy printer and evaluates it, so eager
load-time construction and emitted JAX are the same expressions rendered two ways. A
parallel numpy implementation could drift from the printer silently — the two would only
disagree on some untested edge, and the generated network would differ between a local
run and a swept one.

The fixture is RandomReservoir's construction, so the properties asserted are the ones
the generator actually promises: the post-hoc spectral-radius rescale, the sparsity, and
within-backend reproducibility.
"""

from pathlib import Path

import numpy as np
import pytest

from tvbo.graph_generators.procedural import (
    ProceduralError,
    materialize,
    partition,
)

RESERVOIR = {
    "parameters": {"sparsity": 0.1, "spectral_radius": 0.95, "n_nodes": 60},
    "derived": {
        # `seed_offset` selects the sub-stream. It must be stated: two seeded steps
        # sharing an offset would be perfectly correlated, and drawing them sequentially
        # from one stream (rather than from offset-derived ones) changes every value.
        "raw": {"type": "sample", "seed_offset": 0,
                "distribution": {"name": "Normal", "parameters": {"mean": 0.0, "std": 1.0}}},
        # The original writes `sample(Uniform(0,1)) < sparsity`; expressed with the draw on
        # the right as `sparsity > draw`, which is the same relation.
        "mask": {"type": "stochastic_mask", "of": "sparsity", "comparison": "gt",
                 "seed_offset": 1,
                 "distribution": {"name": "Uniform", "parameters": {"lo": 0.0, "hi": 1.0}}},
        "masked": {"equation": {"rhs": "raw * mask"}},
        "rho": {"equation": {"rhs": "max(abs(eigvals(masked)))"}},
        # `scale` is a step of its own rather than a parenthesised sub-expression, and that
        # is load-bearing: SymPy canonicalises `masked * (spectral_radius / rho)` into
        # `masked*spectral_radius/rho`, and float multiplication is not associative, so a
        # minority of entries would round differently. Expression syntax cannot carry
        # association through a canonicalising CAS — the DAG can, because each step is
        # evaluated separately. This is what makes the migration bit-identical.
        "scale": {"equation": {"rhs": "spectral_radius / rho"}},
    },
    "outputs": {"weights": {"equation": {"rhs": "masked * scale"}}},
}


def _weights(seed=42, **overrides):
    spec = {**RESERVOIR, "parameters": {**RESERVOIR["parameters"], **overrides}}
    return materialize(spec, seed=seed)["weights"]


# Matrices captured from the PRE-MIGRATION engine (the numpy-primitive-table evaluator),
# stored rather than recomputed on the fly. Comparing against the live old engine would
# stop meaning anything the moment that engine is deleted — the test would quietly be
# comparing the new path to itself and would pass no matter what it produced.
_GOLDEN = Path(__file__).parent / "data" / "random_reservoir_pre_migration.npz"


@pytest.mark.parametrize(
    "seed, n, sparsity, radius",
    [(42, 60, 0.1, 0.95), (1, 40, 0.2, 0.8), (7, 40, 0.2, 0.8), (99, 25, 0.3, 0.5)],
)
def test_printer_path_is_bit_identical_to_the_pre_migration_engine(seed, n, sparsity, radius):
    """The migration must not change a single value of an already-shipped generator.

    Properties agreeing (spectral radius on target, roughly the right density) is NOT
    reproduction: an earlier attempt satisfied both while drawing from a different stream
    scheme, producing a different sparsity pattern entirely. Anyone who had generated a
    reservoir before the migration would silently have got a different network after it.
    """
    expected = np.load(_GOLDEN)[f"{seed}_{n}_{sparsity}_{radius}"]
    got = _weights(seed=seed, n_nodes=n, sparsity=sparsity, spectral_radius=radius)
    np.testing.assert_array_equal(got, expected)


def test_sparsity_pattern_is_identical_not_merely_similar():
    """The pattern is the network's topology; 'about the right density' would hide a rewire."""
    expected = np.load(_GOLDEN)["42_60_0.1_0.95"]
    np.testing.assert_array_equal(_weights() != 0, expected != 0)


def test_spectral_radius_is_rescaled_to_target():
    """The generator's headline promise, computed through the printer path."""
    W = _weights()
    assert float(np.max(np.abs(np.linalg.eigvals(W)))) == pytest.approx(0.95, rel=1e-6)


def test_spectral_radius_target_is_honoured_for_other_values():
    W = _weights(spectral_radius=0.5)
    assert float(np.max(np.abs(np.linalg.eigvals(W)))) == pytest.approx(0.5, rel=1e-6)


def test_sparsity_is_approximately_the_requested_density():
    W = _weights()
    assert (W != 0).mean() == pytest.approx(0.1, abs=0.03)


def test_shape_comes_from_n_nodes():
    assert _weights().shape == (60, 60)
    assert _weights(n_nodes=25).shape == (25, 25)


def test_materialisation_is_reproducible_within_a_backend():
    np.testing.assert_array_equal(_weights(seed=42), _weights(seed=42))


def test_materialisation_is_seed_sensitive():
    assert not np.array_equal(_weights(seed=42), _weights(seed=43))


def test_a_fully_seeded_dag_has_no_deterministic_prefix():
    """Every step of a reservoir descends from the draw, so nothing is hoistable."""
    deterministic, stochastic = partition(RESERVOIR)
    assert deterministic == []
    assert stochastic == ["raw", "mask", "masked", "rho", "scale"]


def test_missing_n_nodes_is_rejected():
    """Size comes from Network.number_of_nodes; guessing it would be a silent wrong shape."""
    spec = {**RESERVOIR, "parameters": {k: v for k, v in RESERVOIR["parameters"].items()
                                        if k != "n_nodes"}}
    with pytest.raises(ProceduralError, match="n_nodes"):
        materialize(spec, seed=1)


def test_a_failing_step_reports_which_step_and_what_was_rendered():
    """A broken procedure must name the step and show the emitted source, not raise
    an opaque NameError from inside an eval."""
    spec = {"parameters": {"n_nodes": 4},
            "derived": {"bad": {"equation": {"rhs": "normalize(nope, 0)"}}}}
    with pytest.raises(ProceduralError, match=r"step 'bad' failed to evaluate"):
        materialize(spec, seed=1)


def test_two_seeded_steps_use_distinct_substreams():
    """Sharing a seed_offset makes two draws perfectly correlated, not independent."""
    same = {**RESERVOIR, "derived": {**RESERVOIR["derived"],
            "mask": {**RESERVOIR["derived"]["mask"], "seed_offset": 0}}}
    assert not np.array_equal(_weights(), materialize(same, seed=42)["weights"])
