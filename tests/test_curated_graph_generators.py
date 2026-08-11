"""The shipped generators resolve through the typed DAG, end to end from a Network.

The unit tests around ``procedural`` pin the resolver against inline fixtures. These pin the artefacts users actually get: the curated YAML under ``tvbo/database/graph_generators/``, resolved the way ``Network`` resolves it. A fixture can agree with the resolver while the shipped YAML says something else entirely — that gap is exactly what the migration off the engine's numpy table could have opened.
"""

from pathlib import Path

import numpy as np
import pytest

from tvbo.classes.network import Network
from tvbo.graph_generators import random_reservoir, weight_shuffle
from tvbo.graph_generators.procedural import ProceduralError

# Captured from the PRE-MIGRATION engine, before the typed DAG existed.
_GOLDEN = Path(__file__).parent / "data" / "random_reservoir_pre_migration.npz"

SOURCE = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]])


def _reservoir_network(n=30, seed=5, **params):
    values = {"sparsity": 0.2, "spectral_radius": 0.9, **params}
    return Network(
        number_of_nodes=n,
        graph_generator={
            "name": "reservoir",
            "type": "RandomReservoir",
            "seed": seed,
            "parameters": {k: {"name": k, "value": v} for k, v in values.items()},
        },
    )


@pytest.fixture
def bound_source(monkeypatch):
    """Bind WeightShuffle's source matrix directly, so the null model needs no I/O."""
    for module in ("tvbo.graph_generators.load_matrix", "tvbo.graph_generators.catalog.load_matrix"):
        monkeypatch.setattr(module, lambda _source: SOURCE, raising=True)


# RandomReservoir — the typed-DAG route
@pytest.mark.parametrize(
    "seed, n, sparsity, radius",
    [(42, 60, 0.1, 0.95), (1, 40, 0.2, 0.8), (7, 40, 0.2, 0.8), (99, 25, 0.3, 0.5)],
)
def test_shipped_yaml_reproduces_the_pre_migration_engine(seed, n, sparsity, radius):
    """Anyone who built a reservoir before the migration must get the same one after it.

    Agreeing on properties — spectral radius on target, roughly the right density — is not reproduction: an earlier attempt satisfied both while drawing from a different stream scheme, so the sparsity pattern (the network's topology) was different. The pattern is therefore pinned bit-for-bit; the weights carry a ``1/max|eigenvalue|`` scale that is not bit-reproducible across LAPACK builds, so they get a tolerance far tighter than any real stream-scheme change but looser than a ULP.
    """
    expected = np.load(_GOLDEN)[f"{seed}_{n}_{sparsity}_{radius}"]
    got = random_reservoir(n_nodes=n, sparsity=sparsity, spectral_radius=radius, seed=seed)["weights"]
    np.testing.assert_array_equal(got != 0, expected != 0)  # topology: exact
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=0)  # weights: LAPACK-ULP tolerant


def test_network_resolution_matches_the_standalone_helper():
    """Both routes reach one resolver, so a swept network equals a scripted one."""
    np.testing.assert_array_equal(
        np.asarray(_reservoir_network().weights),
        random_reservoir(n_nodes=30, sparsity=0.2, spectral_radius=0.9, seed=5)["weights"],
    )


def test_size_comes_from_the_network_not_the_generator():
    assert np.asarray(_reservoir_network(n=17).weights).shape == (17, 17)


def test_a_generator_without_a_node_count_is_rejected():
    """Size has one source. Guessing it would silently build the wrong-sized network."""
    with pytest.raises(ValueError, match="number_of_nodes"):
        Network(
            graph_generator={
                "name": "reservoir",
                "type": "RandomReservoir",
                "parameters": {
                    "sparsity": {"name": "sparsity", "value": 0.2},
                    "spectral_radius": {"name": "spectral_radius", "value": 0.9},
                },
            }
        )


def test_a_zero_node_count_is_rejected_with_the_same_clear_message():
    """`is None` would let 0 through to fail deep inside a step on an empty matrix."""
    with pytest.raises(ValueError, match="positive count"):
        Network(
            number_of_nodes=0,
            graph_generator={
                "name": "reservoir",
                "type": "RandomReservoir",
                "parameters": {
                    "sparsity": {"name": "sparsity", "value": 0.2},
                    "spectral_radius": {"name": "spectral_radius", "value": 0.9},
                },
            },
        )


def test_a_builder_is_not_handed_defaults_its_signature_cannot_accept(monkeypatch):
    """Declared defaults describe the generator's interface, so they apply to both routes.

    A Callable builder may implement only part of that interface, though, and passing it a default it never declared is a TypeError at the call — blaming the recipe for something the curated database supplied.
    """
    import sys
    import types

    seen = {}
    module = types.ModuleType("_tvbo_test_builder")

    def build(source, seed=None):  # deliberately has no `preserve`
        seen["source"] = source
        return {"weights": SOURCE}

    module.build = build
    monkeypatch.setitem(sys.modules, "_tvbo_test_builder", module)

    net = Network(
        number_of_nodes=3,
        graph_generator={
            "name": "null",
            "type": "WeightShuffle",
            "seed": 2,
            "builder": {"name": "build", "module": "_tvbo_test_builder"},
            "parameters": {"source": {"name": "source", "value": "x://y"}},
        },
    )
    assert seen["source"] == "x://y"
    np.testing.assert_array_equal(np.asarray(net.weights), SOURCE)


def test_weight_distribution_overrides_the_declared_default():
    """The parameter is the generator's one configurable family; it must reach the draw."""
    default = random_reservoir(n_nodes=40, sparsity=0.9, spectral_radius=1.0, seed=3)["weights"]
    uniform = random_reservoir(
        n_nodes=40,
        sparsity=0.9,
        spectral_radius=1.0,
        seed=3,
        weight_distribution={"name": "Uniform", "parameters": {"lo": {"value": -1.0}, "hi": {"value": 1.0}}},
    )["weights"]
    assert not np.array_equal(default, uniform)


def test_overriding_the_weights_leaves_the_sparsity_pattern_untouched():
    """`seed_offset` isolates the two draws; sharing a stream would couple them.

    The mask must depend only on the seed and the sparsity, so a study that varies the weight distribution compares distributions over one fixed topology rather than silently rewiring the network at the same time.
    """
    default = random_reservoir(n_nodes=40, sparsity=0.9, spectral_radius=1.0, seed=3)["weights"]
    uniform = random_reservoir(
        n_nodes=40,
        sparsity=0.9,
        spectral_radius=1.0,
        seed=3,
        weight_distribution={"name": "Uniform", "parameters": {"lo": {"value": -1.0}, "hi": {"value": 1.0}}},
    )["weights"]
    np.testing.assert_array_equal(default != 0, uniform != 0)


def test_a_degenerate_draw_is_rejected_rather_than_returned_as_nan():
    """Every edge masked out => spectral radius 0 => a divide the DAG cannot survive.

    The all-NaN matrix it would otherwise produce simulates happily and only surfaces as
    NaN trajectories much later, with nothing pointing back at the connectome.
    """
    with pytest.raises(ProceduralError, match="NaN"):
        random_reservoir(n_nodes=3, sparsity=0.0, spectral_radius=0.9, seed=1)


# WeightShuffle — the documented Callable route
def test_weight_shuffle_resolves_through_its_python_binding(bound_source):
    """The escape hatch is reached by the standard `bindings.python` slot, not a special case."""
    net = Network(
        number_of_nodes=3,
        graph_generator={
            "name": "null",
            "type": "WeightShuffle",
            "seed": 7,
            "parameters": {"source": {"name": "source", "value": "irrelevant://source"}},
        },
    )
    np.testing.assert_array_equal(np.asarray(net.weights), weight_shuffle("irrelevant://source", seed=7)["weights"])


def test_weight_shuffle_through_a_network_preserves_the_null_model(bound_source):
    net = Network(
        number_of_nodes=3,
        graph_generator={
            "name": "null",
            "type": "WeightShuffle",
            "seed": 7,
            "parameters": {"source": {"name": "source", "value": "irrelevant://source"}},
        },
    )
    W = np.asarray(net.weights)
    np.testing.assert_array_equal(W != 0, SOURCE != 0)
    np.testing.assert_array_equal(np.sort(W[W != 0]), np.sort(SOURCE[SOURCE != 0]))


def test_the_declared_preserve_default_reaches_the_callable(bound_source):
    """`preserve` is declared only in the curated YAML, so an omitted one must resolve.

    Without the entry's declared defaults the callable would be invoked with `preserve` missing and fall back to its own signature default — the same answer by luck, until the two disagree.
    """
    net = Network(
        number_of_nodes=3,
        graph_generator={
            "name": "null",
            "type": "WeightShuffle",
            "seed": 1,
            "parameters": {"source": {"name": "source", "value": "irrelevant://source"}},
        },
    )
    np.testing.assert_array_equal(
        np.asarray(net.weights),
        weight_shuffle("irrelevant://source", preserve="binary_mask", seed=1)["weights"],
    )
