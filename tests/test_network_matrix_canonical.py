"""`Network.matrix` is the one connectivity lookup, and every other accessor defers to it.

The regression these freeze: `_weights_matrix` used to consult a legacy cache BEFORE the lazy companion store, while `matrix` consulted the store first. A resolution step that populated the cache from somewhere other than the companion — declaring `parcellation:` alongside `data_file` cached a normative atlas connectome — then made the two accessors return different matrices for the same network. `matrix("weight")` reported the companion's SC while the codegen path silently integrated the atlas's raw streamline counts, so every consistency check passed and the simulation was still wrong. There is no such cache any more: what is resident is `arrays`, keyed by companion path, and it is the one place every accessor reads.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tvbo.classes.network import Network

COMPANION = np.array([[0.0, 0.25], [0.25, 0.0]])
STALE = np.array([[0.0, 182165.0], [182165.0, 0.0]])


class _Store:
    """Minimal stand-in for the lazy companion store attached by `load_network`."""

    def __init__(self, arrays):
        self.arrays = dict(arrays)
        self.names = list(arrays)

    def __contains__(self, name):
        return name in self.arrays

    def __getitem__(self, name):
        return self.arrays[name]


@pytest.fixture
def net_over_a_companion():
    """A network with a stale matrix resident under its own name and the real one in the companion, which is what the resident-wins precedence has to be read against."""
    net = Network.from_matrix(STALE, np.zeros_like(STALE))
    object.__setattr__(net, "_store", _Store({"weight": COMPANION}))
    return net


def test_a_resident_matrix_wins_over_the_companion(net_over_a_companion):
    assert np.array_equal(np.asarray(net_over_a_companion.matrix("weight", format="dense")), STALE)


def test_codegen_path_agrees_with_matrix(net_over_a_companion):
    """The accessor the tvboptim codegen path uses must not diverge from `matrix`."""
    canonical = np.asarray(net_over_a_companion.matrix("weight", format="dense", apply_transforms=False))
    codegen = np.asarray(net_over_a_companion._weights_matrix(apply_transforms=False))
    assert np.array_equal(codegen, canonical)
    assert np.array_equal(codegen, STALE)


def test_deprecated_accessors_warn_and_agree(net_over_a_companion):
    for name in ("weights_matrix", "raw_weights_matrix", "weights"):
        with pytest.deprecated_call():
            value = getattr(net_over_a_companion, name)
        assert np.array_equal(np.asarray(value), STALE), name


def test_alias_spellings_resolve_to_the_same_matrix():
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {})
    object.__setattr__(net, "_store", _Store({"weights": COMPANION}))
    assert np.array_equal(np.asarray(net.matrix("weight", format="dense")), COMPANION)


def test_a_user_set_matrix_wins_over_the_companion_under_any_spelling():
    """Precedence is between SOURCES; a spelling is not a precedence.

    Checking `_arrays[name]` then `_store[name]` one spelling at a time lets the companion file's `weight` beat a user-set `weights` — the same silent shadowing this module exists to remove, arrived at along the alias axis instead of the cache axis.
    """
    user = COMPANION * 7.0
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {"edges/weights": user})
    object.__setattr__(net, "_store", _Store({"weight": COMPANION}))
    assert np.array_equal(np.asarray(net.matrix("weight", format="dense")), user)


def test_primary_weight_selects_the_active_variant():
    variant = COMPANION * 3.0
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {})
    object.__setattr__(net, "_store", _Store({"weight": COMPANION, "shuffled": variant}))
    net.primary_weight = "shuffled"
    assert np.array_equal(np.asarray(net.matrix("weight", format="dense")), variant)


@pytest.mark.parametrize("net", [Network(number_of_nodes=1), Network(number_of_nodes=2)])
def test_an_unconnected_node_set_has_zero_weights_not_absent_ones(net):
    """No edges is zero weights, and `matrix` must say so rather than returning None.

    Every backend run reaches `ns.run_experiment(weights=network.matrix("weight", ...))`, which hands the result to `jnp.array` — so `None` here is `ValueError: None is not a valid value for jnp.array` for any single-node model or uncoupled ensemble, both of which are ordinary declarations (a bifurcation study is exactly one). The canonical accessor has to subsume what the deprecated properties returned, not a subset of it.
    """
    n = net.number_of_nodes
    W = net.matrix("weight")
    assert W is not None
    assert np.array_equal(np.asarray(W), np.zeros((n, n)))
    assert np.array_equal(np.asarray(W), np.asarray(net._weights_matrix()))


def test_an_unconnected_node_set_is_not_run_through_weight_transforms():
    """A transform describes real connectivity; over an all-zero stand-in it would yield nan.

    `W / mean(W[W > 0])` is the shape the docs use, and its denominator is 0 for zero weights.
    """
    net = Network(
        number_of_nodes=2,
        transforms=[{"name": "weight", "equation": {"rhs": "W / mean(W[W > 0])"}}],
    )
    W = np.asarray(net.matrix("weight"))
    assert np.array_equal(W, np.zeros((2, 2))), W
    assert np.isfinite(W).all()


def test_absent_tract_lengths_stay_absent():
    """Lengths get no zero default: "no tract lengths" is not "every tract has length zero".

    Standing in zeros would defeat every `is not None` delay guard downstream.
    """
    assert Network(number_of_nodes=2).matrix("length") is None


def test_the_pytree_payload_wins_under_a_jax_transformation():
    """The live matrices under a JAX transform are the leaves `tree_unflatten` installed, not pre-trace attributes.

    A resident JAX array is what `matrix` hands back, untouched. An accessor that reads around it returns the pre-trace weights and the run completes on stale connectivity.
    """
    traced_w = COMPANION * 5.0
    traced_l = np.full_like(COMPANION, 42.0)
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {"edges/weight": jnp.asarray(traced_w), "edges/length": jnp.asarray(traced_l)})
    assert np.array_equal(np.asarray(net.matrix("weight")), traced_w)
    assert np.array_equal(np.asarray(net.matrix("length")), traced_l)
    assert np.array_equal(np.asarray(net.matrix("weight")), np.asarray(net._weights_matrix()))


LENGTHS = np.array([[0.0, 30.0], [30.0, 0.0]])


def test_lengths_resolve_from_the_companion():
    """Lengths carried the same shadowing as weights — a wrong delay matrix, silently."""
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {})
    object.__setattr__(net, "_store", _Store({"weight": COMPANION, "length": LENGTHS}))
    assert np.array_equal(np.asarray(net.matrix("length", format="dense")), LENGTHS)


def test_a_sidecar_may_spell_lengths_any_known_way():
    """`tractLength` is a legal BEP017 spelling; missing it leaves a delayed network undelayed."""
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    object.__setattr__(net, "_arrays", {})
    object.__setattr__(net, "_store", _Store({"weight": COMPANION, "tractLength": LENGTHS}))
    assert np.array_equal(np.asarray(net.matrix("length", format="dense")), LENGTHS)


def test_every_weight_spelling_carries_the_declared_transform():
    """A lookup and its transform must agree: `matrix("sc")` used to skip weight transforms."""
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    net.add_transform("weight", "weight * 10")
    canonical = np.asarray(net.matrix("weight", format="dense"))
    assert np.array_equal(np.asarray(net.matrix("sc", format="dense")), canonical)
    assert np.array_equal(np.asarray(net.matrix("weights", format="dense")), canonical)


def test_a_single_node_network_is_unconnected_not_absent():
    """The degenerate case every consumer relies on; returning None here broke 158 tests."""
    from tvbo.datamodel.schema import Node

    net = Network(number_of_nodes=1, nodes=[Node(id=0, label="n0")])
    assert np.array_equal(np.asarray(net.matrix("weight")), np.zeros((1, 1)))


def test_the_pytree_payload_is_the_connectivity_under_a_trace():
    """Reading around the installed leaves returns pre-trace weights — silently stale, never an error."""
    net = Network.from_matrix(COMPANION, np.zeros_like(COMPANION))
    traced = np.full((2, 2), 5.0)
    object.__setattr__(net, "_arrays", {"edges/weight": jnp.asarray(traced), "edges/length": jnp.asarray(LENGTHS)})
    assert np.array_equal(np.asarray(net.matrix("weight")), traced)
    assert np.array_equal(np.asarray(net.matrix("length")), LENGTHS)


def test_a_sparse_stored_atlas_connectome_stays_a_sparse_matrix():
    """A parcellation whose database connectome is stored sparse resolves to a sparse N x N matrix, not a 0-d object array wrapping it."""
    from scipy import sparse

    net = Network(
        parcellation={"atlas": {"name": "Schaefer2018"}},
        tractogram={"name": "dTOR"},
        bids={"segmentation": "17Networks", "scale": "600"},
    )
    weights = net.matrix("weight", apply_transforms=False)
    assert sparse.issparse(weights)
    assert weights.shape == (600, 600)
    assert net.lengths_matrix.shape == (600, 600)


def test_load_matrix_keeps_a_sparse_matrix_sparse():
    from scipy import sparse

    net = Network()
    net.load_matrix(sparse.csr_matrix(COMPANION))
    weights = net.matrix("weight", apply_transforms=False)
    assert sparse.issparse(weights)
    assert weights.shape == (2, 2)


def test_an_edge_matrix_file_is_read_target_by_source(tmp_path):
    """A matrix file holds the weights the way every matrix a Network takes does, ``W[i, j]`` being the edge j -> i, so an asymmetric file is neither transposed nor mirrored."""
    matrix = np.array([[0.0, 2.0, 0.0], [5.0, 0.0, 0.0], [0.0, 7.0, 1.5]])
    np.savetxt(tmp_path / "w.csv", matrix, delimiter=",")
    net = Network(edge_matrix_files=[str(tmp_path / "w.csv")])
    np.testing.assert_array_equal(np.asarray(net.matrix("weight", apply_transforms=False)), matrix)
    assert net.number_of_nodes == 3


@pytest.mark.parametrize(
    ("content", "files", "message"),
    [
        ("0,1\n1,0\n", 2, "holds the one weight matrix"),
        ("0,1,2\n1,0,2\n", 1, "not a square matrix"),
        ("0,nan\n1,0\n", 1, "non-finite"),
    ],
)
def test_an_edge_matrix_file_that_is_not_one_finite_square_matrix_is_refused(tmp_path, content, files, message):
    """More than one file, a non-square matrix or a non-finite entry is refused by name rather than read partially."""
    paths = []
    for k in range(files):
        path = tmp_path / f"w{k}.csv"
        path.write_text(content)
        paths.append(str(path))
    with pytest.raises(ValueError, match=message):
        Network(edge_matrix_files=paths)


def test_a_transform_keeps_a_float64_matrix_at_float64():
    """JAX computes in 32 bits by default; a declared transform of a float64 connectome still comes back at float64."""
    weights = np.random.default_rng(0).random((5, 5)) * 3.7
    net = Network.from_matrix(weights, transforms=[{"name": "weight", "equation": {"rhs": "weight / max(weight)"}}])

    transformed = np.asarray(net.matrix("weight", format="dense"))

    assert transformed.dtype == np.float64
    np.testing.assert_allclose(transformed, weights / weights.max(), rtol=0, atol=1e-15)
