"""The curated ``tvbo:functions/NodePermutations`` and ``tvbo:functions/SpinPermutations``: the resampling tables of a travelling-wave detector's two tests, against rules small enough to check by hand.

A table indexes network nodes in gather form, so a field permuted by row r is ``x[index[r]]``. Each row moves the nodes of every group among themselves: uniformly, as Koller et al. (2024) permute each hemisphere's phases, or by a rotation of their sphere positions followed by the nearest node of the same group, as BrainSpace's ``SpinPermutations`` reassigns them for his spin tests. The spin cases that turn on one rotation fix it, and their docstrings work the expected table out.
"""

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from tvbo.analysis import mesh
from tvbo.analysis.mesh import node_permutations, spin_permutations
from tvbo.data import param_io
from tvbo.datamodel import FunctionCall, Parameter

LABELS = np.array(["lh-d", "lh-a", "lh-f", "lh-c", "lh-e", "lh-b", "rh-c", "rh-f", "rh-a", "rh-e", "rh-b", "rh-d"])
GROUPS = np.array([label[:2] for label in LABELS])
_POINTS = np.random.default_rng(0).standard_normal((12, 3))
SPHERE = xr.DataArray(_POINTS / np.linalg.norm(_POINTS, axis=1, keepdims=True), dims=("node", "axis"), coords={"node": LABELS})
RECORDS = {node_permutations: "NodePermutations", spin_permutations: "SpinPermutations"}
BOTH = pytest.mark.parametrize("producer", list(RECORDS), ids=RECORDS.get)
GROUPINGS = {"per_hemisphere": ["lh", "lh", "rh", "rh"], "one_group": "brain"}

QUARTER_TURN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
"""A quarter turn about z acting on row vectors: ``(x, y, z) @ QUARTER_TURN`` is ``(-y, x, z)``."""

c20, s20, c30, s30 = np.cos(np.pi / 9), np.sin(np.pi / 9), np.cos(np.pi / 6), np.sin(np.pi / 6)
HAND = np.array([[1.0, 0.0, 0.0], [c20, s20, 0.0], [0.0, 0.0, 1.0], [0.0, c30, s30]])
"""Four positions on the unit sphere, nodes a to d: a on x, b 20 degrees from a in the xy plane, c on z, and d 30 degrees above the y axis."""


def _draw(producer, labels=LABELS, groups=GROUPS, n=20, seed=7):
    """*producer*'s table for a network whose nodes are *labels*; a spin turns SPHERE's positions, which are labelled and so meet any node order."""
    if producer is spin_permutations:
        return spin_permutations(SPHERE, list(labels), groups, n, seed)
    return node_permutations(list(labels), groups, n, seed)


def _by_label(table, labels):
    """Each node's group and the labels of its sources over the rows, keyed by the node's label, which no order of the nodes changes."""
    groups = table["group"].tolist()
    return {label: (groups[k], labels[table["index"][:, k]].tolist()) for k, label in enumerate(labels)}


def _sources(table, labels, group):
    """Per row, the label suffix of the source of each node of *group*, the nodes in label order: the group's table in its own terms."""
    nodes = [k for k in np.argsort(labels) if labels[k].startswith(group)]
    return [[labels[source][3:] for source in row[nodes]] for row in table["index"]]


def _turn_by(monkeypatch, rotation):
    """Every rotation the spins draw becomes *rotation*."""
    monkeypatch.setattr(mesh, "_rotations", lambda n, seed: np.broadcast_to(rotation, (n, 3, 3)))


def test_each_row_permutes_the_nodes_of_every_group_among_themselves():
    index = node_permutations(list(LABELS), GROUPS, 50, 3)["index"]
    for group in ("lh", "rh"):
        nodes = np.flatnonzero(GROUPS == group)
        np.testing.assert_array_equal(np.sort(index[:, nodes], axis=1), np.broadcast_to(nodes, (50, nodes.size)))


@BOTH
def test_no_row_takes_a_node_from_another_group(producer):
    table = _draw(producer, n=50)
    for group in ("lh", "rh"):
        nodes = np.flatnonzero(GROUPS == group)
        assert np.isin(table["index"][:, nodes], nodes).all()
    assert table["group"].tolist() == GROUPS.tolist()


@BOTH
def test_the_same_seed_gives_the_same_table_and_another_seed_another(producer):
    first, again, other = (_draw(producer, seed=seed)["index"] for seed in (7, 7, 8))
    np.testing.assert_array_equal(again, first)
    assert not np.array_equal(other, first)


@BOTH
def test_no_order_of_the_nodes_or_the_groups_changes_the_table_read_by_label(producer):
    """The network's nodes reversed, which also puts the right hemisphere first, and the groups labelled in the original order: each node keeps its sources, read by label (SO1)."""
    forward = _draw(producer)
    labelled = xr.DataArray(GROUPS, dims="node", coords={"node": LABELS})
    backward = _draw(producer, labels=LABELS[::-1], groups=labelled)
    assert _by_label(backward, LABELS[::-1]) == _by_label(forward, LABELS)


def test_a_groups_rows_do_not_depend_on_the_other_groups():
    """A third group added to the network leaves the rows of the other two as they were, since each group's stream is keyed by the seed and its own label."""
    both = node_permutations(list(LABELS), GROUPS, 20, 7)["index"]
    three = node_permutations([*LABELS, "sc-a", "sc-b", "sc-c"], [*GROUPS, "sc", "sc", "sc"], 20, 7)["index"]
    np.testing.assert_array_equal(three[:, :12], both)


def test_each_group_draws_its_own_permutations():
    """Two groups of six draw different rows, as Koller's left and right hemispheres take successive draws of one generator."""
    table = node_permutations(list(LABELS), GROUPS, 20, 7)
    assert _sources(table, LABELS, "lh") != _sources(table, LABELS, "rh")


def test_every_ordering_of_a_group_is_drawn_about_equally_often():
    """The six orderings of a group of three each come up close to 1,000 times in 6,000 rows: one ordering's count is binomial with an SD of 29, so 850 to 1,150 holds more than five SDs out."""
    index = node_permutations(["x", "y", "z", "u", "v"], ["three", "three", "three", "two", "two"], 6000, 2)["index"]
    orderings, counts = np.unique(index[:, :3], axis=0, return_counts=True)
    assert len(orderings) == 6
    assert ((counts >= 850) & (counts <= 1150)).all()


def test_the_same_rotations_turn_every_group():
    """The right hemisphere given the very positions of the left, suffix for suffix: every row reassigns both alike, as Koller's two fits with one random_state turn both hemispheres by the same rotations."""
    position = {label[3:]: point for label, point in zip(LABELS[:6], SPHERE.values[:6], strict=True)}
    twinned = np.array([position[label[3:]] for label in LABELS])
    table = spin_permutations(twinned, list(LABELS), GROUPS, 20, 7)
    assert _sources(table, LABELS, "lh") == _sources(table, LABELS, "rh")


def test_each_spin_is_a_rotation_and_keeps_the_angle_between_every_two_positions():
    """Before any reassignment, a spin keeps every dot product of two positions, so every angle between them, and has determinant 1, so it never reflects."""
    rotations = mesh._rotations(100, 5)
    points = SPHERE.values
    turned = points @ rotations
    dots = np.broadcast_to(points @ points.T, (100, 12, 12))
    np.testing.assert_allclose(turned @ turned.transpose(0, 2, 1), dots, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=1e-12)


def test_the_rotation_angles_follow_the_uniform_distribution_over_rotations():
    """A rotation drawn uniformly turns by an angle t with density (1 - cos t) / pi on [0, pi], so with distribution (t - sin t) / pi, and 20,000 draws match it by Kolmogorov-Smirnov. The draw is fixed by its seed; a uniform draw falls below p = 1e-4 for one seed in ten thousand."""
    rotations = mesh._rotations(20000, 11)
    angle = np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) / 2, -1, 1))
    assert scipy.stats.kstest(angle, lambda t: (t - np.sin(t)) / np.pi).pvalue > 1e-4


@pytest.mark.parametrize("grouping", list(GROUPINGS))
def test_a_spin_of_zero_angle_is_the_identity(monkeypatch, grouping):
    _turn_by(monkeypatch, np.eye(3))
    index = spin_permutations(HAND, ["a", "b", "c", "d"], GROUPINGS[grouping], 3, 0)["index"]
    np.testing.assert_array_equal(index, np.broadcast_to(np.arange(4), (3, 4)))


@pytest.mark.parametrize(("grouping", "expected"), [("per_hemisphere", [1, 1, 2, 2]), ("one_group", [3, 3, 2, 2])])
def test_a_turned_node_takes_the_nearest_node_of_its_group_so_a_row_may_repeat_one(monkeypatch, grouping, expected):
    """A quarter turn about z takes a to (0, 1, 0), 0.52 from d, 1.15 from b and 1.41 from a and c; b to (-sin 20, cos 20, 0), 0.61 from d, 1.41 from b and c and 1.64 from a; c stays; and d to (-cos 30, 0, sin 30), 1.00 from c, 1.22 from d and 1.9 from a and b.

    In one group a and b both take d, and c and d take c, so a and b are drawn by no node: BrainSpace's default reassignment (``unique=False``) is no permutation. That one group is Koller's whole-brain fit, and a and b, the left hemisphere here, take a node of the right. With a group per hemisphere, {a, b} and {c, d}, a and b take b and c and d take c.
    """
    _turn_by(monkeypatch, QUARTER_TURN)
    index = spin_permutations(HAND, ["a", "b", "c", "d"], GROUPINGS[grouping], 2, 0)["index"]
    np.testing.assert_array_equal(index, [expected, expected])


@BOTH
@pytest.mark.parametrize("case", ["missing_group", "group_of_one", "no_rows", "negative_rows", "no_seed"])
def test_a_table_that_cannot_be_drawn_is_refused(producer, case):
    without_first = xr.DataArray(GROUPS[1:], dims="node", coords={"node": LABELS[1:]})
    groups, n, seed, message = {
        "missing_group": (without_first, 20, 7, "groups has no entry for 1 label"),
        "group_of_one": (np.where(LABELS == "rh-c", "solo", GROUPS), 20, 7, "group 'solo' holds the one node 'rh-c'"),
        "no_rows": (GROUPS, 0, 7, "got n=0, seed=7"),
        "negative_rows": (GROUPS, -3, 7, "got n=-3, seed=7"),
        "no_seed": (GROUPS, 20, None, "got n=20, seed=None"),
    }[case]
    with pytest.raises(ValueError, match=message):
        _draw(producer, groups=groups, n=n, seed=seed)


@BOTH
def test_a_table_named_by_iri_materialises_and_an_observer_takes_its_index(producer, tmp_path):
    """``producer: {iri: tvbo:functions/<record>, output: index}`` binds the table as one array, read back as drawn. Stating no ``output`` makes it a bundle of the two arrays the record lists, from the same artifact, the group labels read back as text."""
    arguments = {"node_labels": LABELS.tolist(), "groups": GROUPS.tolist(), "n": 5, "seed": 7}
    if producer is spin_permutations:
        arguments["sphere"] = SPHERE.values.tolist()
    iri = f"tvbo:functions/{RECORDS[producer]}"
    table = Parameter(name="table", producer=FunctionCall(iri=iri, output="index", arguments=dict(arguments)))
    bundle = Parameter(name="bundle", producer=FunctionCall(iri=iri, arguments=dict(arguments)))

    path, key = param_io.materialise(table, cache_dir=tmp_path)
    bound = param_io.materialise_bundle(bundle, cache_dir=tmp_path)

    np.testing.assert_array_equal(param_io.read_artifact(path, key), producer(**arguments)["index"])
    assert list(bound) == param_io.bundle_outputs(bundle) == ["index", "group"]
    assert {artifact for artifact, _ in bound.values()} == {path}
    assert param_io.read_artifact(*bound["group"]).tolist() == GROUPS.tolist()
