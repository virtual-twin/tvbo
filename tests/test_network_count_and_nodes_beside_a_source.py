"""A ``number_of_nodes`` beside a source is a size, and a node list beside a source is that source's nodes.

At the source's own count the network is the source. Below it, the network keeps a random subset of that many nodes, drawn at build from the ``execution.random_seed`` of the experiment building it and kept as the ``node_selection`` that names them, so a save and its reload keep the same nodes, and a rebuild draws the same ones again unless a ``--set`` of the count or of the seed asks for another subset, which it then draws. Above it, the network is refused, and so is a count beside a ``node_selection`` that keeps another number. A node list with ids beside a source has to be the source's nodes or the ones its ``node_selection`` keeps, and a refused one is named by its labels, or by its ids where it states none. A subset, a count's or a selection's, is reported in one warning that names the network and both counts, and reading back a save that already holds it gives none.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pytest

from tvbo import Network, SimulationExperiment
from tvbo.run.study import apply_overrides, rebuild_invalidated

DESIKAN = "tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat"
SOURCES = pytest.mark.parametrize(
    "source", [{"iri": DESIKAN}, {"parcellation": {"atlas": {"name": "DesikanKilliany"}}}], ids=["iri", "parcellation"]
)


def keeps(name: str, kept: int) -> str:
    """The subset warning for the network *name* keeping *kept* of Desikan-Killiany's 87 nodes, as a pattern."""
    return rf"^{name}: keeps {kept} of its 87 nodes\.$"


def name_of(source: dict) -> str:
    """The name the subset warning gives a network built from *source*: the curated network's label, or its atlas's name when the parcellation names it."""
    return r"DesikanKilliany \(dTOR\)" if "iri" in source else "DesikanKilliany"


@contextmanager
def no_subset_warning():
    """A block in which a subset warning is an error."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*: keeps \d+ of its \d+ nodes\.")
        yield


@pytest.fixture(scope="module")
def curated():
    """Desikan-Killiany as the database holds it."""
    return Network(iri=DESIKAN)


def weights(net) -> np.ndarray:
    """*net*'s weight layer as a dense array."""
    matrix = net.matrix("weight")
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


@SOURCES
def test_a_count_at_the_sources_own_is_the_source(source, curated):
    with no_subset_warning():
        net = Network(**source, number_of_nodes=87)
    assert list(net.node_labels) == list(curated.node_labels) and net.node_selection is None
    np.testing.assert_array_equal(weights(net), weights(curated))


@SOURCES
def test_a_smaller_count_keeps_a_random_subset_of_the_sources_nodes_with_their_weights(source, curated):
    with pytest.warns(UserWarning, match=keeps(name_of(source), 5)):
        net = Network(**source, number_of_nodes=5)
    at = [list(curated.node_labels).index(label) for label in net.node_labels]
    assert net.number_of_nodes == 5 and at == sorted(at)
    assert list(net.node_selection.keep) == list(net.node_labels) and net.node_selection.applied
    np.testing.assert_array_equal(weights(net), weights(curated)[np.ix_(at, at)])


def test_the_subset_is_drawn_from_the_experiments_random_seed():
    """Two experiments at one seed keep the same nodes, another seed others, and a network built on its own draws from the seed an experiment defaults to."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = {
            seed: [
                list(
                    SimulationExperiment(
                        network={"iri": DESIKAN, "number_of_nodes": 5}, execution={"random_seed": s}
                    ).network.node_labels
                )
                for s in (seed, seed)
            ]
            for seed in (0, 1)
        }
        alone = list(Network(iri=DESIKAN, number_of_nodes=5).node_labels)
    assert drawn[0][0] == drawn[0][1] and drawn[1][0] == drawn[1][1]
    assert drawn[0][0] != drawn[1][0] and alone == drawn[0][0]


def test_the_subset_is_drawn_once_and_a_save_and_its_reload_keep_it(tmp_path):
    with pytest.warns(UserWarning, match=keeps(r"DesikanKilliany \(dTOR\)", 5)):
        net = Network(iri=DESIKAN, number_of_nodes=5)
    kept, kept_weights = list(net.node_labels), weights(net)
    net.invalidate_resolution()
    with pytest.warns(UserWarning, match=keeps(r"DesikanKilliany \(dTOR\)", 5)):
        net._resolve()
    assert list(net.node_labels) == kept
    net.save(tmp_path / "subset.yaml")
    with no_subset_warning():
        back = Network.from_file(tmp_path / "subset.yaml")
    assert list(back.node_labels) == kept and list(back.node_selection.keep) == kept
    np.testing.assert_array_equal(weights(back), kept_weights)


def experiment(count: int, seed: int) -> SimulationExperiment:
    """An experiment on Desikan-Killiany cut to *count* nodes, drawn at *seed*."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimulationExperiment(network={"iri": DESIKAN, "number_of_nodes": count}, execution={"random_seed": seed})


@pytest.mark.parametrize(
    ("overrides", "count", "seed"),
    [
        ([("network.number_of_nodes", 10)], 10, 0),
        ([("network.number_of_nodes", 87)], 87, 0),
        ([("execution.random_seed", 1)], 5, 1),
        ([("network.number_of_nodes", 10), ("execution.random_seed", 1)], 10, 1),
        ([("execution.random_seed", 1), ("network.number_of_nodes", 10)], 10, 1),
    ],
    ids=["count", "count-of-the-source", "seed", "count-then-seed", "seed-then-count"],
)
def test_a_set_of_the_count_or_the_seed_draws_the_subset_the_recipe_stating_them_draws(overrides, count, seed):
    """An override is the run of the recipe that states it, so a drawn subset is drawn again from the count and the seed set."""
    exp = experiment(5, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuild_invalidated(apply_overrides(exp, overrides), exp)
    stated = experiment(count, seed).network
    assert exp.network.number_of_nodes == count and list(exp.network.node_labels) == list(stated.node_labels)
    assert (exp.network.node_selection is None) == (count == 87)
    np.testing.assert_array_equal(weights(exp.network), weights(stated))


def test_a_count_written_on_a_network_before_it_is_invalidated_is_the_count_its_rebuild_keeps():
    net = experiment(5, 0).network
    net.number_of_nodes = 10
    net.invalidate_resolution()
    with pytest.warns(UserWarning, match=keeps(r"DesikanKilliany \(dTOR\)", 10)):
        net._resolve()
    assert list(net.node_labels) == list(experiment(10, 0).network.node_labels)


def test_a_reloaded_subset_keeps_its_nodes_whatever_seed_is_set(tmp_path):
    """A save lists the nodes it kept, so its reload is those nodes, and a seed set on the experiment holding it draws nothing."""
    experiment(5, 0).network.save(tmp_path / "subset.yaml")
    with no_subset_warning():
        back = Network.from_file(tmp_path / "subset.yaml")
        kept = list(back.node_labels)
        exp = SimulationExperiment(network=back, execution={"random_seed": 0})
        assert apply_overrides(exp, [("execution.random_seed", 1)]) == []
    assert list(exp.network.node_labels) == kept == list(experiment(5, 0).network.node_labels)


@SOURCES
def test_a_count_above_the_sources_is_refused_naming_both(source):
    with pytest.raises(
        ValueError,
        match=r"states `number_of_nodes: 100`, and .* has 87 nodes\. Beside a source a count is the number of its nodes the network keeps",
    ):
        Network(**source, number_of_nodes=100)


def test_a_node_list_beside_a_source_that_is_its_nodes_is_taken(curated):
    listed = [{"id": i, "label": str(label)} for i, label in enumerate(curated.node_labels)]
    net = Network(iri=DESIKAN, nodes=listed)
    assert list(net.node_labels) == list(curated.node_labels)
    np.testing.assert_array_equal(weights(net), weights(curated))


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        (
            lambda labels: [{"id": i, "label": f"n{i}"} for i in range(3)],
            r"lists 3 node\(s\) beside `iri: .*`, which gives 87; node 0 is 'n0' in the list and 'left-cerebellum-cortex' in the source",
        ),
        (
            lambda labels: [{"id": i, "label": f"r{i}"} for i in range(87)],
            r"lists 87 node\(s\) beside `iri: .*`, which gives 87; node 0 is 'r0' in the list",
        ),
        (
            lambda labels: [{"id": i, "label": label} for i, label in enumerate([labels[1], labels[0], *labels[2:]])],
            r"lists 87 node\(s\) beside `iri: .*`, which gives 87; node 0 is 'left-thalamus' in the list and 'left-cerebellum-cortex' in the source",
        ),
    ],
    ids=["fewer", "renamed", "reordered"],
)
def test_a_node_list_beside_a_source_that_is_not_its_nodes_is_refused_naming_both(listed, message, curated):
    with pytest.raises(ValueError, match=message):
        Network(iri=DESIKAN, nodes=listed([str(label) for label in curated.node_labels]))


def test_a_node_list_with_ids_and_no_labels_beside_a_source_is_refused_naming_the_id():
    """The list's nodes state no label, so the refusal names the first that differs by its id, not by a label the list never wrote."""
    with pytest.raises(
        ValueError,
        match=r"lists 3 node\(s\) beside `iri: .*`, which gives 87; node 0 is the node with id 0 and no label in the list and 'left-cerebellum-cortex' in the source",
    ):
        Network(iri=DESIKAN, nodes=[{"id": i} for i in range(3)])


@SOURCES
def test_a_count_beside_a_selection_keeping_another_number_is_refused_naming_both(source, curated):
    """Beside a source a count and a ``node_selection`` each say how many of its nodes the network keeps, so they agree, or the network is refused."""
    kept = [str(label) for label in curated.node_labels][:3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            ValueError,
            match=r"states `number_of_nodes: 5`, and its `node_selection` keeps 3 of the 87 nodes of .*\. Beside a source both are the number of its nodes the network keeps, so they agree: state 3, or leave the count out\.",
        ):
            Network(**source, number_of_nodes=5, node_selection={"keep": kept})
        net = Network(**source, number_of_nodes=3, node_selection={"keep": kept})
    assert list(net.node_labels) == kept


def test_a_node_list_of_the_nodes_a_selection_keeps_is_taken_and_any_other_refused(curated):
    labels = [str(label) for label in curated.node_labels]
    kept = [labels[0], labels[5]]
    with pytest.warns(UserWarning, match=keeps(r"DesikanKilliany \(dTOR\)", 2)):
        net = Network(
            iri=DESIKAN, node_selection={"keep": kept}, nodes=[{"id": i, "label": label} for i, label in enumerate(kept)]
        )
    assert list(net.node_labels) == kept
    with pytest.raises(ValueError, match=r"lists 2 node\(s\) beside `iri: .*`, which gives 2; node 1 is .* in the list"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Network(
                iri=DESIKAN,
                node_selection={"keep": kept},
                nodes=[{"id": i, "label": label} for i, label in enumerate([labels[0], labels[6]])],
            )


def test_a_node_selection_that_drops_nodes_warns_and_its_save_reads_back_without_a_warning(tmp_path, curated):
    kept = [str(label) for label in curated.node_labels][:4]
    with pytest.warns(UserWarning, match=keeps(r"DesikanKilliany \(dTOR\)", 4)):
        net = Network(iri=DESIKAN, node_selection={"keep": kept})
    net.save(tmp_path / "selected.yaml")
    with no_subset_warning():
        back = Network.from_file(tmp_path / "selected.yaml")
    assert list(back.node_labels) == kept
