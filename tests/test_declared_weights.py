"""A network's weights are its ``weight`` edge, or the edge its ``primary_weight`` names (register WT1).

Setting ``weight`` on a network that declares its weights sets the declared edge, which is the one read. A second transform of the weights names them as its ``output``, a transform's ``name`` being its identifier. No other name is read as the weights. A network that holds an edge of a name a weight is often given (``weights``, ``sc``, ``streamlineCount``, ``streamlineDensity``) and neither a ``weight`` edge nor a ``primary_weight`` is refused naming it, and so is a transform aimed at such a name when no layer of the network has it: either would otherwise read as an unconnected network or a transform of nothing. A BEP017 folder whose measures do not include ``weight`` names its weights in ``structural_measures``. The curated networks whose weights are another edge declare it, so their weights reach a run, a normative read and a save under the name they hold them by.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from tvbo import Network
from tvbo.classes.network import get_normative_connectome_data
from tvbo.utils import edge_label

SMILEY = "tvbo:networks/SmileyNetwork"
HANSEN = (
    "tvbo:networks/tpl-MNI152NLin6Asym_cohort-HCPYA_rec-Hansen2022_atlas-Schaefer2018_seg-7Networks_scale-100_desc-SCFC_relmat"
)
FORMER = ("weights", "sc", "streamlineCount", "streamlineDensity")
M = np.array([[0.0, 2.0, 0.5], [2.0, 0.0, 1.0], [0.5, 1.0, 0.0]])


def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _holding(name, primary=None):
    """A three-node network holding *M* as its edge *name*, declaring *primary* as its weights when given."""
    net = Network(number_of_nodes=3, **({"primary_weight": primary} if primary else {}))
    net.set_matrix(name, M)
    return net


@pytest.mark.parametrize("iri", [SMILEY, HANSEN], ids=["SmileyNetwork", "Hansen2022"])
def test_a_curated_network_whose_weights_are_another_edge_declares_it(iri):
    net = Network(iri=iri)
    assert net.primary_weight == "sc"
    weights = _dense(net.matrix("weight"))
    np.testing.assert_array_equal(weights, _dense(net.matrix("sc")))
    assert weights.any()


@pytest.mark.parametrize(
    "named",
    [
        {"iri": HANSEN},
        {"atlas": "Schaefer2018", "tractogram": "Hansen2022", "segmentation": "7Networks", "scale": "100"},
    ],
    ids=["iri", "entities"],
)
def test_a_normative_read_gives_the_declared_weights(named):
    weights, _ = get_normative_connectome_data(**named)
    own = Network(iri=HANSEN).matrix("weight", apply_transforms=False)
    assert weights.shape == (100, 100)
    np.testing.assert_array_equal(_dense(weights), _dense(own))


def test_a_network_named_by_its_parcellation_reads_the_declared_weights_and_saves_them(tmp_path):
    net = Network(
        parcellation={"atlas": {"name": "Schaefer2018"}},
        bids={"segmentation": "7Networks", "scale": "100"},
        tractogram={"name": "Hansen2022"},
    )
    own = _dense(Network(iri=HANSEN).matrix("weight"))
    np.testing.assert_array_equal(_dense(net.matrix("weight")), own)
    net.save(tmp_path / "parcellated.yaml")
    np.testing.assert_array_equal(_dense(Network.from_file(tmp_path / "parcellated.yaml").matrix("weight")), own)


@pytest.mark.parametrize("name", FORMER)
def test_a_network_holding_a_former_weight_name_and_no_declaration_is_refused(name):
    net = _holding(name)
    refusal = rf"holds the edge '{name}' and no 'weight', and declares no `primary_weight`"
    with pytest.raises(ValueError, match=refusal):
        net.matrix("weight")
    with pytest.raises(ValueError, match=refusal):
        _ = net.has_connectome


@pytest.mark.parametrize("name", FORMER)
def test_a_declared_primary_weight_is_the_weights_and_takes_their_transforms(name):
    net = _holding(name, primary=name)
    net.add_transform("weight", "weight * 10")
    np.testing.assert_array_equal(_dense(net.matrix("weight")), 10 * M)
    np.testing.assert_array_equal(_dense(net.matrix(name)), 10 * M)


def test_a_later_transform_of_the_weights_names_them_as_its_output():
    transforms = [
        {"name": "weight", "equation": {"rhs": "weight * 10"}},
        {"name": "halved", "output": "weight", "equation": {"rhs": "weight / 2"}},
    ]
    net = Network(number_of_nodes=3, transforms=transforms)
    net.set_matrix("weight", M)
    np.testing.assert_array_equal(_dense(net.matrix("weight")), 5 * M)


@pytest.mark.parametrize("name", FORMER)
def test_a_transform_aimed_at_a_former_weight_name_no_layer_has_is_refused(name):
    net = _holding("weight")
    net.add_transform(name, f"{name} * 2")
    advice = r"Aim it at the weights with `name: weight`, or, for a transform after the first, a name of its own and `output: weight`"
    with pytest.raises(
        ValueError, match=rf"A transform of this network is aimed at '{name}', which no layer of it is: .*{advice}"
    ):
        net.matrix("weight")


def test_a_graph_carries_the_declared_weights_once_as_weight():
    attrs = next(iter(_holding("sc", primary="sc").graph.edges(data=True)))[2]
    assert "weight" in attrs and "sc" not in attrs


def test_weights_is_a_layer_name_like_any_other():
    net = _holding("weight")
    net.set_matrix("weights", 2 * M)
    net.add_transform("weight", "weights * 1")
    np.testing.assert_array_equal(_dense(net.matrix("weight")), 2 * M)
    assert edge_label("weights") is None and edge_label("network.weights") is None
    added = Network(number_of_nodes=2)
    added.add_edge(0, 1, weights=0.5)
    with pytest.raises(ValueError, match=r"holds the edge 'weights' and no 'weight'"):
        added.matrix("weight")


@pytest.fixture
def bep017_counts(tmp_path):
    """A BEP017 folder holding streamline counts and tract lengths, and no ``meas-weight``."""
    for measure, matrix in (("streamlineCount", M), ("tractLength", 10 * M)):
        stem = tmp_path / f"atlas-Probe_meas-{measure}_relmat"
        np.savetxt(f"{stem}.dense.tsv", matrix, delimiter="\t")
        (tmp_path / f"{stem.name}.json").write_text(json.dumps({"RelationshipMeasure": measure}))
    return tmp_path


def test_a_bep017_folder_without_weight_names_its_weights(bep017_counts):
    with pytest.raises(
        ValueError,
        match=r"holds the measures streamlineCount, tractLength and no 'weight', and the network states neither `primary_weight` nor `structural_measures`",
    ):
        Network.from_bids(bep017_counts)
    with pytest.raises(ValueError, match=r"and no 'streamlineDensity', the measure its `primary_weight` names"):
        Network.from_bids(bep017_counts, primary_weight="streamlineDensity")
    for net in (
        Network.from_bids(bep017_counts, structural_measures=["streamlineCount", "tractLength"]),
        Network.from_bids(bep017_counts, primary_weight="streamlineCount"),
        Network(bids_dir=str(bep017_counts), primary_weight="streamlineCount"),
    ):
        np.testing.assert_array_equal(_dense(net.matrix("weight")), M)
        np.testing.assert_array_equal(_dense(net.matrix("length")), 10 * M)


def test_a_bep017_folder_finds_the_declared_weights_without_regard_to_case(bep017_counts):
    for net in (
        Network.from_bids(bep017_counts, primary_weight="STREAMLINECOUNT"),
        Network(bids_dir=str(bep017_counts), primary_weight="streamlinecount"),
    ):
        np.testing.assert_array_equal(_dense(net.matrix("weight")), M)
        np.testing.assert_array_equal(_dense(net.matrix("length")), 10 * M)


def test_a_network_whose_weights_are_another_edge_reads_its_bep017_export_back(tmp_path):
    net = Network(iri=HANSEN)
    net.to_bep017(tmp_path)
    back = Network.from_bids(tmp_path, primary_weight="sc")
    own = _dense(net.matrix("weight", apply_transforms=False))
    np.testing.assert_array_equal(_dense(back.matrix("weight")), own)
    assert back.matrix("length") is None


def test_a_save_writes_the_weights_under_the_declared_name(tmp_path):
    net = Network(number_of_nodes=3, primary_weight="sc")
    net.set_matrix("weight", M)
    net.save(tmp_path / "net.yaml")
    with h5py.File(tmp_path / "net.h5") as f:
        assert set(f["edges"]) == {"sc"}
    again = Network.from_file(tmp_path / "net.yaml")
    assert again.primary_weight == "sc"
    np.testing.assert_array_equal(_dense(again.matrix("weight")), M)


def test_setting_the_weights_of_a_network_that_declares_them_sets_the_declared_edge(tmp_path):
    net = Network(iri=HANSEN)
    net.matrix("weight")
    new = np.full((100, 100), 7.0)
    net.set_matrix("weight", new)
    np.testing.assert_array_equal(_dense(net.matrix("weight")), new)
    np.testing.assert_array_equal(_dense(net.matrix("sc")), new)
    net.save(tmp_path / "hansen.yaml")
    np.testing.assert_array_equal(_dense(Network.from_file(tmp_path / "hansen.yaml").matrix("weight")), new)


def test_a_curated_network_saved_and_read_back_keeps_its_declared_weights(tmp_path):
    net = Network(iri=HANSEN)
    weights = _dense(net.matrix("weight"))
    net.save(tmp_path / "hansen.yaml")
    again = Network.from_file(tmp_path / "hansen.yaml")
    assert again.primary_weight == "sc"
    np.testing.assert_array_equal(_dense(again.matrix("weight")), weights)
