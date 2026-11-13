"""``SimulationExperiment.dataset_reconcile_indices`` gives no gather only when the first subject's file is not there; any other error it meets raises, so codegen never compares unaligned observables in silence."""

from __future__ import annotations

import numpy as np
import pytest

from tvbo import Network, SimulationExperiment

pytestmark = pytest.mark.backend_core

QUERY = {"atlas": "Toy", "desc": "FC", "suffix": "relmat"}
LABELS = ["A", "B"]


def _experiment(root, model, subject="01"):
    """An experiment fitting *model* to the FC of *subject* under *root*, its nodes reconciled by label."""
    return SimulationExperiment(
        network=model,
        dataset={"dataset_id": "toy", "bids_root": str(root), "subjects": [{"subject_id": subject}]},
        observations={"fc_target": {"source": ["dataset.subject.fc"], "query": QUERY, "reconcile": "by_label"}},
    )


@pytest.fixture
def root(tmp_path):
    """One subject's FC on the nodes ``B`` and ``A``, in that order."""
    w = np.ones((2, 2)) - np.eye(2)
    folder = tmp_path / "sub-01" / "func"
    folder.mkdir(parents=True)
    Network.from_matrix(w, w, labels=LABELS[::-1]).save(folder / "sub-01_atlas-Toy_desc-FC_relmat.yaml")
    return tmp_path


def _model():
    w = np.ones((2, 2)) - np.eye(2)
    return Network.from_matrix(w, w, labels=LABELS)


def test_the_gather_follows_the_labels(root):
    assert _experiment(root, _model()).dataset_reconcile_indices() == {"fc_target": [0, 1]}


def test_a_subject_whose_file_is_not_there_gives_no_gather(root):
    assert _experiment(root, _model(), subject="02").dataset_reconcile_indices() == {}


def test_a_model_network_naming_an_atlas_several_records_share_raises_the_refusal(root):
    """``Schaefer2018`` without ``seg`` and ``scale`` names twenty atlases; the registry's refusal reaches the caller instead of reading as no gather."""
    model = _model()
    model.parcellation = {"atlas": {"name": "Schaefer2018"}}
    with pytest.raises(ValueError, match="Several curated BrainAtlas records answer to 'Schaefer2018'"):
        _experiment(root, model).dataset_reconcile_indices()
