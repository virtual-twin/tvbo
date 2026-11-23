"""tvb's result carries each observation a run reads from data rather than records, with the value tvboptim's result carries.

A network measure (``network.observations.<measure>``), a pipeline-less connectome matrix (``network.weight``, ``network.length``) and, while a subject is active, a dataset target (``dataset.subject.<measure>``) are bound on the host, not computed: tvboptim hands them to its generated code and returns them under their names, where tvb left them out or, for a dataset target, recorded the raw states under its name. Each case runs the delayed probe on both backends and compares tvb's value with tvboptim's and with the matrix the case declares; the renamed copy of each names its observation and its measure otherwise, so nothing keys on a name. An observation that computes over a network source through a pipeline is refused by name, since the host binds data and evaluates nothing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import Network, SimulationExperiment

from .test_tvb_delayed_clock import _settled

MATRIX = np.array([[1.0, 0.3], [0.7, 1.0]])
"""The network measure's matrix, asymmetric so that a transposed read shows."""

SUBJECT = {"weight": np.array([[0.0, 0.3], [0.3, 0.0]]), "length": np.array([[0.0, 2.5], [2.5, 0.0]])}
"""Subject 01's matrices by name, on the probe's nodes ``r0`` and ``r1``."""

QUERY = {"atlas": "Toy", "desc": "FC", "suffix": "relmat"}

BOUND = {
    "measure": (
        "fc_target",
        {"source": ["network.observations.BoldCorrelation"], "dims": ["node_i", "node_j"]},
        "measure",
        "BoldCorrelation",
    ),
    "measure_renamed": (
        "empirical",
        {"source": ["network.observations.Coherence"], "dims": ["node_i", "node_j"]},
        "measure",
        "Coherence",
    ),
    "connectome": ("sc", {"source": ["network.weight"]}, "connectome", "weight"),
    "connectome_renamed": ("tracts", {"source": ["network.length"]}, "connectome", "length"),
    "dataset": ("fc_subject", {"source": ["dataset.subject.weight"], "query": QUERY}, "dataset", "weight"),
    "dataset_renamed": ("lengths_subject", {"source": ["dataset.subject.length"], "query": QUERY}, "dataset", "length"),
}
"""Per case, the observation's name, its declaration, the kind of data it reads and the name of that data: EI_Tuning's ``fc_target`` declaration, the probe's own weights and lengths, and subject 01's matrices."""

SCALED = {"name": "scaled", "equation": {"rhs": "2 * X"}}
REFUSED = {
    "network_pipeline": (
        {"sc_scaled": {"source": ["network.weight"], "pipeline": [SCALED]}},
        "the observation 'sc_scaled', which declares the source network.weight and the pipeline scaled",
    ),
    "network_pipeline_renamed": (
        {"tracts_scaled": {"source": ["network.length"], "pipeline": [SCALED]}},
        "the observation 'tracts_scaled', which declares the source network.length and the pipeline scaled",
    ),
}
"""Per case, an observation computed over a connectome matrix and what the refusal names."""


def _experiment(case, root):
    """The delayed probe observed by *case*'s observation alone, reading its data under *root*, and the matrix that observation holds."""
    name, observation, kind, label = BOUND[case]
    spec = copy.deepcopy(_settled(0.0))
    spec["observations"] = {name: copy.deepcopy(observation)}
    if kind == "measure":
        spec["network"]["observational_measures"] = [label]
    if kind == "dataset":
        folder = root / "sub-01" / "func"
        folder.mkdir(parents=True, exist_ok=True)
        Network.from_matrix(SUBJECT["weight"], SUBJECT["length"], labels=["r0", "r1"]).save(
            folder / "sub-01_atlas-Toy_desc-FC_relmat.yaml"
        )
        spec["dataset"] = {"dataset_id": "toy", "bids_root": str(root), "subjects": [{"subject_id": "01"}]}
    experiment = SimulationExperiment(**spec)
    if kind == "measure":
        experiment.network.set_array(label, MATRIX)
        return experiment, MATRIX
    if kind == "connectome":
        return experiment, np.asarray(experiment.network.matrix(label, format="dense"))
    return experiment, SUBJECT[label]


@pytest.mark.backend_tvb
@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("case", list(BOUND))
def test_tvb_carries_what_tvboptim_binds_from_data(case, tmp_path):
    """The value on tvb is the one on tvboptim and the declared matrix, which therefore survives the run, under the axes tvboptim names."""
    pytest.importorskip("tvb")
    name, _, kind, _ = BOUND[case]
    run_kwargs = {"active_subject": "01"} if kind == "dataset" else {}
    observed = {}
    for backend in ("tvboptim", "tvb"):
        experiment, expected = _experiment(case, tmp_path)
        observed[backend] = experiment.run(backend, **run_kwargs).integration.observations[name]
    np.testing.assert_array_equal(np.asarray(observed["tvb"]), expected)
    np.testing.assert_array_equal(np.asarray(observed["tvb"]), np.asarray(observed["tvboptim"]))
    assert getattr(observed["tvb"], "dims", None) == getattr(observed["tvboptim"], "dims", None)


@pytest.mark.backend_tvb
@pytest.mark.parametrize("case", list(REFUSED))
def test_tvb_refuses_an_observation_computed_over_network_data(case):
    """The render names the observation, its network source and its pipeline, instead of leaving it out of the result."""
    pytest.importorskip("tvb")
    observations, named = REFUSED[case]
    spec = copy.deepcopy(_settled(0.0))
    spec["observations"] = copy.deepcopy(observations)
    with pytest.raises(NotImplementedError, match=rf"^the tvb backend has no monitor for {named}\. "):
        SimulationExperiment(**spec).render_code("tvb")
