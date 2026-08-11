"""BIDS (BEP034) export of a TimeSeries."""

import json

import numpy as np
import pytest

from tvbo import Network
from tvbo.data.types import TimeSeries


@pytest.fixture
def unlabelled_series():
    """A three-node series whose space labels are integers, as an unnamed Network yields."""
    network = Network(number_of_nodes=3)
    data = np.zeros((20, 1, 3, 1))
    return TimeSeries(
        time=np.arange(20, dtype=float),
        data=data,
        network=network,
        sample_period=1.0,
        labels_dimensions={"State Variable": ["V"], "Space": list(np.arange(3))},
    )


def read_sidecars(root, kind):
    """Every JSON sidecar written under ``root/<subject>/<kind>``."""
    return [json.loads(p.read_text()) for p in root.rglob(f"{kind}/*.json")]


def test_integer_space_labels_export_as_strings(tmp_path, unlabelled_series):
    """NodeLabels is typed list[str], so integer node indices must be coerced.

    A Network built without region names labels its nodes with numpy integers.
    Passing those straight through made NetworkSidecar raise a pydantic
    string_type ValidationError, which broke every export of an unnamed network.
    """
    unlabelled_series.to_bids(output_dir=str(tmp_path), subject="01")

    sidecars = [s for s in read_sidecars(tmp_path, "net") if "NodeLabels" in s]
    assert sidecars, "no net/ sidecar carrying NodeLabels was written"
    for sidecar in sidecars:
        assert sidecar["NodeLabels"] == ["0", "1", "2"]
        assert all(isinstance(label, str) for label in sidecar["NodeLabels"])
