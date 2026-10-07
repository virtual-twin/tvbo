"""A BIDS connectome arrives with its units declared; TVBO now reads them.

`to_bids` has always written `MeasureUnits` into each relmat sidecar from the edge's `unit`. Nothing read it back, so a network that round-tripped through BIDS came home unitless and fell to the `mm` default whatever its sidecar said — and `distance_unit` is what divides `conduction_speed` to give delays, so the default being wrong is not a labelling problem, it is a thousandfold error in every delay.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tvbo.classes.network import Network

SHIPPED = "tvbo/database/networks/bids/dk_average"


@pytest.fixture
def bids_dir(tmp_path):
    """A two-measure BEP017 directory whose lengths are declared in metres."""
    weights = np.eye(3)[::-1]
    lengths = np.full((3, 3), 0.05)
    for measure, matrix, unit in (
        ("streamlineCount", weights, "arbitrary"),
        ("tractLength", lengths, "m"),
    ):
        stem = tmp_path / f"atlas-Probe_meas-{measure}_relmat"
        np.savetxt(f"{stem}.dense.tsv", matrix, delimiter="\t")
        (tmp_path / f"{stem.name}.json").write_text(json.dumps({"RelationshipMeasure": measure, "MeasureUnits": unit}))
    return tmp_path


def test_the_declared_length_unit_reaches_the_delay_conversion(bids_dir):
    """Lengths in metres against a speed in mm/ms: delays are 1000x, exactly.

    This is the whole point of reading the field. Left at the `mm` default, the same connectome yields delays a thousand times too short, and every number downstream stays plausible.
    """
    network = Network.from_bids(bids_dir, atlas="Probe")

    assert str(network.distance_unit) == "m"
    assert network._unit_conversion_factor("ms") == 1000.0


def test_every_measures_unit_is_recorded_not_just_the_length(bids_dir):
    """`arbitrary` normalises onto the enum rather than being dropped."""
    network = Network.from_bids(bids_dir, atlas="Probe")

    assert network._bids_measure_units == {
        "streamlineCount": "arbitrary_unit",
        "tractLength": "m",
    }


def test_an_explicit_distance_unit_still_wins(bids_dir):
    """The sidecar is a default, not an override — the caller stays in charge."""
    network = Network.from_bids(bids_dir, atlas="Probe", distance_unit="cm")

    assert str(network.distance_unit) == "cm"


def test_a_non_length_unit_is_refused_rather_than_believed(bids_dir):
    """A dataset with no tract lengths puts a count second; it is not a distance.

    Accepting `arbitrary` as `distance_unit` would divide a speed by a count and produce delays in a unit that means nothing — worse than the `mm` default, which is at least wrong in a known direction.
    """
    network = Network.from_bids(bids_dir, atlas="Probe", structural_measures=["tractLength", "streamlineCount"])

    assert str(network.distance_unit) == "mm"


def test_a_sidecar_without_the_field_changes_nothing(bids_dir):
    """Most datasets in the wild declare no units at all."""
    for sidecar in bids_dir.glob("*.json"):
        sidecar.write_text(json.dumps({"RelationshipMeasure": "x"}))

    network = Network.from_bids(bids_dir, atlas="Probe")

    assert str(network.distance_unit) == "mm"
    assert network._bids_measure_units == {}


def test_the_shipped_connectome_declares_what_it_always_assumed():
    """`dk_average` says `mm`, which is the default — so nothing moves today.

    Worth asserting: it is the reason this change is provably inert on the shipped database while being live for any dataset that says something else.
    """
    network = Network.from_bids(
        SHIPPED,
        structural_measures=["streamlineCount", "tractLength"],
        observational_measures=["BoldCorrelation"],
    )

    assert str(network.distance_unit) == "mm"
    assert network._bids_measure_units == {
        "streamlineCount": "arbitrary_unit",
        "tractLength": "mm",
        "BoldCorrelation": "dimensionless",
    }
