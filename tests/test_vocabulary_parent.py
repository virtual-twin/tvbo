"""A vocabulary declares its hierarchy with ``Parameter.parent``: the network parameter that gives each of its entries a parent in a coarser vocabulary, one code per entry into the vocabulary its ``levels`` names, ``-1`` for an entry with no parent.

A hierarchy that does not do this is refused when the network is resolved, naming what is wrong, and a save keeps it in every format, as values or as arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo import Network

REGIONS = ["precentral", "superiorfrontal", "cuneus", "insula"]
LOBES = ["frontal", "occipital"]
CODES = [0, 0, 1, -1]


def _network(**parameters) -> Network:
    """Four nodes, each region its own, whose region vocabulary sits within the lobes; *parameters* replace the declared ones by name."""
    declared = {
        "lobes": {"value": LOBES},
        "region_lobe": {"value": CODES, "levels": "lobes"},
        "regions": {"value": REGIONS, "parent": "region_lobe"},
        **parameters,
    }
    net = Network(
        nodes=[{"id": i, "label": name} for i, name in enumerate(REGIONS)],
        edges=[{"label": "weight"}],
        parameters={name: spec for name, spec in declared.items() if spec is not None},
        node_template={"parameters": {"region": {"measure": "nodes/region", "levels": "regions"}}},
    )
    net.set_matrix("weight", np.ones((4, 4)) - np.eye(4))
    net.set_array("nodes/region", np.arange(4, dtype="int32"))
    return net


def _as_arrays() -> Network:
    """The hierarchy of :func:`_network` with each vocabulary and the codes held as arrays its ``measure`` names."""
    net = _network(
        lobes={"measure": "nodes/lobes"},
        region_lobe={"measure": "nodes/region_lobe", "levels": "lobes"},
        regions={"measure": "nodes/regions", "parent": "region_lobe"},
    )
    net.set_array("nodes/lobes", np.array(LOBES, dtype="S9"))
    net.set_array("nodes/region_lobe", np.array(CODES, dtype="int32"))
    net.set_array("nodes/regions", np.array(REGIONS, dtype="S15"))
    return net


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
@pytest.mark.parametrize("make", [_network, _as_arrays], ids=["values", "arrays"])
def test_a_hierarchy_of_vocabularies_reads_back_from_a_save(tmp_path, make, binary_format):
    net = make()
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    back = Network.from_file(str(tmp_path / "n.yaml"))
    assert (back.parameters["regions"].parent, back.parameters["region_lobe"].levels) == ("region_lobe", "lobes")
    assert [int(code) for code in back._entries(back.parameters["region_lobe"])] == CODES
    assert [e.decode() if isinstance(e, bytes) else str(e) for e in back._entries(back.parameters["regions"])] == REGIONS


@pytest.mark.parametrize(
    ("changed", "refusal"),
    [
        (
            {"region_lobe": None},
            r"names 'region_lobe' as the parameter giving its entries their parents.*no parameter 'region_lobe' with `levels`",
        ),
        ({"region_lobe": {"value": CODES}}, r"no parameter 'region_lobe' with `levels`"),
        (
            {"region_lobe": {"value": CODES[:3], "levels": "lobes"}},
            r"gives 3 parents to the 4 entries of the vocabulary 'regions'",
        ),
        (
            {"region_lobe": {"value": [0, 2, 1, -2], "levels": "lobes"}},
            r"holds the codes \[-2\.0, 2\.0\], which name no entry of the 2 of its vocabulary 'lobes'",
        ),
        ({"region_lobe": {"value": [0, 0.5, 1, -1], "levels": "lobes"}}, r"holds the codes \[0\.5\]"),
        (
            {"region_lobe": {"value": [0, float("nan"), 1, -1], "levels": "lobes"}},
            r"The parameter 'region_lobe' gives the entry 'superiorfrontal' \(1\) of the vocabulary 'regions' the parent nan, which is no code",
        ),
        (
            {"region_lobe": {"value": [0, "a", 1, -1], "levels": "lobes"}},
            r"the entry 'superiorfrontal' \(1\) .* the parent 'a', which is no code",
        ),
    ],
    ids=["no-parameter", "no-levels", "too-few-codes", "outside", "not-whole", "nan", "text"],
)
def test_a_hierarchy_that_does_not_give_each_entry_a_parent_is_refused(changed, refusal):
    with pytest.raises(ValueError, match=refusal):
        _network(**changed)


@pytest.mark.parametrize(
    ("codes", "refusal"),
    [
        (np.array([0, 0, 1, 2], dtype="int32"), r"holds the codes \[2\.0\]"),
        (np.array([0, np.nan, 1, -1]), r"the parent nan, which is no code"),
    ],
    ids=["outside", "nan"],
)
def test_a_hierarchy_held_as_arrays_set_after_the_network_is_built_is_checked_when_saved(tmp_path, codes, refusal):
    """The arrays are not held when the network is resolved, so the check falls to the save, which writes nothing."""
    net = _as_arrays()
    net.set_array("nodes/region_lobe", codes)
    with pytest.raises(ValueError, match=refusal):
        net.save(tmp_path / "n.yaml")
    assert not list(tmp_path.iterdir())


def test_a_network_taking_another_s_categorical_attribute_takes_the_vocabularies_above_it():
    """As a network named by its atlas takes the curated network's declarations: the region codes, the region vocabulary, the codes giving each region its lobe and the lobes."""
    target = Network(nodes=[{"id": i, "label": name} for i, name in enumerate(REGIONS)])
    target._take_declarations(_network(), with_arrays=True)
    assert {"regions", "region_lobe", "lobes"} <= target.parameters.keys()
    target._check_levels()
