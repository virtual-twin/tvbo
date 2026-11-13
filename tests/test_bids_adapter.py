"""``tvbo.adapters.bids.get_unique_entity_values``, public API with a page in the reference, reads each value an entity takes in a BIDS layout once."""

from __future__ import annotations

import json

import pytest
from bids import BIDSLayout

from tvbo.adapters.bids import get_unique_entity_values

pytestmark = pytest.mark.backend_core


def test_each_value_an_entity_takes_is_read_once(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "two subjects", "BIDSVersion": "1.10.0"}), encoding="utf-8"
    )
    for subject, session in [("01", "a"), ("01", "b"), ("02", "a")]:
        anat = tmp_path / f"sub-{subject}" / f"ses-{session}" / "anat"
        anat.mkdir(parents=True)
        (anat / f"sub-{subject}_ses-{session}_T1w.nii.gz").touch()
    layout = BIDSLayout(tmp_path, validate=False)
    assert get_unique_entity_values(layout, "subject") == {"01", "02"}
    assert get_unique_entity_values(layout, "session") == {"a", "b"}
    assert get_unique_entity_values(layout, "acquisition") == set()
