"""A ``Phenotype`` whose ``data_file`` ends in ``.tsv`` writes and reads the BIDS phenotype pair.

The table is what a BIDS tool reads (``participant_id`` as ``sub-<id>``, one column per measure, ``n/a`` for a missing value) and the ``.json`` column sidecar is the ``measure_specs`` one to one: ``Description``, ``Units`` as the curated unit's UCUM code (the unit as written when QUDT has none) and ``TermURL`` as the ``iri`` expanded to a URL. The h5 companion is untouched by the choice.
"""

import json

import numpy as np
import pytest

from tvbo.classes.phenotype import Phenotype


def _phenotype(data_file):
    return Phenotype(
        dataset_id="cohort",
        data_file=data_file,
        subjects=["01", "ctrl03"],
        measures=["age", "iq"],
        measure_specs=[
            {"name": "age", "iri": "loinc:30525-0", "unit": "year", "description": "Age at scan"},
            {"name": "iq", "iri": "https://www.cognitiveatlas.org/concept/id/trm_521ef89ce5e84", "unit": "dimensionless"},
        ],
    )


def test_tsv_companion_round_trips_and_is_bids(tmp_path):
    scores = {"age": np.array([31.0, 0.30000000000000004]), "iq": np.array([np.nan, 112.0])}
    _phenotype("cohort.tsv").to_file(tmp_path / "cohort.yaml", values=scores)

    lines = (tmp_path / "cohort.tsv").read_text().splitlines()
    assert lines[0] == "participant_id\tage\tiq"
    assert lines[1].split("\t") == ["sub-01", "31.0", "n/a"]
    sidecar = json.loads((tmp_path / "cohort.json").read_text())
    assert sidecar == {
        "age": {"Description": "Age at scan", "Units": "a", "TermURL": "http://loinc.org/rdf/30525-0"},
        "iq": {"Units": "dimensionless", "TermURL": "https://www.cognitiveatlas.org/concept/id/trm_521ef89ce5e84"},
    }

    back = Phenotype.from_file(tmp_path / "cohort.yaml")
    np.testing.assert_array_equal(back.get("age"), scores["age"])
    np.testing.assert_array_equal(back.get("iq"), scores["iq"])
    assert back.measure_spec("age").iri == "loinc:30525-0"


def test_h5_companion_is_unchanged(tmp_path):
    scores = {"age": np.array([31.0, 45.0]), "iq": np.array([99.0, 112.0])}
    _phenotype("cohort.h5").to_file(tmp_path / "cohort.yaml", values=scores)
    assert (tmp_path / "cohort.h5").exists() and not (tmp_path / "cohort.tsv").exists()
    np.testing.assert_array_equal(Phenotype.from_file(tmp_path / "cohort.yaml").get("iq"), scores["iq"])


def test_tsv_rows_must_be_the_records_subjects(tmp_path):
    _phenotype("cohort.tsv").to_file(tmp_path / "cohort.yaml", values={"age": [1.0, 2.0], "iq": [3.0, 4.0]})
    tsv = tmp_path / "cohort.tsv"
    tsv.write_text(tsv.read_text().replace("sub-ctrl03", "sub-ctrl04"))
    with pytest.raises(ValueError, match="participant_id"):
        Phenotype.from_file(tmp_path / "cohort.yaml").get("age")


def test_unknown_companion_extension_is_refused(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        _phenotype("cohort.csv").to_file(tmp_path / "cohort.yaml", values={"age": [1.0, 2.0], "iq": [3.0, 4.0]})


def test_values_in_memory_are_checked_and_written(tmp_path):
    fields = _phenotype("cohort.tsv").model_dump(exclude_none=True)
    with pytest.raises(ValueError, match="expected shape"):
        Phenotype.from_values({"age": [31.0], "iq": [1.0, 2.0]}, **fields)

    phenotype = Phenotype.from_values({"age": [31.0, 45.0], "iq": [100.0, 112.0]}, **fields)
    np.testing.assert_array_equal(phenotype.get("iq"), [100.0, 112.0])
    phenotype.to_file(tmp_path / "cohort.yaml")
    np.testing.assert_array_equal(Phenotype.from_file(tmp_path / "cohort.yaml").get("age"), [31.0, 45.0])
