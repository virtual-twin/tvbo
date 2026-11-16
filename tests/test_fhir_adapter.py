"""The FHIR adapter reads a hospital Bundle into a Dataset and a Phenotype, and writes a Phenotype back as a transaction Bundle that base R4 accepts.

Every Bundle the adapter writes is parsed by ``fhir.resources`` (R4B), which checks element names, types and cardinalities. The content checks here are what that cannot see: the research label on every resource, the preliminary status, the subject attribution and the round trip of the values.
"""

import json

import numpy as np
import pytest

from tvbo.adapters.fhir import RESEARCH_LABEL, from_fhir, to_fhir
from tvbo.classes.phenotype import Phenotype

r4b = pytest.importorskip("fhir.resources.R4B.bundle")


def _observation(code, subject, value, unit="a", display="Age", system="http://loinc.org"):
    return {
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": system, "code": code, "display": display}]},
            "subject": {"reference": subject},
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
        }
    }


def _bundle(*observations):
    patients = [
        {
            "fullUrl": "urn:uuid:p1",
            "resource": {
                "resourceType": "Patient",
                "id": "p1",
                "identifier": [{"system": "urn:example:pseudonym", "value": "01"}],
                "gender": "female",
            },
        },
        {"fullUrl": "urn:uuid:p2", "resource": {"resourceType": "Patient", "id": "p2", "identifier": [{"value": "ctrl03"}]}},
    ]
    return {"resourceType": "Bundle", "type": "collection", "entry": [*patients, *observations]}


def _hospital_bundle():
    sex = {
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "76689-9"}]},
            "subject": {"reference": "Patient/p1"},
            "valueCodeableConcept": {"text": "female"},
        }
    }
    return _bundle(_observation("30525-0", "Patient/p1", 31), _observation("30525-0", "urn:uuid:p2", 45), sex)


def test_bundle_becomes_dataset_and_phenotype(tmp_path, caplog):
    dataset, phenotype = from_fhir(_hospital_bundle(), dataset_id="ward")

    assert list(dataset.subjects) == ["01", "ctrl03"]
    assert phenotype.subjects == ["01", "ctrl03"] and phenotype.measures == ["30525-0"]
    spec = phenotype.measure_spec("30525-0")
    assert (spec.iri, spec.unit, spec.description) == ("loinc:30525-0", "year", "Age")
    np.testing.assert_array_equal(phenotype.get("30525-0"), [31.0, 45.0])
    assert "76689-9" in caplog.text and "not a quantity" in caplog.text

    phenotype.to_file(tmp_path / "ward.yaml")
    assert (tmp_path / "ward.tsv").read_text().splitlines()[1] == "sub-01\t31.0"
    sidecar = json.loads((tmp_path / "ward.json").read_text())["30525-0"]
    assert sidecar == {"Description": "Age", "Units": "a", "TermURL": "http://loinc.org/rdf/30525-0"}


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ((_observation("30525-0", "Patient/p1", 31, system="http://hospital.example/codes"),), "hospital.example"),
        ((_observation("30525-0", "Patient/nobody", 31),), "no Patient"),
        ((_observation("30525-0", "Patient/p1", 31), _observation("30525-0", "Patient/p1", 32)), "two values"),
        ((_observation("30525-0", "Patient/p1", 31), _observation("30525-0", "Patient/p2", 380, unit="mo")), "two different"),
    ],
)
def test_ambiguous_or_foreign_observations_are_refused(observations, message):
    with pytest.raises(ValueError, match=message):
        from_fhir(_bundle(*observations), dataset_id="ward")


def test_display_variants_share_a_measure_and_an_absent_value_is_skipped():
    absent = _observation("30525-0", "urn:uuid:p2", None, display="Age at visit")
    del absent["resource"]["valueQuantity"]["value"]
    observations = (
        _observation("30525-0", "Patient/p1", 31),
        _observation("8302-2", "Patient/p2", 170, unit="cm", display="Height"),
    )
    later = _observation("30525-0", "urn:uuid:p2", 45, display="Age at visit")
    _, phenotype = from_fhir(_bundle(*observations, absent, later), dataset_id="ward")
    np.testing.assert_array_equal(phenotype.get("30525-0"), [31.0, 45.0])
    assert phenotype.measure_spec("30525-0").description == "Age"


def _condition(subject, *codings, status=None):
    condition = {"resourceType": "Condition", "subject": {"reference": subject}, "code": {"coding": list(codings)}}
    if status:
        condition["verificationStatus"] = {"coding": [{"code": status}]}
    return {"resource": condition}


ICD10GM = {"system": "http://fhir.de/CodeSystem/bfarm/icd-10-gm", "version": "2025", "code": "G20.9"}
PARKINSON = {"system": "http://snomed.info/sct", "code": "49049000"}


def test_conditions_become_subject_diagnoses():
    conditions = (
        _condition("Patient/p1", ICD10GM, PARKINSON),
        _condition("urn:uuid:p1", PARKINSON),
        _condition("Patient/p2", PARKINSON, status="refuted"),
    )
    dataset, _ = from_fhir(_bundle(*conditions), dataset_id="ward")
    assert dataset.subjects["01"].diagnosis == ["sct:49049000"]
    assert dataset.subjects["ctrl03"].diagnosis is None

    with pytest.raises(ValueError, match="none of"):
        from_fhir(_bundle(_condition("Patient/p1", ICD10GM)), dataset_id="ward")
    with pytest.raises(ValueError, match="no Patient"):
        from_fhir(_bundle(_condition("Patient/nobody", PARKINSON)), dataset_id="ward")


ONE_ROW = {"30525-0": [31.0], "score": [2.0]}
TWO_ROWS = {"30525-0": [1.0, 2.0], "score": [1.0, 2.0]}


def _phenotype(subjects=("01",), values=ONE_ROW, **kwargs):
    return Phenotype.from_values(
        values,
        dataset_id="predicted",
        data_file="predicted.tsv",
        subjects=list(subjects),
        measures=list(values),
        measure_specs=[
            {"name": "30525-0", "iri": "loinc:30525-0", "unit": "year", "description": "Predicted age"},
            {"name": "score", "iri": "sct:273249006", "unit": "dimensionless"},
        ],
        provenance={
            "date_created": "2026-11-07T10:00:00+01:00",
            "generated_by": "tvbo 0.9.0",
            "derived_from": "ward.tsv",
            "activities": [
                {
                    "name": "exp-1",
                    "iri": "tvbo:exp/predicted/exp-1",
                    "started_at": "2026-11-07T09:00:00+01:00",
                    "ended_at": "2026-11-07T10:00:00+01:00",
                }
            ],
        },
        **kwargs,
    )


def _validated(bundle):
    return [entry.resource for entry in r4b.Bundle.model_validate(bundle).entry]


def _by_type(bundle, kind):
    return [e for e in bundle["entry"] if e["resource"]["resourceType"] == kind]


def test_phenotype_becomes_a_valid_bundle_and_round_trips():
    phenotype = _phenotype(["01", "ctrl03"], {"30525-0": [31.0, 45.0], "score": [np.nan, 2.5]})
    bundle = to_fhir(phenotype, subject_system="urn:example:pseudonym")

    resources = _validated(bundle)
    assert [r.get_resource_type() for r in resources] == [
        "Patient",
        "Patient",
        "Observation",
        "Observation",
        "Observation",
        "Provenance",
    ]
    assert all(r.meta.security[0].code == RESEARCH_LABEL["code"] for r in resources)
    observations = [r for r in resources if r.get_resource_type() == "Observation"]
    assert {o.status for o in observations} == {"preliminary"}
    age = observations[0]
    assert (age.code.coding[0].system, age.code.coding[0].code) == ("http://loinc.org", "30525-0")
    assert (age.valueQuantity.value, age.valueQuantity.code, age.valueQuantity.system) == (
        31.0,
        "a",
        "http://unitsofmeasure.org",
    )
    assert age.method.text == "tvbo:exp/predicted/exp-1"
    assert age.subject.reference == bundle["entry"][0]["fullUrl"]
    assert bundle["entry"][0]["request"]["ifNoneExist"] == "identifier=urn:example:pseudonym|01"
    provenance = resources[-1]
    assert len(provenance.target) == 5 and provenance.agent[0].who.display == "tvbo 0.9.0"
    assert provenance.entity[0].what.display == "ward.tsv"
    assert to_fhir(phenotype, subject_system="urn:example:pseudonym") == bundle

    dataset, back = from_fhir(bundle, dataset_id="back")
    assert list(dataset.subjects) == ["01", "ctrl03"]
    np.testing.assert_array_equal(back.get("30525-0"), [31.0, 45.0])
    np.testing.assert_array_equal(back.get("273249006"), [np.nan, 2.5])
    assert (back.measure_spec("30525-0").unit, back.measure_spec("273249006").iri) == ("year", "sct:273249006")


def test_cohort_table_is_attributed_to_a_group():
    from tvbo.datamodel.pydantic import Cohort

    cohort = Cohort(cohort_id="patients", label="PD patients", members=["01", "ctrl03"], diagnosis=["sct:49049000"])
    bundle = to_fhir(_phenotype(["patients"], {"30525-0": [38.0], "score": [1.0]}), cohort=cohort)

    _validated(bundle)
    patients = [e["fullUrl"] for e in _by_type(bundle, "Patient")]
    (group,) = _by_type(bundle, "Group")
    assert group["resource"]["actual"] is True
    (characteristic,) = group["resource"]["characteristic"]
    assert characteristic["valueCodeableConcept"]["coding"] == [PARKINSON] and characteristic["exclude"] is False
    assert [m["entity"]["reference"] for m in group["resource"]["member"]] == patients
    assert {e["resource"]["subject"]["reference"] for e in _by_type(bundle, "Observation")} == {group["fullUrl"]}

    with pytest.raises(ValueError, match="one row"):
        to_fhir(_phenotype(["01", "ctrl03"], TWO_ROWS), cohort=cohort)


def test_report_needs_one_subject_and_a_measure_needs_a_code(tmp_path):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4")
    resources = _validated(to_fhir(_phenotype(), report=report))
    diagnostic = next(r for r in resources if r.get_resource_type() == "DiagnosticReport")
    assert len(diagnostic.result) == 2 and diagnostic.presentedForm[0].contentType == "application/pdf"
    assert diagnostic.presentedForm[0].data == b"%PDF-1.4"

    with pytest.raises(ValueError, match="one subject"):
        to_fhir(_phenotype(["01", "ctrl03"], TWO_ROWS), report=report)
    uncoded = _phenotype()
    uncoded.measure_specs[1].iri = "https://www.cognitiveatlas.org/concept/id/trm_521ef89ce5e84"
    with pytest.raises(ValueError, match="no code"):
        to_fhir(uncoded)
    uncoded.measure_specs[1].iri = "http://loinc.org/rdf/30525-0"
    with pytest.raises(ValueError, match="share the code"):
        to_fhir(uncoded)


def test_provenance_period_orders_by_time_and_needs_an_offset():
    phenotype = _phenotype()
    activities = phenotype.provenance.activities
    activities.append(type(activities[0])(name="exp-2", started_at="2026-11-07T08:30:00Z"))
    provenance = _validated(to_fhir(phenotype))[-1]
    assert provenance.occurredPeriod.start.isoformat() == "2026-11-07T09:00:00+01:00"

    phenotype.provenance.date_created = "2026-11-07"
    with pytest.raises(ValueError, match="no UTC offset"):
        to_fhir(phenotype)
