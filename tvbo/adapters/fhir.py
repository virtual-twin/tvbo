# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""HL7 FHIR R4 conversion for TVBO: a hospital ``Bundle`` becomes a ``Dataset`` and a ``Phenotype``, and a ``Phenotype`` of coded results becomes a transaction ``Bundle``.

One rule in both directions: a coded quantity is a FHIR ``Observation``, one subject and one measure value each, and a coded diagnosis is a ``Condition``. Importing reads every ``Patient`` as a ``Subject``, every ``Condition`` as one of its ``diagnosis`` terms and every ``Observation`` with a ``valueQuantity`` as one cell of a ``Phenotype``, whose ``MeasureSpec`` carries the code as ``iri`` and the UCUM unit as ``unit``. Exporting writes one ``Observation`` per cell, attributed to a ``Patient`` per subject or, for a cohort-level table, to a ``Group`` of the cohort's members that states the cohort's ``diagnosis`` as its characteristic, together with a ``Provenance`` from the phenotype's own provenance record and, on request, a ``DiagnosticReport``. Every exported resource is ``status: preliminary`` and labelled healthcare research (``HRESCH``), since TVBO is not a certified medical device.

FHIR JSON is plain JSON, so this module reads and writes dicts and has no runtime dependency; the test suite validates what it writes with ``fhir.resources``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CODE_SYSTEMS", "UCUM_SYSTEM", "RESEARCH_LABEL", "from_fhir", "to_fhir"]

CODE_SYSTEMS = {"loinc": "http://loinc.org", "sct": "http://snomed.info/sct"}
"""The FHIR ``system`` URI of each terminology prefix the schema declares; the RDF IRI of the same code is the prefix's expansion."""

UCUM_SYSTEM = "http://unitsofmeasure.org"

RESEARCH_LABEL = {
    "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
    "code": "HRESCH",
    "display": "healthcare research",
}
"""The purpose-of-use security label every exported resource carries in ``meta.security``."""

_SYSTEM_PREFIXES = {system: prefix for prefix, system in CODE_SYSTEMS.items()}

_DIAGNOSIS = {"coding": [{"system": CODE_SYSTEMS["sct"], "code": "439401001"}], "text": "diagnosis"}
"""The ``Group.characteristic.code`` a cohort's diagnosis is stated under."""

_NOT_A_DIAGNOSIS = {"refuted", "entered-in-error"}


def from_fhir(bundle: dict | str | os.PathLike, *, dataset_id: str):
    """The ``Dataset`` and ``Phenotype`` a FHIR ``Bundle`` holds, as ``(dataset, phenotype)``.

    Every ``Patient`` becomes a ``Subject`` named by its first ``identifier.value``, in order of appearance, and every ``Condition`` adds its code to that subject's ``diagnosis``, once per code; a ``Condition`` whose ``verificationStatus`` is ``refuted`` or ``entered-in-error`` states no diagnosis and is skipped. Every ``Observation`` with a ``valueQuantity`` becomes one cell of the phenotype: the measure is named by the code, its ``MeasureSpec`` carries the code as a CURIE under :data:`CODE_SYSTEMS` (``loinc:30525-0``), the quantity's UCUM code as ``unit`` and the display of the code's first coding as ``description``, which a later display does not contradict. A cell no observation fills is ``nan``. The phenotype's values are in memory; ``to_file`` writes them with its BIDS ``.tsv`` companion, which ``data_file`` names after the dataset.

    A code is read from the first coding under a system :data:`CODE_SYSTEMS` lists, so a local or ICD-10-GM coding beside a SNOMED CT one is passed over. The quantity's UCUM code is read back to the curated unit it names (``a`` is ``year``) and recorded as written when it names none. A ``Condition`` or ``Observation`` whose subject is no ``Patient`` of the bundle, or whose code has no coding under those systems, is refused by name, as is an ``Observation`` whose code arrives with two different codings or units or that gives a subject a second value for the same code. An ``Observation`` without a ``valueQuantity`` value (a coded value such as sex assigned at birth, a panel, or a quantity absent for a ``dataAbsentReason``) is no quantity and is skipped, logged at WARNING with its code.
    """
    from tvbo.classes.phenotype import Phenotype
    from tvbo.datamodel.pydantic import Dataset, Subject
    from tvbo.utils.units import unit_from_ucum

    document = _load(bundle)
    entries = [e for e in document.get("entry", []) if "resource" in e]
    patients: dict[str, str] = {}
    for entry in entries:
        resource = entry["resource"]
        if resource.get("resourceType") == "Patient":
            subject_id = _identifier(resource)
            for ref in (f"Patient/{resource['id']}" if resource.get("id") else None, entry.get("fullUrl")):
                if ref:
                    patients[ref] = subject_id
    subjects = list(dict.fromkeys(patients.values()))
    diagnoses: dict[str, dict[str, None]] = {}
    cells: dict[str, dict[str, float]] = {}
    specs: dict[str, dict] = {}
    for entry in entries:
        resource = entry["resource"]
        kind = resource.get("resourceType")
        if kind == "Condition":
            status = {c.get("code") for c in resource.get("verificationStatus", {}).get("coding", [])}
            if status & _NOT_A_DIAGNOSIS:
                logger.info("Condition %s is %s and states no diagnosis", resource.get("id"), sorted(status))
                continue
            diagnoses.setdefault(_subject(resource, patients), {})[_curie(_coding(resource, _codings(resource)))] = None
            continue
        if kind != "Observation":
            continue
        codings = _codings(resource)
        if "value" not in resource.get("valueQuantity", {}):
            logger.warning(
                "Observation %s has no valueQuantity value and is skipped: it is not a quantity", codings[0]["code"]
            )
            continue
        coding = _coding(resource, codings)
        subject = _subject(resource, patients)
        quantity = resource["valueQuantity"]
        code = quantity.get("code")
        spec = {"iri": _curie(coding), "unit": unit_from_ucum(code) or code or quantity.get("unit")}
        known = specs.setdefault(coding["code"], {**spec, "description": coding.get("display")})
        if {k: known[k] for k in spec} != spec:
            raise ValueError(
                f"Observation code {coding['code']} arrives with two different codings or units: {known} and {spec}"
            )
        column = cells.setdefault(coding["code"], {})
        if subject in column:
            raise ValueError(
                f"subject {subject!r} has two values for {coding['code']}: {column[subject]} and {quantity['value']}"
            )
        column[subject] = float(quantity["value"])

    phenotype = Phenotype.from_values(
        {m: [cells[m].get(s, np.nan) for s in subjects] for m in specs},
        dataset_id=dataset_id,
        data_file=f"{dataset_id}.tsv",
        subjects=subjects,
        measures=list(specs),
        measure_specs=[{"name": m, **spec} for m, spec in specs.items()],
        category="clinical",
    )
    dataset = Dataset(
        dataset_id=dataset_id,
        subjects={s: Subject(subject_id=s, diagnosis=list(diagnoses.get(s, {})) or None) for s in subjects},
    )
    return dataset, phenotype


def to_fhir(phenotype, *, cohort=None, subject_system: str | None = None, report: str | os.PathLike | None = None) -> dict:
    """A transaction ``Bundle`` of the coded quantities *phenotype* holds.

    Each subject becomes a ``Patient`` whose identifier is the subject id under *subject_system*, the namespace the receiving system knows the pseudonyms by; with a *subject_system* the ``Patient`` is a conditional create, so a server that already holds the pseudonym links to it instead of creating a second patient. Each cell that is not ``nan`` becomes an ``Observation``: ``code`` from the measure's ``iri``, a CURIE under :data:`CODE_SYSTEMS` or the URL it expands to, ``valueQuantity`` with the unit's UCUM code, and ``subject`` the row's ``Patient``. With *cohort*, the table is cohort-level: its one row is the cohort's, listed in ``subjects`` under the cohort id, and the ``Observation``s are attributed to a ``Group`` of the cohort's members instead. *report* names a rendered report file, embedded in a ``DiagnosticReport`` over every ``Observation``; a report is one subject's, so it needs a table of one row. The phenotype's ``provenance`` record becomes a ``Provenance`` over every resource of the bundle.

    A measure without a ``MeasureSpec``, or whose ``iri`` is no code under :data:`CODE_SYSTEMS`, is refused by name: there is no ``Observation.code`` to write for it. So are two measures with one code, which no receiver could tell apart.
    """
    entries: list[dict] = []
    subjects = list(phenotype.subjects)
    if cohort is None:
        subject_refs = {s: _add_patient(entries, s, subject_system) for s in subjects}
    else:
        if subjects != [cohort.cohort_id]:
            raise ValueError(
                f"a cohort-level table has one row, the cohort's ({cohort.cohort_id!r}); this one lists {subjects}"
            )
        members = [_add_patient(entries, m, subject_system) for m in cohort.members]
        subject_refs = {cohort.cohort_id: _add(entries, _group(cohort, members))}

    method = _method(phenotype.provenance)
    observations = []
    codes: dict[tuple, str] = {}
    for measure in phenotype.measures:
        spec = phenotype.measure_spec(measure)
        if spec is None:
            raise ValueError(f"measure {measure!r} declares no MeasureSpec, so it has no Observation.code")
        coding = _fhir_coding(spec.iri, f"measure {spec.name!r}")
        if spec.description:
            coding["display"] = spec.description
        key = (coding["system"], coding["code"])
        if codes.setdefault(key, measure) != measure:
            raise ValueError(
                f"measures {codes[key]!r} and {measure!r} share the code {spec.iri}, so their Observations cannot be told apart"
            )
        unit = _quantity_unit(spec.unit)
        for subject, value in zip(subjects, phenotype.values[measure], strict=True):
            if not np.isnan(value):
                observation = _observation(coding, {"value": float(value), **unit}, subject_refs[subject], method)
                observations.append(_add(entries, observation))

    if report is not None:
        if len(subjects) != 1:
            raise ValueError(f"a DiagnosticReport is one subject's; this table has {len(subjects)} rows")
        _add(entries, _diagnostic_report(phenotype, subject_refs[subjects[0]], observations, Path(report)))
    targets = [e["fullUrl"] for e in entries]
    _add(entries, _provenance(phenotype.provenance, targets))
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def _load(bundle) -> dict:
    """*bundle* as a dict: a mapping as given, a path read as JSON."""
    if isinstance(bundle, dict):
        return bundle
    return json.loads(Path(bundle).read_text(encoding="utf-8"))


def _identifier(patient: dict) -> str:
    """The first identifier value of *patient*, the pseudonym a ``Subject`` is named by."""
    for identifier in patient.get("identifier", []):
        if identifier.get("value"):
            return str(identifier["value"])
    raise ValueError(f"Patient {patient.get('id')!r} has no identifier, so it names no subject")


def _codings(resource: dict) -> list[dict]:
    """The codings of *resource*'s code that carry a code; a resource with none is refused."""
    codings = [c for c in resource.get("code", {}).get("coding") or [] if c.get("code")]
    if not codings:
        raise ValueError(f"{resource.get('resourceType')} {resource.get('id')!r} has no code")
    return codings


def _coding(resource: dict, codings: list[dict]) -> dict:
    """The first of *resource*'s *codings* under a system :data:`CODE_SYSTEMS` lists."""
    for coding in codings:
        if coding.get("system") in _SYSTEM_PREFIXES:
            return coding
    systems = [c.get("system") for c in codings]
    raise ValueError(
        f"{resource['resourceType']} code {codings[0]['code']} is under {systems}, none of {sorted(CODE_SYSTEMS.values())}"
    )


def _curie(coding: dict) -> str:
    """A FHIR coding under :data:`CODE_SYSTEMS` as the CURIE the schema spells it, ``loinc:30525-0``."""
    return f"{_SYSTEM_PREFIXES[coding['system']]}:{coding['code']}"


def _subject(resource: dict, patients: dict[str, str]) -> str:
    """The subject id of the ``Patient`` *resource* refers to, refused when the bundle holds no such ``Patient``."""
    reference = resource.get("subject", {}).get("reference")
    if reference not in patients:
        raise ValueError(
            f"{resource['resourceType']} {resource.get('id')!r}: subject {reference!r} is no Patient of this Bundle"
        )
    return patients[reference]


def _fhir_coding(iri, owner: str) -> dict:
    """The ``Coding`` of *iri*, a CURIE under :data:`CODE_SYSTEMS` or the URL the schema expands it to; *owner* names what carries it in the refusal."""
    from tvbo.datamodel.dialect import compact_curie

    prefix, _, code = compact_curie(iri).partition(":")
    if prefix not in CODE_SYSTEMS or not code:
        raise ValueError(f"{owner}: iri {iri!r} is no code under {sorted(CODE_SYSTEMS)}")
    return {"system": CODE_SYSTEMS[prefix], "code": code}


def _meta() -> dict:
    return {"security": [dict(RESEARCH_LABEL)]}


def _add_patient(entries: list[dict], subject_id: str, system: str | None) -> str:
    """Add the ``Patient`` of *subject_id*, conditionally created on its identifier when *system* names the pseudonym namespace."""
    identifier = {"value": subject_id}
    if system:
        identifier["system"] = system
    patient = {"resourceType": "Patient", "meta": _meta(), "identifier": [identifier]}
    return _add(entries, patient, if_none_exist=f"identifier={system}|{subject_id}" if system else None)


def _group(cohort, member_refs: list[str]) -> dict:
    group = {
        "resourceType": "Group",
        "meta": _meta(),
        "type": "person",
        "actual": True,
        "name": str(cohort.label or cohort.cohort_id),
        "member": [{"entity": {"reference": ref}} for ref in member_refs],
    }
    if cohort.description:
        group["code"] = {"text": str(cohort.description)}
    if cohort.diagnosis:
        group["characteristic"] = [
            {
                "code": _DIAGNOSIS,
                "valueCodeableConcept": {"coding": [_fhir_coding(d, f"cohort {cohort.cohort_id!r} diagnosis")]},
                "exclude": False,
            }
            for d in cohort.diagnosis
        ]
    return group


def _quantity_unit(unit) -> dict:
    """The unit part of a ``valueQuantity``: the unit as written and, when the curated unit has one, its UCUM code."""
    from tvbo.utils.units import unit_to_ucum

    if not unit:
        return {}
    code = unit_to_ucum(unit)
    return {"unit": str(unit), **({"system": UCUM_SYSTEM, "code": code} if code else {})}


def _observation(coding: dict, quantity: dict, subject_ref: str, method: dict | None) -> dict:
    observation = {
        "resourceType": "Observation",
        "meta": _meta(),
        "status": "preliminary",
        "code": {"coding": [coding]},
        "subject": {"reference": subject_ref},
        "valueQuantity": quantity,
    }
    if method:
        observation["method"] = method
    return observation


def _method(provenance) -> dict | None:
    """The run that produced the table, as the ``Observation.method`` text: the first activity's IRI or name."""
    for activity in (provenance.activities if provenance else None) or []:
        name = activity.iri or activity.name
        if name:
            return {"text": str(name)}
    return None


def _diagnostic_report(phenotype, subject_ref: str, observation_refs: list[str], report: Path) -> dict:
    import base64
    import mimetypes

    content_type = mimetypes.guess_type(report.name)[0] or "application/octet-stream"
    data = base64.b64encode(report.read_bytes()).decode("ascii")
    return {
        "resourceType": "DiagnosticReport",
        "meta": _meta(),
        "status": "preliminary",
        "code": {"text": str(phenotype.label or phenotype.dataset_id)},
        "subject": {"reference": subject_ref},
        "result": [{"reference": ref} for ref in observation_refs],
        "presentedForm": [{"contentType": content_type, "data": data, "title": report.stem}],
    }


def _provenance(record, target_refs: list[str]) -> dict:
    """The phenotype's ``Provenance`` record as a FHIR ``Provenance`` over *target_refs*; without a record, the bare attribution to TVBO."""
    from tvbo.data.provenance import now
    from tvbo.datamodel.pydantic import Provenance

    record = record or Provenance()
    activities = record.activities or []
    provenance = {
        "resourceType": "Provenance",
        "meta": _meta(),
        "target": [{"reference": ref} for ref in target_refs],
        "recorded": _instant(record.date_created or now()).isoformat(),
        "agent": [{"who": {"display": str(record.generated_by or "tvbo")}}],
    }
    if record.derived_from:
        provenance["entity"] = [{"role": "derivation", "what": {"display": str(record.derived_from)}}]
    starts = [_instant(a.started_at) for a in activities if a.started_at]
    ends = [_instant(a.ended_at) for a in activities if a.ended_at]
    if starts or ends:
        period = {}
        if starts:
            period["start"] = min(starts).isoformat()
        if ends:
            period["end"] = max(ends).isoformat()
        provenance["occurredPeriod"] = period
    return provenance


def _instant(stamp) -> datetime:
    """A provenance timestamp as an aware datetime; one without a UTC offset is refused, since a FHIR instant needs it and inventing one would shift the time."""
    instant = datetime.fromisoformat(str(stamp))
    if instant.tzinfo is None:
        raise ValueError(f"provenance timestamp {stamp!r} has no UTC offset, which a FHIR instant needs")
    return instant


def _add(entries: list[dict], resource: dict, if_none_exist: str | None = None) -> str:
    """Append *resource* as a POST entry of a transaction bundle, conditional on the search *if_none_exist* when given, and return the ``urn:uuid:`` reference the other entries name it by, derived from its content so a bundle is the same bytes for the same table."""
    full_url = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(resource, sort_keys=True))}"
    request = {"method": "POST", "url": resource["resourceType"]}
    if if_none_exist:
        request["ifNoneExist"] = if_none_exist
    entries.append({"fullUrl": full_url, "resource": resource, "request": request})
    return full_url
