"""Tests for the ingested NeuroML-core ontology module and contract index.

Covers the two artifacts emitted by ``scripts/ontology/gen_neuroml.py``:

- ``ontology/tvb-o-neuroml.ttl`` — the semantic module (classes, ``subClassOf``
  from ``extends``, ``skos:exactMatch`` cross-reference to NeuroML).
- ``tvbo/data/ontology/neuroml_contracts.json`` — the accumulated contract index
  the adapter loads to ground its base-type emission.

These assert the committed artifacts have the structure the adapter and the ontology merge depend on; a final determinism check (gated on pylems + the jar)
guards against generator drift.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import tvbo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TTL_PATH = REPO_ROOT / "ontology" / "tvb-o-neuroml.ttl"
CONTRACTS_PATH = pathlib.Path(tvbo.__file__).resolve().parent / "data" / "ontology" / "neuroml_contracts.json"

NML = Namespace("https://w3id.org/tvbo/neuroml/")
NEUROML2 = Namespace("http://www.neuroml.org/schema/neuroml2#")

# The synapse branch is the first ingested vertical slice; its chain anchors the structural assertions below.
SYNAPSE_CHAIN = [
    "baseConductanceBasedSynapse",
    "baseVoltageDepSynapse",
    "baseSynapse",
    "basePointCurrent",
    "baseStandalone",
]


@pytest.fixture(scope="module")
def ttl() -> Graph:
    assert TTL_PATH.exists(), f"missing generated module {TTL_PATH} (run `make gen-neuroml`)"
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    return g


@pytest.fixture(scope="module")
def contracts() -> dict:
    assert CONTRACTS_PATH.exists(), f"missing contract index {CONTRACTS_PATH} (run `make gen-neuroml`)"
    return json.loads(CONTRACTS_PATH.read_text())


class TestNeuroMLOntologyModule:
    """The Turtle module merged into ``tvbo.owl``."""

    def test_has_many_domain_classes(self, ttl):
        classes = set(ttl.subjects(RDF.type, OWL.Class))
        assert len(classes) > 200, "expected the full NeuroML2 core hierarchy"

    def test_synapse_subclass_chain(self, ttl):
        """``extends`` is faithfully rendered as a navigable ``subClassOf`` chain."""

        def parent(cls):
            supers = list(ttl.objects(cls, RDFS.subClassOf))
            return supers[0] if supers else None

        cls = NML["baseConductanceBasedSynapse"]
        walked = []
        while cls is not None:
            walked.append(str(cls).rsplit("/", 1)[-1])
            cls = parent(cls)
        assert walked == SYNAPSE_CHAIN

    def test_exactmatch_crossref_to_neuroml(self, ttl):
        """Both the tvbo-scoped and the direct NeuroML IRI denote the type."""
        matches = set(ttl.objects(NML["baseConductanceBasedSynapse"], SKOS.exactMatch))
        assert NEUROML2["baseConductanceBasedSynapse"] in matches

    def test_local_slot_annotations(self, ttl):
        exposes = {str(o) for o in ttl.objects(NML["baseConductanceBasedSynapse"], NML.exposes)}
        assert "g" in exposes
        event_ports = {str(o) for o in ttl.objects(NML["baseSynapse"], NML.eventPort)}
        assert "in" in event_ports

    def test_plumbing_types_excluded(self, ttl):
        """RDF/Dublin-Core metadata plumbing is not part of the ontology."""
        for junk in ("notes", "annotation", "dc_title", "rdf_RDF"):
            assert (NML[junk], RDF.type, OWL.Class) not in ttl


class TestNeuroMLContracts:
    """The accumulated JSON contract index consumed by the adapter."""

    def test_synapse_contract_accumulated(self, contracts):
        c = contracts["baseConductanceBasedSynapse"]
        assert c["exposures"] == {"g": "conductance", "i": "current"}
        assert c["requirements"] == {"v": "voltage"}
        assert c["event_ports"] == {"in": "in"}
        assert c["parameters"] == {"gbase": "conductance", "erev": "voltage"}
        assert c["chain"] == SYNAPSE_CHAIN

    def test_inheritance_extends_parent(self, contracts):
        """A concrete synapse inherits its base contract and adds its own params."""
        c = contracts["expOneSynapse"]
        assert c["extends"] == "baseConductanceBasedSynapse"
        assert {"gbase", "erev"} <= set(c["parameters"])  # inherited
        assert "tauDecay" in c["parameters"]  # own

    def test_cell_synapse_attachment_derived(self, contracts):
        """Synapse-hosting is declared on concrete cells (not the abstract base)."""
        assert contracts["iafRefCell"]["attachments"] == {"synapses": "basePointCurrent"}

    def test_contract_covers_module_classes(self, contracts, ttl):
        classes = {str(s).rsplit("/", 1)[-1] for s in ttl.subjects(RDF.type, OWL.Class)}
        assert set(contracts) == classes


class TestGeneratorDeterminism:
    """Regenerating from the jar must reproduce the committed artifacts byte-for-byte."""

    def test_regeneration_matches_committed(self, tmp_path):
        pytest.importorskip("lems")
        pytest.importorskip("pyneuroml")
        import subprocess
        import sys

        ttl_out = tmp_path / "nml.ttl"
        json_out = tmp_path / "nml.json"
        script = REPO_ROOT / "scripts" / "ontology" / "gen_neuroml.py"
        try:
            subprocess.run(
                [sys.executable, str(script), "-o", str(ttl_out), "--contracts", str(json_out)],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            pytest.skip("jNeuroML jar not available")
        except subprocess.CalledProcessError as exc:
            if b"No jNeuroML" in exc.stderr:
                pytest.skip("jNeuroML jar not available")
            raise
        assert ttl_out.read_text() == TTL_PATH.read_text(), "tvb-o-neuroml.ttl drifted from generator"
        assert json_out.read_text() == CONTRACTS_PATH.read_text(), "neuroml_contracts.json drifted from generator"
