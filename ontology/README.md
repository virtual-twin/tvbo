# TVB-O Ontology

Source modules for TVB-O (The Virtual Brain Ontology). Everything here merges
into one artifact, `tvbo.owl`, which the runtime loads and which we submit
upstream (W3C / community ontology hubs).

## Modules

Two kinds of file live here: **generated** modules (rebuilt from the LinkML
schema or the YAML database — never hand-edit them) and **hand-authored**
modules (OWL statements LinkML cannot express).

| File | Origin | Purpose |
|------|--------|---------|
| `tvb-o-struct.owl` | generated — `make gen-owl` from `schema/tvbo_datamodel.yaml` | T-box: classes, properties, ranges, domains. The structural data model. |
| `tvb-o.shacl.ttl` | generated — `make gen-shacl` from the schema | SHACL shapes for instance-level validation. |
| `tvb-o-data.ttl` | generated — `make gen-abox` from the YAML database | A-box individuals: studies, models, parameters. |
| `tvb-o-biology.ttl` | generated — `make gen-abox` (`--bio-output`) | Biological grounding for the A-box. |
| `tvb-o-axioms.ttl` | hand-authored | OWL axioms LinkML cannot capture: subclass hierarchies, equivalences, disjointness. Layered on `tvb-o-struct.owl`. |
| `tvb-o-bifurcation.ttl` | hand-authored | Axiomatised bifurcation / special-point taxonomy, defined by codimension, eigenvalue signature, and branch geometry (single source of truth for the backend label maps). |
| `tvb-o-neuroml.ttl` | generated — `make gen-neuroml` from the jNeuroML core types | The NeuroML2 core LEMS `ComponentType` hierarchy as OWL classes (`extends`→`subClassOf`), each `skos:exactMatch`-linked to its canonical NeuroML IRI. The reference from `tvbo.owl` to NeuroML-core, beside the GO link. Companion `tvbo/data/ontology/neuroml_contracts.json` (the accumulated contract index) is emitted in the same pass and loaded by the NeuroML adapter. |
| `tvb-o-neuroml-mappings.ttl` | hand-authored | tvbo-core ↔ NeuroML/LEMS alignment: meta-model correspondences (`Dynamics`↔`ComponentType`, `StateVariable`/`Parameter`/`DerivedVariable`/`Event`↔LEMS peers) and role correspondences (synapse/cell/population/input branches), via SKOS mapping relations. |
| `tvb-o-clinical.ttl` | generated from a PubMed + full-text survey (2026) | Clinical-applications addon: published TVB studies → clinical domains (ICD-11, MeSH) → the neural-mass model each uses. |
| `tvb-o-clinical-nmm.ttl` | hand-authored | `tvbc:NeuralMassModel ⊑ tvbo:Dynamics` plus clinical-NMM links. |
| `tvbo.owl` | generated — `make gen-merged` | Merge of every module above. The complete ontology; also copied to `tvbo/data/ontology/tvbo.owl`, which the runtime loads. |
| `*.ru` | hand-authored | SPARQL update queries applied after the merge (`fix-punning.ru`, `clinical-postmerge.ru`). |

## Workflow

1. Change the structural model in the LinkML schema (`schema/tvbo_datamodel.yaml`),
   then regenerate with `make gen-owl` and `make gen-shacl`.
2. For statements LinkML cannot express (equivalences, disjointness, axiomatised
   taxonomies), edit `tvb-o-axioms.ttl` or add a dedicated module such as
   `tvb-o-bifurcation.ttl`.
3. Rebuild the A-box from the YAML database with `make gen-abox`.
4. Merge everything into `tvbo.owl` with `make gen-merged`. That target runs the
   ROBOT merge, applies the `.ru` updates, reasons with ELK, and copies the
   result to the runtime path.
5. Submit `ontology/` upstream.

## Generation

```bash
make gen-owl      # → tvb-o-struct.owl   (T-box from the schema)
make gen-shacl    # → tvb-o.shacl.ttl    (SHACL shapes from the schema)
make gen-abox     # → tvb-o-data.ttl + tvb-o-biology.ttl (A-box from YAML)
make gen-neuroml  # → tvb-o-neuroml.ttl + neuroml_contracts.json (NeuroML-core; needs the neuroml extra)
make gen-merged   # → tvbo.owl           (merge + reason; also packaged for runtime)
```

The generated outputs are checked in so external consumers need no LinkML or
ROBOT toolchain. CI rebuilds them on every PR and fails if a committed copy
drifts from its regenerated form.
