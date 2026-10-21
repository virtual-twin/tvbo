# openMINDS_tvbo Examples

This directory contains example JSON-LD instances conforming to the openMINDS_tvbo schema.

## Files

### `simulationExperiment_jansenrit.jsonld`

A complete `SimulationExperiment` instance demonstrating:
- **Jansen-Rit neural mass model** with all 6 state variables
- Full parameter specification with units and domains
- Sigmoid transfer function definition
- Coupled equations with coupling input terms
- Output/observable variable (`v_pyr`)
- **Network** based on Desikan-Killiany parcellation (68 regions)
- Global coupling and conduction speed parameters
- **Heun integration** with stochastic noise
- **Multiple monitors**: temporal average + BOLD fMRI
- **Software environment** specification (TVB-library via conda)

### `simulationStudy_epilepsy.jsonld`

A `SimulationStudy` containing multiple experiments for epilepsy surgical planning:
- **Epileptor neural mass model** with fast/slow seizure dynamics
- Patient-specific personalized connectome
- **Heterogeneous parameters**: different excitability (x0) per region
- **Node-level customization**: epileptogenic zone specification
- SANDS coordinate references for node positions
- Multiple simulation experiments:
  1. **Baseline**: Reproducing patient's seizure dynamics
  2. **Virtual resection**: Testing proposed surgery
  3. **Alternative strategy**: Testing selective hippocampectomy
- Docker-based execution environment

## Schema Compliance

These examples use JSON-LD with:
- `@context` mapping namespaces to openMINDS vocabularies
- `@type` corresponding to openMINDS schema types
- `@id` providing persistent identifiers for resources
- Embedded vs linked types following schema definitions

## Validation

To validate against the schema:

```python
import json
import jsonschema

# Load example
with open("simulationExperiment_jansenrit.jsonld") as f:
    instance = json.load(f)

# Load schema (after resolving $refs)
with open("../schemas/simulationExperiment.schema.tpl.json") as f:
    schema = json.load(f)

# Validate (note: openMINDS templates need preprocessing)
# jsonschema.validate(instance, schema)
```

## Usage with The Virtual Brain

These examples can be converted to TVB-compatible configurations:

```python
from tvbo.datamodel import SimulationExperiment
from tvbo.export import to_tvb_simulator

# Parse the JSON-LD
with open("simulationExperiment_jansenrit.jsonld") as f:
    data = json.load(f)

# Convert to TVBO datamodel
experiment = SimulationExperiment.from_dict(data)

# Export to TVB simulator
simulator = to_tvb_simulator(experiment)
```
