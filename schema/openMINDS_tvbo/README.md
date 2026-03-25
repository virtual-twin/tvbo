# openMINDS_tvbo

OpenMINDS-compatible schema extension for The Virtual Brain Ontology (TVBO).

This schema module extends the openMINDS metadata framework to support computational neuroscience simulations using The Virtual Brain platform and related tools.

## Auto-Generation from LinkML

**The schemas in this module are auto-generated from the LinkML source of truth.**

```bash
# Regenerate schemas after modifying tvbo_datamodel.yaml
python generate_openminds.py

# Preview what would be generated
python generate_openminds.py --dry-run
```

The generator script (`generate_openminds.py`) reads the LinkML schema and:
1. Maps TVBO classes to openMINDS schema format
2. Automatically links to existing openMINDS types (SANDS, core, computation)
3. Skips internal helper classes
4. Preserves inheritance and categories

## Dependencies

- [openMINDS_core](https://github.com/openMetadataInitiative/openMINDS_core) - Core metadata schemas
- [openMINDS_computation](https://github.com/openMetadataInitiative/openMINDS_computation) - Computational activity schemas
- [openMINDS_SANDS](https://github.com/openMetadataInitiative/openMINDS_SANDS) - Spatial Anchor Data Schema for brain atlases

## Design Principles

This schema module follows a **no-redundancy** policy:
- Reuse existing openMINDS schemas wherever possible
- Only define new schemas for concepts unique to brain network simulation
- Link to `sands:` types for neuroanatomy, `core:` for files/actors, `computation:` for environments

## Type Mappings to Existing openMINDS

The following TVBO/LinkML types map to existing openMINDS types (no new schemas generated):

| LinkML Type | openMINDS Type |
|-------------|----------------|
| `BrainAtlas` | `sands:BrainAtlas` |
| `BrainAtlasVersion` | `sands:BrainAtlasVersion` |
| `CommonCoordinateSpace` | `sands:CommonCoordinateSpace` |
| `ParcellationEntity` | `sands:ParcellationEntity` |
| `ParcellationEntityVersion` | `sands:ParcellationEntityVersion` |
| `Coordinate` | `sands:CoordinatePoint` |
| `File` | `core:File` |
| `FileBundle` | `core:FileBundle` |
| `DOI` | `core:DOI` |
| `Person` | `core:Person` |
| `Organization` | `core:Organization` |

## Generated Schema Categories

### Dynamics (7 schemas)
Neural dynamics and model specification:
- `dynamics` - Neural mass model or dynamical system
- `stateVariable` - State variable with differential equation
- `derivedVariable` - Algebraic variable derived from state
- `coupling` - Inter-node coupling function
- `couplingInput` - Coupling input channel specification
- `noise` - Stochastic noise process
- `fieldStateVariable` - Field-like state variable for PDEs

### Network (6 schemas)
Brain network and connectivity:
- `network` - Brain network/connectome specification
- `node` - Network node with local dynamics
- `edge` - Directed/undirected edge with coupling
- `parcellation` - Brain atlas parcellation reference
- `tractogram` - Tractography data source
- `regionMapping` - Vertex-to-region mapping

### Simulation (6 schemas)
Simulation configuration and execution:
- `simulationExperiment` - Complete simulation (extends `computation:Simulation`)
- `simulationStudy` - Collection of related experiments
- `integrator` - Numerical integration method
- `monitor` - Observation/recording specification
- `stimulus` - External stimulation protocol
- `processingStep` - Processing pipeline step

### Mathematical (9 schemas)
Mathematical objects:
- `equation` - Mathematical equation representation
- `parameter` - Numerical parameter with constraints
- `derivedParameter` - Parameter computed from others
- `function` - Reusable mathematical function
- `range` - Value range with optional step
- `distribution` - Statistical distribution
- `conditionalBlock` - Conditional equation block
- `costFunction` - Optimization cost function
- `temporalApplicableEquation` - Time-dependent equation

### Data (4 schemas)
Data and software:
- `timeSeries` - Simulation output time series
- `softwareEnvironment` - Execution environment specification
- `softwareRequirement` - Software dependency
- `softwarePackage` - Software package identity

### PDE/Field (5 schemas)
Partial differential equations and spatial fields:
- `pDE` - PDE problem definition
- `pDESolver` - PDE solver configuration
- `spatialDomain` - Spatial domain specification
- `spatialField` - Spatial field data
- `mesh` - Spatial mesh/grid

### Other (8 schemas)
Additional schemas:
- `observationModel` - Base observation model
- `downsamplingModel` - Downsampling specification
- `modelFitting` - Model fitting configuration
- `fittingTarget` - Fitting target specification
- `boundaryCondition` - PDE boundary condition
- `differentialOperator` - Differential operator
- `neuralMassModel` - Legacy neural mass model (deprecated)
- `randomStream` - Random number generator state

## Customizing the Generator

Edit `generate_openminds.py` to modify:

- `OPENMINDS_TYPE_MAPPINGS`: Map LinkML types to existing openMINDS types
- `OPENMINDS_CATEGORIES`: Assign openMINDS categories to types
- `OPENMINDS_EXTENDS`: Specify schema inheritance
- `SKIP_CLASSES`: Internal classes to skip

## Version

See `version.txt` for the current schema version.
