# Network Builder Features for TVBO Platform

## Overview
Enhanced the Odoo TVBO Configurator with comprehensive network configuration capabilities, supporting three distinct modes for defining brain networks.

## Features Implemented

### 1. **Standard Network Mode** (Tractogram + Parcellation)
Load or configure networks based on neuroimaging data:

**Features:**
- Select from existing networks in the database
- Configure new networks with:
  - Parcellation/Atlas selection (Desikan-Killiany, Destrieux, HCP-MMP1, Yeo 7/17)
  - Tractogram source (DTI, DSI, HCP Average, Custom Upload)
  - Number of regions
  - Global coupling strength
  - Conduction speed (mm/ms)
  - Normalization options (by region, maximum, tract length)

**Use Case:** Standard neuroscience workflows using tractography and brain atlases

### 2. **YAML File Mode**
Load network definitions from external YAML files:

**Features:**
- File upload (.yaml, .yml)
- Direct YAML content paste
- Parse and validate YAML structure
- Support for LinkML-compliant network definitions

**Example YAML:**
```yaml
label: My Network
nodes:
  - id: 0
    label: Node_A
    dynamics:
      name: Generic2dOscillator
      dataLocation: database/models/Generic2dOscillator.yaml
    position:
      x: 0.0
      y: 0.0
      z: 0.0
edges:
  - source: 0
    target: 1
    weight: 0.8
    delay: 2.0
```

**Use Case:** Reusable network configurations, version control, external tool integration

### 3. **Custom Network Builder** (Node/Edge Classes)
Interactive builder for custom networks:

**Node Configuration:**
- Unique ID
- Label/name
- Brain region
- 3D position (x, y, z coordinates using Coordinate class)
- Dynamics model selection from database
- Visual grid layout with 8 columns

**Edge Configuration:**
- Source node ID
- Target node ID
- Connection weight
- Transmission delay (ms)
- Physical distance (mm)
- Coupling function type (Linear, Sigmoidal, Hyperbolic Tangent, etc.)
- Visual grid layout with 7 columns

**Global Parameters:**
- Global coupling strength
- Conduction speed

**Use Case:** Custom heterogeneous networks, small-scale testing, research prototypes

## Implementation Details

### Files Modified
- `platform/odoo-addons/tvbo/static/src/js/model_builder.js`
  - Added `initializeNetworkTab()` with mode switching
  - Implemented `collectNetworkConfig()` for all three modes
  - Updated `generateYamlPreview()` to render network configurations

### Data Collection
The `collectNetworkConfig()` function returns structured data matching the LinkML schema:

**Standard Mode:**
```javascript
{
  mode: 'standard',
  network_id: string,
  parcellation: string,
  tractogram: string,
  number_of_regions: integer,
  global_coupling_strength: float,
  conduction_speed: float,
  normalization: string
}
```

**YAML Mode:**
```javascript
{
  mode: 'yaml',
  yaml_content: string,
  source: 'yaml_file'
}
```

**Custom Mode:**
```javascript
{
  mode: 'custom',
  label: string,
  description: string,
  nodes: [
    {
      id: integer,
      label: string,
      region: string,
      position: { x: float, y: float, z: float },
      dynamics: { name: string, dataLocation: string }
    }
  ],
  edges: [
    {
      source: integer,
      target: integer,
      weight: float,
      delay: float,
      distance: float,
      coupling: { name: string }
    }
  ],
  number_of_nodes: integer,
  global_coupling_strength: float,
  conduction_speed: float
}
```

## Integration with LinkML Schema

The custom network builder generates data that conforms to the TVBO LinkML schema:

- **Nodes:** Match the `Node` class with `position` as `Coordinate` (x, y, z)
- **Edges:** Match the `Edge` class with source, target, coupling, weight, delay, distance
- **Network:** Match the `Network` class with nodes, edges, and global parameters
- **Dynamics references:** Use `dataLocation` field to link to model YAML files

## User Workflow

1. Navigate to **Network** tab in Simulation Configurator
2. Select network configuration mode from dropdown
3. Configure network using selected mode:
   - **Standard:** Choose existing or configure with parcellation/tractogram
   - **YAML:** Upload file or paste content
   - **Custom:** Build node-by-node and edge-by-edge
4. Configuration automatically included in YAML preview
5. Export or save complete simulation experiment

## Benefits

✅ **Flexibility:** Three distinct modes cover different use cases
✅ **Standards-compliant:** Follows LinkML schema and SANDS coordinate system
✅ **Reusability:** YAML mode enables sharing and version control
✅ **Precision:** Custom mode allows exact network specification
✅ **Integration:** Seamlessly works with existing dynamics and monitor configuration
✅ **Zero redundancy:** Dynamics referenced by dataLocation, not duplicated

## Future Enhancements

- Real-time network visualization (3D node positions)
- YAML syntax validation and error reporting
- Import from standard neuroimaging formats (GIfTI, NIfTI)
- Network templates library
- Coupling function library with visual editor
- Auto-calculate delays from distances and conduction speed
- Network statistics and validation (connectivity matrix, degree distribution)
