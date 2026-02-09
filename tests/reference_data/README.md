# Reference Data for NetworkDynamics.jl Comparison Tests

This directory contains Julia scripts that generate HDF5 reference data
from the original NetworkDynamics.jl tutorials.

## Generating reference data

```bash
cd /path/to/tvbo
julia --project=@. tests/reference_data/generate_nd_references.jl
```

This creates three HDF5 files (gitignored):
- `diffusion_reference.h5` — 20-node Barabási-Albert diffusion
- `kuramoto_reference.h5` — 8-node Kuramoto on Watts-Strogatz ring
- `fitzhugh_nagumo_reference.h5` — 90-node FHN on AAL brain atlas

## Running comparison tests

```bash
pytest tests/test_networkdynamics_comparison.py -v
```

Tests that don't require HDF5 data (code generation, YAML specs, URL validation)
always run. Tests that compare against reference data are skipped if HDF5 files
are not present.

## Required Julia packages

NetworkDynamics, Graphs, OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK,
SimpleWeightedGraphs, StableRNGs, HDF5, DelimitedFiles
