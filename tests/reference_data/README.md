# Reference data

Committed artifacts that tests compare against, one directory per corpus.

| Directory | Frozen artifact | Test module |
| --- | --- | --- |
| `codegen/` | The source `Dynamics.render_code` emits for every curated model, per format | `tests/test_codegen_golden_corpus.py` |
| `numerical/` | The output of the curated specs in `numerical/specs/`, values and coordinate labels alike | `tests/test_numerical_golden_corpus.py` |
| *(this directory)* | HDF5 dumps from the NetworkDynamics.jl tutorials — generated, not committed | `tests/test_networkdynamics_comparison.py` |

## Golden corpora

`codegen/` and `numerical/` are golden corpora: they pin what TVBO promises to produce and fail when it changes. The shared harness lives in `tests/golden.py`, and each module's docstring states what its corpus covers and why.

Re-baseline with `pytest <module> --regenerate-golden`. This overwrites every reference with whatever the current code produces and asserts nothing, so the run is forced to exit non-zero — a re-baseline is a change to TVBO's promised output and belongs in its own reviewed commit, never mixed into the change that caused it.

## NetworkDynamics.jl comparison

```bash
julia --project=@. tests/reference_data/generate_nd_references.jl
```

Creates three gitignored HDF5 files:

- `diffusion_reference.h5` — 20-node Barabási-Albert diffusion
- `kuramoto_reference.h5` — 8-node Kuramoto on Watts-Strogatz ring
- `fitzhugh_nagumo_reference.h5` — 90-node FHN on AAL brain atlas

Tests needing no HDF5 data (code generation, YAML specs, URL validation) always run; the comparison tests skip when the files are absent.

Required Julia packages: NetworkDynamics, Graphs, OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK, SimpleWeightedGraphs, StableRNGs, HDF5, DelimitedFiles.
