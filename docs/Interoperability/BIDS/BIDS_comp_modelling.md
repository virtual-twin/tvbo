# BIDS Computational Modelling (BEP034)

BEP034 is the BIDS Extension Proposal for computational modelling data. It is still a draft, and this page describes what it currently specifies, what TVBO does, and where the two differ.

## What BEP034 specifies

The schema compiled from the BEP034 pull request defines one datatype, `model`, holding two file kinds: `eq` for equations and `param` for parameters, both in [LEMS](http://lems.github.io/LEMS) XML. There is no other datatype, suffix or extension in the proposal.

Relationship matrices are explicitly out of scope. BEP034 delegates them to **BEP017**, which defines the `_relmat` suffix, the `meas-` entity, and extensions `.tsv`, `.h5` and `.zarr` across any datatype. An implementation storing connectivity as part of a BEP034 dataset follows BEP017 for file naming, placement and tabular structure. `Network.to_bep017` does this.

## What TVBO stores, and why

A simulation experiment produces observations over a parameter space. BEP034 has no concept for that, so TVBO writes one self-describing HDF5 container per experiment: every recorded observation is a labelled variable, and the swept parameters are shared coordinates.

The alternative permitted by BEP034 is one file per parameter cell. For a study sweeping 4 initial conditions by 39 coupling values by 10 conduction speeds by 10 trials, that is 15,600 files, and it discards the grid structure that makes the result readable. The container keeps it, and `xarray.open_dataset` reads the file without TVBO installed.

Model provenance travels with the result rather than in a separate directory. Each container is accompanied by a JSON metadata sidecar and a frozen, re-runnable recipe, so the model that produced a result cannot drift away from it.

## Where TVBO diverges from BEP034

| Topic | BEP034 | TVBO |
|---|---|---|
| Model definition | `model/` datatype, LEMS XML | the recipe beside the result, LEMS as an optional export |
| Simulation output | not addressed | one gridded container per experiment |
| Parameter sweeps | not addressed | swept parameters as named coordinates |
| Connectivity | delegated to BEP017 | BEP017 via `Network.to_bep017` |


## Proposed alignment

We consolidate on the container layout described above, and would propose matching changes to BEP034 itself. Nothing has been filed with the BIDS project.

For the authoritative description of relationship matrices, see BEP017 in the BIDS documentation.
