# Data provenance

Replication of Jansen & Rit, *Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns*, **Biological Cybernetics 73**, 357–366 (1995), [doi:10.1007/BF00199471](https://doi.org/10.1007/BF00199471).

**This study sources no external data.** Every quantity it needs — the model equations, all parameter values, the two-column connectivity and the stimulus trains — is stated in the paper and transcribed into `Jansen1995.yaml`, with the paper's own line references kept beside each value. There is no connectome to download, no atlas to fetch, and nothing here that could not be redistributed.

That is why this directory holds only this file. It exists rather than being omitted because "no external inputs" is itself a provenance claim, and one a reader should not have to infer from an empty directory.

## If that changes

Anything obtained rather than computed belongs here, documented with its true upstream source, its licence, and the exact steps to fetch and regenerate it — never committed, since `.gitignore` tracks only this README. Material published by the work being reproduced (its PDF, figures or source data) belongs in `original_study/` instead, kept apart so it is never mistaken for an output of this study.
