## Snakemake SLURM profile, shipped inside an emitted kit.
## Inputs (all resolved by the emitter — no lookups here):
##   jobs           int   max SLURM jobs queued/running at once
##   container      str|None  the `container:` directive, if the kit declares one
##   container_args str   extra `apptainer exec` flags (binds), "" when none
##   container_args_yaml str  container_args as a ready-quoted YAML scalar
##   partition      str|None  SLURM partition
##   account        str|None  SLURM account
##   retries        int
# Snakemake profile — submit each rule to SLURM via the executor plugin.
# Run (login node):  snakemake --profile profile
# Requires:          pip install snakemake-executor-plugin-slurm
executor: slurm
jobs: ${jobs}                      # max SLURM jobs queued/running at once
slurm-logdir: logs             # per-rule SLURM logs -> <kit>/logs/rule_<name>/ (visible, not .snakemake/)
% if container:
## A `container:` directive is declared, so run every rule inside it via Apptainer
## (provided on the compute nodes) — no venv/module activation needed. A registry
## reference is converted to a SIF first; a path to a prebuilt image is used as-is.
## The CLI option is nargs="+" (a set of deployment methods), so this must be a YAML
## list, not a scalar.
# run each rule inside the declared container:
software-deployment-method:
  - apptainer
% if container_args:
## The container sees only $HOME and $PWD by default. On a site whose home holds
## symlinks into other filesystems (e.g. ~/.cache -> scratch) those links dangle
## in-container until the target filesystem is bound too, and a library that touches
## such a path at import time fails before the task starts.
## Pre-quoted by the emitter: container_args is the verbatim escape hatch, so it may
## itself contain a double quote (e.g. --env FOO="bar"), which would close this scalar
## early and make the profile unparseable.
# extra `apptainer exec` flags (bind mounts for site filesystems)
apptainer-args: ${container_args_yaml}
% endif
% endif
## Only open `default-resources:` when something goes under it — a key whose sole
## content is a comment parses as null, not as an empty mapping.
% if partition or account:
default-resources:
% if partition:
  slurm_partition: ${partition}
% endif
% if account:
  slurm_account: ${account}
% endif
% else:
# no cluster identity declared; set workflow.slurm.partition / .account
# (or --set slurm.partition=…) and re-emit to get a default-resources block
% endif
retries: ${retries}
keep-going: True
scheduler: greedy              # ILP scheduler stalls on large fan-outs (1000s of jobs); greedy scales
