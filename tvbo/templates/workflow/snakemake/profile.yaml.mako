## Snakemake SLURM profile, shipped inside an emitted kit.
## Inputs (all resolved by the emitter — no lookups here):
##   jobs           int   max SLURM jobs queued/running at once
##   container      str|None  the `container:` directive, if the kit declares one
##   container_args str   extra `apptainer exec` flags (binds), "" when none
##   container_args_yaml str  container_args as a ready-quoted YAML scalar
##   partition      str|None  SLURM partition
##   account        str|None  SLURM account
##   retries        int
# Snakemake profile: submit each rule to SLURM. The kit README says how to run it.
executor: slurm
jobs: ${jobs}                      # max SLURM jobs queued/running at once
slurm-logdir: logs             # per-rule SLURM logs -> <kit>/logs/rule_<name>/ (visible, not .snakemake/)
% if container:
## The CLI option takes a set of deployment methods, so this must be a YAML list rather than a scalar.
# run each rule inside the declared container:
software-deployment-method:
  - apptainer
% if container_args:
## Pre-quoted by the emitter, since this verbatim escape hatch may itself contain a double quote that would end the scalar early.
# extra `apptainer exec` flags (bind mounts for site filesystems)
apptainer-args: ${container_args_yaml}
% endif
% endif
## Only open `default-resources:` when something goes under it: a key holding only a comment parses as null.
% if partition or account:
default-resources:
% if partition:
  slurm_partition: ${partition}
% endif
% if account:
  slurm_account: ${account}
% endif
% else:
# no cluster identity declared; set workflow.slurm.partition / .account and re-emit
% endif
retries: ${retries}
keep-going: True
scheduler: greedy              # ILP scheduler stalls on large fan-outs (1000s of jobs); greedy scales
