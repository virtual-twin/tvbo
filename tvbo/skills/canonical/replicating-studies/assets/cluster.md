# Phase 8 reference — scaling a replication out to a cluster

Read this only once Phase 8 of **replicating-studies** has established that one
node genuinely will not do. Everything here assumes the recipe already runs
locally: a cluster kit is an emit of the same recipe, never a rewrite of it.

## Phase 8 — Scale out to a cluster (ONLY when one node genuinely won't do)

**Skip this phase unless the work is irreducibly large** — a per-subject cohort (one
independent fit × N subjects) or a fit whose single run is itself heavy. First try NOT
to need it: a big *graph* → `graph_representation: sparse` + vectorized coupling; a big
*parameter grid* → a streaming reduced observable (Phase 4). Both routinely turn a
"needs HPC" run into minutes on one GPU, numerically identical (~1e-16). Assess this
before packaging anything.

REQUIRED output: a packed kit + a `docs/analysis/cluster-run.md` (the run route + site facts).

- **The kit is the same recipe, one command — no drivers, no bash.** `tvbo workflow
  snakemake <Study>.yaml -o <out> --pack` emits the whole study as ONE Snakemake DAG
  (one rule per experiment; dataset experiments fan out per subject; a `from_experiment`
  dependency becomes the DAG edge). Everything stays declarative in the recipe's
  `workflow:` block: runtime env via `workflow.container: docker://…` (each rule runs
  inside it via Apptainer — no venv/module activation); per-subject inputs via
  `Dataset.bundle: true` (`--pack` copies them in and rewrites `bids_root` relative);
  custom builders/analysis via `code_source:`; per-rule resources
  (`cpus_per_task`/`mem`/`time`/`partition`) via `workflow.slurm`. The kit is one
  `.tar.gz`; `tvbo workflow submit <kit>` runs it. This is invariant #1 (one recipe,
  no drivers) extended to the cluster — never hand-write sbatch.
- **Every run-time knob is a `--set` on the emit, never a recipe hand-edit or a hand-written
  sbatch.** The corollary of "no sbatch": any per-run override — swap the whole runtime
  substrate, retarget the queue, resize a job — is a flag on `tvbo workflow snakemake`, so the
  recipe stays the portable source of truth and the same study emits for CPU-container *and*
  GPU-venv without editing it. A **GPU run** is exactly this: drop the container and point at a
  `jax[cuda]` venv — `--set container= --set slurm.venv=/path/to/.venv --set slurm.partition=gpu
  --set slurm.gres=gpu:1 --set slurm.mem=… --set slurm.time=…` (a RECENT
  `snakemake-executor-plugin-slurm` turns the rule's `gres` resource into `--gres` itself; an
  OLDER plugin has no native `gres` and silently drops it, so **verify the GPU was actually
  allocated** — `squeue -o '…%b'` must show `gres/gpu:N` and `scontrol show job` `ReqTRES` must
  list it. A GPU node with no gres attached gives `cuInit … CUDA_ERROR_NO_DEVICE … Falling back
  to cpu`: a silent CPU run that "succeeds" wrong. On a GPU node let JAX auto-detect — do **not**
  force `JAX_PLATFORMS=cuda`, which drops the CPU device a `jax.debug.print` progress callback
  needs; use `cuda,cpu` if you must set it, and match the `jax-cuda12-*` plugin to `jaxlib`). Env
  vars are `--set 'slurm.env=[{name: …, value: …}]'`. Install the venv from a **compute node**
  (`srun`), never the login node.
- **Provide a `slurm.venv` and the kit TRUSTS it — no `setup.sh`, so the venv must already carry
  the study's deps.** When the recipe declares `requirements:`, a kit normally ships a `setup.sh`
  that layers them into a `--system-site-packages` venv. That is correct over a *base* interpreter
  (a conda env, a container image) but BROKEN over a provided venv: nested venvs don't chain
  site-packages, so pip re-resolves the FULL stack and its CPU `jaxlib` shadows the venv's
  `jax[cuda]` (silent CPU run) — and running it on the login node blows the ~20 MB `/tmp` cap
  ("Disk quota exceeded"). So when `slurm.venv` is set the emitter skips the env layer entirely:
  the venv IS the declared environment. Provision that venv once, from a compute node, with every
  requirement in it. Separately, `tvbo workflow submit` shells out to `snakemake`, so run it from
  an env that has BOTH `tvbo` AND `snakemake` + the slurm executor plugin — a `snakemake`-only
  orchestration env cannot drive it, and neither can a `tvbo`-only one.
- **On a GPU the compile peak is DEVICE memory, not host `mem`.** The same wide-vmap-long-scan
  spike that sizes `slurm.mem` also sizes GPU VRAM: a whole-panel grid can compile to tens of GB
  on-device and OOM a mid-VRAM card (`bfc_allocator … ran out of memory`, `hlo_rematerialization:
  Can't reduce memory use below …`). Fix by shrinking the on-device batch (a smaller `n_parallel`,
  or the kit's per-attempt nvmap retry-shrink) OR by requesting a larger-VRAM GPU (`--set
  slurm.gres=gpu:<big-model>:1`) — the latter fits the full batch without guessing a size.
- **Size `slurm.time` off the measured per-batch rate, not a guess — a single vmap grid can't
  resume mid-run.** A sweep that streams and compiles fine can still blow a short default
  walltime: a whole-panel grid runs its cells in on-device batches at a steady rate (e.g.
  ~2 min/batch × ~200 batches ≈ 6–7 h), and a 4 h `slurm.time` TIME-LIMIT-kills it near ~60 %
  — wasted, because a backend-vectorized grid is ONE job with no mid-grid checkpoint (unlike a
  *fanned* sweep, where each cell is its own resumable job and Snakemake just re-runs the killed
  ones). Read the rate off the first few batches (the log's `[+Ns] … batch k/N` line, or the
  `--benchmark` TSV) and set `--set slurm.time=` with headroom before it hits the wall.
- **Prove the memory/streaming fix — don't eyeball it — with engine-native benchmarking.**
  `tvbo workflow snakemake … --benchmark` (or `--set benchmark=true`) attaches Snakemake's
  native `benchmark:` directive to every rule: a per-cell TSV (wall time, `max_rss`/`vms`/
  `uss`/`pss` MB, io, cpu_time) written next to each output, whether run locally or as a SLURM
  job — one row per cell, so a fanned sweep benchmarks every cell. This is how you turn "reason
  about resident memory" into a *measured* peak (a streaming BOLD fit that would OOM at hundreds
  of GB materialized shows a ~GB peak in the TSV), and how you size `slurm.mem` honestly.
- **Size per-rule memory off the COMPILE peak, and set it PER EXPERIMENT — not one global
  number.** Streaming bounds the *runtime* trajectory, but what OOM-kills a whole-brain fit is
  usually ELSEWHERE: XLA/LLVM **compiling** a wide-vmapped long-scan graph (a G-sweep ×10, a seed
  ensemble ×50) spikes far above the resident set — a 379-node fit that streams at ~2–6 GB still
  needs ~32 GB to compile, and `float64` roughly doubles that. So an 8 GB request that ran the
  tuning fine dies *later* with `Failed to materialize symbols` / `LLVM Cannot allocate memory`.
  Express it as a modest **global `workflow.slurm` baseline overridden per experiment**: each
  heavy experiment carries `workflow: {slurm: {mem: 32G, cpus_per_task: 4}}` (deep-merged over the
  study block — only the set leaves change, partition/time/env inherit; DRY via a shared YAML
  anchor), while the light ones (a DM circuit, a forward run) stay at the baseline. Ship only the
  Snakefile when just the resources change — never re-extract the tarball over a running kit
  (clobbers completed results + snakemake state).
- **A dry run does NOT execute anything — smoke-test ONE experiment in the container
  FIRST.** `tvbo workflow submit --dry-run` (snakemake `-n`) only resolves the DAG
  (wildcards, inputs, resources); no `tvbo run` executes, so it cannot catch a runtime
  bug. A per-rule bug fails all N jobs identically (we once launched 1106 that all died
  the same way). Before the real submit, run a single experiment end-to-end inside the
  SIF (`apptainer exec --bind … <sif> tvbo run <Study>.yaml --experiment <id>`), then its
  dependents, then the full submit. This is Phase 7's "run END-TO-END, not `from_file`"
  at cluster scale. **A *fit* can't be "run once" to smoke-test it** — its whole cost is the
  tuning iterations. Cap them: `tvbo run … --smoke` (= `--max-iterations 1`) or
  `--max-iterations N` reaches the post-tuning evaluation in one/N iterations (the recipe
  untouched), which is how you verify a long fit runs and *streams within memory* in minutes
  rather than days. At kit level it is a run modifier like any other: `--smoke` /
  `--set smoke=true` / `--set max_iterations=N` on `tvbo workflow snakemake`.
- **The container filesystem is READ-ONLY — a bug class that ONLY bites in-container.**
  Anything writing into the installed package or `$HOME/.cache` at import/run time
  fails only inside the SIF, never locally or in a dry run: codegen compiling templates
  into the package dir, `templateflow`/`matplotlib` writing caches, a `$HOME` that
  symlinks into another filesystem (the link dangles in-container). Fixes: writable
  temp dirs for codegen caches, and **bind the site filesystem** (`--bind /data/…`,
  declared in `workflow.slurm` container args). The single-experiment smoke test
  surfaces every one at once.
- **Know which fixes need a container rebuild vs a re-emit.** The container runs the
  *pushed* branch; your emitter is your *working tree*. A schema or codegen-**template**
  change takes effect only after push → image rebuild → SIF re-pull; an emit-side change
  (freezing/packaging in the CLI) just needs a re-emit of the kit. Confirm a fix is
  actually live before assuming — and when you re-pull an image, assert the fix is
  present (a tag can rebuild to stale cached content; a SIF is named by the URL hash, so
  it lands at the same path — force the pull).
- **Ship the kit dual-mode so a version-skewed node can still run YOUR code —
  `--code-source {frozen,spec}`.** A Snakemake study kit emits, inside the kit
  itself, BOTH the frozen pre-rendered `scripts/<exp>` and its own `spec/` copy, and each rule
  can run either: **spec** (default)
  re-generates the backend code from the spec at run time (needs a node `tvbo` whose codegen
  matches the emit-time behaviour); **frozen** runs the pre-rendered script as-is via `tvbo run
  --rendered scripts/<exp>`, so the reducer/streaming logic is already baked into the script and
  the node's `tvbo` needs no matching codegen. This is the clean fix for the *version-skew* trap
  above — when the cluster's released `tvbo` lags a codegen feature the recipe relies on (a new
  streaming reducer), emit `--code-source frozen` and the node runs the frozen code with no
  container rebuild. Set the emit-time default (`tvbo workflow snakemake … --code-source
  frozen`) or override per submission (`tvbo workflow submit … --code-source frozen`, or
  `TVBO_CODE_SOURCE=frozen snakemake …`); a rule with no `scripts/<exp>` (a cross-experiment
  analysis has no standalone sim to render) falls back to spec automatically, and `frozen`
  cannot honour a run-time flag that *changes* codegen (`--set integration.*`, `--pin` on a
  non-vectorized axis) — use `spec` for those. `frozen` and `spec` are byte-identical for a
  deterministic experiment (kit anatomy + the full contract: `docs/CLI/workflow-kits.qmd`).
- **The frozen kit can run a DIFFERENT float precision than your dev run — pin it, or a stiff
  fit silently NaNs on the cluster.** Frozen and spec agree with each other, but both honour the
  recipe's `execution.precision` (which may be `float32`), whereas in-process `experiment.run()`
  hardcodes `enable_x64=True` → **float64**. So you develop and validate in float64 (stable) while
  the cluster kit runs float32 — and a gradient-based whole-brain FIC/EI fit is only *marginally*
  stable in float32: it survives one `cpus_per_task` and NaNs under another (the XLA reduction
  order shifts). The tell is a fit that ran finite once and NaNs on resubmit with nothing changed
  but the cpu count — **the jax version and the cpu count are the red herrings; precision is the
  cause.** Fix declaratively: `execution.precision: float64` AND `JAX_ENABLE_X64=1` in
  `workflow.slurm.env` (forces x64 at runtime on the *already-frozen* scripts, so you re-ship only
  the Snakefile, no re-render). Diagnose by A/B-ing `JAX_ENABLE_X64` 0 vs 1 with everything else
  fixed. (Durable framework fix: make `experiment.run()` respect the declared precision so the two
  paths can't diverge.)
- **Run the orchestrator on a COMPUTE node, not the login node.** Login nodes are
  cgroup-capped (a per-PROCESS memory limit — 128 MB on BIH) that SIGKILLs a long
  `snakemake` — the tell is the driver dying with **exit 137 at DAG build**, seconds
  after launch, while the node's *system* RAM is plentiful (it is the per-process cap,
  not the machine). A tmux-on-login driver is NOT reliable for this: it can squeak under
  the cap one day and get killed the next as the DAG or `.snakemake/` metadata grows. Put
  `tvbo workflow submit` inside a small **sbatch DRIVER job** (`--mem=4G --cpus-per-task=2
  --time=1-00:00:00`, body = `source venv; snakemake --unlock; tvbo workflow submit .`);
  the orchestrator then submits the array via **nested sbatch (a job submitting jobs —
  verified to work on BIH)**, and the driver survives ssh drops AND login reboots (a tmux
  session survives neither cleanly). It is resumable (snakemake skips completed outputs,
  so a walltime cap or a crash just means resubmit). Never install or build on the login
  node — `tar` included (extraction is compute; do it under `srun` or on a transfer node).
- **`bundle: true` data extracted on the cluster gets REAPED mid-run unless you refresh
  its mtime.** Symptom: the first wave of a fan-out succeeds, then every later subject
  fails with `ValueError: No file for sub-… matching query` and the bundled `dataset/`
  dirs are all empty. Scratch auto-cleanup deletes files whose **mtime > 14 days**, and
  `tar`/`rsync` PRESERVE the archived mtime — a `functional_connectomes` sidecar created
  months ago arrives already "expired", so the nightly reaper purges the whole bundle an
  hour into the run, leaving the empty dirs behind (jobs that loaded their file before the
  reaper ran are the only survivors). Extract with **`tar -m`** (or `find <kit> -exec
  touch {} +` after) so every file gets a CURRENT mtime, and stage under
  `~/scratch/<yourdir>/`, never `~/`. This bit a 1096-subject cohort: 100 subjects
  finished, ~996 failed `No file matching query`, and the fix was re-extract-with-`-m`
  (the source `.tar.gz` was still on scratch, so no re-transfer) — then snakemake resumed
  and filled in the missing subjects. Verify before a long run: `find <kit>/…/dataset
  -name '*<suffix>*' | wc -l` equals the subject count, not zero.
- **Big, flaky transfers: chunk + checksum.** A multi-hundred-MB kit over an unreliable
  link won't survive one `scp`/`rsync` stream (macOS ships `openrsync`, which doesn't
  resume); split into ~32 MB chunks, size-verify each with retries, reassemble, then
  **sha256 the result against the source** — a stale-but-right-sized kit passes a
  byte-count glance (we shipped one twice before checking the hash). Iterate with small
  (spec-only) uploads, not the full kit.
- **Pick the tier by measuring the per-step rate, not by assuming a GPU wins.** A 379-node
  delayed whole-brain fit measured **72.6 us/step on an A40** against **229 us/step on one
  CPU core** — a whole GPU for ~3x a single core, because per-edge delays force an (N, N)
  gather that is dispatch-bound rather than arithmetic-bound. A cohort sized on the
  assumption "GPU is the fast tier" came out at ~4800 GPU-hours; the same work on CPU is
  ~30 000 core-hours, which is 10 days on 128 `medium` cores and costs no scarce hardware.
  Measure one experiment's rate on both tiers before sizing anything.
- **Declare `time:` on every GPU experiment, not just the partition.** A study-wide
  `workflow.slurm.time` is inherited by an experiment whose block sets only
  `partition: gpu`, so a kit can be emitted with `runtime=4320` (3 days) on a scarce card
  while the fits themselves take four hours. The cap is the enforcement, not the estimate: a
  mis-sized job should die at the cap rather than squat. Name the GPU type too when the
  site's tiers are not interchangeable — one node type's plugin can fail to initialise the
  CUDA backend while another works on the same driver.
