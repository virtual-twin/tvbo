# Handoff: emit a streaming HRF-BOLD reducer as backend-independent code

**Status:** design complete, not yet built. Sibling to
[`reducers_unification.md`](reducers_unification.md) (the windowed-FC reducer that
this mirrors). Written 2026-07-21 as a focused-session handoff.

## The overarching goal this serves

The end goal is a **faithful, fully tvbo-native replication of Schirner, Deco & Ritter
(2023)** "Learning how network structure shapes decision-making for bio-inspired
computing" (Nat Commun 14:2963), run at scale on the BIH HPC cluster to **rerun ALL
experiments** — the normative Glasser-379 group fit AND the 1096-subject individual
cohort — reproducing Figs 1–6 from tvbo/tvboptim output (not the paper's replotted
data). The replication is one declarative `SimulationStudy` (`Schirner2023.yaml`) that
emits as a single Snakemake DAG and runs inside the dev container via Apptainer, with
per-subject FC gathered/bundled from the BIDS derivatives — no venv, no bash, all
tvbo-CLI / metadata-native.

This session got the whole pipeline running end-to-end on the cluster: it turned a run
that failed 1106 ways into one blocked by exactly ONE remaining issue — the post-tuning
FC evaluation OOM described below. **This streaming-BOLD reducer is the last thing
standing between the current state and launching the full 1096-subject study.** Every
other blocker (packaging, container, schema, base-sim materialization) is already fixed
and committed on `dev`.

## Why this exists — the concrete failure

Running the Schirner2023 Glasser-379 group fit (`exp_34`) on the BIH cluster OOMs.
Root cause chain, all verified:

- The fit's per-stage duration is Schirner's exact setting: **36,000,000 ms (10 h
  biological time) at dt=1 ms = 36 M steps**. This MUST NOT be shortened — it is the
  original protocol.
- The tuning loop itself streams fine (windowed FC, `use_sliding_window`), but each
  algorithm ends with a **post-tuning FC evaluation** that materializes the whole
  trajectory: `post_model_fn = prepare(..., t1=36_000_000)`, then
  `post_tuning = post_model_fn(state)` → `result.data` of shape
  `(36e6, 4 states, 379 nodes)` ≈ **437 GB** → OOM even at 128 GB.
- The post-tuning FC (`post_tuning_observations`) is the **deliverable** (Fig 2 / 3b),
  so it can't be skipped — only computed without materializing.

Two OTHER fixes for this run are already committed on `dev` (needed a container
rebuild, done): base-sim skip for algorithm experiments (`run_main`), the
`observational_measures` freeze fix, mako read-only cache, `container_binds` schema,
`.cache` bind, spec-load diagnostics. See git log on `dev`. THIS is the last blocker
for `exp_34` (and every long FC fit). Separately, `exp_41/42` (2-node) OOM in an
EI-ratio sweep — a different bug, off the critical path, not covered here.

## The decision (locked with the user)

`exp_34`'s BOLD samples S_e at the 720 ms TR anyway, so computing BOLD via a
**streaming reducer** (fold in-carry, never stack the trajectory) yields a
**bit-identical** FC while cutting 437 GB → ~0.6 GB. The user's hard constraints:

1. **tvbo emits the concrete reducer as code** — no calling tvboptim's
   `streaming_hrf_bold` primitive. Backend-independent, authored from metadata and
   lowered by the sympy printers, exactly like `windowed_fc`. "All concrete classes,
   tvbo writes as code." (This is [[feedback-tvbo-extends-backends-via-codegen]] +
   [[feedback-generalize-backend-independent]].)
2. **Optional / opt-in** — must not change any other use-case. Default behaviour
   (materialize) unchanged; only an experiment that opts in streams.
3. Keep Schirner's 36 M-step duration exactly.

## The reference implementation to REPLICATE (do not call)

tvboptim ships the concrete streaming BOLD as
`tvboptim/observations/tvb_monitors/bold.py::streaming_hrf_bold(monitor, dt)` — read
it; it is the ground truth to match byte-for-byte. It returns an
`(init, update, finalize)` reduce triple:

- carries a downsampled-history **ring buffer** (kernel-length) + a preallocated BOLD
  output buffer;
- per block: subsample by `ds_steps` (dt→downsample_period), `concat([ring, block_ds])`,
  `fftconvolve(signal, hrf, 'valid')`, write BOLD samples at TR boundaries, roll the ring;
- **requires `SubSampling`** downsample (uniform integer stride) — `TemporalAverage`'s
  float-rounded windows are NOT streamable;
- **requires** `block_size` and `n_steps` to be multiples of `period/dt` (720). Note
  36e6 / 720 = 50000 (integer ✓); pick `block_size` a multiple of 720.
- `init` seeds the ring from `monitor.history` (warm start / HRF warm-up, `skip_t`).

## The design — a declarative windowed reducer with a convolution emit

streaming BOLD FITS the existing `windowed_fc` reducer shape (see
`tvbo/database/reducers/windowed_fc.yaml` + `tvbo/codegen/streaming_reducers.py`
`StreamingReducerSpec` + `tvbo/codegen/reducers.py` `resolve_streaming_reducer`). Recast:

- **window `x`** = the ring of downsampled neural history, HRF-kernel-length long;
- **`add` / `evict`** = slide one downsampled sample in/out of the ring;
- **`emit`** = `strided_convolve(x, hrf_kernel, tr_stride) * bold_scaling` — the HRF
  convolution evaluated only at TR boundaries;
- **`emit_kind: stride`** — emit at the 720 ms TR, not every step.

The load-bearing op ALREADY EXISTS and is backend-abstracted:
`strided_convolve(X, k, s)` = `fftconvolve(X, k, 'valid')[s::s]`, implemented as a
printer primitive at `tvbo/codegen/code.py:314` (`_afp_strided_convolve`) with the
per-printer `_strided_convolve`. `concatenate`, `outer`, `matmul` are in `ARRAY_FUNCTIONS`
too. So the arithmetic to WRITE streaming BOLD is already backend-independent — the
authoring form is the gap.

## What has to be built — three framework gaps + wiring

1. **`emit_kind: stride` wiring.** The algorithm template only wires `emit_kind ==
   'window'` (`tvbo-tvboptim-algorithm.py.mako:~296`, gate at
   `streaming_reducers.py:100` returns False for `stride`). Land the stride branch:
   emit the reduced value every TR (`period/dt` steps) instead of every step. This is
   already anticipated in the code comments ("until the template's stride branch lands").

2. **Reducer constants.** `resolve_streaming_reducer` vocab is `state`/`v`/`x` +
   `ARRAY_FUNCTIONS` only (`reducers.py:33-40`). BOLD's `emit` needs the **HRF kernel**
   (a constant array derived from kernel params + `downsample_period`) and the BOLD
   scaling `k_1`,`V_0`. Add a `constants:` slot to `StreamingReducerSpec` (name → expr
   over kernel params/dt), resolved once at reducer construction (mirror how
   `streaming_hrf_bold` closes over `hrf` before `init`). Keep it backend-agnostic
   (printed like the recurrences).

3. **Two-level decimation.** dt → `downsample_period` (intermediate, e.g. 4 ms) → TR
   (720 ms). The ring holds *downsampled* history; `add` folds `subsample(v, ds_steps)`.
   `subsample.yaml` already provides the decimation op.

4. **Post-eval wiring + opt-in.** Reducers today wire only into the algorithm in-loop
   (`use_sliding_window`). Wire the streaming BOLD into the **post-tuning eval**:
   `post_model_fn = prepare(network, get_solver(block_size=<mult of 720>), t1=36e6,
   dt, reduce=<emitted streaming_bold>)` at `tvbo-tvboptim-experiment.py.mako:3608`,
   and compute FC from the streamed BOLD in `tvbo-tvboptim-algorithm.py.mako:~999-1016`
   (FC-only post-eval — the `inp_*` observations need the full trajectory and are NOT
   `exp_34` deliverables; skip them when streaming). Gate ALL of it on an **opt-in**
   flag so no other experiment changes — e.g. a declarative
   `Observation.reduce: streaming` on the `fc`/`bold` observation, or an
   experiment/algorithm `stream_post_evaluation: true`. Default off → materialize
   (today's behaviour, untouched).

## Acceptance criteria (non-negotiable)

- **Byte-identical**: emitted streaming BOLD → FC must equal `compute_fc(materialized
  BOLD)` to f64 tolerance (~1e-12), matching how `windowed_fc` is "byte-identical to
  compute_fc" and `strided_convolve` is ~1e-12 vs FFT. Test both directly (reducer vs
  `streaming_hrf_bold`) and end-to-end (a short EIB fit with/without streaming).
- **Opt-in, zero regression**: with the flag off, generated code and all existing tests
  are unchanged. Run `tests/test_tvboptim_experiments.py` and the codegen suite; the
  `EI_Tuning_FIC_EIB_Optimization` experiment is the canonical exercise.
- **No callable into tvboptim**: `grep -rn streaming_hrf_bold tvbo/` stays 0 — tvbo
  emits the reducer; it does not import/call the primitive.

## Verify on the cluster (repro is staged)

The exact repro is on BIH `hpc-login-2` (NOT login-1 — it wedges; use login-2, same
`/data/cephfs-1`). `~/work/martinl_c/exp34_v2.tgz` = re-emitted `exp_34` spec + code.
The diagnostic `verify34_fix.sbatch` binds a fixed template into the current SIF and
runs `exp_34`; adapt it to bind the new streaming-BOLD codegen and confirm the
post-tuning eval completes in ≤16 GB (it currently OOMs at 128 GB). The SIF is
`761bf447b80bb38258ef704790f279c7.simg`; the mako fix + container fixes are already in
`ghcr.io/virtual-twin/tvbo:dev`. See [[reference-bih-cluster-apptainer-run]].

## Where everything lives (manuscript repo + cluster)

The study is a **separate repo** from tvbo — the `tvbo-manuscript` use-case:

- **Study root:**
  `/Users/leonmartin_bih/projects/TVB-O/tvbo-manuscript/use-cases/replication_studies/Schirner2023/`
- **Recipe (the study spec):** `code/recipe/Schirner2023.yaml` (a `SimulationStudy`;
  `exp_34` is the group fit, `exp_30` the 1096-subject cohort, `exp_40/41/42` the Fig-4
  analyses, `exp_50/51` the DM circuit). Study-level `workflow:` block sets the dev
  container + slurm resources; per-experiment `algorithms:` (fic, fic_eib) live in
  `exp_30–34`.
- **Kit:** `output/cluster/schirner_full.tar.gz` (~536 MB, all 10 experiments as one
  Snakemake DAG, 1096 subjects' Glasser FC bundled, sha256 verified on the cluster).
- **Run docs:** `report/cluster_run.md` (the metadata-native run route, the BIH-CUBI
  facts, the `--bind /data/cephfs-1` requirement), `report/gap_analysis.md`.
- **Cluster setup scripts (committed with the study):** `code/tvbo_setup.sbatch`
  (compute-node SIF pull + orchestrator venv). NB the manuscript-repo changes
  (recipe container/`code_source`/`dataset.bundle`, gap_analysis, cluster_run.md,
  the setup sbatch) are the USER's repo — do not commit them without asking.

Emit / re-emit a single experiment for local codegen inspection (run from the study root):

```bash
cd .../use-cases/replication_studies/Schirner2023
/Users/leonmartin_bih/tools/tvbo/.venv/bin/tvbo workflow snakemake \
    code/recipe/Schirner2023.yaml --experiment 34 -o /tmp/emit34
cd /tmp/emit34 && PYTHONPATH=code \
    /Users/leonmartin_bih/tools/tvbo/.venv/bin/tvbo export tvboptim \
    spec/34/experiment.yaml -o /tmp/gen34.py   # inspect run_fic_eib / post_model_fn
```

Re-pack the full kit (used `--set container=<pre-staged SIF path>` +
`--set container_binds=/data/cephfs-1`); see the git history of `tvbo/cli/workflow.py`
and `report/cluster_run.md` for the exact `--pack` invocation.

Local venv/CLI for all of the above: `/Users/leonmartin_bih/tools/tvbo/.venv/bin/tvbo`
(NEVER run fits locally — the group fit OOMs a laptop; use the cluster diagnostic).

## Follow-up (separate goal): fold these lessons into the `/replicating-studies` skill

Once streaming BOLD lands and the Schirner run completes, harvest this whole exercise
into the replication skill. Edit the **canonical** copy (ships in the wheel), then
`tvbo skills sync` — never the generated `.claude/skills` copy (see
[[reference-skills-two-root-layout]]):

`/Users/leonmartin_bih/tools/tvbo/tvbo/skills/canonical/replicating-studies/SKILL.md`

It already has scale/streaming traps (lines ~219-224, ~315-320) but **no dedicated
cluster scale-out phase**. Add one, plus a "codegen memory traps" note, from what cost
us the most time this session. Concrete, generalizable lessons to encode:

- **Metadata-native cluster kit is one command.** `tvbo workflow snakemake <recipe>
  -o <out> --pack` emits the whole study as one Snakemake DAG. Env via
  `workflow.container:` (dev container + Apptainer), per-subject data via
  `Dataset.bundle: true`, custom builders via `code_source:`, cross-experiment order
  via `initial_state.from_experiment`. No venv, no bash, no hand-written sbatch. See
  [[reference-declarative-workflow-packaging]].
- **A dry run does NOT catch runtime bugs.** `snakemake -n` / `tvbo workflow submit
  --dry-run` resolves the DAG only; no rule executes. **Smoke-test ONE experiment end
  to end inside the container** (`apptainer exec … tvbo run spec/<id>/experiment.yaml`)
  before launching N — otherwise a per-rule bug fails all N identically (we hit this
  with 1106 jobs). Escalate: one experiment → its dependents → then the full submit.
- **The container filesystem is READ-ONLY — a whole bug class.** Anything that writes
  into site-packages or `$HOME/.cache` at import/run time fails only in-container:
  mako compiles templates into the package dir; `templateflow`/`matplotlib` write caches;
  a home that symlinks into another FS dangles. Fixes: writable temp for codegen caches,
  and **`--bind <site-root>`** (BIH: `--bind /data/cephfs-1`) declared via
  `workflow.slurm` container args. See [[reference-bih-cluster-apptainer-run]].
- **Version skew: the container runs PUSHED `dev`, the emitter is your working tree.**
  Schema/codegen changes only take effect after push → CI image rebuild → SIF re-pull.
  Emit-side fixes (freeze/packaging) only need a re-emit. Know which is which before
  assuming a fix is live. Re-pull asserts the fix is present (a SIF is named by the
  md5 of the URL, so a rebuild lands at the SAME path — force it).
- **Orchestrator belongs on a COMPUTE node, not the login node.** BIH login slices are
  cgroup-capped (~192 MiB) and OOM-kill snakemake; DAG resolution that takes 20 s on a
  compute node takes 25 min (or dies) on a starved login node. Run `snakemake`/
  `tvbo workflow submit` inside a `long`-partition job (resumable: snakemake skips done
  outputs). Use login-2 for orchestration if login-1 wedges.
- **Fitting experiments: two full-trajectory-materialization traps.** (1) `mode='all'`
  runs a spurious pre-tuning base sim — skip it for algorithm experiments (fixed:
  `run_main`). (2) The post-tuning FC eval materializes the whole trajectory — needs a
  **streaming reducer** (THIS handoff). Both bite only at the paper's real fitting length
  (long sims for stable FC), so they never show in a short smoke test. Rule for the
  skill: **at fitting/grid scale, every observable that drives a long sim must be a
  streaming reduction; a materialized trajectory is a memory bomb** (generalizes the
  existing Phase-4 streaming note to fits, not just parameter grids).
- **Reliability: big transfers over a flaky link → chunk + verify.** `scp`/`openrsync`
  (macOS) don't resume; split into ~32 MB chunks, size-verify each with retries,
  reassemble, then **sha256-verify** against the source (we shipped a stale-looking-fine
  kit twice before checking the hash). Prefer small (spec-only) uploads for iteration.

Also worth a skill asset: a `cluster-run.md.tmpl` mirroring the Schirner
`report/cluster_run.md` (the metadata-native run route + site facts), so each new
replication gets a ready run-doc.

## Key file:line references

- Post-eval OOM site: `tvbo/templates/tvboptim/tvbo-tvboptim-experiment.py.mako:3608`
  (`post_model_fn = prepare(..., t1=36e6)`); consumed
  `tvbo-tvboptim-algorithm.py.mako:999` (`post_tuning = post_model_fn(...)`), `:1011`
  (`compute_all_observations`).
- Reducer authoring: `tvbo/codegen/streaming_reducers.py` (`StreamingReducerSpec`,
  `register_streaming_reducer`, `lookup_streaming_reducer`, `_load_reducer_recipes`),
  `tvbo/codegen/reducers.py` (`resolve_streaming_reducer`),
  `tvbo/database/reducers/windowed_fc.yaml` (the template to mirror).
- The primitive: `tvbo/codegen/code.py:314` (`_afp_strided_convolve`) + `ARRAY_FUNCTIONS`.
- Reference impl to match: tvboptim `observations/tvb_monitors/bold.py::streaming_hrf_bold`
  (+ `HRFBold`, `FirstOrderVolterraHRFKernel`, `SubSampling`, `TemporalAverage`).
- Existing metadata (does NOT solve it, for context): `bold_hrf_strided.yaml` (strides
  the conv but still needs the full trajectory), `subsample.yaml` (the decimation op).
