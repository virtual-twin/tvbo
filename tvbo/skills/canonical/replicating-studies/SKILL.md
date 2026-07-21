---
name: replicating-studies
description: >-
  How to replicate a published study in TVBO — turn a paper into ONE declarative,
  fully tvbo-native `SimulationStudy` (any kind: single-node bifurcation to whole-brain
  network; forward simulation, parameter sweep, or fit to data) + simple plotting + an
  honest, fully-computed report. Encodes the hard-won rules so the replication is fast and
  trustworthy. Composes the atomic skills (writing-models, running-simulations, writing-reports).
metadata:
  audience: user
  applies_to:
    - "**/*.yaml"
    - "**/*.qmd"
    - "**/*.py"
  tags: [replication, workflow, reporting]
  requires_extras: []
---

# Replicating a study in TVBO

You are reproducing a published paper as a **single declarative TVBO recipe** plus
minimal plotting, with a report whose every number is computed from the run — never
typed by hand. This skill owns the *replication-specific* layer; for the atomic
how-to it defers to **writing-models** (Dynamics YAML), **running-simulations**
(sourcing / CLI / backends), and **writing-reports** (the IMRAD report itself).

**It covers any study expressible as a TVBO `SimulationStudy`** — locate yours on three
axes (do this in Phase 1.5; it decides which phases and backend features apply):

- **Scale** — single node (bifurcation / phase portrait, *no network or coupling*) ·
  few-node motif · large network (abstract graph or brain connectome).
- **Mode** — forward simulation · parameter **sweep / bifurcation / continuation** ·
  **fit / inference** to data (an optimisation experiment, not just forward runs).
- **Data** — self-contained (all params from the paper's equations/tables, *no external
  inputs*) · external inputs required (a network, empirical target, stimulus).

The invariants (MUST-rules) hold for every kind; the examples below are illustrations,
not requirements. Work the *applicable* phases in order — a self-contained bifurcation
study skips the data phase, a fit adds an inference experiment, a single-node study has
no network. Each phase you do has a REQUIRED output. The scorecard in Phase 6 maps 1:1
to the criteria you write in Phase 1, and Phase 7 verification is what stops you trusting
figures that silently integrate the wrong attractor.

## The non-negotiables (MUST)

1. **ONE declarative recipe.** All targeted experiments live in one `<Study>.yaml`
   as metadata (anchors + `<<:` inheritance, `from_experiment` seeding). **No Python
   drivers.** `tvbo run <Study>.yaml` runs everything in dependency order; add
   `--experiment 2,3` to run a subset.
2. **Nothing hardcoded in the report.** Every reported value is computed inline from
   a result container (`output/nc/exp*/…h5`) or the recipe metadata — counts, ⟨Δω⟩,
   decay times, bifurcation thresholds, scaling exponents, spectral peaks, fitted params,
   correlations, whatever the paper reports. If you typed a number into prose, it is a
   bug. (Papers are not ground truth; your own asserted numbers are not either.)
3. **A panel shows TVBO output or an honest placeholder — NEVER the paper's replotted
   source data.** Replotting the source arrays is a dev check that plotting *works*; it is
   never a deliverable panel (it passes off the paper's own numbers as your reproduction).
   If a panel's TVBO data isn't ready, render a labelled placeholder holding its slot in the
   paper's layout. This is the integrity line — do not cross it "just to fill the figure".
4. **Backend-independent metadata, backend chosen by fit.** The YAML states *intent*,
   never one backend's mechanism. The execution backend is picked in Phase 1.5 from the
   targets' feature needs, not defaulted.
5. **FAIR layout, spec separate from code** (copy `assets/skeleton/`): the recipe
   `<Study>.yaml` sits at the **study root** — the spec is not hidden inside `code/`.
   Its callables live in `code/recipe/`, reached declaratively via
   `code_source: ./code/recipe` (a local path, or a git repo — see **running-simulations**),
   so `tvbo run <Study>.yaml` finds them without a driver. `code/` also holds the prep
   script, analysis callables, one reference integration, and `figures/plot.py`;
   `original_study/` the paper + analysis; `input/` the data provenance; `report/` the
   report source.
6. **Nothing large or upstream is vendored — gitignore it and document exact retrieval.**
   Git tracks only what you author: the spec, `code/`, `input/DATA.md`, and the report
   source (`report.qmd` + `references.bib` + the prose writeup). **Everything else is
   gitignored:** `output/` and all generated artifacts (figures, `report.pdf`/logs,
   KPI/targets tables — write them to `output/`, never commit them at the study root),
   the paper's own material under `original_study/`, and raw third-party inputs under
   `input/sourcedata/`. Planning/working docs go to a gitignored `_dev/`. A fresh clone
   is small and reproducible; `DATA.md` says how to obtain every ignored input.
7. **Replication, stated honestly.** Frame it as *replication* (independent code +
   independently-sourced data → same conclusions), not bit-exact *reproduction*. Ship a
   **scorecard** (met / partial / out-of-scope) with a **fidelity tier per target**
   (mechanism-level vs decimal-level, Phase 1.5) and name the **accepted limitations**
   (unavailable exact SC, unpublished-seed realization dependence) up front.
8. **One plotting script**, `code/figures/plot.py`, one `main()` whose sections mirror
   the paper's figure *types* (time series / phase portraits / bifurcation diagrams /
   spectra / sweeps / spatial maps), then compose A/B. Simple matplotlib next to the recipe.
9. **Verify against an independent reference** (Phase 7) before trusting any figure.

---

## Phase 1 — Analyze the paper → `targets.md` + `figures.md`

Read the version of record (put it in `original_study/`, figures as `img/fig*.png`).
Produce two artifacts under `original_study/analysis/`:

- **`targets.md`** — a numbered table of replication targets `T1..Tn`. Each row:
  target · figure(s) · **key verbatim params** (copy them exactly — K, α, τ, seeds,
  transient/window times, step sizes as *printed*) · a **pass/fail validation
  criterion** · a feasibility/tier tag (`core` / `extended`).
- **`figures.md`** — per-figure panel map: panels, axes + ranges, colour convention,
  line styles, any quirks to reproduce as-is (mislabelled axes, unit conventions).

Watch for the trap that the *printed* equation is not the one the figures use (Taher's
Eq. 9 has a √N normalization typo; the figures use the plain std). Record the quantity
the *figures* actually show, with the discrepancy noted.

## Phase 1.5 — Scope, then backend-fit + gaps → `backend-fit.md`

**Scope.** Pick which targets to replicate: **all** (default) or a **selected subset**
(`{T1,T2,T7}`). Only selected targets become experiments in Phase 3. If the scope is
contested, settle it with the user before continuing — do not guess.

**Backend-fit + gaps** (`original_study/analysis/backend-fit.md`). For the selected
targets, build a feature matrix (delays? Lyapunov/Benettin? adiabatic sweep? noise?
multi-mode? time-gated events? sparse coupling?) and pick the execution backend that
supports them — **with rationale**. tvboptim (JAX) is common because delays, Lyapunov
and adiabatic `lax.scan` sweeps are tvboptim-gated today; plain forward sims and
operating points run on any backend. **Surface feature gaps now**: a need not yet
supported (e.g. the Lyapunov exponent of a *delayed* closed loop under `vmap`) BLOCKS
its target — flag it as a framework/schema enhancement before you build, and mark the
target `partial`/`out` in the eventual scorecard. This early gap-finding is what sets
honest expectations instead of surprising you mid-YAML.

**Data obtainability + fidelity tier — decide BEFORE building (the biggest time-saver).**
Tag every target with a **fidelity tier**: *mechanism-level* (a sign / pattern / ordering
that reproduces on any reasonable input — the paper's central claim) vs *decimal-level* (a
specific number that needs the paper's exact input). Then confirm that exact input is
actually obtainable *now* — papers routinely deposit only raw login-walled data (no derived
matrix), link a code repo that 404s, or name the wrong author. If the exact input
(connectome, empirical FC, seeds) is not obtainable, choose a **documented substitute** and
downgrade its decimal-level targets to mechanism-level up front — do not start a hunt for a
file that may not exist. (Koller: the exact HCP-S900 Schaefer-1000 SC was unobtainable → we
used the dTOR Schaefer-1000 SC, which reproduces the in-strength→wave *mechanism* but caps
the edge-level FC number; deciding this early would have saved a long fruitless hunt and set
honest Fig-8 expectations from the start.)

## Phase 2 — Source the data → `DATA.md` (tracked) + gitignored data dirs

**Skip this phase if your study is self-contained** — a bifurcation / phase-portrait /
normal-form study whose every parameter comes from the paper's equations and tables needs no
external data; then `DATA.md` is one line ("no external inputs; all parameters from <paper>
§X / Table N"). Otherwise, for studies with a network, empirical target, or stimulus input:

Write `input/DATA.md` (from `assets/DATA.md.tmpl`) as the **one tracked pointer** to every
input: exact upstream source (author, year, DOI, licence), the sheet/column → paper-quantity
map, checksums, **exact download + regenerate steps**, and which quantities are synthesised
vs sourced. Name the true upstream source, never a derived intermediate.

**Do not vendor sizable or upstream data into git — gitignore it and document how to fetch it.**
Place data by provenance:

- the **paper's own published data** (its source-data workbook/arrays, and your extraction
  of them into `.nc`/etc.) → `original_study/`, with the rest of the paper's material
  (gitignored — it is the paper's content, regenerable from the raw per `DATA.md`), *not*
  `input/derivatives/`;
- **third-party raw inputs** you feed the model (connectomes, atlases) → `input/sourcedata/`
  (gitignored);
- only genuinely small, freely-redistributable open inputs may be carried in git.

`code/`'s prep script reads the raw source and emits the tvbo-ready `Network` (+ small
CSV/npz for inspection) into a gitignored location; do NOT hand-edit derived artifacts.

**(Network studies) pin the parcellation AND its order variant up front, then verify the
mapping.** An atlas often ships in several orderings of the *same* parcels (Schaefer-1000 `7Networks` vs
`17Networks` = identical parcels, different node order); using the wrong one silently
scrambles every brain map and correlation without erroring. Fix the paper's exact atlas +
order variant, align every array **by parcel label** (never by position — guards hemisphere
swaps), and **verify with a self-consistency correlation**: a quantity mapped against itself
under your alignment must give +1.000 (e.g. in-strength computed from the SC vs the paper's
published in-strength). A silent order bug is among the costliest — it yields plausible,
wrong figures.

## Phase 3 — The recipe: one tvbo-native `<Study>.yaml`

See **writing-models** for the Dynamics form and **running-simulations** for sourcing
(inline vs YAML vs `iri`) and the CLI. Replication-specific rules:

- **Encode what the study IS; only `Dynamics` is mandatory.** A single-node study carries no
  `network`/`coupling`; a multi-node study adds them (abstract graph or brain connectome). A
  parameter sweep / bifurcation diagram is an `Exploration` over the swept parameter with a
  fixed-point / eigenvalue / peak observation; a **fit to data is an inference / optimisation
  experiment** (see **running-simulations**) whose target is the recovered parameter +
  goodness-of-fit, not a trajectory. Match the paper's structure, not a template's shape.
- **A swept *branch* is not a plain grid — read `assets/sweeps.md` first.** If the paper's
  sweep tracks a hysteresis / partial-sync branch, a continuation, or a per-value analysis
  (λ₁(K), eigenvalues), that file covers warm-start branch tracking (`sweep_seeding:
  from_previous` + `bidirectional`), restarting a per-point analysis over the recorded branch
  (`from_experiment` / `source_point: branch`, shardable), and IC-seed ensembles
  (`distribution.seed` vs `execution.random_seed`). A product grid over independent cells needs
  none of it.
- **Spec at the root, callables in `code/recipe/` via `code_source`.** The study declares
  `code_source: ./code/recipe` (a local path) or a `{git, ref, subdir}` repo; tvbo puts that
  dir on the import path so `callable: {module: <study>_analysis}` resolves with no driver and
  no vendored package (falls back to a `code/` convention if unset). This keeps the spec at the
  top level while its code lives under `code/`.
- One file; shared `&dynamics` / `&params` / `&network` anchors; per-experiment `<<:`
  overrides. Order experiments so a `from_experiment` source precedes its dependents
  (operating point before its control runs) — then bare `tvbo run <Study>.yaml`
  resolves the seeds in one pass.
- Non-obvious params get a one-line comment tying them to the paper (equation/figure).
- Overriding a param replaces it wholesale (YAML merge is shallow) — restate `unit`/
  `description`, or don't override when the anchor default already matches.
- Encode the *intent* declaratively (gates via a Piecewise/`autonomous:false` RHS,
  adiabatic branch via `Exploration.sweep_seeding: from_previous`, delayed self-terms
  via the coupling graph) — not a backend mechanism.

## Phase 4 — Analysis callables (only for non-closed-form pipelines)

Study-specific reductions — order parameters, bifurcation / fixed-point detection, spectral
peaks, control masks, fit residuals — → pure, backend-agnostic, independently-testable
functions in `code/<study>_analysis.py`, referenced from the recipe via
`callable: {name, module: <study>_analysis}`. Keep them NumPy; carry data as **labelled
xarrays**, never positional reshapes. When aligning a
paper's connectome/observable to your node order, match **by label**, never by position
(guards silent hemisphere/order swaps). Note the host/grid split: *declared* observations
run on the host (plain NumPy is fine); only what you put under `record:` runs inside the
jitted/vmapped grid and must be backend-traceable (a non-traceable recording raises).

Two scale/ensemble traps: (1) a **trial ensemble needs per-cell reseeding** — add an
`execution.random_seed` sweep axis so each trial draws a fresh PRNG key; a codegen-constant
key makes every "trial" identical (a degenerate ensemble that silently reads as zero
variance). (2) **At grid scale, record a reduced/streaming observable, never raw
trajectories** — a full θ/voltage trace over a 15k-point grid is terabytes; a streaming
reduction (e.g. effective frequency accumulated online) keeps resident memory ~constant
(block-size, not trajectory-length), so the whole grid vmaps on one GPU with no sharding.

## Phase 5 — Plotting: one `code/figures/plot.py`

Copy `assets/plot.py.tmpl` (one `main()`, one section per paper figure *type* — time series /
phase portrait / bifurcation diagram / spectrum / sweep / spatial map). `plot.py` writes only
**our reproduction** figures (`FigN_<study>.png`) to `output/figures/`. The A/B pairing with
the paper original is a **report-render concern**, gated for copyright by the Phase-6
internal/public profile (the report's `ab()` helper draws the original only in the internal
build) — do **not** bake the copyrighted original into a committed/shared image. (`assets/
compose_ab.py` pre-composes an `ab_fig{N}.png`; if you use it, that composite embeds the ©
original, so treat it exactly like the internal build — git-ignored, never in the public
report.) Read the native result containers directly. Always draw the paper's full multi-panel
layout; a not-yet-run panel renders a labelled placeholder.

**Render spatial data in the paper's coordinate/surface convention**, not a convenient
substitute — e.g. plot brain-region values at the paper's surface parcel centres (Koller uses
the fsaverage5-*inflated* COM), not the raw MNI centroid; the convention changes the figure's
look and can misalign, so derive the coordinates from the paper's surface and verify the
mapping **by label**. And **guard a multi-panel figure by data-requirement, not per figure**:
a group-level panel often reproduces from group data while its per-subject sibling stays
blocked on per-subject data — split the guard so the reproducible panels ship and only the
blocked ones are placeholders.

Keep any extracted **paper source arrays label-keyed** (`xarray` with named coords), not
encoded into filenames — tvbo's declarative figure spec (`dev/figure-spec-design.md`) will
bind figure data by IRI + `output` + `sel`, so a flat per-panel `.nc` set is a fine stopgap,
but an elaborate filesystem-keyed tree is throwaway. Don't over-build it before the renderer.

## Phase 6 — Report: `report/report.qmd` (every number computed)

See **writing-reports** for the report mechanics: the IMRAD structure, the metrics cell
that computes every number from the containers (nothing hand-typed), the native
`EXP.dynamics.generate_report(...)` equation and parameter render, the three-colour status
callouts, the copyright-safe internal/public profile split, the xelatex/LaTeX rules, and
the anti-slop prose standard. The templates it copies ship in this skill's `assets/`:
`report.qmd.tmpl` plus `_quarto.yml.tmpl` and `_quarto-internal.yml.tmpl`. Copy all three
into `report/`.

Replication-specific rules on top of that mechanics:

- **The scorecard maps 1:1 to `targets.md`.** Every criterion `T1..Tn` from Phase 1 is one
  row (met / partial / out), tagged with its Phase-1.5 **fidelity tier**: *mechanism-level*
  (a sign or ordering that any reasonable input reproduces) vs *decimal-level* (a number that
  needs the paper's exact input). Derive the verdict from the data, never assert it.
- **Reproduction vs. replication (NASEM framing).** Frame the study as replication, not
  bit-exact reproduction, and split the mechanism-level targets (they reproduce) from the
  decimal-level ones (capped by unavailable inputs, stated as accepted limitations). This
  section ties the Discussion's claims back to the Phase-1.5 tiers.
- **The negative result is the integrity test (rule #3).** State it in Results with its
  evidence and interpret it in Discussion; a not-yet-run panel is a labelled placeholder with
  a yellow "missing" callout, never the paper's replotted data passed off as a result.
- **Methods sections map to the earlier phases:** the native equation render, variants,
  coupling, and network/data-provenance (Phase 2), the analyses (Phase 4), the backend
  (Phase 1.5), and the Phase-7 verification. Results holds the scorecard and the per-figure
  `ab()` calls; Discussion holds the NASEM framing, the mechanism of any negative result, and
  the accepted limitations.

## Phase 7 — Verify against an independent reference

Before trusting figures, validate the recipe's core dynamics against a **standalone
reference integration** of the paper's governing equation in `code/<study>_reference.py`
(plain NumPy, or another backend) — recipe output must match it (byte-exact, or within a
stated tolerance). Where feasible, also cross-check via `render_code('tvb')` vs
`render_code('tvboptim')`. This is what catches modelling bugs a PDF can't: e.g.
per-step vs per-stage coupling evaluation converging to a *different attractor*.

**Attribute a residual gap to data vs implementation with a head-to-head.** When a metric
misses the paper's number, install the paper's *own* tooling and run it on your *exact*
inputs before blaming your code. (Koller: running his native `tvb-library` model on the same
substitute SC gave FC r=0.27 — the same as tvboptim's 0.32 — proving the shortfall was the
connectome, not the engine; without it we'd have chased an implementation bug that wasn't
there.)

## Phase 8 — Scale out to a cluster (ONLY when one node genuinely won't do)

**Skip this phase unless the work is irreducibly large** — a per-subject cohort (one
independent fit × N subjects) or a fit whose single run is itself heavy. First try NOT
to need it: a big *graph* → `graph_representation: sparse` + vectorized coupling; a big
*parameter grid* → a streaming reduced observable (Phase 4). Both routinely turn a
"needs HPC" run into minutes on one GPU, numerically identical (~1e-16). Assess this
before packaging anything.

REQUIRED output: a packed kit + a `report/cluster_run.md` (the run route + site facts).

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
- **A dry run does NOT execute anything — smoke-test ONE experiment in the container
  FIRST.** `tvbo workflow submit --dry-run` (snakemake `-n`) only resolves the DAG
  (wildcards, inputs, resources); no `tvbo run` executes, so it cannot catch a runtime
  bug. A per-rule bug fails all N jobs identically (we once launched 1106 that all died
  the same way). Before the real submit, run a single experiment end-to-end inside the
  SIF (`apptainer exec --bind … <sif> tvbo run spec/<id>/experiment.yaml`), then its
  dependents, then the full submit. This is Phase 7's "run END-TO-END, not `from_file`"
  at cluster scale.
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
- **Run the orchestrator on a COMPUTE node, not the login node.** Login nodes are
  cgroup-capped (a per-user memory limit that OOM-kills a long `snakemake`); DAG
  resolution that takes seconds on a compute node crawls or dies on a starved login
  node. Wrap `tvbo workflow submit` in a long-partition job — it is resumable
  (snakemake skips completed outputs, so a walltime cap just means resubmit). Never
  install or build on the login node.
- **Big, flaky transfers: chunk + checksum.** A multi-hundred-MB kit over an unreliable
  link won't survive one `scp`/`rsync` stream (macOS ships `openrsync`, which doesn't
  resume); split into ~32 MB chunks, size-verify each with retries, reassemble, then
  **sha256 the result against the source** — a stale-but-right-sized kit passes a
  byte-count glance (we shipped one twice before checking the hash). Iterate with small
  (spec-only) uploads, not the full kit.

---

## Dynamical & numerical traps (these cost us the most time)

- **The integrator, not the physics, can move the attractor.** An explicit scheme
  (Heun/RK2) at too large a `dt` sustains lightly-damped fast librations at high
  coupling: the *time-averaged* spread climbs and reads like desynchronization, but it is
  numerical. Halve `dt` (or switch to RK4) and confirm the transition / operating point
  are unaffected. A paper's quoted "Δt" is often a Lyapunov / rescaling unit, **not** the
  solver step — do not copy it into `step_size`. Phase 7's reference integration is how
  you tell numerical drift from real dynamics.
- **Seeding a *delayed* system needs the delay HISTORY, not just a snapshot.** A
  `from_experiment` seed carries the state but not the τ seconds of history a delayed term
  reads; feedback engaging against an unfilled buffer spikes or fails to converge. Fill it
  with a τ-second **transient pre-roll** (`transient_time: τ`) that replays the operating
  point; the recorded onset is then `gate.t_on − transient_time` (derive it, never hardcode).
- **Big graphs: make the coupling sparse/vectorized before reaching for HPC.** For an
  N-node grid the per-step dense N×N coupling matmul dominates; `network.graph_representation:
  sparse` (with a factored/angle-addition coupling) turned a multi-hour sweep into minutes
  locally, numerically identical (~1e-16). Assess this first — often no cluster is needed.
  For a big *parameter* grid, pair this with a streaming reduced observable (Phase 4) so vmap
  memory stays bounded and the whole grid fits one GPU.
- **A fit at the paper's real length: EVERY long-running observable must stream, and
  the pre-tuning base sim is spurious.** The Phase-4 streaming rule is not just for
  parameter grids. A fit runs the paper's actual simulation length (long, for stable
  FC/statistics), and a post-hoc observation that stacks the full trajectory at that
  length is enormous: Schirner's 10 h × dt=1 ms × 379 nodes × 4 states ≈ 440 GB for ONE
  FC evaluation → OOM even on a highmem node. Compute BOLD/FC/moments as **streaming
  reductions** (fold-in-carry over a block scan) that never materialize the trajectory —
  the result is byte-identical. Two materialization traps specifically: (1) a fitting
  experiment's *pre-tuning* forward sim is not a deliverable (the tuning algorithm is);
  don't run a full-length materialized base sim before it. (2) the *post-tuning*
  evaluation must stream too. Neither shows in a short smoke test — reason about
  resident memory = `n_steps × n_nodes × n_states × 8 B` up front, and if a needed
  streaming observable doesn't exist yet, that's a Phase-1.5 framework gap.
- **Metastable / FC metrics are duration-, trial-, and regime-sensitive — don't call a
  ceiling early.** A single short run's FC/PLV/order-parameter is noise-dominated (one lucky
  trial read 0.17; the 8-trial mean was 0.09). Match the paper's **full duration and trial
  count**, and locate its **operating regime** (the near-critical (K, v) pocket a paper's 2-D
  grid exists to find) before concluding "structure-limited". We twice declared a gap that
  duration/trials/regime then closed.
- **Reconcile the coupling scale with the paper's weight normalization.** A global K that
  looks orders of magnitude off is usually a normalization convention, not a bug: a coupling
  `a·gx/N` on *raw* SC (in-strength ~1e4) puts the operating point near K~1e-6, vs the paper's
  K~0.03 on *normalized* SC. Match how the paper normalizes weights before sweeping K, or the
  sweep hunts the wrong decade.

## Pitfalls we hit (so you don't)

- **A metric's *definition and the empirical modality it's compared against* are part of the
  claim — read them from the METHODS, not the figure caption.** t_c (1/e vs exponential-fit),
  ⟨Δω⟩ (std about the mean vs the median), λ₁ units; and *what* the sim is compared to
  (Koller's Fig-8 "FC" is band-specific **MEG-PLV**, not fMRI — sim FC is PLV on the
  off-diagonal). Pick a documented definition, state it, compute it. A magnitude that differs
  may be a unit/rescaling convention rather than a physics gap — but **confirm that from the
  methods**, don't assume it (we labelled it "likely" and it stayed unverified).

- **Coupling evaluated once per step** silently integrates a different, multistable
  attractor. Use `Integrator.coupling_evaluation: per_stage` for chaotic/multistable
  systems and verify against the reference (Phase 7).
- **Hardcoded fidelity numbers** creep into captions ("t_c ≈ 2.6 s") and read as
  matches when they aren't. Compute them (Phase 6). A recomputed value that *differs*
  from the paper is honest; a typed one that matches is not.
- **Realization dependence.** Exact solitary counts / magnitudes depend on unpublished
  seeds — count median-relative, state the difference as an accepted limitation, don't
  chase the integer.
- **Geometry / eigenmode decompositions: match the *invariant*, not the magnitudes.**
  Reproduce the paper's exact operator (e.g. an `igl` cotangent-Laplacian at the paper's mesh
  resolution — parcel-level, not a dense-surface substitute); the reproduced result is the
  modal *structure* and where power concentrates (a field living in the lowest spatial-frequency
  modes), while absolute scales (wavelengths) track the surface mesh — inflated meshes differ
  ~1.3× across sources. Report it mechanism-level with the scale caveat.
- **Large or derived array constants: declare their provenance, never inline them.** A mesh
  operator, an empirical matrix, or any precomputed array a model/observation consumes is a
  `Parameter` declared by *where it comes from*, not a literal: `source:` (WHERE) + `measure:`
  (WHICH key) for an existing file, or `producer:` (a `FunctionCall` — HOW to compute it) for
  one derived from the study's own inputs (arguments may reference `network.nodes.position` /
  `network.mesh.*`). Sourced/produced values are resolved lazily and materialised to a
  content-addressed companion — never baked into generated source (a 66 MB operator inlined is
  a source file that will not compile). Reserve inline `value:` for genuine scalars/small
  arrays. This keeps the spec the single source of truth (a pre-built file drifts from the mesh
  it came from) and the emitted code self-contained.
- **Some targets are irreproducible from the paper's OWN source data.** A panel can be
  internally inconsistent in the published workbook (Koller Fig 2e: the per-node spread
  disagrees across the steady-state vs transient windows) — a source-data defect, not a
  model gap. Identify these, scope them `out`, say why; don't chase them.
- **Redundant scripts.** One prep script (emits the tvbo Network directly), one plot
  script. Don't split what one `main()` can do.
- **No dead vendored cruft — but a *live* dependency is not cruft.** Keep ONE pristine copy
  of the paper's own code under `original_study/`; don't duplicate it into `code/`. If the
  paper's algorithm is reused at runtime (e.g. a Helmholtz–Hodge flow-potential), *reference*
  that one copy (put its dir on `sys.path`), don't re-vendor. **Before deleting vendored code
  as "unused", confirm it against the actual RUN paths — run a representative experiment
  END-TO-END, not just `from_file` load.** Loading a study does not import a
  flow-potential/observation callable, so a load-only check will wrongly call a live
  dependency dead (this cost us a broken flow-potential path).
- **Generated files never land in git at the study root.** KPI/targets tables, extracted
  arrays, the report PDF/logs → write them into `output/` (gitignored). A generated file
  tracked at the root reads as a hand-curated deliverable and silently drifts stale.
- **Cross-references.** The report must stand alone — no "as in the sibling X study".
- **Framework gaps surface late** if you skip Phase 1.5. Find them before the YAML.
