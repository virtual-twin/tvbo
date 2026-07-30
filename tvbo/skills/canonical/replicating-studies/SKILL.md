---
name: replicating-studies
description: >-
  How to replicate a published study in TVBO — turn a paper into ONE declarative,
  fully tvbo-native `SimulationStudy` (any kind: single-node bifurcation to whole-brain
  network; forward simulation, parameter sweep, or fit to data) + declarative figures + an
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

You are reproducing a published paper as a **single declarative TVBO recipe** — its
experiments **and** its figures (a `figures:` block, rendered by codegen) — with a report
whose every number is computed from the run, never typed by hand. This skill owns the
*replication-specific* layer; for the atomic
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

1. **ONE declarative recipe, rooted at `<Study>.yaml`.** All targeted experiments **and**
   the figures that read them are metadata under a single root `<Study>.yaml` (anchors +
   `<<:` inheritance, `from_experiment` seeding; a `figures:` block — non-negotiable #8).
   It need **not** be monolithic: split a large spec with **`!include`** — the root stays a
   thin entry that `!include`s reusable fragments (a shared Dynamics, an algorithm block,
   per-experiment files) from sibling spec files at the root (never under `code/`), and
   references curated components by `iri:`. **No Python drivers.** `tvbo run <Study>.yaml`
   runs every experiment in dependency order and then renders the declarative figures — one
   command produces results **and** figures; add
   `--experiment 2,3` to run a subset (`--no-figures` to skip rendering).
2. **Nothing hardcoded in the report.** Every reported value is computed inline from
   a result container (`output/nc/exp*/…h5`) or the recipe metadata — counts, ⟨Δω⟩,
   decay times, bifurcation thresholds, scaling exponents, spectral peaks, fitted params,
   correlations, whatever the paper reports. If you typed a number into prose, it is a
   bug. (Papers are not ground truth; your own asserted numbers are not either.) The rule
   is **asymmetric**: the *paper's* quoted values stay literals (you can't recompute someone
   else's number), the bug is a hardcoded *result of yours* — so **audit before shipping**:
   grep the prose for numeric literals and classify each as yours (compute it into `M`) or the
   paper's (quote it). A report can read as fully computed and still hide a typed peak or step
   size (see **writing-reports**).
3. **A panel shows TVBO output or an honest placeholder — NEVER the paper's replotted
   source data.** Replotting the source arrays is a dev check that plotting *works*; it is
   never a deliverable panel (it passes off the paper's own numbers as your reproduction).
   If a panel's TVBO data isn't ready, render a labelled placeholder holding its slot in the
   paper's layout. This is the integrity line — do not cross it "just to fill the figure".
4. **Backend-independent metadata, backend chosen by fit.** The YAML states *intent*,
   never one backend's mechanism. The execution backend is picked in Phase 1.5 from the
   targets' feature needs, not defaulted.
5. **FAIR layout — spec (metadata) at the root, code in `code/`** (copy `assets/skeleton/`):
   the recipe `<Study>.yaml` sits at the **study root**, never inside `code/` — the spec is
   backend-independent metadata, kept separate from code (that split is the point). Its
   callables — model builders, analysis callables, **and the bespoke figure panels/transforms** —
   live **flat in `code/`**, made importable by the zero-config `code/` convention: loading the
   study puts `code/` on the path, so every `module:` / `callable:` / `code_modules:` resolves by
   bare name — no driver, no `PYTHONPATH`, no `code_source`. (Set `code_source` **only** to point
   the importable code *elsewhere* — a git repo or a shared directory — never at a local `code/`
   subfolder; a `code/recipe/` split buys nothing and breaks imports if the line is forgotten.)
   `code/` also holds the prep script and one reference integration; `original_study/` holds the
   **paper's own material ONLY** (fully git-ignored); `input/` the data provenance; `report/` the
   report source **and everything we author or generate** — including our replication analysis under
   `report/analysis/` (targets, figures, backend-fit, adherence). Rendered figures and their
   generated `plot_<name>.py` scripts land in the gitignored `figures/` — images at its
   root, scripts in `figures/scripts/`.
6. **Nothing large or upstream is vendored — gitignore it and document exact retrieval.**
   Git tracks only what you author: the spec, `code/`, `input/DATA.md`, and the report source
   (`report.qmd` + its `report_internal.qmd` wrapper + `_quarto.yml` + `references.bib` +
   `report/analysis/`). **Everything else is gitignored:** `output/` and all generated artifacts
   (figures, `report.pdf`/logs, KPI/targets tables — write them to `output/`, never commit them at
   the study root), **all of `original_study/`** (the paper's own material — fully ignored; nothing
   of ours lives there), and raw third-party inputs under `input/sourcedata/`. Planning/working docs
   go to a gitignored `_dev/`. A fresh clone is small and reproducible; `DATA.md` says how to obtain
   every ignored input. (**`.gitignore` has no inline/trailing comments** — a `#` after a pattern
   becomes part of the pattern and silently breaks it, e.g. an un-ignored `figures/` or a dropped
   `original_study/` exclusion; keep every comment on its own line.)
7. **Replication, stated honestly.** Frame it as *replication* (independent code +
   independently-sourced data → same conclusions), not bit-exact *reproduction*. Ship a
   **scorecard** (met / short / out / blocked -- see below) with a **fidelity tier per target**
   (mechanism-level vs decimal-level, Phase 1.5) and name the **accepted limitations**
   (unavailable exact SC, unpublished-seed realization dependence) up front.
8. **Figures are declared metadata, not a plotting script.** Each paper figure is a
   `Figure` in the study's `figures:` block (layout mosaic + panels + PROV `used`
   bindings); `tvbo figure render <Study>.yaml` — run automatically by
   `tvbo run <Study>.yaml` — emits a self-contained render script per figure **and**
   runs it. Grammar panels (`cartesian`/`heatmap`) need **zero** plotting code; only a
   genuinely bespoke panel interior is a registered `@bsplot.register_panel` callable in a
   `code_modules` module under `code/`. **No hand-written `main()` plotting driver.**
9. **Verify against an independent reference** (Phase 7) before trusting any figure.

---

## Phase 1 — Analyze the paper → `targets.md` + `figures.md`

Read the version of record (put it in `original_study/`, figures as `img/fig*.png` — the paper's
own material, fully git-ignored). Produce two artifacts of **your own** under `report/analysis/`
(tracked — our work, not the paper's):

- **`targets.md`** — a numbered table of replication targets `T1..Tn`. Each row:
  target · figure(s) · **key verbatim params** (copy them exactly — K, α, τ, seeds,
  transient/window times, step sizes as *printed*) · a **pass/fail validation
  criterion** · a feasibility/tier tag (`core` / `extended`).
- **`figures.md`** — per-figure panel map: panels, axes + ranges, colour convention,
  line styles, any quirks to reproduce as-is (mislabelled axes, unit conventions).
- **`methods-vs-code.md`** — the **divergence register** (REQUIRED whenever the study ships
  code; see below). Started in Phase 1, grown through every later phase, and surfaced as a
  first-class section of the report.

### The divergence register — why this is a headline deliverable, not bookkeeping

Whenever a paper ships code, **assume its prose and its source describe different models until
you have checked**, and keep a register of every place they do. This is not incidental: it is
the failure mode a declarative recipe exists to remove, so documenting it is a primary result of
the replication, not a footnote. In TVBO the spec **is** the executable artifact — there is no
second description that can drift — and the register is the evidence for that claim.

Classify each entry, because the classes have different detectability and different fixes:

| Class | What it is | Typical tell |
|---|---|---|
| **A. Value drift** | same symbol, different number in code | a constants file disagrees with a table |
| **B. Algorithm substitution** | code computes a *different operation* than the printed equation | an "integral" implemented as a least-squares solve |
| **C. Undocumented configuration** | a choice the paper never states at all | which of several shipped bases; how many modes; which mask |
| **D. Underdetermined prose** | text admits several readings, one correct | where an average sits relative to a nonlinear step |
| **E. Convention traps** | same name, different meaning across files | id numbering, time units, initial conditions |

Record for each: what Methods says · what the code does · **how you established it** (read vs
verified) · whether it changes a reported number. Keeping "read" and "verified" distinct is what
stops the register from becoming a second layer of assumptions — see the assumption-labelling
rule in Phase 7.

Two lessons from Pang2023, where 14 divergences were found and 8 changed a number: the ones that
bite hardest are **C** (four cases — including the paper using *two different eigenmode bases*
for different figures and saying so nowhere) because nothing in the text hints they exist; and
**B** is the most damaging to a reader, because someone implementing the printed equation will
not reproduce the figures. Note also that the register is only *visible* for open deposits — a
paper without released code has the same drift and no way to see it, which is worth saying
plainly in the report rather than implying the open paper is the sloppy one.

Watch for the trap that the *printed* equation is not the one the figures use (Taher's
Eq. 9 has a √N normalization typo; the figures use the plain std). Record the quantity
the *figures* actually show, with the discrepancy noted.

## Phase 1.5 — Scope, then backend-fit + gaps → `backend-fit.md`

**Scope.** Pick which targets to replicate: **all** (default) or a **selected subset**
(`{T1,T2,T7}`). Only selected targets become experiments in Phase 3. If the scope is
contested, settle it with the user before continuing — do not guess.

**Backend-fit + gaps** (`report/analysis/backend-fit.md`). For the selected
targets, build a feature matrix (delays? Lyapunov/Benettin? adiabatic sweep? noise?
multi-mode? time-gated events? sparse coupling?) and pick the execution backend that
supports them — **with rationale**. tvboptim (JAX) is common because delays, Lyapunov
and adiabatic `lax.scan` sweeps are tvboptim-gated today; plain forward sims and
operating points run on any backend.

**Pin the model CLASS from the supplement, not the main text — it decides the backend features.**
A biophysical paper's main text rarely tabulates the network; the operative parameters *and the
model class* live in the SOM / supplementary methods. Get that document, and cross-check a
published reproduction's parameter file (a NEST/Brian2/Auryn repo), flagging where the
reproduction re-tuned a value. The class is a first-order fork the backend-fit turns on:
current-based (`τ_m dV = -V + I`, drives in mV) vs conductance-based (`C dV = g_L(E_L-V) + …`);
instantaneous δ-PSC (a v-jump per spike, no synaptic time constant) vs kinetic conductance;
Gaussian white-noise vs Poisson external drive; sparse-random vs all-to-all wiring. Reading the
class off the main text — or off a template that happens to be conductance-based/Poisson — is how
you build a plausible *wrong* model; decide it here, and Phase 7 verifies it.

**Spiking / event-driven targets pick a spiking backend, and the two do different jobs.** A
single spike-driven synapse or small event-driven model — a Tsodyks–Markram short-term-plasticity
synapse driven by a defined spike train, an EPSP train — runs on the **NeuroML** backend
(`run("neuroml")`): the STP synapse is a first-class component, `neuroml:blockingPlasticSynapse`
+ a `modes.plasticityMechanism` of `tsodyksMarkramDepMechanism` (depression) or
`tsodyksMarkramDepFacMechanism` (facilitation), mapping `initReleaseProb=U`, `tauRec=τ_D`,
`tauFac=τ_F` (template `docs/Interoperability/NeuroML/examples/Ex7_STP.qmd`); the presynaptic
drive is a `neuroml:spikeGenerator` (regular) or `spikeArray` (preset times). A **recurrent
spiking network** (thousands of LIF neurons, structured populations, population activity) is the
**Brian2** path instead. The native Brian2 backend runs: all-to-all conductance synapses
(lowered to O(N) population-sum hubs, incl. a multi-gate saturating NMDA), **sparse random /
one-to-one connectivity** (`connectivity: random` + a `connection_probability` in the edge's
`parameters`, or `one_to_one`), **instantaneous (δ) current-based PSCs** (a spike jumps `v_post`
directly, no synaptic time constant — the Amit–Brunel / Mongillo form), **per-synapse short-term
facilitation/depression**, **Gaussian white-noise membrane drive** (`StateVariable.noise.intensity`
on a current-based cell that declares `tau_m`), and **timed current-pulse stimulation**
(`neuroml:pulseGenerator` — item cue / nonspecific readout). What Brian2 still does NOT accept is a
*defined spike source* (`spikeGenerator`/`spikeArray`) — that single-synapse-driven-by-a-preset-train
case stays on NeuroML. Spike **rasters persist to the container** (`spikes__<pop>__t/i`,
`firing_rate`, `population_size`, `populations`/`duration_ms` attrs); a spiking figure binds those.
A **synapse-internal trace the figure shows (mean u/x) should be MEASURED, not reconstructed**:
declare `record: true` on the synapse's state variable and the Brian2 backend attaches a
clock-driven, zero-delivery *observation probe* (a sampled copy of the population's synapses, same
STP driven by the same spikes), persisting the population-mean as `synapse__<pop>__<var>` — the
state the network actually integrated, on a byte-identical run (the probe delivers nothing). Prefer
this over replaying the Tsodyks–Markram recurrence on the spikes in a `code/` fn; keep that
reconstruction only as the **fallback** where a backend can't expose synapse state (the NeuroML
single-synapse path), and cross-check the two agree (they did, to ~0.01). This is a case of the
next rule — *measured beats a parallel analytic model of the same quantity*; the gap (record
synapse state) was an addable backend primitive, not a permanent "u/x are analytic" limitation.
(Event-driven STP records a frozen staircase under a plain StateMonitor, so the clock-driven probe
is what makes the continuous "u held through the delay" trace measurable at all.) Different
experiments in one recipe can use different backends (a rate
reduction on tvboptim beside its spike-level companion on neuroml). Study-loader gotcha: a NeuroML cell's nested channel goes
under `modes:`, not `components:` — `components` is a LinkML alias the strict study loader
rejects (only the `from_string` doc path accepts it).

**Surface feature gaps now**: a need not yet
supported (e.g. the Lyapunov exponent of a *delayed* closed loop under `vmap`) BLOCKS
its target — flag it as a framework/schema enhancement before you build, and mark the
target `partial`/`out` in the eventual scorecard. This early gap-finding is what sets
honest expectations instead of surprising you mid-YAML. **But a gap is often an *addable
general primitive*, not a permanent blocker** — an instantaneous δ-PSC synapse, a white-noise
membrane drive, a timed current pulse, sparse random connectivity are backend features any study
of that class wants. If the missing piece generalizes, add it root-cause to the backend (with a
regression test) and un-block the target; reserve `partial`/`out` for gaps that don't generalize,
need heavy new machinery, or that you won't build. State intent in the metadata either way (the
YAML declares a δ-jump or a `noise.intensity`, not a backend mechanism).

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

A third input class sits between obtainable and unobtainable: one the paper **synthesises from
a distribution it gives but a seed it doesn't** (an artificial bimodal-Gaussian net power, a
random connectome ensemble). You can match the *distribution* — and hence the mechanism —
faithfully, but the *realization*, and any count or threshold read off it (the number of
critical/solitary nodes), is realization-dependent by construction and cannot be bit-exact.
Tag those targets mechanism-level here, reproduce the distribution's construction exactly (its
deterministic structure too — e.g. generator/consumer roles from the real grid), and **never
tune the synthesis seed to hit the paper's integer** — that is fitting, not replication.
Contrast it with the study's *deterministic* inputs, which stay decimal-level. (Taher: P^G is a
symmetric random bimodal the paper never deposited → 6 vs the paper's 9 solitary is an honest
realization gap; the real-data P^R reproduces its 11 exactly on the same simulator — which is
what *proves* the gap is the data, not the code.)

A fourth: **a deposit routinely ships the OPTIMUM but not the search that found it.** Pang2023
deposits the fitted model's FC/FCD and a 2-element `KS`, and nothing of the 20-point `r_s`
landscape those came from — that curve exists only as a published raster (Extended Data Fig 10).
So a sweep target's *shape* can be compared only figure-to-figure while its *optimum* compares
numerically. Tag it accordingly, and when you do read values off their raster, say so — reading
a curve by eye is an observation, not a measurement, and must not be presented beside computed
numbers as though it were one. (Ours reproduced their descending limb almost exactly under a
one-grid-step shift while the ascending limb was far shallower — enough to state as a lead, not
enough to claim a mechanism.)

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
  (`from_experiment` / `source_point: branch`, shardable), and IC ensembles — deterministic
  (`initial_conditions.<state_var>`, an evenly-spaced grid over one state variable's initial value,
  for a paper's linspace IC fan) or stochastic (`distribution.seed` vs `execution.random_seed`).
  A product grid over independent cells needs none of it.
- **Spec at the root, callables flat in `code/` (zero-config).** Loading the study puts `code/`
  on the import path, so `callable: {module: <study>_analysis}` and a figure's `code_modules:`
  resolve by bare name — no driver, no `PYTHONPATH`, no vendored package. Set `code_source` ONLY
  to point the importable code elsewhere — a `{git, ref, subdir}` repo or a shared directory —
  never a local `code/` subfolder. This keeps the spec (backend-independent metadata) at the top
  level and its code under `code/`.
- **Reuse within a file** with shared `&dynamics` / `&params` / `&network` anchors +
  per-experiment `<<:` overrides; **reuse across files** with `!include path.yaml`, which
  substitutes the whole included document at that position (a `dynamics:`, an algorithm
  block, a whole experiment). `!include` resolves through the same load path as a monolithic
  study (byte-identical materialisation), so a big study can be a thin root that `!include`s
  per-experiment / shared-component files. (`!include` takes a whole file — there is no
  `#fragment` selector; to reuse one block, put that block in its own file and include it.)
  **The included file is the bare VALUE at that position, not a study-shaped document**: a
  `figures: !include figures.yaml` needs `figures.yaml` to be a bare *list* of Figures (its
  prose header becomes `#` comments), not a second spec with its own `title:`/`label:`/
  `figures:` keys. A sibling `figures.yaml` that the root spec never `!include`s is **invisible
  to `tvbo run`** — the study renders no figures and the drift is silent, because rendering that
  file directly still works. If you can't `tvbo run <Study>.yaml` and get the figures, they are
  not in the recipe.
  Order experiments so a `from_experiment` source precedes its dependents (operating point
  before its control runs) — then bare `tvbo run <Study>.yaml` resolves the seeds in one pass.
  When regime experiments differ only in one value buried inside an otherwise-identical block
  (a µ_ext inside the cell `Dynamics`), **lift it OUT to a declarative input** — a full-duration
  `pulseGenerator` drive — so the whole recurrent network is one shared anchor and only the small
  drive differs per experiment. This both compacts the recipe (a 4-regime spiking study collapses
  from 4× the network to ~1×) and is the more faithful encoding (an external drive is an input, not
  a cell property).
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

Two detector traps (both silent, both cost a session here): (3) **an outlier/critical-node
detector calibrated on one condition can quietly fail another — verify its count on EACH.** An
absolute `|x| > thr` test measures from a fixed zero; a parallel condition that shifts the
population baseline breaks it. Taher's solitary detector returned 6 nodes for P^G (locked bulk
at ω≈0) but *all 438* for P^R, whose synchronised bulk co-rotates at −0.11 Hz. Make such tests
**relative to the population's own baseline** (deviation from the profile median/mode), and
check the returned count against the paper *for every condition* (P^G *and* P^R, control *and*
patient), never just the first. (4) **When two callables compute the same quantity, they must
use the same criterion.** A control *mask* and a solitary *ordering* that both mean "which nodes
are solitary" drifted apart (one median-relative, one absolute) — reproducing one condition and
breaking the other. Grep for siblings and align them.

## Phase 5 — Figures: declare them in the study's `figures:` block

Figures are **metadata**, rendered by codegen — not a hand-written plotting script. Each
paper figure is a `Figure` in `<Study>.yaml`'s `figures:` list (schema `schema/figure.yaml`;
design `dev/figure-spec-design.md`). `tvbo figure render <Study>.yaml` — run automatically by
`tvbo run <Study>.yaml` — emits a self-contained, editable `figures/scripts/plot_<name>.py`
**and** runs it,
producing `<name>.png` in `figures/`. Iterate one figure fast with `tvbo figure render` (the
results stay put; only the plot re-runs). Copy `assets/figures.snippet.yaml` for the block and
`assets/figures.py.tmpl` for the panel module.

**`<study>/figures/` is THE render target — one place, gitignored.** The rendered
`<name>.png` sits at its root and its generated `plot_<name>.py` in
`figures/scripts/`, so the directory the report and reviewers browse holds IMAGES,
not twice as many files. That subdirectory is **not** called `code/`: in a study
that name means the authored, tracked, importable code the recipe references by bare
module name, and a generated artifact must not borrow it. Not `output/figures/`
(that is the results tree) and not `code/figures/`. Everything downstream reads from there: the
report's `FIGS = Path("../figures")`, and any script that still writes a supplement image writes
there too, so a figure and the report that embeds it can never point at different copies. Add
`figures/` to the study `.gitignore` — the `<name>.png` **and** the generated
`scripts/plot_<name>.py`
are both regenerable artifacts.

**A `Figure` is layout + binding + style; keep compute and plotting code out of it.**
- **Layout is metadata:** `layout` (bsplot mosaic string, e.g. `aab/ccb` — letters = panel
  keys, `/` = new row, repeated letters span, `.` = empty), `width`/`height` (mm), `dpi`,
  `font_size`, `height_ratios`/`width_ratios`, `style` (`.mplstyle` paths), `spines`,
  `panel_numbers`/`panel_number_format`/`panel_number_loc`. Set the paper's physical size and
  type scale here, once — never in code.

**Size, aspect and type size are MEASURED off the original, not guessed — get them right on
the first render.** Three defaults are wrong for a replication and cost a re-render every time:

- **Aspect.** Derive `height` from the original's own pixel aspect: `height = width ×
  (h_px / w_px)` of the paper's figure scan (a Nature two-column figure is `width: 183`, a
  single-column one `88`). Then set **`trim_margins: false`** — the default trims to content
  (`bbox_inches="tight"`), which re-crops the saved PNG and makes its aspect drift away from
  the declared `width × height`, which is exactly the "why is my figure the wrong shape"
  symptom. Check it: the rendered PNG's `h/w` must equal `height/width` to ~1 %.
  **But `trim_margins: false` also exposes a layout failure that trimming used to hide.**
  A dense mosaic (many panels, a very short row, empty `.` cells) can starve matplotlib's
  constrained-layout solver — it warns `axes sizes collapsed to zero` and silently drops
  EVERY axes back to the default 12 % subplot margins, giving a figure ringed by ~8 % white
  that neither `set_layout_engine("none")` nor `subplots_adjust` nor `height_ratios` will
  shift. Diagnose by measuring the ink bounding box of the PNG; if padding is ~8 % on all
  four sides while a sibling figure is at ~1 %, that is this bug and not your margins. The
  fix is to let it trim and declare the size OVERSIZE so the trimmed result lands on the
  target (iterate twice — measure, rescale, re-render), and **re-measure after any type-size
  change**, because the trim moves with it.
- **Panel proportions.** A mosaic alone distributes rows/columns EQUALLY. If the paper's rows
  or columns are unequal (a short schematic row above tall matrix panels), measure their pixel
  extents in the original and set `height_ratios` / `width_ratios` — otherwise every panel is
  subtly the wrong shape even though the figure size is right.
- **Type size — the single most repeated defect in this skill's history. VERIFY IT IN PIXELS,
  every time, before you call a figure done.** Set **`font_size: 9`** for a 183 mm figure and
  **10** for an 88 mm one as the *starting* value, never below 8. Then MEASURE, because two
  independent traps make the declared number a lie:

  1. **A study `.mplstyle` silently overrides `font_size`.** If the study ships a style file
     that sets `font.size`, it used to be applied *after* the declared size and won — so
     `font_size: 8` in the spec rendered as whatever the style said (Pang2023: 5 pt) with
     nothing in the spec, the log, or the emitted script to show it. tvbo now applies the
     `.mplstyle` FIRST and the declared `font_size` last (regression-tested), but **any style
     file you write must still be checked**: grep it for `font.size`, `axes.labelsize`,
     `xtick.labelsize`.
  2. **A point size is meaningless without the width it was measured at.** Apparent size is
     the ratio of glyph height to figure WIDTH. Pixel forensics on a 120 mm (1.5-column)
     original that you then reproduce at 183 mm yields type ~1.5× too small — and the same
     error scales every `linewidth`, `markersize` and tick length in the file. If a
     `.mplstyle` is derived from measurements, record the width they were taken at and
     rescale by `target_mm / measured_mm`.

  **The check (do it, don't assume):** binarise the rendered PNG and the paper's own scan,
  take connected components with `5 <= h <= 40 px`, and compare the modal glyph height as a
  PERCENTAGE OF IMAGE WIDTH. That ratio is resolution-independent, so it compares a 953 px
  scan with a 2161 px render directly. Journals run ~0.7–0.85 %; land within that or slightly
  above. **Aim a little above the original** — a Nature figure's 6 pt labels are legible at
  183 mm in print and illegible on screen at 2000 px, and every replication we have shipped
  erred small, three of them after this rule was already written down.
- **Grammar panels need zero code.** A `cartesian` or `heatmap` panel binds data through its
  `layers`: `used: {iri: tvbo:exp/<Study>/exp-3, output: <var|observation__name>, sel: {dim: label}}`
  (label-keyed, never positional — this binding **is** the PROV `used` edge), plus `mark`
  (`line`/`scatter`/`rule`/`band`; implied for heatmap) and `encoding: {x, y, color}` naming
  container dims/coords. `transform:` names an optional presentation-only reduction. Bind an
  **in-study** experiment by id — `used: {experiment: 3}` — rather than spelling a full `iri`:
  it needs no hardcoded study key and registers the run-order dependency (that experiment runs
  before the figure). Reserve an explicit `iri` for a curated/external container.
- **Only a bespoke interior is code.** A `custom` panel sets `render: <fn>` + `opts:`, where
  `<fn>` is a `@bsplot.register_panel` callable `fn(fig, ax, ctx)` in a module named in the
  figure's `code_modules:` (a flat file in `code/`, e.g. `code/<study>_figures.py`). It reads its resolved layers with
  `bsplot.load_layer(ctx["layers"][i])` and draws. A reused reduction is a
  `@bsplot.register_transform` `fn(da)->da`. This is the escape hatch — reach for it only when
  the grammar genuinely can't express the panel (twin axes, connectome, brain surface, dense
  nested subgrids), not by default.

**Compute lives upstream, never in the figure** (the ladder, design decision #4): prefer an
**Observation** declared on the experiment (plot-ready, recorded as `observation__<name>`) →
else a **declarative reduction** in the tvbo schema → and only last a registered `transform`.
The `Figure` stays presentation-only.

**Integrity (#3) is declarative too.** A panel whose TVBO data isn't ready sets
`placeholder: "<label>"` — it draws a labelled placeholder in the paper's layout, never the
paper's replotted source data. **Guard by data-requirement, not per figure:** a group-level
panel often reproduces from group data while its per-subject sibling is blocked — give only the
blocked panels a `placeholder`, so the reproducible ones ship.

**Render spatial data in the paper's coordinate/surface convention**, not a convenient
substitute — e.g. plot brain-region values at the paper's surface parcel centres (Koller uses
the fsaverage5-*inflated* COM), not the raw MNI centroid; the convention changes the figure's
look and can misalign, so derive the coordinates from the paper's surface and verify the
mapping **by label** (a `custom` surface/heatmap panel's job).

**A/B compose stays a report concern**, not a `Figure`: the study renders only *our*
reproduction; the side-by-side against the paper original is drawn in the **report** (the
`ab()` helper / `assets/compose_ab.py`), gated for copyright by the Phase-6 internal/public
profile — do **not** bake the © original into any committed/shared image or into a `Figure`.

**Every figure carries an original caption — `Figure.description` is it.** Write each figure a
`description:` in the `figures:` block: an original sentence or two describing what OUR
reproduction shows (the quantities, panels, what to read off it), in your own words. It is the
figure's caption and it is **public-facing** — so (a) NEVER paste the paper's caption verbatim
(plagiarism, and it usually describes panels/data you didn't reproduce), and (b) NEVER use the
internal A/B framing ("left: paper, right: ours", "paper original beside") — that composite
exists only in the internal build; the caption describes the standalone reproduction. The report
**auto-renders** `Figure.description` as the caption via a `figcap()` helper (single source of
truth, no retyping) — through an **`#| output: asis`** cell so the description renders as markdown
and may use **LaTeX math** (`$I_0$`, `$\sigma$`), NOT an inline `` `{python}` `` (whose output is
verbatim and would print literal `$`/`**`); see **writing-reports**. Keep the description in LaTeX
+ ASCII, never Unicode.

**Figures are LaTeX-compatible — no Unicode glyphs in a panel.** Circled numbers (①②③),
Unicode Greek/subscripts (σ, I₀, λ₁, Γ) and em-dashes are font-fragile (render as tofu, or just
look wrong) and are not portable. Mark sampled points with a **numbered colored disc** — a filled
circle marker plus a plain white digit centered on it (`ax.plot(x, y, "o", color=…, ms=…)` +
`ax.text(x, y, "2", ha="center", va="center", color="white", fontweight="bold")`) — matching the
paper's style, not a `①` glyph. Use matplotlib **mathtext** for symbols (`$\sigma$`, `$I_0$`,
`$\Gamma$`, `$t/T$`), and a hyphen, not an em-dash, in titles. This is the figure-side of the same
LaTeX-not-Unicode rule the report captions follow.

**A marked/sampled point must match what the paper says it IS — read the figure description, then
verify via the panel it feeds.** When a `custom` panel marks sample points on a curve (three
periodic orbits 1/2/3 on a period-vs-parameter branch), select each from the paper's figure
caption/description, not a guessed heuristic: the description names what the point is — its period
band **and its morphology** ("point 2 = the *mid* orbit, an asymmetric spike with a slow rise") —
and that fixes which branch point to take. Then verify against the panel the marker drives: the
marked orbit's waveform sub-panel must show the described shape. (An argmin-on-period heuristic put
our "2" at the bottom-corner fold — a too-symmetric spike; the description's "asymmetric spike +
slow rise" is what identified the elbow one bend up as the right orbit.) This is Phase 7's
shape-check applied to marker placement — the caption/description is the oracle for *which* point,
not just how to word it.

**External published paper data binds by IRI too.** When a panel pairs TVBO output against the
paper's own figure data, wrap that data as an external `Dataset` and bind
`used: {iri: tvbo:dataset/<Study>_source, output: <var>, sel: {figure: 6, panel: c}}` — the
same declarative path, figure/panel as coordinates you `sel` into. Until wrapped, a **flat,
label-keyed** per-panel `.nc` set (`xarray` named coords, not filesystem-keyed) is an accepted
stopgap; don't build an elaborate filename tree — it's throwaway once the `Dataset` binding lands.

## Phase 6 — Report: `report/report.qmd` (every number computed)

**The report MUST carry a "Where the Methods and the code diverge" section** whenever the study
ships code — a summary table by class (A–E) with counts, the two or three entries that would
silently produce a wrong figure spelled out, and a short paragraph on why one declarative
description removes the whole class. This is a headline result, so give it a numbered
section of its own rather than burying it in Limitations; the full evidence lives in
`report/analysis/methods-vs-code.md`. State plainly that the divergences are *visible* only
because the deposit is open — otherwise the section reads as a criticism of the most transparent
papers.

See **writing-reports** for the report mechanics: the IMRAD structure, the metrics cell
that computes every number from the containers (nothing hand-typed), the native
`EXP.dynamics.generate_report(..., citeformat="quarto")` equation and parameter render, the
three-colour status callouts, the copyright-safe internal/public split, references as Quarto's
auto-appended bibliography, the LaTeX rules, and the anti-slop prose standard. The templates
it copies ship in this skill's `assets/`: `report.qmd.tmpl`, `report_internal.qmd.tmpl`, and
`_quarto.yml.tmpl`. Copy all three into `report/` (as `report.qmd`, `report_internal.qmd`,
`_quarto.yml`). One Quarto project renders BOTH PDFs from a single `quarto render` (in `report/`,
no file arg): `report.qmd` holds the whole report and carries NO front matter, `report_internal.qmd`
is a thin `{{< include report.qmd >}}` wrapper that draws the paper's © figures for A/B checking,
and `_quarto.yml` lists both and holds the shared `format: pdf` (xelatex) + `bibliography:`. The build
branches on `QUARTO_DOCUMENT_FILE`; no `--profile`, no post-render hook (see the header comment in
`_quarto.yml.tmpl`).

**Stage every figure into `report/_figures/` and embed it from there, never through a link up
into `../figures/`.** `tvbo.utils.report.report_figure` does the staging, decides per build
whether the © original is opened at all, and composes the A/B pair — one implementation for every
study, so no report grows its own `ab()` again. Loop the recipe's own `figures:` block
(`figures_in_paper_order`, `figure_title`, `figure_caption`) rather than a hand-written list of
stems and captions, and derive each figure's status callout from `figure_targets(fig,
TARGET_ROWS)` so it cannot disagree with the scorecard.

**Captions read the LOADED study, never a YAML file.** `figure_caption` resolves off
`SimulationStudy.from_file("../<Study>.yaml").figures`, so it keeps working however the spec is
split — a caption helper that raw-parses `figures.yaml` breaks the moment the figures move into
the recipe or behind an `!include`, and it silently returns `""` rather than failing.

**Migrating an older report off the profile split**: `report-src.qmd` → `report.qmd` with its
front matter *moved* into `_quarto.yml` (the file must carry none, or the wrapper's `output-file`
is overridden), add the `report_internal.qmd` wrapper, and **delete** `_quarto-internal.yml`
together with any `post-render:` hook or `make_internal_report.py`-style generator — the two
entries in `render:` replace all of it. Flip `INTERNAL` to `tvbo.utils.report.is_internal()`,
repoint `FIGS` at `../figures`, and render with a bare `quarto render`.

Replication-specific rules on top of that mechanics:

- **A shortfall is one of three things, and the scorecard must not merge them.** `met` /
  **`short`** (attempted, did not meet its criterion -- the only true replication failure) /
  **`out`** (judged to test nothing the other targets do not; declared unattempted) /
  **`blocked`** (in scope, but an input cannot be obtained). Written as one bucket, a scope
  decision reads as a failure and -- worse -- a failure can hide inside a scope decision. Check
  by reading each reason: if it describes an obstacle ("needs data that is not released") the
  row is `blocked`, not `out`; if it describes a result, it is `short`.

- **The scorecard maps 1:1 to `targets.md`.** Every criterion `T1..Tn` from Phase 1 is one
  row, tagged with its Phase-1.5 **fidelity tier**: *mechanism-level*
  (a sign or ordering that any reasonable input reproduces) vs *decimal-level* (a number that
  needs the paper's exact input). Derive the verdict from the data, never assert it.
  Mechanically: give `targets.md` a `Status` column and **read that file** —
  `report.read_md_tables(<path>)` returns each table's rows as `{header: cell}` dicts, so the
  scorecard, the tally by scope, and the "which targets fell short" list are all computed
  from the one file whose criteria were written before anything ran. Typing the tally into
  prose is the same defect as typing a result (non-negotiable #2): it drifts the first time a
  target changes verdict, and nothing catches it. **A file you compute from is an input —
  validate the parse.** Three rows of our `targets.md` had run two cells together into one, so
  those rows silently shifted a column left and their Scope/Fidelity/Status read as each other's
  neighbour; the table still rendered, and the tally was simply wrong. Check that every row
  parsed to the full header width and that each value falls in its expected vocabulary
  (`core|extended|out`, `mech|dec`, `met|partial|out`) before believing the counts.
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
stated tolerance). **If the paper states a closed form** — a steady-state law, an iterative
amplitude recurrence — that closed form IS the oracle: compare the recipe output against it
directly. A self-contained study has no external input to cap the number, so the agreement is
exact to integration tolerance (the decimal-level targets pass), not merely mechanism-level.
Where feasible, also cross-check via `render_code('tvb')` vs `render_code('tvboptim')`. This is what catches modelling bugs a PDF can't: e.g.
per-step vs per-stage coupling evaluation converging to a *different attractor*.

**Attribute a residual gap to data vs implementation with a head-to-head.** When a metric
misses the paper's number, install the paper's *own* tooling and run it on your *exact*
inputs before blaming your code. (Koller: running his native `tvb-library` model on the same
substitute SC gave FC r=0.27 — the same as tvboptim's 0.32 — proving the shortfall was the
connectome, not the engine; without it we'd have chased an implementation bug that wasn't
there.)

### When the paper deposits its own ANALYSIS OUTPUTS, demand identity (r = 1, RMSE ~1e-15)

Many deposits ship not just inputs but the authors' own *derived* arrays (accuracy curves,
power spectra, permutation sets). That converts verification from "do we agree roughly?" into
an exact test: run **our** implementation on **their** inputs and require machine precision.
Write it as a standing harness (`code/verify_identity.py`) that prints one table, because it
is the thing you re-run after every refactor. Classify each check up front — mixing the
classes is how a replication overclaims:

| class | meaning | criterion |
|---|---|---|
| `identity` | deterministic, same inputs, same algorithm | RMSE ≲ 1e-12. **A failure is OUR bug.** |
| `convergent` | deterministic but solver-tolerance-limited | agreement stated *with its floor* |
| `stochastic` | depends on an unpublished seed | distributional only — matching an exact number would mean we tuned to it |

Identity is a *discriminating instrument*, not a rubber stamp — it localises bugs that a
correlation would hide. Four traps it caught in one study (Pang2023), each of which would
have produced plausible, wrong figures:

- **The deposit ships several versions of "the same" array.** The basis under
  `results/basis_geometric_*` differed from `template_eigenmodes/*_emode_200.txt` by 4.2e-2.
  Both look right; only one gives identity (5.6e-16 vs 2.6e-6). **Try every candidate and let
  identity pick** — never assume the obviously-named file is the one the figures used.
- **Order of a nonlinear step.** A normalised power spectrum averaged over subjects is NOT
  the spectrum of the subject-averaged map: r = 0.885 vs r = 1.0000000000. Whenever a
  statistic normalises, establish *where* the averaging happens; the paper's prose often
  won't say, and only identity distinguishes them.
- **"Improving" the reference algorithm breaks it.** Symmetrising a Gram matrix before
  solving is numerically defensible and *wrong here* — port the reference's arithmetic
  exactly (`(Ψ'Ψ)\(Ψ'y)`), because identity against it is the criterion.
- **Masked/NaN vertices silently poison a least-squares solve.** One NaN turns an entire
  reconstruction into NaN. Restrict to the analysis mask the paper uses (its cortex mask),
  and treat an all-NaN result as a convention bug, not a data problem.

Two mechanical ones worth a checklist line: when loading a `.mat`/HDF5 reference, select the
dataset **by name** (`eig_vec`), never "the first key" — sibling arrays like `eig_val` sort
first and load silently; and MATLAB HDF5 arrives **transposed**, so confirm orientation
against a known dimension rather than by eye.

**A cross-check experiment should RECORD on the grid it will be compared against.** When one
experiment exists to bound another's error, declare its observation at the *other* run's
sampling period (`iri: tvbo:SubSample`, `period: <the other run's dt>`, `reduce: streaming`)
rather than recording its own — much finer — solver step. The two then share one time
coordinate by construction, so the comparison needs no interpolation and no positional
decimation, and the container stops being an artifact in its own right: Pang2023's vertex-space
check went 2.3 GB → 151 MB and 10 min → 1m23, because the *write*, not the solve, was eight of
those ten minutes. Recording every step of a 32,492-node field "in case we need it" is also how
you stall the whole machine — that write filled the page cache and collapsed throughput for
unrelated work that followed, which reads as a hung job rather than as the disk-bound write it is.

**Report a cross-check that does not converge AS unresolved.** Do not quote a bound from a
diverged run, and do not quietly drop the target. Say what was measured (the step sizes tried,
where it left the physical range, the growth rate at each), separate what that *does* exonerate
(here: the analysis chain, verified end to end on the diverged container) from what stays open
(the discretisation), and mark the row `partial`/open in the scorecard. An unresolved
verification honestly reported is a result; a missing one is a gap in the replication.

### When NO output data is shipped, an unverified convention is an ASSUMPTION — label it

The identity checks above only exist because that deposit happened to include the authors'
derived arrays. **Most do not.** The failure mode is subtle and expensive: with nothing to
test against, a plausible reading of the Methods gets written into `targets.md` as though it
were established, every downstream number inherits it, and the report states it as fact.

The tell is that the paper's prose *underdetermines* the computation. "The power spectrum of
the group-averaged maps" does not say whether the averaging precedes or follows a nonlinear
normalisation — and those differ by r = 0.885 vs 1.0. Prose almost never pins down: where an
average sits relative to a nonlinear step; which of several shipped files is "the" basis;
whether an analysis runs on all vertices or a cortex mask; 0- vs 1-based indices; whether a
"correlation" is over vertices or parcels.

So, when you cannot verify:

1. **Write the assumption down as an assumption**, in `targets.md`, next to the target it
   feeds — not as a statement of what the paper did. Phrase it "we read X as Y; not
   verifiable from the deposit".
2. **Enumerate the plausible alternatives you rejected**, and say why. If you cannot name an
   alternative, you have not understood the choice well enough to make it.
3. **Test sensitivity.** Compute the target under each candidate convention. If they agree
   to within the reported precision, the ambiguity is harmless — say so and move on. If they
   disagree materially, that is a *first-class limitation* of the replication, and the
   scorecard must show the range, not one arbitrarily-chosen member of it.
4. **Never let an assumption harden into an assertion** through repetition. A convention you
   guessed in Phase 1 is still a guess in Phase 6 unless something verified it in between.

This is the same discipline as **doubting a claimed discrepancy** — default to "we may have
misread this", and make the uncertainty visible instead of resolving it silently.

### For a LINEAR model, don't fit a scale — invert the transfer function

When a replication's output has the right shape but the wrong magnitude, the instinct is to
report a best-fit scale factor. For a linear model that is the weak measurement, because the
fit absorbs every other residual — basis truncation above all — and lands on a number that is
neither the true scale nor obviously wrong. In Pang2023 the forward fit read 1.85 against a
4–8 % truncation floor, and sat unexplained for a long time.

Invert the model instead. A linear system's own transfer function is exactly invertible, so
the deposited OUTPUT determines the INPUT that produced it:
`Q(ω) = Φ(ω)·[−ω² + 2iωγ_s + γ_s²(1 + r_s²λ)]/γ_s²`. That returned a flat boxcar of amplitude
**10.00 ± 0.05** where the Methods said 20 — a factor of exactly 2, settled in one step.

Two reasons this beats fitting:

- **It is truncation-consistent.** The same basis appears on both sides, so the error that
  contaminates a forward comparison cancels instead of biasing the estimate.
- **The recovered input's SHAPE is a self-test of the whole model.** A flat rectangle can only
  come out if `γ_s`, the damping term, the eigenvalues and the stiffness are all right; any
  error makes the recovery frequency-shaped. So the measurement validates the model and
  quantifies the discrepancy at once — you are not merely asserting agreement.

Generalises to any linear or linearised stage: a haemodynamic convolution, a filter, a modal
projection. Where the model is nonlinear, invert around the operating point and say so.

**Port a statistical procedure from the reference implementation, not from its description.**
A spin test is the canonical example: naive nearest-neighbour matching of rotated parcels is
*not a permutation* (parcels get duplicated and dropped), which biases the null; the published
method (Váša `rotate_parcellation.m`) does a greedy "most distant minimum" assignment
**without replacement**. Also force `det = +1` — the QR of a Gaussian matrix can be a
*reflection*, which is not a rotation of the sphere. Where the deposit ships its own
permutation set, use **theirs** to verify your statistic, which isolates the test from your
RNG; then check your own generator separately (every row a true permutation).

**Measure the layout, then eyeball the shape.** Declare each figure's published counterpart
with `reference_image: original_study/img/fig_0N.png` and run `tvbo figure compare
<Study>.yaml`: it decomposes both images into panel boxes (recursive XY-cut), matches them by
overlap, and writes a per-panel offset table plus a side-by-side overlay. Page **aspect** is
the number to read first — it is exactly reproducible and it catches the whole class of "the
figure is the wrong shape" that survives every value check. A deliberate aspect difference
(a panel of the paper's you do not draw) is fine, but it belongs in the figure's
`description:` as a stated departure, not as an unexplained 1.14-against-1.75. The panel
counts often disagree because a published raster's panels touch where yours have gutters;
read the offsets only where the counts agree. Identifying the counterpart is itself worth the
few minutes: deposits number their images `fig_01…fig_NN` with no mapping to "Extended Data
Fig 10", the offset from main-text numbering is *not* uniform, and the only reliable way is to
open the candidates — doing so is what turned Pang2023's `r_s` landscape from an
uncomparable panel into one measurable at aspect 1.272 against 1.280.

**Eyeball every reproduced panel's *shape* against the paper — the A/B internal composite is
the instrument, not a formality.** Inline-computed numbers (non-negotiable #2) catch a wrong
*value*, but a curve that plateaus where the paper's descends, a flipped monotonicity, a sign
error, or a saturated axis still *computes* a number and sails through a value check. Lay the
reproduction beside the original panel-for-panel and confirm the qualitative shape before
declaring a figure done — a mismatch there is a modelling/analysis bug the reference
integration alone won't surface. (Taher Fig 9(d): one strategy curve sat as a flat plateau
instead of the paper's staircase descent — the visible tell of a broken solitary set, invisible
in the scalar metrics.) This is also the moment a stale caption shows up: prose written before a
later fix (a "not yet wired" follow-up that since shipped) must be reconciled with what the
panel now shows.

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
- **Every run-time knob is a `--set` on the emit, never a recipe hand-edit or a hand-written
  sbatch.** The corollary of "no sbatch": any per-run override — swap the whole runtime
  substrate, retarget the queue, resize a job — is a flag on `tvbo workflow snakemake`, so the
  recipe stays the portable source of truth and the same study emits for CPU-container *and*
  GPU-venv without editing it. A **GPU run** is exactly this: drop the container and point at a
  `jax[cuda]` venv — `--set container= --set slurm.venv=/path/to/.venv --set slurm.partition=gpu
  --set slurm.gres=gpu:1 --set slurm.mem=… --set slurm.time=…` (the SLURM executor turns
  `gres` into `--gres` itself; on a GPU node let JAX auto-detect — do **not** force
  `JAX_PLATFORMS=cuda`, which drops the CPU device a `jax.debug.print` progress callback needs;
  use `cuda,cpu` if you must set it, and match the `jax-cuda12-*` plugin to `jaxlib`). Env vars
  are `--set 'slurm.env=[{name: …, value: …}]'`. Install the venv from a **compute node**
  (`srun`), never the login node.
- **Prove the memory/streaming fix — don't eyeball it — with engine-native benchmarking.**
  `tvbo workflow snakemake … --benchmark` (or `--set benchmark=true`) attaches Snakemake's
  native `benchmark:` directive to every rule: a per-cell TSV (wall time, `max_rss`/`vms`/
  `uss`/`pss` MB, io, cpu_time) written next to each output, whether run locally or as a SLURM
  job — one row per cell, so a fanned sweep benchmarks every cell. This is how you turn "reason
  about resident memory" into a *measured* peak (a streaming BOLD fit that would OOM at hundreds
  of GB materialized shows a ~GB peak in the TSV), and how you size `slurm.mem` honestly.
- **A dry run does NOT execute anything — smoke-test ONE experiment in the container
  FIRST.** `tvbo workflow submit --dry-run` (snakemake `-n`) only resolves the DAG
  (wildcards, inputs, resources); no `tvbo run` executes, so it cannot catch a runtime
  bug. A per-rule bug fails all N jobs identically (we once launched 1106 that all died
  the same way). Before the real submit, run a single experiment end-to-end inside the
  SIF (`apptainer exec --bind … <sif> tvbo run spec/<id>/experiment.yaml`), then its
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
  `--code-source {frozen,spec}`.** A Snakemake study kit emits BOTH the frozen pre-rendered
  `scripts/<exp>` and the `spec/<exp>`, and each rule can run either: **spec** (default)
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

- **Size `step_size` from the STIFFEST thing the experiment actually integrates — not from the
  paper's fitted parameter, and not from the sibling experiment whose `integration:` block you
  inherited.** A step chosen for the optimum is wrong for the sweep that visits the rest of the
  grid, and wrong again for the same equation solved in a different space. Both failures are
  SILENT: the sweep returns plausible numbers from the cells that happened to converge. Two
  measured cases from Pang2023, both from one inherited anchor. (1) The resting model's fastest
  mode is `γ_s·√(1 + r_s²·λ_max)` — 114 Hz at the fitted `r_s` = 28.9 mm but **390 Hz at the
  grid's 100 mm**. At the single run's 0.5 ms every cell from `r_s` = 76 mm up returned a growing
  fraction of non-finite modes (11 % → 47 %) while the low-`r_s` cells looked perfectly healthy —
  and the *converged* part of the landscape was distorted too: halving to 0.25 ms did not merely
  remove NaNs, it sharpened the optimum from KS 0.065 in a flat well to **0.029 against 0.068 at
  its neighbour**, moving the very quantity the paper's optimisation minimises. (2) The same PDE
  on the mesh instead of in a 200-mode basis: the truncated basis stops at |λ| = 0.044 mm⁻² while
  the full cotangent LBO reaches **16.0 mm⁻²**, ~360× stiffer, and the inherited 0.1 ms step
  diverged to 1e116. Measure the operator's spectral radius
  (`scipy.sparse.linalg.eigsh(L, k=1, which='LM')`), form `dt·ω`, pick the step from that, then
  confirm the boundary empirically — a sweep locates its own (ours sat at `dt·ω ≈ 0.9` for Heun).
  Give the swept or differently-discretised experiment its OWN `integration:` block
  (`<<: *anchor` + an overriding `step_size:`) and say why in a comment, or a reader reads the
  difference as drift rather than as the measurement it is.
- **A stability claim needs the FULL production window — a short probe proves nothing.** We
  tested the vertex-space run over 20 ms, watched it decay, and declared the finer step stable;
  over the declared 100 ms it holds to ~25 ms and then passes 1e7. A marginal instability grows
  per STEP, so its blow-up *time* scales with the step — which is also the diagnostic that
  separates it from a sign/operator error: a genuine positive eigenvalue blows up at the same
  time whatever the step, whereas ours slowed from 2150 s⁻¹ to 735 s⁻¹ when the step shrank 5×.
  Measure that growth rate at two steps before concluding which failure you have.
- **A swept cell must be the SAME computation as the single run — check the frame count, not the
  code.** The two paths differ structurally: a single run integrates the transient separately and
  streams only the main window, while a sweep folds transient + main into ONE window and asks the
  reducer to drop the transient. If that `skip` is accepted and ignored, the sweep silently keeps
  `skip/stride` extra leading samples — 1,338 BOLD frames where the same experiment run alone
  gives 1,200 — and every FC/FCD statistic is then computed over a window contaminated by the
  start-up transient the single run discards. After any sweep, assert the per-cell shape equals
  the base run's before believing a landscape.
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
  **You request streaming declaratively — `reduce: streaming` on the observation** (opt-in,
  byte-identical to the post-scan value to f64 rounding, zero effect on any other
  observation), which folds it into the integrator carry as an (init, update, finalize)
  reducer via `prepare(reduce=…)` instead of stacking a trajectory. Supported for the
  HRF-Volterra BOLD pipeline (the resolver lifts the kernel, downsample stride, TR stride and
  Volterra `k_1`/`V_0` from the declared pipeline), for cumulative **mean / std / variance**
  aggregations (Welford, folded per block), and for a **matrix co-moment FC** (`compute_fc` — a
  running covariance emitted at the end, never a trajectory) — the last is what turns Schirner's
  ~440 GB FC evaluation into a ~GB peak. Byte-identical noise-off; with tvboptim's *per-block*
  noise draw the realization shifts with block size (ergodically vanishing — the same accepted
  tradeoff as the shipped BOLD stream), so treat a noisy streamed metric as distributional, not
  bit-exact across block sizes. **A streamed observation must decimate
  by a stride/`subsample`, never `temporal_average`** — a stride is block-additive so it is
  identical whether or not it is folded in-carry, whereas `temporal_average` is not (and
  `temporal_average(1)` is not even the identity — it shifts by one). Verify it reaches the
  streaming post-eval within memory *without* running the whole fit via `--smoke` (below).
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
- **A near-bifurcation operating point is implementation-specific — re-tune it to the phenomenon,
  with precedent.** When a paper selects a regime with a control parameter sitting near a
  bifurcation (a background drive µ that flips activity-silent → persistent → asynchronous, a
  coupling at a synchronization onset), the paper's *exact* value need not reproduce that regime in
  YOUR discretization — a δ-PSC / Euler network's transition sits at a different µ than the paper's
  kinetic/exact one. Re-tune the control parameter to reproduce the *phenomenon* (the regime and its
  ordering), document the shift, and cite the precedent: published reproductions routinely re-tune
  the same knob (the Mongillo NEST reproduction shifted µ_ext ≈0.5 mV after changing the PSC kernel;
  ours shifted comparably, activity-silent at 22.4 not the SOM's 23.1 mV). Faithful = the phenomenon
  at a re-tuned operating point, not a byte-identical control value — decimal- vs mechanism-level
  (Phase 1.5) applied to a control parameter, stated as such in the scorecard. Locate the transition
  with a quick 3–4 point scan of the control parameter *before* committing the recipe value.

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
  **networked** systems and verify against the reference (Phase 7). It re-evaluates the
  *network* coupling term at each integrator stage, so it is a **no-op for a single node**
  (no network coupling to re-evaluate) — there the attractor-moving knob is `dt` (RK4 / halve
  the step), not per_stage. Don't reach for it to explain a single-node discrepancy.
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
- **Redundant scripts.** One prep script (emits the tvbo Network directly); figures are
  the declarative `figures:` block, not scripts. Don't hand-write per-figure `plot_*.py`
  or an A/B compose driver — the renderer emits the plot scripts, and bespoke panel code
  lives in ONE `code_modules` module in `code/`. (`plot_<name>.py` in `figures/scripts/`
  is *generated*; never author or commit it.)
- **Moving a module changes what `Path(__file__).parents[N]` means — grep for the climb
  BEFORE you flatten.** Study code routinely locates the study root by climbing from its own
  file (`_ROOT = Path(__file__).resolve().parents[2]`, written when it lived in `code/recipe/`).
  Flattening it into `code/` makes every such climb overshoot by one, so paths resolve into the
  *sibling-studies* directory. The failure is loud only if nothing exists there — otherwise you
  silently read another study's tree. After any move, `grep -rn "parents\[" code/`, fix each N,
  and re-run one figure end-to-end to confirm the containers still resolve.
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
- **A lineage of related papers → sibling studies sharing a curated model; pin every
  original-figure lookup.** When one model spans several papers (a foundation and its
  successor, e.g. a synapse used first at the single-synapse level then in a network), make
  each paper its own self-contained study and share the model by a curated `iri:` — don't
  cram both into one recipe (the scales and reports differ). Keep only the paper being
  replicated under that study's `original_study/`; when it also holds a precursor/successor's
  figures, an unpinned `original_study.rglob("fig_03.png")` in the report's `ab()` silently
  grabs the WRONG paper's `fig_03.png`. Pin the lookup to the specific paper dir
  (`glob("Author1997*")/"img"`), and eyeball the internal A/B once to confirm the original is
  the right figure.
- **A run persists a container ONLY with `-o`, and figures read whatever container is on disk —
  fresh or stale.** Two silent failure modes. (1) `tvbo run` *without* `-o` computes the result
  and DISCARDS it, so a re-run after a recipe change leaves the OLD container in place and every
  figure/report reads STALE data — you then reason about the new recipe from the previous run's
  output. This is the costliest silent trap here: it produced a whole wrong "the backend can't
  reproduce this" diagnosis before the container turned out to be days old. Always pass
  `-o output/nc`, and before trusting a figure confirm its container is FRESH — the file timestamp
  is from this run and its dims/coords match the current recipe (the exploration axis you just
  changed is the dim you now see), not a leftover. (The CLI now warns on a no-`-o` run, but the
  discipline is: persist, then verify freshness.) (2) A pure forward run that only records a raw
  trajectory (no exploration, no declared observation) — e.g. a NeuroML EPSP-train run — must
  still write `output/…_result.h5`; confirm `wrote [...]` is non-empty (a figure binding
  `iri: tvbo:result/<Study>/exp-N` can't resolve an unwritten container). Run END-TO-END, not
  `from_file`.
- **Re-running an experiment does NOT invalidate the analyses computed from it.** An analysis
  container carries no link back to the result it was derived from, and any "run what is missing"
  pass skips whatever already exists — so after re-running an experiment the figures render THIS
  run's dynamics against the PREVIOUS run's analyses, and nothing raises. Delete the dependent
  containers explicitly before recomputing, and take the dependency set from the study's own
  schedule (the `after` stage of `_study_analysis_stages`, which is transitive — for Pang2023 it
  correctly caught 17 including second-order ones like the FCD landscape and the myelin
  correlation) rather than hand-listing, which misses exactly the ones you did not think of. Then
  confirm the invalidation *worked* by checking that an unchanged quantity comes back identical —
  trusting the pass is how you end up believing a stale number twice.
- **Two runs of the same field may name the same axis differently — reconcile by NAME, never
  broadcast.** A modal run projected onto the surface lands on `vertex`; the mesh run calls the
  same axis `node`. Subtracting them as they arrive broadcasts into a 32,492 × 32,492 outer
  product instead of an elementwise difference — 8 GB and a meaningless answer, with no error
  raised. Match the non-shared dims by size, rename, transpose, and only then subtract. Where the
  two sample a *shared* axis differently (a stiffer run needs a finer step), align on its
  COORDINATE — `.sel(time=…, method="nearest", tolerance=…)` — never by decimating positionally,
  and better still make the coincidence structural (next bullet).
- **A single-value exploration axis silently OVERRIDES the base parameter — never use one as
  ensemble scaffolding.** An `Exploration` axis with one `explored_values` entry (or a 1-point
  domain) still *writes that value over* the Dynamics/Coupling parameter it names. So a stand-in
  axis added only to give a trial-only ensemble a `space` (a) runs the whole study at the axis's
  value, not the model's — a typo or a stale number (`explored_values: [-1.76]` where the model
  sets `-1.76128`) silently integrates the wrong regime, and reads as a backend failure — and
  (b) is unnecessary. Express the ensemble with the mechanism that actually varies it: `n_trials`
  (+ a per-SV `distribution`) for a stochastic IC ensemble, or an `initial_conditions.<state_var>`
  sweep for a deterministic one (`assets/sweeps.md`). To pin a parameter, set it on the
  Dynamics/Coupling, never as a degenerate axis.
- **Framework gaps surface late** if you skip Phase 1.5. Find them before the YAML.
