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

**This skill is a spine plus six reference files.** Everything below is the spine: the
invariants, and the decision each phase has to make. The mechanics of a phase live in a
reference file next to this one, meant to be read when you reach that phase — or handed to a
subagent doing that phase alone.

| file | read it when |
|---|---|
| `assets/sweeps.md` | the paper's sweep is a branch / continuation / IC ensemble, not a product grid (Phase 3) |
| `assets/figures.md` | writing the `figures:` block — layout keys, the size/aspect/type-size protocol, panel binding (Phase 5) |
| `assets/verification.md` | building the oracle — identity harness, assumption labelling, free conventions, linear inversion (Phase 7) |
| `assets/deposit-forensics.md` | a number of yours disagrees with a published one and the deposit ships their arrays (Phase 7) |
| `assets/cluster.md` | packing and submitting a cluster kit (Phase 8) |
| `assets/traps.md` | something returns plausible-but-wrong numbers — indexed by symptom at the end of this file |

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
   `original_study/` exclusion; keep every comment on its own line.) **And a negation cannot
   rescue a file under a directory an ANCESTOR `.gitignore` excluded** — git does not descend into
   an excluded directory, so `!original_study/analysis/**` in the study's own file is dead the
   moment a parent ignores bare `original_study`. Three of our studies wrote exactly that pair and
   silently kept their targets table, figure map and adherence notes untracked for weeks. Do not
   try to carve an exception: authored work goes in `report/analysis/`, which nothing ignores.
   Verify rather than assume — `git check-ignore -v <path>` names the file and line that won.
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
10. **NEVER give a study its own execution machinery, and never add one to tvbo to serve a
    study.** Parallelism, batching, device placement and scheduling belong to the framework —
    tvboptim's `vmap`/`pmap` and `n_parallel` for a sweep, `tvbo workflow` for a cluster fan-out.
    A replication declares WHAT is computed; HOW MANY PROCESSES is not metadata and does not
    belong in the recipe, in `code/`, or in a new tvbo utility. Concretely forbidden: a process
    pool, a thread pool, a `joblib` fan-out, a bespoke worker loop, a hand-written `sbatch`.
    The pull is real, because an `Analysis` is the one place the recipe still permits arbitrary
    Python (`is_a: FunctionCall` → a `code/` callable), so the framework cannot parallelise it
    and it is tempting to do it yourself. Do not. When an analysis is too slow, in order:
    **(a) profile it** — the bottleneck is usually I/O layout or redundant algebra, not cores
    (in Pang2023 the per-subject cost was 52 % sparse ARPACK, 15 % an HDF5 chunking pathology
    and 24 % a nested sweep recomputing its own Gram; the algebra alone gave 10×);
    **(b) make it declarative** so the backend can run it, the way a symbolic `Observation`
    with `reduce: streaming` is lowered into the jitted grid;
    **(c) accept that it is serial and say so.** A sparse eigensolve, a CIFTI read and a mesh
    rotation are host-side by nature and no backend will batch them.
    What a study MAY do is make its own work resumable — cache per unit, skip what is cached —
    because that is a property of the computation, not a scheduler. The cost of ignoring this
    is not hypothetical: a five-process pool bolted onto one analysis returned **1.28×** on a
    loaded machine and drove it into swap exhaustion until the OS killed the job.

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
| **F. Unreleased** | the model or step the paper compares against is not in the deposit at all | a competitor whose figures are drawn from frozen arrays; no source anywhere for its symbols |

**F is the one class you can find with no deposit to read, and it is usually the sharpest.** The
others need code to compare prose against; F needs only the published equations and the published
parameter values — put one into the other and see whether the reported operating point exists. In
Pang2023 it does not: the competitor mass model was never deposited (a grep of every `.m` and
`.py` for any inhibitory symbol returns nothing, and its panel is drawn from frozen arrays), and
its four published weights, entered into the paper's own equations, yield no 3 Hz fixed point —
the feedback-inhibition relation returns `w_IE` = 8.933 where the paper prints 7.13, and at 7.13
the only roots are 0.002, 0.384 and 561 Hz. Note the order that makes such a claim safe: every
input was falsified first (our solver reproduces the precursor paper's operating point, the
symbols were re-extracted from the supplement's equation objects, the normalisation and target
rate were confirmed against a third paper) — see the "doubt your own discrepancy" rule. The
finding is not that a number is wrong but that **it cannot be checked**, and a target resting on
it is `out` with that as its reason, not `short`.

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

**The register is hand-maintained, so guard it like data.** It has no upstream and no
regenerating script, which means a structural failure in it is silent: a `divergence_register`
parse once returned 183 rows all of one class and the report printed that as its headline. Add
it to the identity harness — ids unique, more than one class, `material ≤ scored` — and let a
duplicate id fail the build rather than the reader. (Two different rows both numbered `C10` sat
there through several sessions of both being cited.)

**When a row is superseded, REWRITE it and say what it replaced.** A register entry is a dated
measurement, not a permanent fact, and its worst failure mode is hardening into a documented
"impossible" that stops anyone re-measuring. Pang2023's D4 recorded that no function of the
deposit's own affinity matrix reproduces the published variance ("every candidate gives ≈ 4 %")
and that embedding it does not return its own gradients ("median |r| 0.42"). Both were true when
written and both were artifacts of a configuration two changes later superseded — the dense
graph, and a voxel ordering that was simply wrong. They then blocked the right investigation for
a while, because the handoff said the step was unverifiable. Keep a short **"corrections to
earlier claims (do not re-introduce)"** list beside the register, with the superseded sentence
quoted and the number that replaces it, so the next session inherits the correction rather than
the claim.

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
target `blocked` in the eventual scorecard. This early gap-finding is what sets
honest expectations instead of surprising you mid-YAML. **But a gap is often an *addable
general primitive*, not a permanent blocker** — an instantaneous δ-PSC synapse, a white-noise
membrane drive, a timed current pulse, sparse random connectivity are backend features any study
of that class wants. If the missing piece generalizes, add it root-cause to the backend (with a
regression test) and un-block the target; reserve `blocked` for gaps that don't generalize or
need heavy new machinery, and `out` for the ones you judged not worth building because the other
targets already test what they would. State intent in the metadata either way (the
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

**Match the MODALITY the Methods name, not the one that is easiest to load.** "The same data"
from one release comes in forms that are not interchangeable: a volumetric NIfTI and the
grayordinate CIFTI derived from it differ in voxel set, in coverage and — the one that bites —
in spatial smoothness. Pang2023's Methods say connectopic mapping was applied to *"the volumetric
voxel-wise resting-state fMRI data"*; using the CIFTI subcortical grayordinates instead covered
88 % of the ROI's voxels rather than 100 %, and produced a similarity field whose spatial
autocorrelation was several times shorter, which was the whole of a target's shortfall for two
sessions. **Record the modality in `DATA.md` as a decision, with the Methods sentence quoted**,
and treat a coverage mismatch against the deposit's own arrays (their matrix is N × N over the
full mask, yours is over a subset) as the first evidence that you took a different file.

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

## Phase 4 — Analyses: declare the non-simulation results too

Study-specific reductions — order parameters, bifurcation / fixed-point detection, spectral
peaks, control masks, fit residuals — → pure, backend-agnostic, independently-testable
functions in `code/<study>_analysis.py`, referenced from the recipe via
`callable: {name, module: <study>_analysis}`. Keep them NumPy; carry data as **labelled
xarrays**, never positional reshapes. When aligning a
paper's connectome/observable to your node order, match **by label**, never by position
(guards silent hemisphere/order swaps). Note the host/grid split: *declared* observations
run on the host (plain NumPy is fine); only what you put under `record:` runs inside the
jitted/vmapped grid and must be backend-traceable (a non-traceable recording raises).

**A result the report quotes but no simulation produces is an `Analysis` in the study's
`analyses:` block — not a script you ran once.** This is the same rule as non-negotiable #1
applied to the half of a replication that is *analysis* rather than dynamics: a paper's basis
solve, a group average, a gradient decomposition, a cohort statistic. An `Analysis` `is_a
FunctionCall`, so it names the `code/` callable and its arguments, and each argument is either a
literal `{value: …}` or a declared input — `{used: {analysis: <name>, output: <var>}}` for
another analysis, `{used: {experiment: <id>, output: <var>}}` for a run:

```yaml
analyses:
  - name: fig1_basis_raw
    label: "That basis on the 29,696 cortex vertices"
    description: >-
      The solve returns all 32,492 vertices (zeros in the wall); the analyses run on the
      cortex vertices only ...
    callable: {name: mask_vertices, module: pang2023_analysis}
    arguments:
      data: {used: {analysis: fig1_basis_solve, output: emodes}}
      mask: {value: "input/sourcedata/templates/surfaces/fsLR_32k_cortex-lh_mask.txt"}
```

Declaring it buys four things a script cannot: it runs in dependency order alongside the
experiments under a bare `tvbo run <Study>.yaml`; its output persists as a container a figure
can bind (`used:` is the same PROV edge Phase 5 uses); the `used:` graph is what makes staleness
computable at all (the transitive `after` set of the bullet on re-running experiments); and the
`description:` is where the *convention* the analysis embodies gets stated as metadata — which
of two bases, which mask, which polarity — instead of living as a comment in a function.

Re-derive one with `tvbo run <Study>.yaml --analysis <name>`; it re-runs that analysis only and
names the downstream containers it has just made stale. It is local-engine only and refuses to
combine with any flag that selects or reshapes simulation work (`--experiment`, `--shard`,
`--rendered`, `--limit`, `--smoke`, `--set`, `--pin`, …), because an ignored flag there would
exit 0 having simulated nothing — a "success" on a cluster. An `Analysis` is also the one place
the recipe still permits arbitrary Python, so non-negotiable #10 lands hardest here: make it
resumable if it is slow, never parallel.

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
breaking the other. Grep for siblings and align them. (5) **A statistic that picks an EXTREMUM
over a whole trace needs a physically admissible window.** A latency read as a global `argmax`
will happily select numerical ripple that precedes the event: Pang2023's time-to-peak put one
region's peak *before* the stimulus could reach it, on a 341× smaller amplitude than the real
arrival, because a 200-mode truncation leaves pre-arrival oscillation everywhere. Restrict the
search to what the physics allows (`t ≥ t_on + d_min/(γ_s r_s)`) — and before calling the
difference a discrepancy, **run the same statistic on the deposit's own arrays**: theirs had the
artefact too (7 of 180 regions non-causal, and the correction moves their own published P from
0.034 to 0.093), which turns "our number disagrees" into a documented property of the published
definition. The companion instinct to resist: **do not drop the inconvenient unit.** Check the
released code for an actual exclusion first; when there is none — as here — the outlier is
telling you the statistic is wrong, not that the region is bad. (6) **A cohort reduction must be
NaN-aware AND must report the N it actually scored.** In any real cohort some units are
incomplete — a subject missing two of the paper's tasks, a parcel outside the mask. A plain
`mean` propagates that one gap into the whole column, and, worse, a paired test computed
afterwards divides by a denominator counting units it never scored, so the statistic is wrong in
a direction no NaN reveals. Aggregate with `nanmean`/`nanstd`, drop non-finite pairs *before* the
test, and carry the surviving N into the prose. (`xarray`'s `.mean(dim=)` already skips NaN for
float dtypes; the trap is the raw-NumPy path beside it, which is why the two must be checked
separately.) Pang2023's T32 lost one subject's EMOTION and RELATIONAL contrasts out of 255, and
that alone NaN'd the entire task column.

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

**A `Figure` is layout + binding + style; keep compute and plotting code out of it.** The
mechanics are in **`assets/figures.md`** — every `layout` key, the size/aspect/type-size
measurement protocol, how a grammar panel binds its data through `layers`/`used`/`encoding`,
when a bespoke `@bsplot.register_panel` interior is warranted, how to choose *which* point a
marker marks, and how the paper's own published data binds by IRI. Read it before writing the
block: three of its rules — derive `height` from the original's pixel aspect, set
`trim_margins: false`, and VERIFY type size in pixels rather than trusting the declared
`font_size` — are the most-repeated defects in this skill's history, and every one of them
costs a re-render.

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

## Phase 6 — Report: `report/report.qmd` (every number computed)

**The report MUST carry a "Where the Methods and the code diverge" section** whenever the study
ships code — a summary table by class (A–E) with counts, the two or three entries that would
silently produce a wrong figure spelled out, and a short paragraph on why one declarative
description removes the whole class. This is a headline result, so give it a numbered
section of its own rather than burying it in Limitations; the full evidence lives in
`report/analysis/methods-vs-code.md`. State plainly that the divergences are *visible* only
because the deposit is open — otherwise the section reads as a criticism of the most transparent
papers.

**Every choice RECOVERED by matching the deposit must be disclosed in the REPORT, not only in
the register — and the report must say what it does and does not license.** This is the single
easiest way for a replication to overclaim, and it is easy precisely because nothing looks
wrong: the recipe is honest, the register is honest, the report quotes computed numbers, and a
reader still cannot tell that part of the pipeline was tuned until it matched. Pang2023 shipped
a report saying "T30 met, all six numbers within 3.1 points" while three determining choices —
the graph sparsification, the variance weight and window, and the input field's smoothing — had
each been fixed by scoring against the authors' own arrays, and the words "sparsified",
"smoothed" and "1/λ" appeared nowhere in the PDF.

Write the disclosure as its own Limitations bullet, and give it all three parts:

- **Name each recovered choice and read its value FROM THE SPEC**, never typed — the same rule
  as any other number (`analysis_argument(<analysis>, <argument>)` off the loaded study, so the
  prose cannot outlive a change to the recipe).
- **State the consequence in the reader's terms**: the agreement is evidence about the *paper's*
  pipeline — it says these steps are what the published figure was made with — and it is **not**
  an independent check of yours, because yours was tuned until it matched.
- **State the counterweight, if you have one.** What in the chain was *not* fitted (masks,
  meshes, the eigensolve, the correlation), which alternative causes were excluded first, and
  what makes the choice a recovery rather than a fit — a single value repairing several
  independent structures at once, with the answer flat over a range, is the strongest form.

The same bullet is where a **declared deviation from the paper's own code** belongs (a corrected
statistic, a causality constraint), with the uncorrected numbers computed and printed beside the
corrected ones so the reader can see both.

**A number computed in a scratchpad is not a computed number.** Diagnostics done outside the
recipe — a sensitivity sweep, a cross-tabulation against the deposit — produce exactly the kind
of striking figure that ends up typed into prose, where nothing recomputes it and nothing catches
it drifting. Promote them before they enter the report, and there are only two homes:

- **A declared analysis**, when the diagnostic runs on your own artifacts. A mode-count
  sensitivity sweep becomes `<fig>_sensitivity` with its own arguments, and if it needs
  something the recipe does not otherwise declare (a basis carried past the model's mode count)
  it declares that too, labelled as being for the check and nothing else.
- **The identity harness**, when it must read the deposit — the one place where that is
  legitimate. Give the harness's report object a `values` dict alongside its pass/fail rows,
  persist it with the summary, and bind the numbers by name. Keep one genuine assertion in the
  check so it can still fail (the deposit's own vector through your code must return the
  deposit's own published number); the rest are measurements.

Two mechanical points that bite here. Shared scalars a diagnostic must agree with the model
about (`γ_s`, the stimulus window, the step size) should be **YAML-anchored once and referenced
twice**, so the check cannot drift from the model it is testing — but an alias resolves only
*after* its anchor, and `analyses:` precedes `experiments:` in the recipe, so the anchor has to
sit at the first use, in the analysis. And **audit the prose for typed decimals before shipping**:
strip the `{python}` spans, regex the remainder for numbers, and classify every hit as the
*paper's* (a literal, correctly) or *yours* (a bug). The one-line audit found eleven of ours in
prose that had been written the same afternoon.

See **writing-reports** for the report mechanics: the IMRAD structure, the metrics cell
that computes every number from the containers (nothing hand-typed), the whole Methods
rendered in one call by `STUDY.report("qmd", level=3)` — deduplicated across experiments that
share a model, every table captioned, `part: supplementary` demoting an experiment's paragraph
without hiding it — the three-colour status callouts, the copyright-safe internal/public split, references as Quarto's
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
  (`core|extended|out`, `mech|dec`, `met|short|out|blocked`) before believing the counts.
- **Score every signed difference against the noise floor of the quantity it is a difference
  OF — and doubt your own POSITIVE results, not only the discrepancies.** A margin between two
  stochastic models measured at one seed is not a result; it is one draw. Declare an
  `execution.random_seed` ensemble (Phase 4) for each model and report the margin in units of
  the seed spread. Pang2023's wave-vs-mass-model comparison looked like three wins at seed 0;
  against a ten-seed floor, edge FC (−0.005, 4/10 seeds) and node FC (+0.078, 7/10) are **not
  established** and only the FCD KS survives (−0.223, 10/10, 6.2 sd). Two claims we would have
  shipped. The same rule chooses the right floor for a *cohort* claim: a parcel bootstrap
  answers a within-subject question, so an ordering asserted across subjects needs a paired
  between-subject comparison (30 subjects, sign test) — which is what finally established the
  link our bootstrap could not. And when a single realisation *is* what a panel shows, put the
  ensemble's `mean ± sd` beside it so the reader sees the width.
- **Establish a direction along the whole nuisance axis, not at one point on it — then look for
  a measure the artefact you fear would break the SAME way.** A comparison almost always sits on
  an axis nobody is claiming anything about (mode count, window length, parcellation
  granularity). One p at one point is a choice the reader cannot audit; sweeping the axis and
  reporting where the direction is *established* is auditable and strictly more informative,
  because it also shows where the direction reverses. Pang2023's T32 went from "individual beats
  template" to "the individual basis is ahead at 44 of the 50 mode counts, the template at only
  3 — all of them N = 5–12 — and the margin decays to nothing by N = 500": the same data, with
  its own limits attached. Then the stronger move, when a *positive* result of yours could be
  explained away by an artefact in your own pipeline: find a second measure that artefact must
  corrupt in the same direction, and show it goes the other way. A misaligned or badly
  conditioned per-subject basis loses on every measure — so individual-ahead on task maps *and*
  template-ahead on resting FC cannot both come from bad alignment. That turns an unfalsifiable
  "our solves are fine" into an internal control a reader can check. Two measures pointing in
  opposite directions is evidence; two pointing the same way is also consistent with a bug.
- **Prose that quotes a count of your own artifact must be COMPUTED from that artifact.**
  Non-negotiable #2 is usually read as "don't type a simulation result", but the version that
  actually bites is a sentence that was *true when written* and then aged: our abstract said
  "fourteen places where they diverge; eight change a number" while the register it cited had
  grown to 39 rows and 18 material verdicts. Nothing failed; the headline finding of the
  replication was simply wrong by a factor of nearly three. Parse the artifact once in the setup
  chunk and quote it inline everywhere — abstract, section prose and caption from the one parse,
  never one computed table beside a hand-written summary of it.
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
- **One `STUDY.report()` call writes the model half of Methods**, however many experiments the
  recipe has — never a `for exp in EXPS: exp.render(...)` loop, which reprints the same model
  once per experiment. Each experiment's own `description:` is what the section says about it,
  so write that field as publishable prose in the recipe rather than editing the render.

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

**When the deposit ships the authors' own derived arrays, verification becomes exact** — run
our implementation on their inputs and require machine precision. Write it as a standing
harness (`code/verify_identity.py`) that prints one table, because it is what you re-run after
every refactor. Classify every check before writing it, because mixing the classes is how a
replication overclaims: `identity` (deterministic, same algorithm — RMSE ≲ 1e-12, and **a
failure is OUR bug**), `convergent` (solver-tolerance-limited — agreement stated with its
floor), `stochastic` (depends on an unpublished seed — distributional only, since matching an
exact number would mean we tuned to it).

Two rules hold whatever the deposit contains:

- **A check that cannot run must FAIL, not vanish.** A summary reading "50 checks, 0 failing"
  has to mean fifty were attempted. A check that raises before registering itself reports
  success for itself forever, and ours guarded the sign vectors of every displayed basis.
- **A convention you cannot verify is an ASSUMPTION — write it down as one**, in `targets.md`
  beside the target it feeds, with the alternatives you rejected and a sensitivity test. Most
  deposits ship no derived outputs at all, so this is the common case, not the exception, and
  an assumption hardening into an assertion through repetition is the standard way a
  replication states something it never established.

**`assets/verification.md`** has the instrument itself: the four identity traps (several
versions of "the same" array, where a nonlinear step sits relative to an average, why
"improving" the reference algorithm breaks it, NaN poisoning a least-squares solve), harness
construction (parse the spec with the loader not a regex, cover every artifact the report
*quotes*, compare a quantity against a reference of ITSELF), the assumption-labelling
procedure, gauging a free convention on the DISPLAY path and never inside the solver,
inverting a linear stage instead of fitting a scale, and verifying a figure's shape rather
than only its numbers.

**When a number of ours disagrees with a published one, `assets/deposit-forensics.md` is the
playbook** — inventory the deposit by content rather than by filename, find an order-invariant
oracle before trusting any positional comparison, measure the statistic's own stability under
a choice the paper never fixes before blaming your implementation, prove a step inert with
algebra rather than hunting it with sweeps, compare two derived matrices as a function of a
covariate, and keep "their result" and "their printed p" as separate claims. It operates under
the default this skill states everywhere: **a claimed discrepancy is our bug until a
falsification test says otherwise.**

## Phase 8 — Scale out to a cluster (ONLY when one node genuinely won't do)

**Skip this phase unless the work is irreducibly large** — a per-subject cohort (one
independent fit × N subjects) or a fit whose single run is itself heavy. First try NOT to need
it: a big *graph* → `graph_representation: sparse` + vectorized coupling; a big *parameter
grid* → a streaming reduced observable (Phase 4). Both routinely turn a "needs HPC" run into
minutes on one GPU, numerically identical (~1e-16). Assess this before packaging anything.

REQUIRED output: a packed kit + a `report/cluster_run.md` (the run route + site facts).

**The kit is the same recipe, one command — no drivers, no bash.** `tvbo workflow snakemake
<Study>.yaml -o <out> --pack` emits the whole study as ONE Snakemake DAG (one rule per
experiment; dataset experiments fan out per subject; a `from_experiment` dependency becomes the
DAG edge), and `tvbo workflow submit <kit>` runs it. Everything stays declarative in the
recipe's `workflow:` block. This is invariant #1 extended to the cluster — never hand-write
sbatch — and its corollary is that every run-time knob (swap the runtime substrate, retarget
the queue, resize a job, go to GPU) is a `--set` on the emit, never a recipe hand-edit.

**`assets/cluster.md` is the operational detail**, and a first cluster run needs all of it:
sizing per-rule memory off the COMPILE peak rather than the streaming runtime, benchmarking to
measure that peak instead of reasoning about it, the frozen-vs-spec dual-mode kit for a
version-skewed node, the float32 divergence that NaNs a stiff fit on the cluster but never
locally, the read-only container filesystem, which fixes need an image rebuild vs a re-emit,
why a dry run catches no runtime bug (smoke-test one experiment in the container first), and
running the orchestrator off the login node.

---

## Symptom index — something is wrong, start here

Every entry below **raises nothing**: plausible numbers, a clean summary, a figure that
renders. That is exactly why they are indexed by symptom and not by topic — you cannot go
looking for the section whose name you do not yet know. Find the line matching what you are
seeing, then read its entry in full in **`assets/traps.md`** (dynamical and numerical traps
first, then workflow pitfalls).

**The dynamics are plausible and wrong**

- Some sweep cells return a growing non-finite fraction while the low-parameter cells look healthy → the step was sized for the paper's *fitted* parameter, not the stiffest thing the grid visits.
- Halving the step does not merely remove NaNs, it *moves the optimum* → same cause; the converged part of the landscape was distorted too.
- The same equation diverges when solved on the mesh instead of in a truncated basis → the full operator is ~360× stiffer than the truncated one; measure the spectral radius, don't inherit the `integration:` anchor.
- A run that decayed over a short probe blows up over the production window → a marginal instability grows per STEP; measure the growth rate at two steps to tell it from a sign/operator error.
- A sweep's per-cell frame count differs from the same experiment run alone → the sweep folded transient + main into one window and the reducer's `skip` was ignored; every FC/FCD statistic is then contaminated.
- Spread climbs with coupling and reads as desynchronization → an explicit scheme at too large a `dt` sustaining numerical librations.
- A `from_experiment` seed spikes or fails to converge → a delayed system needs the τ seconds of HISTORY, not a state snapshot.
- A chaotic or multistable *network* settles on a different attractor → coupling evaluated once per step; needs `coupling_evaluation: per_stage` (a no-op for a single node — there the knob is `dt`).
- The operating point sits decades away from the paper's K → a weight-normalisation convention, not a bug.
- The paper's exact control value gives the wrong regime → a near-bifurcation operating point is discretisation-specific; re-tune to the phenomenon and cite the precedent.
- An FC/PLV/order-parameter number well below the paper's → duration, trial count and operating regime, before "structure-limited".

**It ran out of memory, or took absurdly long**

- A full-length FC evaluation OOMs at hundreds of GB → the observable must stream (`reduce: streaming`), and the *pre-tuning* forward sim of a fit is not a deliverable at all.
- A dense N×N coupling matmul dominates every step → `graph_representation: sparse` before any thought of HPC.
- A stage looks I/O-bound → profile it on a COLD cache; ours was one linear-algebra call at 16.6 s of a 19 s subject while a 438 MB read cost 0.2 s.
- A comment explains why a slow path is necessary → that is a claim; measure the risk it names before accepting *or* removing it.

**The artifact is stale, or is the wrong file**

- You are reasoning about a new recipe from an old run's output → `tvbo run` without `-o` DISCARDS the container and leaves the previous one on disk.
- Figures show this run's dynamics against the previous run's analyses → re-running an experiment does not invalidate the analyses derived from it; take the dependency set from the study's own transitive `after` stage.
- A direct Python call returns the new value while the run reads the old → a cache keyed on inputs, not code; know which of the three remaining holes you are in.
- A PDF predates the figures it embeds, or a comparison artifact predates both → the framework's staleness detector only sees what a run touched; audit container → figure → compare → PDF at the end of every session.
- An A/B composite shows the wrong paper's figure → an unpinned `rglob("fig_03.png")` across a lineage of sibling studies.
- A documented "we established this is impossible" → a measurement with a date on it; any change that could bear on it invalidates it.

**The edit did not take, or landed somewhere else**

- A script reports success and the intended edit never appears (or the file explodes to megabytes) → `str.replace` on a computed slice that evaluated to the empty string.
- Paths silently resolve into a sibling study's tree → `Path(__file__).parents[N]` after the module moved; grep for the climb.
- A partly-refilled cache mixes two algorithms inside one cohort mean → deleting the cache is part of editing the callable that fills it; key the path on the choice that changed.

**The claim is wrong even though the numbers are right**

- A metric matches in shape but not magnitude → the definition and the empirical modality it is compared against are part of the claim; read them from the Methods.
- An exact count differs from the paper's integer → realization dependence on an unpublished seed; state it, never tune the seed to hit it.
- Wavelengths differ ~1.3× across mesh sources → match the *invariant* of a geometric decomposition, not the magnitudes.
- The whole study ran at an exploration axis's value rather than the model's → a single-value axis still OVERRIDES the parameter it names; express an ensemble with `n_trials` or an `initial_conditions` sweep.
- An 8 GB outer product instead of an elementwise difference → two runs naming the same axis differently; reconcile by NAME, align by COORDINATE.
- A published panel is internally inconsistent with the paper's own workbook → a source-data defect; scope it `out` and say why.
- A generated source file will not compile → a large derived array was inlined instead of declared by `source:`/`producer:`.

**Also in `assets/traps.md`**: keep generated files out of git at the study root; track
`report/analysis/` from the first commit (it is the only copy of the register, the targets
table and the figure map); the report must stand alone with no cross-references to a sibling
study; hand-written `plot_*.py` and A/B compose drivers are redundant; a *live* vendored
dependency is not cruft, so confirm against the actual run paths END-TO-END before deleting
it; and framework gaps surface late if you skip Phase 1.5 — find them before the YAML.
