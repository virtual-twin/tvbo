# Symptom index — dynamical, numerical and workflow traps

These cost this project the most time, and they share a shape: **nothing raises**.
A wrong step size, a stale container, a single-value exploration axis and a
mis-keyed cache all return plausible numbers. So the entry point is the symptom,
not the topic — find yours in the spine's symptom index, then read its entry here
in full.

## Dynamical & numerical traps (these cost us the most time)

- **A published "unstable branch" your continuation cannot find may be the paper's ROOT SOLVER
  terminating off-root — replay the paper's own procedure and classify every returned point by
  its residual.** MATLAB's `fsolve` (and any dogleg/trust-region equation solver) terminates at
  stationary points of the least-squares merit ½‖f‖² — points with `J^T f = 0` but `f ≠ 0` — and
  a model sitting just below a saddle-node threshold has a whole valley of them: the fold's
  ghost. Because `J^T f = 0` with `f ≠ 0` forces a SINGULAR Jacobian, an eigenvalue check labels
  every such terminus "not stable", so unfiltered fsolve output plots as a coexisting *unstable
  fixed-point branch* that no root of the equations backs. Deco2014's Fig-2c E-E branch is
  exactly this: the E-E fixed point is provably unique (order-preserving lfp/gfp bracket,
  gap ≤ 2e-14), yet random-seed fsolve reproduces the published open circles as merit termini —
  onset G = 1.44 vs the paper's 1.47, ‖f‖∞ ≈ 1e-3 on an equation scale of 2e-3, |Re λ| ≤ 7e-8/s
  vs 3/s for genuine saddles. The discipline: (1) classify solver output by ‖f‖∞ against the
  equation scale, never by the solver's own success flag in either direction; (2) polish stalls
  with `least_squares` and check `|J^T f|` — a converged nonzero-residual stationary point is a
  well-defined mathematical object you can trace in the parameter, compare to the published
  branch, and draw (labelled as a solver terminus, never as a state); (3) run BOTH controls
  before claiming the artifact — a positive one where genuine coexisting roots exist and must be
  found as true roots, and a variant test showing no plausible mis-transcription of the
  equations makes the terminus an exact root; (4) state the conclusion as an inference about
  unpublished code unless the paper's solver script is released.

- **Never locate a fold window by scanning the drive axis, and never trust an inner
  elimination you did not residual-check — both misplace the bistability threshold, and a
  misplaced threshold reads as a self-contradiction.** Two measured failure modes from one
  Deco2014 question ("at which J_NMDA does the node fold?"): a *damped* fixed-point iteration
  for the slaved variable stopped converging exactly where the recurrence gets interesting
  (residual up to 0.1 in S_I) and its error wiggles counted as extra nullcline crossings,
  moving the onset from the true 0.28 to 0.20; and an honest scan with a Newton-solved inner
  variable still stepped over windows narrower than its grid (8e-6 nA at the cusp against a
  5e-4 step). The discipline: invert the fixed-point condition into the closed-form drive
  locus `x*(S) = H⁻¹(r(S)) − c(S)` and read the window edges off its interior extrema — exact,
  catches windows of any width, no scan — and Newton-solve every inner elimination. Then say
  which AXIS each threshold lives on: a node that folds in *drive*, inside a window of
  deficits that the network's additive coupling can never deliver, does not make the network
  fold in *G* — "the node folds from 0.28" and "the network never folds up to 0.70" are
  simultaneously true, and prose that omits the axis reads as 0.70-versus-0.28 nonsense.

- **Before a native analysis observation is allowed to overturn a cross-variant ordering,
  reproduce it host-side at the same operating point — a mistargeted stimulus lowering
  produces smooth, plausible, wrong curves.** Deco2014's Fig-6f Fisher information came out
  inverted (FIC lowest where the paper has it highest) from the recipe-native path, while a
  host recomputation at the identical operating point — one that reproduced the native E-E
  and FFI curves EXACTLY — put FIC highest, matching the paper. The inversion traced to
  stimulus-targeting regressions in the backend (the event's `target_regions` dropped, so
  external inputs broadcast to every node; the fisher observation's node mask resolved
  empty), not to the model — and the broken value was unfalsifiable from its own output:
  smooth, decreasing, right units, right order of magnitude. The discipline: any analysis
  observation that carries a conclusion gets an independent host-side recomputation (same
  Jacobian convention, same noise, same operating point), checked PER VARIANT — matching for
  two variants of three validates nothing about the third when the third exercises a
  different code path (here: the FIC constraint solve).

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

- **A fan container can hold every cell correct and still be scrambled — the cell ADDRESSES
  are a separate thing to verify.** Symptom: two analyses that score the same quantity with
  literally the same callable disagree at a shared operating point (our EDF10 landscape read
  0.18 where the seed ensemble read 0.076). Diagnose with shapes-encode-configuration: a
  physical per-cell statistic (BOLD std) must vary smoothly along a parameter axis and be
  near-constant along a seed axis — a PERIODIC pattern along the parameter axis is the
  signature of a flat cell order refolded positionally under the wrong axis order (period =
  n_outer/gcd tells you which). Repair needs no re-simulation once the permutation is pinned
  (verify cell-identity, smoothness, and a ratio against an independent single-axis container
  at the shared point BEFORE swapping). Put that triple in the identity harness so a future
  scramble fails the build; the root fix belongs in the framework's assembler (key cells BY
  VALUE and RAISE when a coordinate can't be matched — a silent positional fallback is how
  this shipped).

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
  trusting the pass is how you end up believing a stale number twice. The hole is specifically
  **`tvbo run --experiment N`**: a whole-study run recomputes every analysis, but a partial one
  deliberately does not, so the containers keep the previous run's numbers beside a fresh result.
  It now names the affected set (transitively, off the study's own `used:` edges) and tells you to
  refresh — heed that warning rather than rendering on top of it.
- **A cache is keyed on INPUTS, so editing a `code/` callable invalidates only what the key
  actually covers — know which is which.** This is the same defect as the bullet above, one level
  deeper, and it is worth knowing by name because the caches are invisible. Since the Pang2023
  incident tvbo closes the worst of it: a `producer:` parameter's artifact digest now includes a
  hash of the **source of the module defining the producer**, so editing that file yields a
  different artifact rather than a stale hit — on the **next process**, which is what every
  `tvbo run` is. Three holes remain, and all are silent:
  - the digest hashes only the producer's **own file**, so an edit to a helper in a *sibling*
    module under `code/` is still invisible;
  - within one long-lived process (a Jupyter kernel, a report render) the digest is **pinned to
    the source the module was loaded from**, because Python does not re-execute an imported
    module: the edited function is not running either, so the artifact keeps matching the code
    that filled it. Restart the kernel, exactly as you would to pick up the edit itself;
  - a study's own `.npz` solve cache is keyed on its path, and an **analysis container** on its
    name — neither hashes anything about the code, so re-deriving one is a deliberate act:
    `tvbo run <Study>.yaml --analysis <name>` (which re-runs only that analysis and names the
    downstream containers it just made stale), or delete the file.

  Untreated, the symptom is that an experiment reads the artifact from *before* the edit while a
  direct Python call to the same function returns the new value — two answers from one function,
  and the run is the one that is wrong. In Pang2023 this drove the wave model with a pre-edit
  stimulus projection for a whole afternoon and read as an unexplained "the run resolved a
  different basis". Diagnose with file mtimes (`ls -la ~/.tvbo/constants` against the edit's
  timestamp), and prefer arguments over code for anything you expect to vary, since an argument
  IS in the key.
- **A content-addressed cache must key its MEMORY and its DISK copy on the same thing.** Adding
  the code digest to the artifact path but not to the in-process cache key is worse than not
  adding it at all: a session that materialises, has its producer edited underneath it, and
  materialises again computes the new path from the new source while the in-memory cache still
  answers on the old one — writing pre-edit arrays under a digest that asserts they are
  post-edit. Every later run then reads that file and trusts it. Whenever you add a term to a
  cache key, grep for every other place that key is constructed.

  Two traps follow, and the second is subtle enough to have been got wrong twice here. First,
  the term must describe **what the process will actually do**, not what is on disk: a digest
  re-read from the file each time claims the artifact matches code that Python is not running.
  Second, a test that edits the source mid-process is testing a **reload that never happens**,
  so it can only pass by faking one — which is how a fix that closed nothing passed its own
  test. Assert the invariant directly instead: that the memory key and the artifact path carry
  the **same** digest, and that the digest tracks the loaded source rather than the file. Test
  the end-to-end invalidation where it is actually defined — across two processes, i.e. with
  the caches cleared between the two calls.
- **NEVER text-edit a spec or a report artifact with `str.replace` on a computed slice.** The
  idiom `old = t[t.index(A):t.index(B)]` returns the **empty string** whenever `B` precedes `A` in
  the file — a table row that got reordered is enough — and `t.replace("", new)` then inserts
  `new` between **every character**, so a 33 KB register becomes 81 MB of interleaved garbage and
  the intended edit never lands. It is silent: the script prints its success message. Use the
  Edit tool (it fails on a non-unique or absent match) or anchor on a full unique line; if a
  script must do it, `assert old` before replacing, and re-parse the artifact afterwards with
  whatever the report uses to read it. Recovery, if it happens: the original bytes are all still
  there, so `corrupt.replace(new, "")` returns the file exactly — confirm with
  `len(corrupt) == len(recovered) + (len(recovered) + 1) * len(new)`.
- **Track `report/analysis/` from the first commit — it is the only copy.** The register, the
  targets table and the figure map are authored deliverables with no upstream and no regenerating
  script. A study left untracked "until it is ready" has no recovery path for exactly the files
  that cannot be recomputed, and one bad `str.replace` (above) is then unrecoverable except by
  luck. Gitignore the heavy generated trees, commit the analysis prose early.
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
- **A documented "we established this is impossible" is a measurement with a date on it.**
  Handoff notes and register rows harden fast: once `figure-state.md` says a step is
  unverifiable, nobody re-measures it, and the claim outlives the configuration it was measured
  in. **Any change that could bear on a recorded negative result invalidates it** — re-run the
  test rather than inheriting the conclusion. Two of Pang2023's flatly-stated impossibilities
  ("the gradients are decimal-level unobtainable", "no spectrum reproduces the published
  variance") were both measured on a graph construction that a later fix superseded, and both
  fell in an afternoon once re-measured. Cheap habit: when writing a negative result, record the
  *configuration* it holds under in the same sentence.
- **A comment justifying a slow or awkward path is a claim — measure it before accepting it,
  and before "optimising" it away.** `_subject_similarity` carried a deliberate note that the
  full SVD was kept because η² is not invariant to a per-component sign flip and a cheaper Gram
  eigendecomposition "would silently change the similarity". True, and the size was never
  measured: rms 3.6e-4 against the matrix's own sd of 0.047 (r = 0.99997), while the discarded
  factor cost 16.6 s of every 19 s subject — 20 hours against 40 minutes over a 765-subject
  cohort. Measure the stated risk, put the number in the docstring, then decide. (The converse
  discipline holds too: neither sign convention is canonical, so the difference is a property of
  the printed method, which is a register line, not a silent choice.)
- **Profile before you optimise, on a COLD cache.** The same run looked I/O-bound and was not:
  reading a 438 MB CIFTI took 0.2 s from an external volume, while one linear-algebra call took
  16.6 s of a 19 s subject. A one-file timing harness (print elapsed after each stage) settled in
  30 seconds what an afternoon of restructuring the recipe would have got wrong.
- **Deleting a cache is part of editing the callable that fills it.** Caches key on inputs, not
  on code, so a changed algorithm silently reuses the old answer — and a *partly* refilled cache
  is worse than a stale one, because it mixes two algorithms inside one cohort mean. Key the
  cache path on the choice that changed (`..._fwhm-4.npz`) so the two cannot mix, and delete the
  superseded files explicitly, saying how many and why.
- **The framework's staleness detector only sees what a run touched — audit the whole chain at
  the end.** Artifacts *outside* the container tree go stale silently: a figure-comparison
  summary the internal report quotes was six days old while the figures it described had both
  changed, and the PDFs predated the figures they embed. Close every session with an ordering
  check — newest container ≤ oldest figure ≤ compare artifacts ≤ both PDFs — and rebuild until
  it holds. Beware that *running* the detector can create the staleness it reports: recomputing
  one analysis to test it invalidates everything downstream.
- **Framework gaps surface late** if you skip Phase 1.5. Find them before the YAML.

- **A quantity can match the published material at r = 1.000 and still be in the wrong unit.** Symptom:
  every correlation, p-value and scorecard verdict is right, and one axis stops well short of the
  paper's — ours reached 38 where theirs reached 80. A monotone rescaling leaves Pearson and
  Spearman untouched, so nothing statistical can see it; only a *range* comparison can. Check a
  landmark the paper prints in that unit (canonical iTBS is "30 pulses / train"; we read 15) and
  put the derived quantity in the published-data oracle table so a match is confirmed by range as well
  as by rank.

- **A panel's title names a column; its axis range names the quantity — and the two disagree
  more often than you would expect.** Symptom: your rendered field looks right in shape and is
  off by orders of magnitude in scale. Kadak's per-connection panels are titled
  `coupling.xx.nu_post` in three figures and plot the absolute weight, the signed change and the
  unsigned relative magnitude respectively; the published material's plotting cells read a *differenced*
  frame whose columns kept the original names. Identify the quantity by putting each candidate
  through and comparing against the published tick labels on all ten panels at once, then
  register the mismatch as a convention trap.

- **A colourbar that factored out a shared multiplier is a silently wrong axis.** Symptom: a
  slim bar reads "3, 1, 0" for a field spanning 3e4, or "3, 0, -1" for one spanning 1e-4, with
  the exponent nowhere on the figure. Always read a bar's printed numbers against its layer's
  own min/max before believing the panel.

- **Geometry read inside a panel callable is pre-layout geometry.** Symptom: hand-placed labels
  land inside the plot, or a "square" panel is an ellipse. `ax.get_position()` at draw time
  returns the box before the layout pass has run. Use `set_aspect`/`box_aspect` and let the
  engine solve it; anything that genuinely must run after the tidy-up belongs in the renderer's
  post-format pass, not in the panel.

- **A tick formatter asked for a string before the first draw answers the same for every axis.**
  Symptom: a pass meant to fix two ticks that print one number instead re-rounds axes that were
  already correct — 0.00135 becomes 0.0014 everywhere. A `ScalarFormatter` carries state the
  draw establishes, so a check that consults it early sees every axis as degenerate. Judge the
  DRAWN labels in a pass over the finished figure, and reach the cells through `child_axes` —
  `fig.axes` does not contain a grid's insets.

- **Your panel's axis limits are the paper's frame, not your data's extent.** Symptom: the
  reference marks in your panel sit at different fractions of the width than the same marks in
  the original, and the clouds "look shifted". Calibrate: two marks whose data values you know
  give a pixel→data map for the published panel, and the frame falls out of it. Ours auto-scaled
  to the responsive subset (1–15.5 Hz) where the paper framed the whole protocol space (0–20).

- **Editing importable framework code while a run is in flight kills the run, and the failure
  names a symbol you just created.** Symptom: `ImportError: cannot import name '<the function
  you added ten seconds ago>'` from experiments that had not started yet, while the ones already
  running finish fine. Codegen re-reads its **template per experiment**, but the template's
  helper module was imported **once** at process start, so a long run ends up executing a new
  template against an old module. Nothing about the edit is wrong; its timing is. Treat a
  launched sweep as a freeze on everything importable that it touches — the study's `code/`, the
  templates, the framework — and hold the edit until it lands. The cost is measured in whole
  conditions: ours lost a sweep and five baselines and then wiped a results tree on top.

- **A waiter that watches PIDs reports "done" for a run that crashed.** Symptom: the chained
  step deletes the derived results and re-derives them from a container that is missing half its
  experiments, so every downstream number is quietly computed on a subset. `wait` returning is
  evidence that the processes ENDED, not that they SUCCEEDED. Gate the chain on a completeness
  check of the artifacts themselves — every expected experiment group present, with the expected
  cell count — and make that gate the thing that decides whether the destructive step runs.

- **A truncated read is not evidence, however plausible the fragment.** Symptom: a confident
  conclusion about what a published cell computes, drawn from a slice that ended mid-line at
  `x = df` where the file continues `_full.apply(...)`. Sizing a file before reading it whole is
  right (a 459-line file can be 95 % data); founding a claim on the part you happened to read is
  not. When the claim is about one assignment, match that assignment with a regex over every
  cell and read the whole match.

- **A multi-file edit script that asserts before it writes leaves the earlier files unwritten.**
  Symptom: the report references a column that the analysis module never gained, because the
  heredoc's third replacement failed its assertion and aborted the whole script after the first
  two had only been computed in memory. Either write each file as you finish it, or verify every
  match before writing any. Confirm with a grep for the new symbol, never with the exit code.

- **A panel that blanks its slot gets the slot back from the format pass.** Symptom: a colour
  bar sits inside a ghost frame carrying its own 0–1 ticks, and the tick options declared on that
  panel shape the ghost instead of the bar. A scale/legend/grid drawer calls `ax.axis("off")` on
  its host axes and then a figure-wide tidy-up re-derives ticks for **every** axes in the figure,
  including the one just blanked. Re-apply the blanking after the format pass, and let a scale
  panel hand back the axes its bar actually lives on so a declared frame lands there. One caveat
  that will bite: `Axes3D` reports `axison == False` by construction (it draws its own frame), so
  a blanket "re-blank everything that was off" erases every 3-D panel — exclude them explicitly.

- **Hiding a panel's tick labels asserts a shared scale the axes do not have.** Symptom: three
  panels of the same quantity, only the leftmost labelled, and their limits differ by a third —
  the reader compares heights that are not comparable. Hiding labels is a *display* change;
  sharing a scale is a *limits* change, and only the second makes the first honest. Declare the
  group (`share_y: ["c,g,h"]`) so every panel in it ends on the union of the group's limits.
  Never paper over it with a literal `ylim`: the run that follows moves the data and the literal
  silently clips it.

- **An out-of-view tick label still has a window extent.** Symptom: an overlap detector reports
  collisions between labels that are nowhere in the rendered PNG, and you chase them for an hour.
  A locator emits ticks past the view limits; matplotlib does not draw those, but they remain
  `Text` artists with `get_visible() == True` and a real bbox. Filter tick labels by the axis's
  view interval, and skip the artists of any axes that is switched off, before measuring anything.

- **A size solver that reads a stale PNG walks the declared size off a cliff.** Symptom: a
  figure's `height:` is negative in the spec and matplotlib refuses the figsize. The solver
  corrects `declared += (target - measured)`; when the render fails it leaves the previous PNG in
  place, the solver measures that, and every iteration corrects against a size it never produced.
  Gate on the output's mtime: no fresh file, no correction — restore what was declared and stop.

- **A stored column can contradict the columns it was computed from, and nothing raises.**
  Symptom: one condition of a sweep is a dramatic outlier — the response inverts, the responsive
  count halves, a correlation goes non-significant — and it reads as a resonance crossing or a
  genuine null. Before believing it, recompute the derived column from its own container's
  inputs. Ours reproduced the stored `power_modulation` to 1e-15 in eleven containers and failed
  on the twelfth for all 432 cells, matching `(post - pre)/post` exactly where the recipe declares
  `(pre - post)/pre`. The cause was codegen emitting a NAMED argument positionally, so which array
  landed where was decided by the order the datamodel yielded the arguments mapping — and
  regenerating the datamodel mid-sweep reversed it. Three lessons, in order of durability:
  **(a)** put the self-consistency check in the gate that stands before the destructive re-derive,
  not in your head — the old gate gave the corrupt container a clean bill because it was complete
  and carried every trace; **(b)** enumerate the exposure rather than guess it — only a callable
  taking TWO OR MORE observation-valued arguments can be scrambled this way, which in a whole
  study was one observation; **(c)** prove a codegen fix at RUNTIME on a short experiment written
  into a scratch container, and compare it against a container written before the regression —
  ours came back bit-identical, which is the only evidence that actually settles it.

- **`--experiment 15,14` does not run 15 first.** Symptom: you reorder the argument to get one
  experiment early and nothing changes. The list is a FILTER; the run follows the study's own
  declaration order. When one experiment gates something you want hours early — the last two
  unscored targets, a figure that has never rendered — give it its own invocation and chain the
  rest after it. Two sequential calls cost nothing and moved a container from 03:30 to 01:30.

- **One `tvbo run` cannot fill the machine, and a serial sweep leaves most of it idle.** Symptom:
  a six-experiment job is projected to finish in eleven hours while `top` shows four of twelve
  cores busy. A single run draws roughly four cores whatever you give it, so N experiments in one
  invocation is N sequential four-core jobs. Split them across parallel invocations — three jobs
  of two took the same work from ~13:00 to ~04:00 — and size the split by cores, not by taste:
  jobs x cores-per-job should land at or just under the core count. Check free memory BEFORE
  committing (52 % free, peak footprints of 4.8/1.2/1.4/1.3 GB, each job under its own guard),
  because the failure mode of over-splitting is a swap death, not a slow run.

- **Never regenerate the datamodel, or touch importable framework code, while a run is in
  flight.** Symptom: experiments written before some moment are right and the ones after are
  wrong, with no code change to blame. Templates are re-read per experiment and the generated
  datamodel is imported once per process, so a regeneration mid-sweep changes what later
  experiments emit while the sweep looks untouched. This is how an argument-ordering latency
  became a data corruption: the bug existed all along and was harmless until a regeneration
  reversed a mapping's order at 20:29, splitting one sweep into a correct half and a wrong half.
  If you must fix the framework during a run, finish the run first, or accept that everything
  after the edit needs re-running and gate on it.

- **A recipe description that narrates HOW its result is produced goes stale the day the
  pipeline moves — and the report renders it verbatim.** Symptom: the report's Methods
  (rendered from the experiment's `description`) contradict its own Results about how an
  experiment ran. Deco2014's exp-71 description said its FIC contrast was "run by the Brian2
  reference builder" months after the experiment ran natively, and quoted the reference run's
  correlations (0.038 → 0.019) beside a container holding different ones — every number and
  every pipeline claim in a description is a copy nothing regenerates. Descriptions state the
  current contract and the PAPER's targets only; a measured value of OURS belongs in the result
  container the report computes from, never in the description. When a pipeline goes native or
  a driver is replaced, grep the recipe's descriptions for the old mechanism's name in the same
  commit.
