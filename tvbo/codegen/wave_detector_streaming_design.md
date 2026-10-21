# Tier 2 — streaming GPU wave detector (Koller Fig-6 `cortical_wave_metrics`)

> **STATUS: WIRED INTO exp_41 + runs end-to-end (2026-08-05).** The declarative detector
> (`wave_operators` producer + `detect_matmul` transcription + masked pearson + both permutation
> surrogates + grouped `wave` reducer + group-vmap + finalize) is byte-identical to the CPU path
> to ~4e-17, now PROMOTED to permanent manuscript tests
> (`Koller2024/code/tests/test_wave_detector.py`) and WIRED into `Koller2024.yaml` exp_41 as an
> `!include`d Observation (`code/observations/wave_metrics.yaml`): GPU streaming vmap job, full
> faithful Fig-6 panel (all 4 IF), K union `[1e-6, 10]`. Full `render_code('tvboptim')` +
> a guarded 2-cell run produce a sane per-hemisphere `wave_metrics` (n_groups=2, metric=3).
> Wiring surfaced + fixed FOUR tvbo-core integration gaps: (1) producer-arg resolver parity —
> `network.positions`/`network.instrength` now resolve in `param_io._resolve_ref` via a shared
> `resolve_network_node` the observation path also uses; (2) partition `n_groups` read from the
> materialised gather artifact (`param_io.read_artifact`); (3) `n_groups` deferred on the bare
> `resolve_reduction` predicate call (no experiment); (4) observer DVs emitted in DEPENDENCY order
> (`_toposort_derived`) because the pydantic keyed collection alphabetises DV keys, breaking the
> "declaration order = emit order" assumption. **n_permutations dominates XLA compile** (100→8+ min
> CPU for 2 cells; 10→~25 s) — a compile-time cost, cell-count-independent.
> Remaining: rewrite `koller2024_fig6.py` aggregation for the single gridded result (was per-cell
> h5 glob); pack + GPU cluster run at n_permutations=100.


Scoping for reimplementing the host cortical wave detector as a jittable, vmappable
streaming `Observation`, so the whole exp_41 grid runs as ONE exp_40-scale GPU job
instead of 1560 fanned CPU cells. Supports task #21. Companion to
`reducers_unification.md`, `streaming_bold_handoff.md`.

## 0. Why

Measured per-cell host detector = **313 s, 96.4 % in the permutation surrogate**
(arccos over ~500 verts × k-ring × timepoints × 100 perms). The cost is genuine FLOPs,
not Python overhead — numpy batching was byte-exact but **slower** (410 s), so there is
no CPU win. GPU is the only lever: the surrogate and per-timestep math parallelise
perfectly. Full grid detection today = 135 CPU-hours; on GPU it collapses to minutes.

## 1. Key realization — this fits existing machinery; the "PRNG hard part" is gone

- **Target engine = the integrator-carry reducer**, not the sliding-window one.
  `resolve_reduction` (`tvbo/templates/tvboptim/utils.py`) → `render_reduction`
  (`tvbo-tvboptim-observation.py.mako:633`) emit an `(init, update, finalize)` triple
  consumed by tvboptim `prepare(reduce=…)`. The **same triple serves the host base run
  (whole trajectory as one block) and the grid** (`experiment.mako:2985`), so one
  implementation covers both and byte-validation is direct.
- **No in-reduction PRNG — by design.** Task #7 closed the stochastic-key idea as
  misconceived. The intended surrogate is a **fixed `(n_perm, n)` permutation-index
  table baked as a constant**, and the schema slot already exists:
  `Surrogate` (`schema/tvbo_datamodel.yaml:3006`) with `statistic` / `permute` /
  `permutations` (= name of the observer parameter holding the table). `resolve_reduction`
  already resolves surrogates (`utils.py:1580`) **but no mako emits them** — resolved-but-
  unbuilt. Completing that emitter is the central new work, and it is deterministic +
  byte-checkable, not a novel RNG primitive.

## 2. Architecture — streaming per-timestep fold, θ never materialized

Every timepoint's wave/directed decision is **independent of other timepoints** (the
surrogate shuffles *vertices*, not time), and the detector only ever uses `exp(iθ)`, so
the CPU `np.unwrap` is a no-op we drop. Therefore the detector is a natural per-timestep
fold inside the integration scan, emitting on a `period = downsampling_factor` (the
recurrence reducer already has `monitor`/`period` and `stride` forms,
`observation.mako:914,775`):

Per emitted step `t`, source = θ_t (n_verts,):
1. `asim_obs_t` = angular-similarity(gradient(θ_t))            → (n_interior,)
2. surrogate: `vmap` the fixed `(n_perm, n)` table over θ_t → 100 asim → `max` over
   vertices → compare ≥ |asim_obs_t| → **wave_present_t** (bool)
3. HHD flow-potential `U_t` = `M @ normalize(grad θ_t)`; `corr_t` = pearson(instrength, U_t);
   instrength-permutation surrogate (same table idea) → **sig_corr_t** (bool)
4. fold into carry: `n_wave += wave_present_t`, `n_dir += (sig_corr_t & wave_present_t)`,
   and stash `corr_t`, `wave_present_t` into fixed `(T_ds,)` buffers (16 KB) for the median.

Finalize: `proportion_waves = n_wave/T_ds`, `proportion_directed = n_dir/n_wave`,
`rho = median(corr_t[wave_present])`. Per-cell resident memory = a handful of
(n_verts,) vectors + the constant operators — **duration-independent**, so wide vmap over
grid cells fits GPU (same property that let exp_40's EF grid run in one job).

Runs **twice** (lh, rh) — the model produces whole-brain θ; the observer splits by a
precomputed hemisphere mask. Two operator sets, or one padded-to-max with a hemi mask.

## 3. Constants — one `producer` bundle, already a supported pattern

All fixed operators are precomputed **once on the host via igl** and baked as observer
`parameters` through `param_io` (`producer:` → `_producer_bundle` → lazy `_load_constant`,
`observation.mako:856`). This is exactly the existing `wave_operators` producer pattern —
declare one producer, point several parameters at its named `output:`s. `network.positions`
/ `network.instrength` are auto-embedded (`collect_network_node_arrays`, `utils.py:3531`).

Bundle (per hemisphere): `grad_op` (sparse→dense (3·n_faces, n_verts)); `bc` barycentric;
`div_template` + **padded k-ring index table + mask** (ragged→fixed (n_verts, max_ring));
`boundary_mask`; `M` = Helmholtz-Hodge pseudo-inverse `(−cotmatrix)⁺·div_op` (dense, one
matmul); and the **`(n_perm, n)` permutation table** (the surrogate constant). None of
these enter generated source — they load once as concrete arrays.

## 4. Component → primitive mapping

| CPU step (`koller2024_wave_detection.py`) | jax form | status |
|---|---|---|
| `exp(1j·θ)`, unwrap | `exp`, unwrap dropped (exp is 2π-invariant) | EXISTS (elementwise) |
| `compute_phase_gradient` (grad_op·ce, barycentric) | `matmul`, `take`, complex mul | EXISTS (`matmul`,`take`) |
| `compute_angular_similarity` (k-ring gather, `arccos`, mean) | padded `take` + `clip`+`arccos` + masked `sum_axis`/mean | `clip`/`arccos` EXIST; **k-ring gather-mean = new primitive or padded-take+mask** |
| max-over-vertices FWE | `max` over vertex axis (+ nan→mask) | **masked-max primitive** (small) |
| `compute_helmholtz_hodge_decomposition` | `M @ normalize(pg)` | EXISTS (`matmul`,`normalize`) |
| `_colwise_corr` | `pearson` | EXISTS (`pearson`) |
| permutation surrogate loop | `vmap(table)` + compare + mean → p-value | **Surrogate emitter (resolved-but-unbuilt)** |
| `median(corr_t[wave_mask])` | fixed `(T_ds,)` buffer + masked median at finalize, OR masked histogram-median (`observation.mako:882`) | **masked-median finalize** |

Most primitives already exist in `ARRAY_FUNCTIONS` (`tvbo/parse/expression.py`) with
per-backend printers (`tvbo/codegen/code.py`). New seams are small and follow the blessed
"printer primitive" pattern: a masked/nan `max` over an axis, and the k-ring
gather-and-average (padded index + mask + `sum_axis`).

## 5. Two routes — recommend declarative, keep bespoke as fallback

- **Route A (declarative, recommended).** Express the per-timestep math as array-primitive
  sympy exprs on an `Observation.dynamics` recurrence + a `Surrogate` declaration, and
  **complete the surrogate emitter** in `render_recurrence_reduction`. Aligned with the
  design-lock and the framework's values (backend-independent, symbolic, reusable — the
  surrogate emitter then serves *any* permutation test, not just this detector). More
  upfront primitive work.
- **Route B (bespoke escape hatch).** Add a new `kind` + `render_wave_reduction` mako def
  emitting a hand-written `(init, update, finalize)`, mirroring `render_convolution_reduction`.
  Faster to first-light, but a one-off branch against the grain.

**Plan: A for the surrogate + reductions (reusable), dropping to a small B-style primitive
only where a structural op (k-ring gather, masked-max) doesn't fit an existing function.**

## 6. Concrete gaps to build

1. **Surrogate emitter** — ✅ DONE. `render_surrogate` (observation mako) emits the
   `(vmap(λp: stat(field[p]))(perms) <cmp> stat(field)).mean(...)` fold; `resolve_reduction`
   interleaves the p-value DV into the per-step chain in declaration order and derives the
   nan-aware FWE extremum from `direction`. Schema stays intent-only: `Surrogate.family_wise:
   bool` (NOT a backend reducer name) → resolver maps `greater_equal→nanmax`, `less_equal→
   nanmin`. Byte-validated vs numpy under a shared permutation table, symmetric AND max-T FWE,
   block-invariant (`tests/test_streaming_surrogate_reduction.py`, 18 tests). The one genuinely
   new, reusable piece — serves any permutation null, not just waves.
2. **Structural primitives** — ✅ mostly already exist + tested (`take` = k-ring 2-D gather,
   `sum_axis`, `pearson`, `clip`, `arccos`, `any`/`all`, `max`; k-ring gather-mean composes as
   `sum_axis(ang*nbr_mask,1)/deg`, see `test_wave_array_primitives.py`). The FWE masked-max is
   NOT a separate primitive — it lives inside the surrogate emitter (family-wise extremum over
   the vmap output). No new primitive needed.
3. **`wave_operators` producer** — ✅ DONE + byte-checked. `koller2024_wave_detection.wave_operators`
   splits hemispheres, calls `precompute_wave_operators` per hemi, PADS to a common
   `(n_groups, nv_max, nf_max, K_max)` bundle with masks (`boundary` True on pad, `nbr_mask`/
   `real_mask` 0 on pad, `perms` permutes only the real block), plus `grp_verts`/`instrength`.
   Padding is byte-exact: the padded single-group body (HHD guard `pg/where(nrm>0,nrm,1)` +
   boundary→NaN + mask-weighted pearson) reproduces `detect_matmul`/native pearson to ~1e-16 on
   unequal hemispheres (scratchpad `check_wave_operators.py`). No new primitives — NaN + `nanmax`
   + `any()` handle vertex masking for free. The producer LARGELY reuses what already exists:
   the manuscript detector ships `precompute_wave_operators(v, f, k_ring)` (igl → grad_op, bc, hhd_op, nbr_faces, nbr_mask,
   div_tmpl, boundary + the matmul-split `bc_op`/`deg`/`grad_k`/`hhd_k`/`dt_k`) AND
   `detect_matmul(theta, ops)` — a per-timestep reference using ONLY those named operators and
   ONLY tvbo primitives (matmul, cos/sin, sqrt, `take` 2-D gather, clip, arccos, `sum_axis`),
   byte-validated vs the reference detector. So the producer = wrap `precompute_wave_operators`
   per hemisphere + pad to common (nv,nf,K) + add the permutation table + expose via the tvbo
   `producer:` path; the declarative body = `detect_matmul` transcribed expression-for-expression
   + the two surrogates. NEXT.
   RESOLVED (hemisphere → group axis) = **vmap-over-groups**, ✅ built + tested. The per-timestep
   body is written ONCE for a single group (= detect_matmul + the 1-D surrogates) and
   `jax.vmap`'d over the partition axis; group-indexed operators carry a leading group axis and
   are sliced per call, `group_vmap.gather` selects each group's vertices from the whole-brain
   source. Chosen over unrolled (duplicated body, hardcoded count) and over padded-batched
   operators (would force a per-group family reduce + a batched gather primitive): vmap-over-
   groups keeps the surrogate a clean per-vertex max-T INSIDE the vmap, so render_surrogate +
   family_reduce stay untouched. General to any partition (2 hemispheres or N parcels); the group
   count is data, not code — tvbo's own exploration-grid-is-a-vmap pattern one level down.
   `render_wave_reduction` gained an optional `group_vmap: {gather, over}`; byte-checked vs a
   numpy group-loop (`test_group_vmap_matches_numpy_group_loop`).
4. **Grouped output shape + masked-median finalize** — ✅ DONE (the open architectural
   question, now resolved + built). New `kind: 'wave'` reducer (`render_wave_reduction`): the
   metrics collapse BOTH time and the node axis into per-GROUP scalars, so the output is keyed
   `(n_groups, metric=3)`, NOT `template.shape[-1]`. A monitor-style `(n_ds, n_groups)` buffer
   per named per-step output (`corr`/`wave_present`/`sig_corr`, each a `(n_groups,)` vector from
   the declarative DV chain) is reduced at finalize to `proportion_waves = nw/T`,
   `proportion_directed = sum(sig&wave)/nw`, `rho = nanmedian(where(wave, corr, nan))` — the
   masked median is EXACT (no binning; `nanmedian` over a tiny buffer, not the histogram), and
   `nw==0 → NaN` matches the CPU `if nw else np.nan`. Byte-validated vs Koller's finalize
   (`tests/test_streaming_wave_reduction.py`, 7 tests). Hemispheres = 2 groups via a partition
   mask (padded-per-hemi operators feed the `(n_groups,)` body). Only the outer carry is
   bespoke; the per-step math stays the declarative chain + surrogate emitter.
5. **Recipe** — exp_41 gains `record: [cortical_wave_metrics]` (the streaming observer) and
   drops the workflow fan-out; grid runs like exp_40 (one GPU job). exp_41 mem → GPU node.
   GATED on the fig6_lowK probe (don't accelerate a null result).

## 7. Validation

- **Deterministic primitives** (gradient, angular sim, HHD, corr) — byte-validate the
  emitted jax against the CPU `cortical_wave_metrics` on a fixed phase field, primitive by
  primitive (the Tier-1 exercise already proved the batched *math* is byte-exact).
- **Surrogate** — feed the **same fixed `(n_perm, n)` table** to both CPU and GPU paths →
  the surrogate becomes deterministic → byte-comparable `exceed`/`proportion_waves`. (The
  CPU stochastic `rng` and GPU differ only in *which* perms; equal perms ⇒ equal output.)
- **End-to-end** — one exp_41 cell GPU vs CPU with a shared permutation table: identical
  metrics; then a grid slice GPU vs the fanned CPU results (statistical, since production
  perms differ).

## 8. Milestones / effort / risk

1. `wave_operators` producer bundle + byte-check each operator vs igl on host. (~1 d)
2. Deterministic per-timestep observer (gradient→angular-sim→HHD→corr) as declarative
   record; byte-validate vs CPU sans surrogate. (~2 d, incl. 2 structural primitives)
3. Surrogate emitter + fixed permutation-table constant; byte-validate with shared table. (~2 d)
4. Masked-median finalize; full one-cell GPU==CPU. (~1 d)
5. Recipe swap + grid run + Fig-6 re-score. (~1 d)

**Effort ≈ 1 week.** Risks: (a) masked-median in a fixed carry is the fiddliest finalize;
(b) ragged→padded k-ring must match igl's neighbour sets exactly (byte-check at step 1);
(c) GPU memory for wide vmap × 100-perm surrogate — bounded by chunking grid cells, per-cell
footprint is duration-independent.

## 9. Interim

Until Tier 2 lands, throughput is purely slurm concurrency: 135 CPU-hours ÷ N jobs
(≈1.5 h at 100 concurrent). No CPU-side speedup exists (Tier 1 closed).
