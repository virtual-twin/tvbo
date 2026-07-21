# Handoff: emit a streaming HRF-BOLD reducer as backend-independent code

**Status:** design complete, not yet built. Sibling to
[`reducers_unification.md`](reducers_unification.md) (the windowed-FC reducer that
this mirrors). Written 2026-07-21 as a focused-session handoff.

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
