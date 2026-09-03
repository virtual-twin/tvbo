"""Regression guards for the fic_eib tuning codegen.

- compile-once (Bug 2): a multi-stage `fic_eib` schedule routes its tuning ``lax.scan``
  through ONE module-level ``_<algo>_tuning_core`` that XLA-compiles once across stages,
  with per-stage-varying scalars (eta, resync period, ring window) threaded TRACED rather
  than baked — otherwise the scan recompiles once per stage (the 47-min compile).
- NaN guard: the host path fails loud on a non-finite estimate (a diverged fit must never
  be written to disk as a success).
- update_every: a batched-update cadence knob — absent leaves the rendered code
  byte-identical (no gate); declared gates every update-rule write TRACED through the core.
"""

import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.datamodel import AlgorithmStage, Parameter

_EXP = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"


def _multistage_experiment(n_iter=3):
    """FIC_EIB with a tiny 2-stage schedule whose window VARIES (-> maxwin M-ring)."""
    exp = SimulationExperiment.from_file(_EXP)
    fe, fic = exp.algorithms["fic_eib"], exp.algorithms["fic"]
    fe.stages = [
        AlgorithmStage(
            n_iterations=n_iter, arguments=[Parameter(name="eta", value=0.10), Parameter(name="window_size", value=4)]
        ),
        AlgorithmStage(
            n_iterations=n_iter, arguments=[Parameter(name="eta", value=0.05), Parameter(name="window_size", value=8)]
        ),
    ]
    fe.n_iterations = fic.n_iterations = n_iter
    return exp


def test_tuning_scan_hoisted_to_module_level_core():
    """run_<algo> calls a module-level jitted core; it no longer holds a bare scan."""
    code = _multistage_experiment().render_code("tvboptim")
    assert "def _fic_eib_tuning_core_impl(" in code
    assert "_fic_eib_tuning_core = jax.jit(" in code
    assert "_ls_final, _ys_all = _fic_eib_tuning_core(" in code

    run = code[code.index("def run_fic_eib(") : code.index("def _fic_eib_tuning_core_impl(")]
    assert "jax.lax.scan(_tuning_step" not in run, "run_<algo> must delegate the scan to the core"


def test_core_threads_per_stage_scalars_traced():
    """Per-stage-varying scalars enter the core TRACED; model_fn is a STATIC arg."""
    code = _multistage_experiment().render_code("tvboptim")
    sig = code[code.index("def _fic_eib_tuning_core_impl(") :]
    sig = sig[: sig.index("):")]
    for traced in ("eta", "_resync_period", "ws0", "use_ring"):
        assert traced in sig, f"core must take {traced} as an argument"
    core = code[code.index("_fic_eib_tuning_core = jax.jit(") :]
    core = core[: core.index(")\n") + 1]
    assert '"model_fn"' in core, "model_fn must be a STATIC arg for the jit cache to key stably"


def test_eta_call_site_passes_variable_not_literal():
    """Bug 1 guard: the update call passes the `eta` variable, never a baked float."""
    code = _multistage_experiment().render_code("tvboptim")
    call = code[code.index("new_wLRE = wLRE_update(") :]
    call = call[: call.index(")")]
    assert "eta," in call and "0.1" not in call and "0.05" not in call


def test_nan_guard_emitted_default_on():
    """A tuning algo emits a fail-loud non-finite-estimate guard on the host path."""
    code = _multistage_experiment().render_code("tvboptim")
    assert "_nonfinite_estimates" in code and "jnp.isfinite" in code
    assert "tuning diverged: non-finite estimate" in code


def test_update_every_gate_absent_when_undeclared():
    """No update_every hyperparameter -> NO gate (rendered code stays the online path)."""
    code = _multistage_experiment().render_code("tvboptim")
    assert "_apply_update" not in code
    assert "update_every" not in code


def test_update_every_gate_emitted_and_traced_when_declared():
    """Declaring update_every gates every update-rule write, threaded TRACED via the core."""
    import re

    exp = _multistage_experiment()
    exp.algorithms["fic_eib"].hyperparameters.append(Parameter(name="update_every", value=20))
    code = exp.render_code("tvboptim")

    assert "_apply_update" in code
    assert "% jnp.maximum(jnp.asarray(update_every" in code, "gate must use the traced ue"
    # every update-rule write (J_i, wLRE, wFFI) is gated by the cadence predicate
    assert len(re.findall(r"jnp\.where\(\s*_apply_update", code)) == 3
    # threaded TRACED into the compile-once core (not baked), like eta / resync period
    assert "_canon_tree(update_every)" in code
    core_sig = code[code.index("def _fic_eib_tuning_core_impl(") :]
    core_sig = core_sig[: core_sig.index("):")]
    assert "update_every" in core_sig


def test_nan_guard_reads_final_state_not_rec_buffer():
    """The guard must check the RETURNED estimate (state.*), not the subsampled __rec buffer.

    __rec is recorded only at ``(i+1)%save_every==0 or i==0``; when save_every does not divide n_iterations the final iterate is unrecorded, so a divergence in the last steps would leave __rec finite while the written estimate (state) is NaN.
    """
    code = _multistage_experiment().render_code("tvboptim")
    blocks, start = [], 0
    while (i := code.find("_nonfinite_estimates = [", start)) != -1:
        j = code.index("]", i)
        blocks.append(code[i : j + 1])
        start = j + 1
    assert blocks, "no NaN guard emitted"
    for b in blocks:
        assert "__rec" not in b, "guard must read the final state estimate, not __rec"
        assert "state.dynamics" in b or "state.coupling" in b
    # fic_eib's coupling estimates (wLRE / wFFI) must be guarded from state.coupling
    assert any("state.coupling" in b for b in blocks)


@pytest.mark.slow
def test_multistage_tuning_core_compiles_once():
    """The fic_eib tuning core compiles ONCE across the 2 stages (Bug 2 fix)."""
    pytest.importorskip("tvboptim")
    import functools

    import jax

    _orig = jax.jit
    compiles = {}

    def _counting_jit(fn=None, /, *a, **k):
        def _wrap(f):
            name = getattr(f, "__name__", "anon")

            @functools.wraps(f)  # keep signature so jax.jit static_argnames still binds
            def _counted(*aa, **kk):  # runs only at trace time == once per compilation
                compiles[name] = compiles.get(name, 0) + 1
                return f(*aa, **kk)

            return _orig(_counted, *a, **k)

        return _wrap if fn is None else _wrap(fn)

    jax.jit = _counting_jit
    try:
        _multistage_experiment().run("tvboptim", mode="all", quiet=True)
    finally:
        jax.jit = _orig

    core = [k for k in compiles if "fic_eib_tuning_core" in k]
    assert core, f"tuning core never jitted; saw {sorted(compiles)}"
    assert compiles[core[0]] == 1, f"tuning core recompiled per stage ({compiles[core[0]]}x)"


def test_stage_reset_absent_by_default():
    """A schedule that does not declare reset_state stays one continuous trajectory."""
    code = _multistage_experiment().render_code("tvboptim")
    assert "_carry_tuned_" not in code
    assert '_sd.get("reset_state")' not in code
    assert "_stage_monitors0" not in code


def test_stage_reset_carries_only_tuned_targets():
    """reset_state restarts each stage from the entry state, carrying every update-rule target.

    The dynamical state, window buffer and monitors must come from the algorithm's entry state, the noise key must restart from the run seed (so each stage samples the same realisation), and each tuned parameter must be grafted from the previous endpoint.
    """
    exp = _multistage_experiment()
    for stage in exp.algorithms["fic_eib"].stages:
        stage.reset_state = True
    code = exp.render_code("tvboptim")

    assert "def _carry_tuned_fic_eib(" in code
    # every update-rule target crosses the boundary: the two coupling matrices + J_i
    assert "_tuned.coupling.EIBLinearCoupling.wLRE" in code
    assert "_tuned.coupling.EIBLinearCoupling.wFFI" in code
    assert "_tuned.dynamics.J_i" in code
    # ...and everything else is restored from the entry state / initial buffers
    assert "_carry_tuned_fic_eib(algo_state, _stage_state)" in code
    assert "_stage_monitors = _stage_monitors0" in code
    assert "_stage_bold_buffer = _stage_bold_buffer0" in code
    # noise restarts from the run seed rather than being folded per stage
    assert 'if _sd.get("reset_state")' in code
    assert "jax.random.fold_in(algo_key, _si)" in code
