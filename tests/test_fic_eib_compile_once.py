"""Regression guard for the compile-once tuning core (Bug 2).

A multi-stage `fic_eib` schedule must route its tuning ``lax.scan`` through ONE
module-level ``_<algo>_tuning_core`` that XLA-compiles once across stages, with the
per-stage-varying scalars (eta, resync period, ring window) threaded TRACED rather
than baked — otherwise the scan recompiles once per stage (the 47-min compile).
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
        AlgorithmStage(n_iterations=n_iter, arguments=[
            Parameter(name="eta", value=0.10), Parameter(name="window_size", value=4)]),
        AlgorithmStage(n_iterations=n_iter, arguments=[
            Parameter(name="eta", value=0.05), Parameter(name="window_size", value=8)]),
    ]
    fe.n_iterations = fic.n_iterations = n_iter
    return exp


def test_tuning_scan_hoisted_to_module_level_core():
    """run_<algo> calls a module-level jitted core; it no longer holds a bare scan."""
    code = _multistage_experiment().render_code("tvboptim")
    assert "def _fic_eib_tuning_core_impl(" in code
    assert "_fic_eib_tuning_core = jax.jit(" in code
    assert "_ls_final, _ys_all = _fic_eib_tuning_core(" in code

    run = code[code.index("def run_fic_eib("):code.index("def _fic_eib_tuning_core_impl(")]
    assert "jax.lax.scan(_tuning_step" not in run, "run_<algo> must delegate the scan to the core"


def test_core_threads_per_stage_scalars_traced():
    """Per-stage-varying scalars enter the core TRACED; model_fn is a STATIC arg."""
    code = _multistage_experiment().render_code("tvboptim")
    sig = code[code.index("def _fic_eib_tuning_core_impl("):]
    sig = sig[:sig.index("):")]
    for traced in ("eta", "_resync_period", "ws0", "use_ring"):
        assert traced in sig, f"core must take {traced} as an argument"
    core = code[code.index("_fic_eib_tuning_core = jax.jit("):]
    core = core[:core.index(")\n") + 1]
    assert '"model_fn"' in core, "model_fn must be a STATIC arg for the jit cache to key stably"


def test_eta_call_site_passes_variable_not_literal():
    """Bug 1 guard: the update call passes the `eta` variable, never a baked float."""
    code = _multistage_experiment().render_code("tvboptim")
    call = code[code.index("new_wLRE = wLRE_update("):]
    call = call[:call.index(")")]
    assert "eta," in call and "0.1" not in call and "0.05" not in call


@pytest.mark.slow
def test_multistage_tuning_core_compiles_once():
    """The fic_eib tuning core compiles ONCE across the 2 stages (Bug 2 fix)."""
    pytest.importorskip("tvboptim")
    import jax

    _orig = jax.jit
    compiles = {}

    def _counting_jit(fn=None, /, *a, **k):
        def _wrap(f):
            name = getattr(f, "__name__", "anon")

            def _counted(*aa, **kk):  # runs only at trace time == once per compilation
                compiles[name] = compiles.get(name, 0) + 1
                return f(*aa, **kk)

            _counted.__name__ = name
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
