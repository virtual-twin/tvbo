"""Every host-side solve a generated module executes goes through ``jax.jit``.

``prepare()`` hands back an UN-jitted callable. Invoked directly it dispatches op by op for every step of the solve, which costs about 4.5x what the same call costs compiled: measured on a 379-node network, one prepared solve ran at 1023 us per integration step raw against 229 us under ``jax.jit``. Nothing failed and nothing warned — the run simply took four and a half times longer, which reads as "this model is expensive" rather than as a defect.

Two templates emit these calls and they compile at different levels, so the invariant is checked at the level each one uses. ``run_<algo>`` rebinds its prepared callables to jitted ones on receipt, so its bare calls are already compiled; the experiment template cannot, since the same ``model_fn`` also reaches a loss function the optimizer traces and an exploration that builds its own vmap — there, every host-side evaluation is wrapped at the call.

These assert on the emitted source and on the templates because a run cannot tell compiled from interpreted: both compute the same numbers.
"""

import re

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment, database_path

TEMPLATES = database_path.parent / "templates" / "tvboptim"

# Names that hold a callable straight from `prepare()`, in both the local and the `sim_result.<name>` spelling. Invoking one of these directly runs the solve interpreted.
PREPARED = ("model_fn", "post_model_fn", "model_fn_init", "_stream_fn", "stream_fn")
_CALL = re.compile(r"(?:^|[^_a-zA-Z])(" + "|".join(PREPARED) + r")\(")


def _code_lines(path):
    """Template lines with comment-only text dropped, so prose naming a call is not read as one."""
    for n, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if code.strip():
            yield n, code


@pytest.fixture(scope="module")
def code():
    """A curated experiment that carries an algorithm, so both templates are emitted."""
    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "EI_Tuning_FIC_EIB_Optimization.yaml"))
    exp.configure()
    return exp.render_code("tvboptim")


def test_the_compiled_runner_is_emitted(code):
    assert "def _run_compiled(" in code
    assert "jax.jit(fn)(state)" in code


def test_no_prepared_callable_is_invoked_raw_in_the_experiment_template():
    """The experiment template has no jit-on-receipt to fall back on, so every call must be wrapped.

    This is the guard against an eighth call site being added raw: it sweeps every line of the template for a call on any name that holds a prepared callable, attribute spelling included, rather than naming the sites that were fixed. The exploration and loss paths bind their own aliases (`_expl_model_fn`, `_opt_model_fn`, ...) and call them from inside a `@jax.jit`, which is why those names are deliberately not on the list.
    """
    raw = [f"{n}: {c.strip()}" for n, c in _code_lines(TEMPLATES / "tvbo-tvboptim-experiment.py.mako") if _CALL.search(c)]
    assert raw == [], "prepared solve invoked without _run_compiled:\n" + "\n".join(raw)


@pytest.mark.parametrize("name", ["model_fn", "post_model_fn"])
def test_the_algorithm_template_compiles_its_prepared_callables_on_receipt(name):
    """``run_<algo>`` rebinds each to a jitted one, which is why its bare calls need no wrapper."""
    src = (TEMPLATES / "tvbo-tvboptim-algorithm.py.mako").read_text()
    assert re.search(rf"^    {name} = (None if {name} is None else )?jax\.jit\({name}\)", src, re.M)


def test_the_tuning_core_still_receives_the_raw_callable(code):
    """It keys its own jit cache on that identity; handing it the jitted one would break stage reuse."""
    assert "_raw_model_fn = model_fn" in code
    assert "model_fn=_raw_model_fn" in code


def test_the_loss_function_still_receives_the_raw_callable():
    """The optimizer traces the loss itself — wrapping the solve inside it would nest jit in grad.

    Asserted on the template because the loss branch is emitted only for a fitted experiment.
    """
    src = (TEMPLATES / "tvbo-tvboptim-experiment.py.mako").read_text()
    assert "compute_all_observations(_opt_model_fn(state), state, _opt_transient)" in src
