"""Emitted branch code must *mean* what the `Piecewise` it came from means.

The golden corpus freezes the text every emitter produces. Text identity is not meaning:
a `Piecewise` lowered to the wrong arithmetic is frozen just as faithfully as one lowered to the right arithmetic, and stays frozen. This module closes that gap for the one lowering that cannot be read at a glance — LEMS, which has no ternary and no `Piecewise` and must express branch selection as a sum of Heaviside-gated terms.

It caught four shipped models. `tent_map`, `pomeau_manneville_map`, `Hopfield` and `EpileptorCodim3` each emitted their else-branch *un-gated*, so it was added to whichever branch was taken rather than replacing it — the tent map differed from its own definition over 63% of its domain. Two independent defects produced that:

* `Piecewise` is first-match-wins, so every term after the first is reachable only when no
  earlier condition held. Summing the terms un-gated is correct only when the conditions
  happen to be exclusive *and* the default is zero.
* The branch value was interpolated without parentheses, so a value that prints as a sum
  escaped its own gate: `H(x .gt. 0.5) * -4*x + 3` gates `-4*x` and adds `3` always.

The backend-independent printers are checked alongside it. They lower to `where`/`ifelse` and have no equivalent trap, which is exactly why they belong here: this is the test that says so rather than assuming it.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp

from tvbo.codegen.code import get_printer

x, mu, thr, a, b, c, d = sp.symbols("x mu thr a b c d")
SYMBOLS = (x, mu, thr, a, b, c, d)

CASES = {
    "two-arm, non-zero default": sp.Piecewise((mu * x, x < 0.5), (mu * (1 - x), True)),
    "one arm, zero default": sp.Piecewise((-sp.Rational(1, 10) * x**7, x < 0), (0, True)),
    "four arms": sp.Piecewise((a, x < -1), (b, x < 0), (c, x < 1), (d, True)),
    "overlapping arms, first match wins": sp.Piecewise((a, x < 1), (b, x < 2), (c, True)),
    "additive branch values": sp.Piecewise((a + b, x > thr), (c + d, True)),
    "conjunctive condition": sp.Piecewise((a, sp.And(x > 0, x < 1)), (b, True)),
    "disjunctive condition": sp.Piecewise((a, sp.Or(x < -1, x > 1)), (b, True)),
    "negated condition": sp.Piecewise((a, sp.Not(x > 0)), (b, True)),
}

_LEMS_RELATIONS = ((".geq.", ">="), (".leq.", "<="), (".neq.", "!="), (".gt.", ">"), (".lt.", "<"), (".eq.", "=="))


def _lems_as_python(text: str) -> str:
    """Read emitted LEMS math back as an evaluable Python expression.

    Each `H(...)` argument is wrapped before `.and.`/`.or.` are translated: LEMS `x .gt. 0 .and. x .lt. 1` is one boolean expression, but Python's `&` binds tighter than `>`, so a direct substitution would mean `x > (0 & x) < 1`.
    """
    for lems, python in _LEMS_RELATIONS:
        text = text.replace(lems, python)
    text = text.replace("^", "**")

    out, cursor = [], 0
    for match in re.finditer(r"\bH\(", text):
        if match.start() < cursor:
            continue
        depth, end = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            end += 1
        out.append(text[cursor : match.start()] + f"HEAVISIDE(({text[match.end() : end - 1]}))")
        cursor = end
    text = "".join(out) + text[cursor:]
    return text.replace(".and.", ") & (").replace(".or.", ") | (").replace(".not.", "~")


def _sample(seed: int = 0, n: int = 20_000) -> dict:
    """Random values for every symbol, spread wide enough to reach every branch."""
    rng = np.random.default_rng(seed)
    return {s.name: rng.normal(0.0, 1.5, n) for s in SYMBOLS}


def _reference(expr, env) -> np.ndarray:
    """What SymPy itself says the expression evaluates to."""
    values = sp.lambdify(SYMBOLS, expr, "numpy")(*[env[s.name] for s in SYMBOLS])
    return np.broadcast_to(np.asarray(values, dtype=float), np.shape(env["x"]))


@pytest.mark.backend_core
@pytest.mark.parametrize("label", list(CASES), ids=list(CASES))
def test_lems_branches_evaluate_as_the_piecewise_does(label: str):
    """Emitted LEMS agrees with SymPy at every sampled point, exactly."""
    expr = CASES[label]
    emitted = get_printer("lems").doprint(expr)

    env = _sample()
    env["HEAVISIDE"] = lambda condition: np.where(condition, 1.0, 0.0)
    produced = np.broadcast_to(
        np.asarray(eval(_lems_as_python(emitted), {}, env), dtype=float),  # noqa: S307
        np.shape(env["x"]),
    )
    expected = _reference(expr, env)

    wrong = ~np.isclose(produced, expected, rtol=1e-12, atol=1e-12)
    if wrong.any():
        i = int(np.argmax(np.abs(produced - expected)))
        point = {s.name: round(float(env[s.name][i]), 4) for s in SYMBOLS}
        pytest.fail(
            f"{wrong.sum()}/{wrong.size} sampled points disagree\n"
            f"  emitted : {emitted}\n"
            f"  at      : {point}\n"
            f"  sympy   : {expected[i]:.10g}\n"
            f"  lems    : {produced[i]:.10g}"
        )


@pytest.mark.backend_core
@pytest.mark.parametrize("fmt", ["numpy", "jax", "tvb"])
@pytest.mark.parametrize("label", list(CASES), ids=list(CASES))
def test_array_backends_branch_as_the_piecewise_does(fmt: str, label: str):
    """`np.where` / `jnp.where` / TVB `where` lowering agrees with SymPy, on arrays.

    Arrays rather than scalars on purpose: `where` evaluates *both* arms and selects elementwise, so a lowering that is right for a scalar can still be wrong for a vector.

    `tvb` is evaluated by `numexpr` — the evaluator TVB actually hands an `Equation.equation` string to — rather than by `eval`. That is the whole point of the target: `numexpr` rejects a Python `a if c else b` and Python's `and`/`or` outright, so a stimulus lowered that way is not merely differently spelled, it does not run.

    The tolerance follows the backend's own precision — JAX computes in float32 unless x64 is enabled, which this must not turn on process-wide. What is under test is which arm each element takes, and picking the wrong arm moves a value far more than a rounding.
    """
    expr = CASES[label]
    emitted = get_printer(fmt).doprint(expr)

    env = _sample(seed=1)
    if fmt == "tvb":
        numexpr = pytest.importorskip("numexpr")
        raw = numexpr.evaluate(emitted, {s.name: env[s.name] for s in SYMBOLS})
    else:
        if fmt == "jax":
            import jax.numpy as jnp

            env["jnp"] = jnp
        else:
            env["np"] = np
        raw = eval(emitted, {}, env)  # noqa: S307

    tolerance = 1e-6 if np.asarray(raw).dtype == np.float32 else 1e-12
    produced = np.asarray(raw, dtype=float)
    expected = _reference(expr, env)
    np.testing.assert_allclose(produced, expected, rtol=tolerance, atol=tolerance, err_msg=emitted)


@pytest.mark.backend_core
def test_zero_default_is_not_emitted():
    """A zero default adds nothing to a sum, so it is left out rather than gated.

    Without this, every single-branch expression grows a `(1 - H(…)) * 0` tail — six curated models carry one.
    """
    emitted = get_printer("lems").doprint(CASES["one arm, zero default"])
    assert "* 0" not in emitted and not emitted.rstrip().endswith("+ 0"), emitted


@pytest.mark.backend_core
def test_branch_value_cannot_escape_its_gate():
    """A branch whose value prints as a sum stays inside its own `H(...) *` term."""
    emitted = get_printer("lems").doprint(sp.Piecewise((-4 * x + 3, x > sp.Rational(1, 2)), (0, True)))
    assert emitted == "H(x .gt. 1/2) * (3 - 4*x)", emitted


@pytest.mark.backend_core
@pytest.mark.parametrize(
    "module, names",
    [
        (
            "tvbo.classes.equation",
            [
                "piecewise2numpy",
                "convert_ifelse_to_np_where",
                "convert_numpy_where_to_sympy",
                "extract_parts_from_numpy_where",
                "get_latex_equation",
                "render_latex_equations",
            ],
        ),
        ("tvbo.codegen.templater", ["boolean2bitwise", "equation2class", "model2class", "get_model_info"]),
    ],
)
def test_no_second_lowering_path_survives(module: str, names: list[str]):
    """Nothing may re-enter the string-rewriting path a `Piecewise` used to take.

    A `Piecewise` is now built in exactly one place (`parse_eq`), printed in exactly one place (`print_Piecewise`, through each printer's `_where3`), and rendered for humans in exactly one place (`sympy.latex`). Each of these names was the second of one of those, reachable by importing it — which is how the two spellings drifted apart in the first place. Asserting their absence is what keeps the count at one.
    """
    import importlib

    imported = importlib.import_module(module)
    present = [name for name in names if hasattr(imported, name)]
    assert not present, f"{module} still exposes {present}"
