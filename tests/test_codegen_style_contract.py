"""The house-style contract for generated source, enforced per backend.

TVBO hands users generated code to read, review and attach to papers, so it is held
to the same bar as the rest of the package. This module states that bar as an
executable contract and checks every backend against it:

* the source parses as the language its :class:`~tvbo.export.registry.ExportFormat`
  declares — a formatter that cannot read our output means we emitted a broken
  program, and the user would have hit it later with a worse message;
* generated Python is byte-identical to its ``black`` form, so formatting is
  canonical rather than a matter of which template fragment wrote a line;
* generated Python is clean under :data:`RUFF_RULES` — the blocking subset CI
  already applies to TVBO's own source, plus ``F401``/``F841``.

``F401`` and ``F841`` are load-bearing rather than cosmetic. Statements with no
effect reach generated code when a template emits something the spec did not ask
for: a parameter unpack for a coupling that has no parameters, an import a partial
carries whether or not its feature is used, a state binding the pre-expression never
reads. Requiring the emitted module to be free of them forces the emission to be
*gated* on what the spec needs, and keeps it gated.

``F821`` (inside ``F82``) is the strongest of the three: an undefined name in
generated code is a ``NameError`` waiting for whoever runs it.

This suite freezes the CONTRACT, not the bytes. The complementary corpora that
freeze emitted code and simulation output live in ``test_codegen_golden_corpus.py``
and ``test_numerical_golden_corpus.py``. The three fail on disjoint classes of
regression: a spec can render byte-identically and still violate the contract only
if the contract changed, but a repair that cleans up emitted code changes the bytes
without touching the contract.

:data:`KNOWN_VIOLATIONS` records the pairs that do not meet the contract yet, each
mapped to the exact rule codes it still trips. The record is pinned from both sides —
a code that appears without being recorded fails, and a recorded code that stops
appearing fails too — so repairing an emitter reports as a failure asking for the
entry to be updated, the same reconciliation the golden corpora use for their
unrenderable pairs. A silently skipped backend cannot be told from one nobody noticed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tvbo import SimulationExperiment
from tvbo.export import resolve

RUFF_RULES = "E9,F63,F7,F82,F401,F841"
"""Rule selection for generated Python.

``E9,F63,F7,F82`` is exactly the blocking gate ``.github/workflows/ci.yml`` runs
over TVBO's own source, so generated code is held to the standard its generator is.
``F401``/``F841`` extend it to the unused imports and dead bindings that ungated
emission produces.
"""

FORMATS = ("tvb", "jax", "tvboptim", "julia", "neuroml", "lems")
"""Backends that render from a core install, so the contract is checkable anywhere."""

KNOWN_VIOLATIONS: dict[str, tuple[str, ...]] = {}
"""Pairs that parse and are black-clean but still trip lint, mapped to the codes.

Empty, and meant to stay so: every backend meets the contract in full. Emitters gate on
what the spec needs, and :mod:`tvbo.codegen.prune` removes the imports and pure
scaffolding bindings that a finished module turns out not to refer to.

The mapping is kept rather than deleted because it is what makes a violation temporary.
The recorded codes are exact, in both directions:
:func:`test_generated_python_is_lint_clean` fails on a code that is NOT recorded, and
:func:`test_known_violations_are_reconciled` fails on a recorded code that no longer
trips. Neither an unnoticed regression nor a stale entry can survive.
"""


KURAMOTO_FACTORED = {
    "id": 1,
    "label": "coupling with no parameters, factored source-only pre",
    "dynamics": {
        "name": "MiniKuramoto",
        "system_type": "continuous",
        "output": ["theta"],
        "parameters": {"K": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "theta": {
                "equation": {"rhs": "K * c"},
                "initial_value": 0.0,
                "coupling_variable": True,
            },
        },
    },
    "network": {
        "number_of_nodes": 4,
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["theta"],
                "pre_expression": {"rhs": "[sin(theta), cos(theta)]"},
                "post_expression": {"rhs": "cos(theta_i)*gx_0 - sin(theta_i)*gx_1"},
            }
        },
    },
    "integration": {
        "method": "heun",
        "step_size": 0.1,
        "duration": 1.0,
        "transient_time": 0.0,
        "unit": "s",
    },
}
"""A coupling carrying NO parameters, whose pre-expression is a factored list.

This is the shape that used to emit the bare ``= p.`` — an empty parameter unpack —
and reference ``theta``, ``theta_i``, ``gx_0`` and ``gx_1`` without binding any of
them. Four defects that ``black`` and ``F821`` catch between them, in a spec small
enough to read.
"""

SPECS = {"kuramoto_factored": KURAMOTO_FACTORED}

CASES = [(name, fmt) for name in sorted(SPECS) for fmt in FORMATS]
IDS = [f"{n}.{f}" for n, f in CASES]
PY_CASES = [c for c in CASES if resolve(c[1]).language == "python"]
PY_IDS = [f"{n}.{f}" for n, f in PY_CASES]


def _ruff() -> str:
    """Path to the ruff executable, or skip when the `lint` extra is not installed."""
    exe = shutil.which("ruff")
    if exe is None:
        pytest.skip("ruff not installed — `pip install -e '.[lint]'` to run this gate")
    return exe


def _render(name: str, fmt: str) -> str:
    """Render spec *name* for backend *fmt*, skipping when the backend is absent."""
    try:
        return SimulationExperiment(**SPECS[name]).render_code(format=fmt)
    except ImportError as exc:  # optional backend dependency missing
        pytest.skip(f"{fmt} backend unavailable: {exc}")


def _lint(code: str, ruff: str) -> list[str]:
    """Return the ruff rule codes *code* trips, deduplicated and sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "generated.py"
        path.write_text(code)
        proc = subprocess.run(
            [ruff, "check", "--isolated", "--select", RUFF_RULES,
             "--output-format=json", str(path)],
            capture_output=True, text=True,
        )
    return sorted({item["code"] for item in json.loads(proc.stdout or "[]")})


@pytest.mark.parametrize("name,fmt", CASES, ids=IDS)
def test_generated_source_parses(name, fmt):
    """Every backend emits source its own language can parse.

    ``render_code`` raises :class:`GeneratedSourceError` when it cannot, so simply
    rendering is the assertion. This is what caught the JAX empty parameter unpack.
    """
    assert _render(name, fmt).strip(), f"{fmt} rendered an empty module"


@pytest.mark.parametrize("name,fmt", PY_CASES, ids=PY_IDS)
def test_generated_python_is_black_clean(name, fmt):
    """Generated Python already IS its black form — formatting is canonical."""
    import black

    code = _render(name, fmt)
    assert code == black.format_str(code, mode=black.FileMode()), (
        f"{fmt} output is not black-clean; the central format gate in "
        f"tvbo.export.registry.render should have made it so"
    )


@pytest.mark.parametrize("name,fmt", PY_CASES, ids=PY_IDS)
def test_generated_python_is_lint_clean(name, fmt):
    """Generated Python trips no rule in :data:`RUFF_RULES`.

    Pairs listed in :data:`KNOWN_VIOLATIONS` are allowed to trip exactly the codes
    recorded for them, and no others — a NEW violation in a known-bad pair still
    fails.
    """
    known = KNOWN_VIOLATIONS.get(f"{name}.{fmt}", ())
    codes = _lint(_render(name, fmt), _ruff())
    unexpected = [c for c in codes if c not in known]
    assert not unexpected, (
        f"{fmt} generated code trips {', '.join(unexpected)}"
        + (f" beyond its recorded {', '.join(known)}" if known else "")
    )


@pytest.mark.parametrize("case", sorted(KNOWN_VIOLATIONS))
def test_known_violations_are_reconciled(case):
    """No recorded code may outlive the defect it records.

    Asserting only that SOME recorded code still trips would let a partially repaired
    entry go stale — which is how ``F821`` stayed listed for tvboptim after the
    undefined names were fixed. Every recorded code must still be observed.
    """
    name, fmt = case.rsplit(".", 1)
    codes = _lint(_render(name, fmt), _ruff())
    stale = [c for c in KNOWN_VIOLATIONS[case] if c not in codes]
    assert not stale, (
        f"{case} no longer trips {', '.join(stale)} — drop {'it' if len(stale) == 1 else 'them'} "
        f"from KNOWN_VIOLATIONS (remove the entry entirely if nothing is left)."
    )
