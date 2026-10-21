"""Every curated model's AUTO-07p source compiles, and a model with a mode axis is refused before anything is rendered.

The printer checks need no compiler: Fortran's ``merge``, ``max`` and ``min`` require all their values of one type, and its intrinsics take no INTEGER, so an integer literal beside a real one is a compile error. The corpus check compiles each rendered source with ``gfortran -fsyntax-only``, the front end AUTO-07p builds the model with.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import sympy as sp
import yaml

from tvbo.codegen.code import FortranPrinter

MODELS = Path(__file__).resolve().parents[1] / "tvbo" / "database" / "models"


def _modes(path):
    return int((yaml.safe_load(path.read_text()) or {}).get("number_of_modes") or 1)


SINGLE_MODE = sorted(path.stem for path in MODELS.glob("*.yaml") if _modes(path) == 1)
MULTI_MODE = sorted(path.stem for path in MODELS.glob("*.yaml") if _modes(path) > 1)

x = sp.Symbol("x")


@pytest.mark.parametrize(
    ("expr", "printed"),
    [
        (sp.Piecewise((0, x < 0), (2.5 * x, True)), "merge(0.0d0, 2.5d0*x, x < 0)"),
        (sp.atan2(0, x), "atan2(0.0d0, x)"),
        (sp.Max(0, x), "max(0.0d0, x)"),
        (sp.Min(1, x), "min(1.0d0, x)"),
        (x**2, "x**2"),
    ],
)
def test_an_integer_literal_prints_as_a_double_where_fortran_needs_one(expr, printed):
    """An integer exponent stays one: ``x**2.0d0`` would be a real power, undefined for a negative base."""
    assert FortranPrinter().doprint(expr) == printed


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran is not installed")
@pytest.mark.parametrize("name", SINGLE_MODE)
def test_the_auto07p_source_compiles(tmp_path, name):
    from tvbo import Dynamics
    from tvbo.adapters.numcont import NumContAdapter

    model = Dynamics.from_db(name)
    source = tmp_path / "model.f90"
    source.write_text(NumContAdapter(model).render_code(model=model))
    compiled = subprocess.run(
        ["gfortran", "-fsyntax-only", "-ffree-line-length-none", source.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize("name", MULTI_MODE)
def test_a_model_with_modes_is_refused(name):
    from tvbo import Dynamics
    from tvbo.adapters.numcont import NumContAdapter

    model = Dynamics.from_db(name)
    with pytest.raises(ValueError, match="modes"):
        NumContAdapter(model).render_code(model=model)
