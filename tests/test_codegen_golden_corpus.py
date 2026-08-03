"""Golden corpus for generated code — the guard rail that freezes what codegen emits.

TVBO already freezes the *specification* (the YAML under ``tvbo/database/``) and checks
that models *run* (``test_run_models.py`` executes every model for 10 ms). Neither pins
what sits between them: the source that ``Dynamics.render_code`` emits. A refactor of the
symbolic layer can leave every YAML byte-identical and every smoke test green while
silently changing a generated right-hand side — term order, a printer dispatch, an inlined
function, a ``Piecewise`` branch.

This module closes that gap. For every curated model it renders the backend-independent
formats and compares the result, byte for byte, against a committed reference under
``tests/reference_data/codegen/``.

``render_code`` is deterministic: rendering all curated models under two different
``PYTHONHASHSEED`` values produces identical output, so a diff here always means a real
behavioural change, never flakiness.

The model directory is resolved through ``tvbo.data.registry`` rather than from this
file's position in the tree, so the corpus and the database the rest of TVBO reads can
never disagree about which models exist.

Formats needing a backend toolchain (``julia``, ``tvb``) are intentionally absent: this
suite must stay runnable with the core install so it can gate every pull request. See
``tests/golden.py`` for the regeneration and reconciliation semantics.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from tvbo.classes.dynamics import Dynamics
from tvbo.data.registry import database_dir

from .golden import GoldenCorpus

MODEL_ROOT = database_dir("Dynamics")

FORMATS = ("jax", "numpy")


def _model_paths() -> list[Path]:
    """Every curated model, under either YAML spelling.

    ``Spring.yml`` and ``ReducedWongWangFunc.yml`` use ``.yml``; globbing only ``.yaml``
    drops them from the corpus with no signal.
    """
    return sorted(p for ext in ("*.yaml", "*.yml") for p in MODEL_ROOT.rglob(ext))


def _case_id(path: Path, fmt: str) -> str:
    """Stable, filesystem-safe identifier: ``julia__discrete__tent_map.jax``.

    Built from the path *relative to the model root* so two models sharing a stem in
    different subdirectories cannot collide onto one reference.
    """
    rel = path.relative_to(MODEL_ROOT).with_suffix("")
    return f"{'__'.join(rel.parts)}.{fmt}"


def _diff(produced: str, expected: str) -> str | None:
    if produced == expected:
        return None
    return "\n".join(
        difflib.unified_diff(
            expected.splitlines(), produced.splitlines(),
            fromfile="reference", tofile="rendered", lineterm="", n=2,
        )
    )


CORPUS = GoldenCorpus(
    Path(__file__).parent / "reference_data" / "codegen",
    ".txt",
    write=lambda path, code: path.write_text(code, encoding="utf-8"),
    read=lambda path: path.read_text(encoding="utf-8"),
    compare=_diff,
)

CASES = [(path, fmt) for path in _model_paths() for fmt in FORMATS]


def _render(model: Dynamics, fmt: str) -> str:
    return model.render_code(format=fmt)


@pytest.mark.backend_core
@pytest.mark.parametrize(("path", "fmt"), CASES, ids=[_case_id(p, f) for p, f in CASES])
def test_generated_code_matches_golden(path: Path, fmt: str, regenerate: bool):
    """The emitted source for every curated model is byte-identical to its reference."""
    CORPUS.check(
        _case_id(path, fmt),
        _render(Dynamics.from_file(str(path)), fmt),
        regenerate=regenerate,
        what=f"generated {fmt} code",
    )


@pytest.mark.backend_core
def test_corpus_covers_every_model_and_format(regenerate: bool):
    """Every (model, format) pair has a reference, and none outlives its model.

    Reconciling per *case* rather than per model is what makes this meaningful: keyed on
    the model name alone, deleting one format's reference — or adding a third format —
    would leave the corpus reporting full coverage.
    """
    CORPUS.reconcile(
        (_case_id(p, f) for p, f in CASES),
        regenerate=regenerate,
        what="model/format pairs",
    )


@pytest.mark.backend_core
@pytest.mark.slow
@pytest.mark.parametrize("fmt", FORMATS)
def test_rendering_one_model_twice_is_stable(fmt: str):
    """Rendering the *same* ``Dynamics`` instance twice yields identical source.

    Guards the memoisation planned for the symbolic layer. Re-loading the model between
    renders would defeat that: a per-instance cache is rebuilt from scratch and can never
    disagree with itself, so the instance is deliberately reused here.
    """
    for path in _model_paths():
        model = Dynamics.from_file(str(path))
        assert _render(model, fmt) == _render(model, fmt), f"{path.name} [{fmt}]"
