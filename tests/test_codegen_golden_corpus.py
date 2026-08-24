"""Golden corpus for generated code — the guard rail that freezes what codegen emits.

TVBO already freezes the *specification* (the YAML under ``tvbo/database/``) and checks that models *run* (``test_run_models.py`` executes every model for 10 ms). Neither pins what sits between them: the source that ``Dynamics.render_code`` emits. A refactor of the symbolic layer can leave every YAML byte-identical and every smoke test green while silently changing a generated right-hand side — term order, a printer dispatch, an inlined function, a ``Piecewise`` branch.

This module closes that gap. For every curated model it renders the backend-independent formats and compares the result, byte for byte, against a committed reference under ``tests/reference_data/codegen/``.

``render_code`` is deterministic: rendering all curated models under two different ``PYTHONHASHSEED`` values produces identical output, so a diff here always means a real behavioural change, never flakiness.

The model directory is resolved through ``tvbo.data.registry`` rather than from this file's position in the tree, so the corpus and the database the rest of TVBO reads can never disagree about which models exist.

Every format that renders from a core install is frozen: emitting Julia, TVB, tvboptim or NeuroML source needs no Julia, no TVB and no jNeuroML, only the templates in this repository, and those emitters are among the ones under active change. Only ``mtk`` is absent, because it raises for models it cannot express.

A pair with no output to freeze is named rather than quietly dropped, because one missing from the corpus with no explanation cannot be told from one nobody noticed: ``UNRENDERABLE`` where the emitter is broken, ``UNSUPPORTED`` where a backend declines by design. Both are asserted to still raise, so a repair reports as a test failure asking for its reference.

See ``tests/golden.py`` for the regeneration and reconciliation semantics.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from tvbo.classes.dynamics import Dynamics
from tvbo.data.registry import database_dir

from .golden import GoldenCorpus

MODEL_ROOT = database_dir("Dynamics")

FORMATS = ("jax", "numpy", "julia", "tvb", "tvboptim", "neuroml")

UNRENDERABLE = {
    "neuroml__HH_KineticScheme.tvboptim": IndexError,
    "neuroml__hhcell_1.neuroml": AttributeError,
}
"""Pairs whose emitter raises, mapped to the exception it raises.

Excluded from the corpus because there is no output to freeze, and named here because a
silent exclusion cannot be told from an oversight. `test_excluded_pairs_still_raise`
holds them to it: repair the emitter and that test fails, which is the prompt to move the
pair into the corpus.
"""

UNSUPPORTED = {
    f"{name}.tvb": ValueError
    for name in (
        "julia__duffing",
        "julia__forced_pendulum",
        "julia__riddled_basins",
        "julia__ueda",
        "julia__vanderpol",
        "neuroml__HH_Tissue_Q10",
        "neuroml__HodgkinHuxley_Q10",
        "neuroml__IaFCell",
        "neuroml__Izhikevich2007Cell",
        "neuroml__IzhikevichBurst",
        "neuroml__IzhikevichCell",
    )
}
"""Pairs a backend declines by design, mapped to the exception it raises.

Every one is a non-autonomous system meeting TVB, whose ``Model.dfun`` takes no time
argument: the five Julia models force an oscillator with `cos(omega*t)`, and the six
NeuroML cells carry a `pulseGen` written inline as a `Piecewise` in `t`. There is nowhere
to put `t`, so the emitter says so instead of emitting an unbound name.

Kept apart from `UNRENDERABLE` because the two ask for opposite things. A pair leaves that
list when someone fixes the emitter; a pair leaves this one only when the *model* changes —
by moving its time dependence into a stimulus, which TVB applies outside `dfun`.
"""

EXCLUDED = {**UNRENDERABLE, **UNSUPPORTED}
"""Every pair with no reference to freeze, whatever the reason."""


def _model_paths() -> list[Path]:
    """Every curated model.

    One spelling, ``.yaml``, which is also the only one `tvbo.data.registry` resolves — so a model this corpus renders is a model `Dynamics.from_db` can reach.
    """
    return sorted(MODEL_ROOT.rglob("*.yaml"))


def _case_id(path: Path, fmt: str) -> str:
    """Stable, filesystem-safe identifier: ``julia__discrete__tent_map.jax``.

    Built from the path *relative to the model root* so two models sharing a stem in different subdirectories cannot collide onto one reference.
    """
    rel = path.relative_to(MODEL_ROOT).with_suffix("")
    return f"{'__'.join(rel.parts)}.{fmt}"


def _diff(produced: str, expected: str) -> str | None:
    """A readable account of the difference, or `None` when there is none.

    Falls back to a whitespace-visible comparison when the line diff comes out empty, which happens for a change `splitlines` cannot see — a trailing newline, trailing spaces. Left as-is those failures report an empty diff, telling the reader nothing.
    """
    if produced == expected:
        return None
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            produced.splitlines(),
            fromfile="reference",
            tofile="rendered",
            lineterm="",
            n=2,
        )
    )
    if lines:
        return "\n".join(lines)
    return (
        "  differs only in whitespace invisible to a line diff\n"
        f"  reference ends {expected[-40:]!r}\n"
        f"  rendered  ends {produced[-40:]!r}"
    )


CORPUS = GoldenCorpus(
    Path(__file__).parent / "reference_data" / "codegen",
    ".txt",
    write=lambda path, code: path.write_text(code, encoding="utf-8"),
    read=lambda path: path.read_text(encoding="utf-8"),
    compare=_diff,
)

CASES = [(path, fmt) for path in _model_paths() for fmt in FORMATS if _case_id(path, fmt) not in EXCLUDED]


def _render(model: Dynamics, fmt: str) -> str:
    return model.render_code(format=fmt)


@pytest.mark.backend_core
@pytest.mark.parametrize(("path", "fmt"), CASES, ids=[_case_id(p, f) for p, f in CASES])
def test_generated_code_matches_golden(path: Path, fmt: str, regenerate: bool):
    """The emitted source for every curated model is byte-identical to its reference.

    Renders twice from one `Dynamics` and asserts both agree before comparing. That is the determinism the frozen references rest on — without it a diff here could not be told from run-to-run jitter — and it is checked in the same test so it cannot end up behind a marker no CI shard enables. The instance is deliberately reused: a per-instance cache rebuilt from scratch can never disagree with itself, so re-loading would prove nothing.
    """
    model = Dynamics.from_file(str(path))
    produced = _render(model, fmt)
    assert produced == _render(model, fmt), f"{path.name} [{fmt}] does not render deterministically"

    CORPUS.check(
        _case_id(path, fmt),
        produced,
        regenerate=regenerate,
        what=f"generated {fmt} code",
    )


def _equations_of(model) -> dict:
    """Every stored right-hand side, as text — what a mutating render would disturb."""
    return {
        f"{coll}.{name}": str(getattr(getattr(element, "equation", None), "rhs", ""))
        for coll in ("state_variables", "derived_variables", "derived_parameters")
        for name, element in (getattr(model, coll, None) or {}).items()
    }


@pytest.mark.backend_core
@pytest.mark.parametrize("fmt", FORMATS)
def test_rendering_does_not_modify_the_model(fmt: str):
    """Rendering is a query: it reads the model and leaves it as it found it.

    Guards the property every other test here depends on. A renderer that normalises what it reads emits a file that depends on how many times it has been called — and the symbolic layer's cache, which keys on the model's content, is unsound the moment reading can change it.
    """
    for path in _model_paths():
        if _case_id(path, fmt) in EXCLUDED:
            continue
        model = Dynamics.from_file(str(path))
        before = _equations_of(model)
        _render(model, fmt)
        assert _equations_of(model) == before, f"{path.name} [{fmt}] was modified by rendering"


@pytest.mark.backend_core
def test_every_model_reports_without_modifying_itself():
    """Every curated model renders a report, and reporting leaves the model alone.

    Two properties in one pass because they were found together. Reporting mutated the model exactly as rendering once did; and ten models could not produce a report at all, because APA formatting took a first initial from every author and BibTeX's `and others` idiom parses as an author with no first name.

    Report output itself is not frozen, so this asserts the two properties rather than the text. Failures are not swallowed: a model that cannot report is the bug this covers.
    """
    for path in _model_paths():
        model = Dynamics.from_file(str(path))
        before = _equations_of(model)

        report = model.generate_report(format="markdown")

        assert report, f"{path.name} produced an empty report"
        assert _equations_of(model) == before, f"{path.name} was modified by reporting"


@pytest.mark.backend_core
@pytest.mark.parametrize("case_id", sorted(EXCLUDED))
def test_excluded_pairs_still_raise(case_id: str):
    """The excluded pairs still fail, and for the reason recorded against them.

    Turns the exclusion list into a claim the suite checks rather than a comment that can go stale: when an emitter is repaired this fails, which is how the pair gets a reference instead of staying invisible.
    """
    name, _, fmt = case_id.rpartition(".")
    path = MODEL_ROOT / (name.replace("__", "/"))
    model = Dynamics.from_file(f"{path}.yaml")

    with pytest.raises(EXCLUDED[case_id]):
        _render(model, fmt)


@pytest.mark.backend_core
def test_corpus_covers_every_model_and_format(regenerate: bool):
    """Every (model, format) pair has a reference, and none outlives its model.

    Reconciling per *case* rather than per model is what makes this meaningful: keyed on the model name alone, deleting one format's reference — or adding a third format — would leave the corpus reporting full coverage.
    """
    CORPUS.reconcile(
        (_case_id(p, f) for p, f in CASES),
        regenerate=regenerate,
        what="model/format pairs",
    )
