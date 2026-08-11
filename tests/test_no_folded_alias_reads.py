"""No source reads a slot alias off a constructed object.

The TVBO dialect folds a declared alias into its canonical slot at construction, so the
alias is NOT an attribute afterwards. `getattr(obj, "<alias>", default)` therefore returns
the default forever — silently, and the defaults are plausible, which is how three of these
survived: Brian2 scaled every declared time unit by 1, `run_cuda` integrated at 0.1
whatever the recipe said, and every Dirichlet boundary condition came out 0.0.

Reading the alias as a *fallback* after the canonical name is fine and common
(`number_of_nodes or number_of_regions`): the alias branch is dead, not wrong.
"""

import re
from pathlib import Path

import pytest

from tvbo.datamodel import tvbo_datamodel as dm
from tvbo.datamodel.dialect_tables import SLOT_ALIASES

ROOT = Path(__file__).resolve().parents[1] / "tvbo"

SKIP = ("datamodel/schema.py", "datamodel/pydantic.py", "datamodel/tvbo_datamodel",
        "datamodel/dialect")

ACCESSORS = ("getattr", "slot", "_p")

EXEMPT = {
    # `sim` is a TVB Simulator being imported FROM, not a TVBO object; its integrator
    # really does carry `dt`.
    ("adapters/tvb.py", "dt"),
    # `_res` is a tvboptim solution object, which carries its own `dt`.
    ("templates/tvboptim/tvbo-tvboptim-experiment.py.mako", "dt"),
}

# A name that is a real slot on SOME generated class cannot be judged from the text alone:
# `target_variable` is an `Edge` alias but `Event`'s own slot.
_REAL_SLOTS = set()
for _name in dir(dm):
    _cls = getattr(dm, _name)
    if isinstance(_cls, type):
        _REAL_SLOTS |= set(getattr(_cls, "__dataclass_fields__", {}))

CHECKED = {alias: canonical
           for mapping in SLOT_ALIASES.values()
           for alias, canonical in mapping.items()
           if alias not in _REAL_SLOTS}


def _sources():
    for path in sorted(ROOT.rglob("*")):
        if path.suffix in (".py", ".mako") and not any(s in str(path) for s in SKIP):
            yield path


def _reads(text, name):
    """Line numbers where *name* is read through one of the accessor helpers."""
    pattern = re.compile(
        r"(?:%s)\(\s*[^,()]+?\s*,\s*['\"]%s['\"]" % ("|".join(ACCESSORS), re.escape(name))
    )
    return {i for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)}


def _canonical_reads(text, canonical):
    """Line numbers where *canonical* is read at all — by accessor or as an attribute.

    A fallback often reaches the canonical slot directly once the alias branch has taken
    the optional path: ``getattr(dyn, "components", None) or dyn.modes``.
    """
    attribute = re.compile(r"\.%s\b" % re.escape(canonical))
    direct = {i for i, line in enumerate(text.splitlines(), 1) if attribute.search(line)}
    return _reads(text, canonical) | direct


def _bare_alias_reads(path):
    """Alias reads in *path* with no canonical read of the same slot nearby.

    A five-line window, because the `canonical or alias` fallback is routinely wrapped
    across lines by the formatter.
    """
    text = path.read_text(errors="ignore")
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    found = []
    for alias, canonical in CHECKED.items():
        if (rel, alias) in EXEMPT:
            continue
        canonical_lines = _canonical_reads(text, canonical)
        for line in _reads(text, alias):
            if not any(abs(line - other) <= 5 for other in canonical_lines):
                found.append((rel, line, alias, canonical))
    return found


@pytest.mark.backend_core
def test_no_source_reads_a_folded_alias_without_the_canonical_slot():
    """The guard. A finding is either a real bug or an entry in `EXEMPT` with a reason."""
    offenders = [f for path in _sources() for f in _bare_alias_reads(path)]

    assert not offenders, "\n".join(
        f"{rel}:{line} reads the folded alias {alias!r} — the dialect renames it to "
        f"{canonical!r} at construction, so this always returns its default"
        for rel, line, alias, canonical in offenders
    )


def test_the_guard_can_see_a_bare_alias_read(tmp_path):
    """Guards the guard: a checker that matches nothing would pass silently forever."""
    alias, canonical = next(iter(CHECKED.items()))
    offender = tmp_path / "offender.py"
    offender.write_text(f'x = getattr(obj, "{alias}", None)\n')

    found = _bare_alias_reads(offender)

    assert [f[2] for f in found] == [alias]
    assert found[0][3] == canonical


def test_the_guard_accepts_the_canonical_first_fallback(tmp_path):
    """`canonical or alias` is the idiom the codebase uses; it must not be flagged."""
    alias, canonical = next(iter(CHECKED.items()))
    ok = tmp_path / "ok.py"
    ok.write_text(
        f'x = (\n    getattr(obj, "{canonical}", None)\n    or getattr(obj, "{alias}", None)\n)\n'
    )

    assert _bare_alias_reads(ok) == []


def test_ambiguous_names_are_excluded_from_the_check():
    """A name that is also a real slot somewhere cannot be judged statically.

    `Edge` aliases `target_variable` to `target_var`, but `Event` and `TuningObjective`
    declare `target_variable` as their own slot — flagging every read of it would demand
    a rename that breaks the classes that own the name.
    """
    assert "target_variable" not in CHECKED
    assert "target_variable" in _REAL_SLOTS
    assert "time_scale" in CHECKED
