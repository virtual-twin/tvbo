"""No source reads a slot alias off a constructed object.

The TVBO dialect folds a declared alias into its canonical slot at construction, so the
alias is NOT an attribute afterwards. `getattr(obj, "<alias>", default)` therefore returns
the default forever — silently, and the defaults are plausible, which is how three of these
survived: Brian2 scaled every declared time unit by 1, `run_cuda` integrated at 0.1
whatever the recipe said, and every Dirichlet boundary condition came out 0.0.

Reading the alias as a *fallback* after the canonical name is fine and common
(`number_of_nodes or number_of_regions`): the alias branch is dead, not wrong.
"""

import ast
import re
from pathlib import Path

import pytest

from tvbo.datamodel import tvbo_datamodel as dm
from tvbo.datamodel.dialect_tables import SLOT_ALIASES

ROOT = Path(__file__).resolve().parents[1] / "tvbo"

SKIP = ("datamodel/schema.py", "datamodel/pydantic.py", "datamodel/tvbo_datamodel", "datamodel/dialect")

ACCESSORS = ("getattr", "slot", "_p")

EXEMPT = {
    ("adapters/tvb.py", "sim.integrator", "dt"): "a TVB Simulator being imported FROM; its integrator owns `dt`",
    ("templates/tvboptim/tvbo-tvboptim-experiment.py.mako", "_res", "dt"): "a tvboptim solution object, which owns `dt`",
}
"""``(path, object, alias) -> why THAT object is not a TVBO one``, so the alias is its name.

Keyed on the object, not the file: exempting every `dt` read in `adapters/tvb.py` would
also excuse a future one on a genuine TVBO `Integrator`, which is the bug this exists for."""


def _real_slot_names():
    """Every slot name some generated class actually declares.

    A name that is a real slot somewhere cannot be judged from the text alone:
    ``target_variable`` is an ``Edge`` alias but ``Event``'s own slot, so flagging every
    read of it would demand a rename that breaks the class owning the name.

    Read from either generated representation. Asking only for ``__dataclass_fields__``
    would return an empty set the day the datamodel becomes Pydantic — which is this
    branch's own direction — and an empty set silently widens the check onto names that
    classes legitimately own, failing the hook on correct code.
    """

    def declared(cls, attribute):
        """*cls*'s field names under *attribute*, or none.

        Guarded because the generated enums resolve **any** attribute as an enum code and
        raise ``ValueError`` for one that names no permissible value.
        """
        try:
            return set(getattr(cls, attribute, None) or {})
        except Exception:
            return set()

    names = set()
    for name in dir(dm):
        cls = getattr(dm, name)
        if isinstance(cls, type):
            names |= declared(cls, "__dataclass_fields__") | declared(cls, "model_fields")
    return names


_REAL_SLOTS = _real_slot_names()

CHECKED = {
    alias: canonical for mapping in SLOT_ALIASES.values() for alias, canonical in mapping.items() if alias not in _REAL_SLOTS
}


def _sources():
    for path in sorted(ROOT.rglob("*")):
        if path.suffix in (".py", ".mako") and not any(s in str(path) for s in SKIP):
            yield path


ACCESSOR_READ = re.compile(r"(?:%s)\(\s*(?P<obj>[^,()\n]+?)\s*,\s*['\"](?P<name>\w+)['\"]" % "|".join(ACCESSORS))
"""``getattr(obj, "name"`` and friends, for templates that no parser will accept.

The object group admits neither parentheses nor a newline: widening it let a call with no
comma at all run on until some later string literal and report an alias read that was
never written."""

ATTRIBUTE_READ = re.compile(r"(?P<obj>[A-Za-z_][\w.]*)\.(?P<name>\w+)\b")
"""``obj.name`` — how a fallback usually reaches the canonical slot once the alias missed."""

WINDOW = 5
"""Lines either side that still count as "beside": a wrapped `canonical or alias` fallback."""


def _parsed_reads(text):
    """``(accessor reads, attribute reads)`` from the syntax tree, or ``None`` if unparsable.

    Parsing rather than matching, because the object decides whether a canonical read
    excuses an alias read beside it, and text cannot say what the object *is*:
    ``nodes[0]``, ``get(obj)`` and a call wrapped over three lines are all ordinary here,
    and a regex either misreads them or refuses them.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    accessors, attributes = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attributes.setdefault((ast.unparse(node.value), node.attr), set()).add(node.lineno)
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            key = node.args[1]
            if name in ACCESSORS and isinstance(key, ast.Constant) and isinstance(key.value, str):
                accessors.setdefault((ast.unparse(node.args[0]), key.value), set()).add(node.lineno)
    return accessors, attributes


def _matched_reads(text, *patterns):
    """``{(object, name): {line, …}}`` by regex, for text the parser cannot take."""
    reads = {}
    for pattern in patterns:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            reads.setdefault((match["obj"].strip(), match["name"]), set()).add(line)
    return reads


def _reads(text):
    """``(accessor reads, reads that can excuse one)`` however *text* can be read."""
    parsed = _parsed_reads(text)
    if parsed is not None:
        accessors, attributes = parsed
        return accessors, {**attributes, **accessors}
    accessors = _matched_reads(text, ACCESSOR_READ)
    return accessors, _matched_reads(text, ACCESSOR_READ, ATTRIBUTE_READ)


def _bare_alias_reads(path):
    """Alias reads in *path* with no canonical read *of the same object* beside them.

    Same object, not merely the same name somewhere near: canonical slots include `rhs`,
    `nodes` and `step_size`, which appear on most lines of an adapter, so a proximity test
    on the name alone excuses nearly every alias read there is.
    """
    text = path.read_text(errors="ignore")
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    accessors, canonicals = _reads(text)
    found = []
    for (obj, name), lines in accessors.items():
        canonical = CHECKED.get(name)
        if canonical is None or (rel, obj, name) in EXEMPT:
            continue
        beside = canonicals.get((obj, canonical), set())
        for line in sorted(lines):
            if not any(abs(line - other) <= WINDOW for other in beside):
                found.append((rel, line, name, canonical))
    return sorted(found, key=lambda f: f[1])


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
    ok.write_text(f'x = (\n    getattr(obj, "{canonical}", None)\n    or getattr(obj, "{alias}", None)\n)\n')

    assert _bare_alias_reads(ok) == []


@pytest.mark.parametrize(
    "label,body",
    [
        ("no alias read at all", 'a = slot(figure)\nb = fmt(x, "dt")\n'),
        ("nested accessor", 'x = getattr(getattr(exp, "dt", None), "foo", None)\nstep = exp.step_size\n'),
        ("subscripted object", 'step = nodes[0].step_size\nx = getattr(nodes[0], "dt", None)\n'),
        ("call result", 'cfg = get(obj).step_size\nx = getattr(get(obj), "dt", None)\n'),
    ],
)
def test_the_guard_does_not_fire_on_these(label, body, tmp_path):
    """Shapes that are correct code. This runs as a pre-push hook, so a false positive
    blocks a push on work that has nothing wrong with it — which is worse than a miss."""
    source = tmp_path / "source.py"
    source.write_text(body)

    assert _bare_alias_reads(source) == [], label


def test_the_slot_index_is_not_empty():
    """An empty index would widen the check onto names classes own, and fail correct code."""
    assert len(_REAL_SLOTS) > 100
    assert {"name", "value", "rhs"} <= _REAL_SLOTS


def test_a_canonical_read_of_a_DIFFERENT_object_does_not_excuse_the_alias(tmp_path):
    """The excuse is same-object, not same-name-nearby.

    Canonicals like `rhs`/`nodes`/`step_size` appear on most lines of an adapter, so a
    proximity test on the name alone excused nearly every alias read in the file.
    """
    alias, canonical = next(iter(CHECKED.items()))
    offender = tmp_path / "offender.py"
    offender.write_text(f'y = other.{canonical}\nx = getattr(obj, "{alias}", None)\n')

    assert [f[2] for f in _bare_alias_reads(offender)] == [alias]


def test_ambiguous_names_are_excluded_from_the_check():
    """A name that is also a real slot somewhere cannot be judged statically.

    `Edge` aliases `target_variable` to `target_var`, but `Event` and `TuningObjective`
    declare `target_variable` as their own slot — flagging every read of it would demand
    a rename that breaks the classes that own the name.
    """
    assert "target_variable" not in CHECKED
    assert "target_variable" in _REAL_SLOTS
    assert "time_scale" in CHECKED
