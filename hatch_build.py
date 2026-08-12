"""Generate the LinkML datamodel — build hook *and* standalone script.

``tvbo/datamodel/schema.py`` and ``tvbo/datamodel/pydantic.py`` are pure artifacts
generated from ``schema/tvbo_datamodel.yaml`` (and its imports). They are **not**
tracked in git (see ``.gitignore``): every wheel / sdist / editable build regenerates
them here, so they can neither conflict during merges nor drift out of sync with the
schema. Consequently ``linkml`` (the heavy generator toolkit) is a *build-time* only
dependency (``[build-system].requires``); importing the generated classes needs just
the lightweight ``linkml-runtime``.

This single file is used two ways so from-source and build-time codegen are byte-identical:
  * as a **hatchling build hook** (wheel / sdist / editable builds), and
  * as a **plain script** (``python hatch_build.py``) — the ``gen-linkml`` Makefile
    target, i.e. the entry point for a from-source checkout without an install.

Determinism: the Python generator emits a ``# Generation date:`` header line — we strip
it so the generated modules are byte-reproducible across builds.
"""
from __future__ import annotations

import re
from pathlib import Path

_NONDETERMINISTIC_PREFIX = "# Generation date:"


def generate_datamodel(root: str | Path) -> None:
    """Write the generated datamodel from ``schema/tvbo_datamodel.yaml``:

    * ``tvbo/datamodel/schema.py``                  — LinkML Python dataclasses,
    * ``tvbo/datamodel/pydantic.py``                — Pydantic models,
    * ``tvbo/datamodel/tvbo_datamodel.schema.json`` — JSON Schema for the
      ``tvbo validate`` CLI (checked with the lightweight ``jsonschema`` lib, so
      validation needs no runtime ``linkml``).
    """
    # Imported lazily so this module is importable without `linkml` (the heavy,
    # build-time-only generator) — e.g. when hatchling merely inspects the hook.
    import json

    from linkml.generators.jsonschemagen import JsonSchemaGenerator
    from linkml.generators.pydanticgen import PydanticGenerator
    from linkml.generators.pythongen import PythonGenerator

    root = Path(root)
    schema = root / "schema" / "tvbo_datamodel.yaml"
    if not schema.is_file():
        raise FileNotFoundError(
            f"Cannot generate the datamodel: schema source {schema} is missing. "
            "Ensure `schema/**` ships in the sdist so builds-from-sdist can regenerate."
        )
    out_dir = root / "tvbo" / "datamodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    shortcuts, aliases, keyed = _dialect_tables(schema)
    mixins = _mixins(root, schema)
    _write(out_dir / "dialect_tables.py", _render_dialect_tables(shortcuts, aliases, keyed))
    _write(
        out_dir / "schema.py",
        _with_behaviour(PythonGenerator(str(schema)).serialize(), mixins) + _INSTALL_DIALECT,
    )
    _write(
        out_dir / "pydantic.py",
        _with_dialect_and_behaviour(PydanticGenerator(str(schema)).serialize(), mixins),
    )

    # JSON Schema — relax `additionalProperties: false → true` everywhere so
    # validation stays lenient, mirroring the previous
    # `JsonschemaValidationPlugin(closed=False)`. `sort_keys` keeps it reproducible.
    js = json.loads(JsonSchemaGenerator(str(schema)).serialize())
    _relax_additional_properties(js)
    _drop_redundant_anyof_type(js)
    (out_dir / "tvbo_datamodel.schema.json").write_text(
        json.dumps(js, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# `boundaries` also implies `enforce: clamp`; yaml_loader._fold_state_variable_domains owns it.
_SEMANTIC_ALIASES = ("range", "boundaries")


def _dialect_tables(schema: Path) -> tuple[dict, dict, dict]:
    """``(scalar shortcuts, slot aliases, keyed collections)`` — the dialect, read off the schema.

    LinkML treats ``aliases:`` as documentation — its loaders key on the canonical slot
    name, so a declared alias is inert and raises ``unexpected keyword argument``. Each
    class's ``__init__`` already receives exactly its own slots, which makes it the one
    place where an alias can be resolved without guessing whether a mapping is an
    instance or a keyed collection, and without a free-form key (a parameter named
    ``dt``) ever being mistaken for a slot. Every construction path — the LinkML
    loaders, ``cls(**data)``, nested and inlined members, subclasses — goes through it.

    It also applies LinkML's ``simple_dict_value`` annotation, which marks the slot a
    bare scalar stands for: ``omega: 0.0628`` means ``{value: 0.0628}`` and
    ``equation: "x+2"`` means ``{rhs: "x+2"}``. LinkML specifies this for keyed
    collections (``inlined_as_simple_dict``) but ``linkml_runtime``'s dataclass loader
    does not implement it, so it is applied here — and extended to single-valued
    inlined slots, which the spec does not cover. Reference slots are skipped: their
    scalar is the target's *identifier*, not a value to wrap
    (``FreeParameter.parameter: ReducedWongWangEIB.J_i``). Inlined-ness is asked of the
    schema (``is_inlined``) rather than read off ``slot.inlined``, because a class-ranged
    slot whose range has an identifier is a reference by default while leaving
    ``inlined`` unset — six such slots were being wrapped, and since the wrapper then had
    to fit a string-ranged slot it landed as the literal ``"JsonObj(value='x')"``.

    Last, it records each keyed collection's identifier slot, so a member written under
    its key can be given the name the key already states. The generated dataclasses do
    that themselves in ``__post_init__``; the generated Pydantic models do not, and a
    curated entry spelled the way this project requires — ``TR: {value: 720.0}``, no
    redundant inner ``name`` — was accepted by one form and rejected by the other.
    """
    from linkml_runtime.utils.schemaview import SchemaView

    view = SchemaView(str(schema))

    shortcut_of: dict[str, str] = {}
    for cls_name in view.all_classes():
        # The annotation must be declared ON this class: induced slots inherit, and a
        # subclass that redefines what a bare scalar means (DerivedParameter, whose
        # scalar is an `equation`, not Parameter's `value`) must not silently take the
        # parent's.
        for slot_name in view.class_slots(cls_name, direct=True):
            slot = view.induced_slot(slot_name, cls_name)
            if slot.annotations and "simple_dict_value" in slot.annotations:
                shortcut_of[cls_name] = slot.name
                break
    lifts: dict[str, dict[str, str]] = {}
    for cls_name in view.all_classes():
        slot_lifts = {
            slot.name: (
                shortcut_of[str(slot.range)],
                bool(slot.multivalued),
                view.get_identifier_slot(str(slot.range), use_key=True) is not None,
            )
            for slot in view.class_induced_slots(cls_name)
            if str(slot.range) in shortcut_of and view.is_inlined(slot)
        }
        if slot_lifts:
            lifts[cls_name] = slot_lifts

    keyed: dict[str, dict[str, str]] = {}
    for cls_name in view.all_classes():
        members = {}
        for slot in view.class_induced_slots(cls_name):
            # `inlined_as_list` is a collection whose spelling IS a list; it has no keys.
            if not slot.multivalued or slot.inlined_as_list or not view.is_inlined(slot):
                continue
            identifier = view.get_identifier_slot(str(slot.range), use_key=True)
            if identifier is not None:
                members[slot.name] = identifier.name
        if members:
            keyed[cls_name] = members

    table: dict[str, dict[str, str]] = {}
    for cls_name in view.all_classes():
        amap = {
            alias: slot.name
            for slot in view.class_induced_slots(cls_name)
            for alias in (slot.aliases or [])
            if alias != slot.name and alias not in _SEMANTIC_ALIASES
        }
        # An alias that is a real slot of this same class is that slot's own name here.
        own = {s.name for s in view.class_induced_slots(cls_name)}
        amap = {a: c for a, c in amap.items() if a not in own}
        if amap:
            table[cls_name] = amap
    return lifts, table, keyed


def _render_dialect_tables(shortcuts: dict, aliases: dict, keyed: dict) -> str:
    """The dialect tables as their own module.

    Kept apart from ``schema.py`` so the Pydantic models can read them without importing
    the dataclasses: that import is the coupling the migration exists to remove, and it
    would be a cycle once ``schema.py`` installs the dialect from the same place.
    """
    return f'''"""Generated dialect tables — see :mod:`tvbo.datamodel.dialect`.

``SCALAR_SHORTCUTS`` maps ``{{class: {{slot: (slot the bare scalar stands for,
multivalued, keyed)}}}}``, from ``annotations.simple_dict_value`` on each range class.
``SLOT_ALIASES`` maps ``{{class: {{alias: canonical slot}}}}`` from the schema's ``aliases:``.
``KEYED_COLLECTIONS`` maps ``{{class: {{slot: the member slot its key states}}}}``.
"""

SCALAR_SHORTCUTS = {shortcuts!r}

SLOT_ALIASES = {aliases!r}

KEYED_COLLECTIONS = {keyed!r}
'''


#: Appended to the generated ``schema.py``; the dataclasses fold the dialect in ``__init__``.
_INSTALL_DIALECT = """

from tvbo.datamodel.dialect import install_on_dataclasses as _install_dialect

_install_dialect(globals())
"""

#: Attached to every class the schema gives an ``iri``; see :mod:`tvbo.behaviour._enrich`.
_ENRICHABLE_MIXIN = "tvbo.behaviour._enrich.IriEnrichable"


def _mixins(root: Path, schema: Path) -> dict[str, list[str]]:
    """``{class: dotted paths of the mixins it takes}``, most specific first.

    Two rules, each derived rather than listed. A class takes its behaviour mixin because
    one is named after it, and it takes :class:`IriEnrichable` because the schema gives it
    an ``iri`` — a record that may name an entity elsewhere is a record that can be filled
    from one. Only the classes that declare the slot *directly* are named: a subclass
    inherits the base it was given, and naming it again would put the same mixin twice in
    one class statement.

    Behaviour comes first so a behaviour mixin can refine an enrichment method and reach
    the generic one through ``super()``.
    """
    from linkml_runtime.utils.schemaview import SchemaView

    view = SchemaView(str(schema))
    mixins = {cls: [path] for cls, path in _behaviour_mixins(root, schema).items()}
    for cls_name in view.all_classes():
        if "iri" in view.class_slots(cls_name, direct=True):
            mixins.setdefault(cls_name, []).append(_ENRICHABLE_MIXIN)
    return mixins


def _behaviour_mixins(root: Path, schema: Path) -> dict[str, str]:
    """``{class: dotted path of its behaviour mixin}``, discovered from ``tvbo/behaviour``.

    Behaviour is attached where the class is generated rather than in a runtime subclass,
    so it reaches every object from every construction path — loaded, nested, or built by
    hand — instead of only the ones some call site remembered to wrap.

    Which class a mixin belongs to is read from its own name: ``EventBehaviour`` attaches
    to ``Event``. Deliberately NOT declared in the schema, which is language-neutral and
    is also what the OWL export is generated from; a Python import path there would be one
    language's mechanism stated as if it were a fact about the model.

    Discovery parses the modules rather than importing them, so the build hook stays free
    of the package's runtime dependencies. A mixin naming a class the schema does not
    define is an error, since it would otherwise attach to nothing and fail silently.
    """
    import ast

    from linkml_runtime.utils.schemaview import SchemaView

    known = set(SchemaView(str(schema)).all_classes())
    mixins: dict[str, str] = {}
    for module in sorted((root / "tvbo" / "behaviour").glob("*.py")):
        if module.stem.startswith("_"):
            continue
        for node in ast.parse(module.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Behaviour"):
                continue
            target = node.name[: -len("Behaviour")]
            if target not in known:
                raise RuntimeError(
                    f"{module.name} defines {node.name}, but the schema has no class "
                    f"{target!r} for it to attach to."
                )
            mixins[target] = f"tvbo.behaviour.{module.stem}.{node.name}"
    return mixins


def _inject_bases(code: str, additions: dict[str, list[str]]) -> str:
    """Prepend bases to generated class statements.

    LinkML emits these classes from its own template and no generator hook reaches inside
    them, so the class statement is rewritten directly. Each anchor must match exactly
    once: a template change then fails the build here, loudly, rather than silently
    producing classes that have quietly lost their dialect or their behaviour.
    """
    for cls_name, bases in additions.items():
        pattern = re.compile(rf"^class {re.escape(cls_name)}\((?P<bases>[^)]*)\):", re.M)
        matches = pattern.findall(code)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one generated `class {cls_name}(...)`, found "
                f"{len(matches)} — the LinkML template or the schema changed shape."
            )
        code = pattern.sub(
            lambda m: f"class {cls_name}({', '.join(bases)}, {m.group('bases')}):",
            code,
            count=1,
        )
    return code


#: Prepended to ``pydantic.py`` as ``ConfiguredBaseModel``'s base, folding before validation.
_DIALECT_BASE = '''from pydantic import model_validator as _model_validator

from tvbo.datamodel.dialect import normalize as _normalize_dialect


class _DialectBase(BaseModel):
    """Accepts the TVBO dialect — declared aliases and bare-scalar shortcuts.

    Runs before validation, once per constructed model, so Pydantic's own recursion
    reaches every nested member; the validator only ever handles its own level.
    """

    @_model_validator(mode="before")
    @classmethod
    def _fold_tvbo_dialect(cls, data):
        if isinstance(data, dict):
            return _normalize_dialect(cls.__name__, dict(data))
        return data


'''


def _mixin_imports(mixins: dict[str, list[str]]) -> str:
    """``from <module> import <Mixin>`` for each declared mixin."""
    return "".join(
        f"from {path.rsplit('.', 1)[0]} import {path.rsplit('.', 1)[1]}\n"
        for path in sorted({path for paths in mixins.values() for path in paths})
    )


def _mixin_bases(mixins: dict[str, list[str]]) -> dict[str, list[str]]:
    """``{class: mixin class names}`` — the dotted paths reduced to what a base is written as."""
    return {cls: [path.rsplit(".", 1)[1] for path in paths] for cls, paths in mixins.items()}


def _with_behaviour(code: str, mixins: dict[str, list[str]]) -> str:
    """Give each annotated dataclass its mixins.

    Both generated forms take the same mixins, so behaviour reaches an object whether it
    came from LinkML's loader (a dataclass) or from Pydantic validation. Without this the
    dataclasses would lose every helper the moment it moved out of a runtime subclass,
    since that is still what the loaders construct.
    """
    if not mixins:
        return code
    code = code.replace(
        'metamodel_version = "', _mixin_imports(mixins) + '\nmetamodel_version = "', 1
    )
    return _inject_bases(code, _mixin_bases(mixins))


def _with_dialect_and_behaviour(code: str, mixins: dict[str, list[str]]) -> str:
    """Give the generated models the dialect, and each annotated class its mixins.

    The mixins are imported beside ``_DialectBase``, above every generated class, so a
    mixin module must not import the datamodel at module scope — the same discipline
    :mod:`tvbo.datamodel.dialect` follows.
    """
    code = code.replace(
        "class ConfiguredBaseModel(BaseModel):",
        _mixin_imports(mixins) + _DIALECT_BASE + "class ConfiguredBaseModel(BaseModel):",
        1,
    )
    additions = {"ConfiguredBaseModel": ["_DialectBase"]}
    additions.update(_mixin_bases(mixins))
    return _inject_bases(code, additions)


def _relax_additional_properties(node) -> None:
    """Recursively rewrite ``additionalProperties: false`` → ``true`` (open validation)."""
    if isinstance(node, dict):
        if node.get("additionalProperties") is False:
            node["additionalProperties"] = True
        for value in node.values():
            _relax_additional_properties(value)
    elif isinstance(node, list):
        for value in node:
            _relax_additional_properties(value)


def _drop_redundant_anyof_type(node) -> None:
    """Strip the redundant sibling ``type`` LinkML stamps beside ``anyOf``.

    ``JsonSchemaGenerator`` emits a slot's base range as a sibling ``type`` even
    when the slot declares ``any_of``. In JSON Schema a sibling ``type`` conjoins
    with ``anyOf``, so the base range silently *narrows* the union: ``n_parallel``
    (``any_of: [integer, string]``, base range ``string`` from ``default_range``)
    rejects ``1`` with "1 is not of type 'string'". Only a *scalar* base-range stamp
    (``string``/``integer``/``number``/``boolean``) is this redundant, wrong sibling;
    a structural ``type: object``/``array`` beside ``anyOf`` (a class-level rule) is a
    real constraint, so it is left intact.
    """
    _SCALAR_STAMP = {"string", "integer", "number", "boolean"}
    if isinstance(node, dict):
        if "anyOf" in node and node.get("type") in _SCALAR_STAMP:
            node.pop("type", None)
        for value in node.values():
            _drop_redundant_anyof_type(value)
    elif isinstance(node, list):
        for value in node:
            _drop_redundant_anyof_type(value)


def _write(target: Path, code: str) -> None:
    body = "".join(
        line
        for line in code.splitlines(keepends=True)
        if not line.startswith(_NONDETERMINISTIC_PREFIX)
    )
    target.write_text(body, encoding="utf-8")


try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # invoked as a plain script (no hatchling on the path)
    pass
else:

    class DatamodelBuildHook(BuildHookInterface):
        """Regenerate the datamodel before every wheel / sdist / editable build."""

        PLUGIN_NAME = "custom"

        def initialize(self, version: str, build_data: dict) -> None:
            generate_datamodel(self.root)


if __name__ == "__main__":
    generate_datamodel(Path(__file__).resolve().parent)
