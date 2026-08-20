"""Generate the LinkML datamodel and materialize the authored records — build hook *and* standalone script.

``tvbo/datamodel/schema.py`` and ``tvbo/datamodel/pydantic.py`` are pure artifacts generated from ``schema/tvbo_datamodel.yaml`` (and its imports). They are **not** tracked in git (see ``.gitignore``): every wheel / sdist / editable build regenerates them here, so they can neither conflict during merges nor drift out of sync with the schema. Consequently ``linkml`` (the heavy generator toolkit) is a *build-time* only dependency (``[build-system].requires``); importing the generated classes needs just the lightweight ``linkml-runtime``.

It also copies the instance documents authored beside the schema (``schema/study_layout.yaml``, the ground truth for the study directory layout) into the package tree, so the runtime resolves them at one import-relative path whether tvbo was installed from a wheel or editable.

This single file is used two ways so from-source and build-time codegen are byte-identical:
  * as a **hatchling build hook** (wheel / sdist / editable builds), and
  * as a **plain script** (``python hatch_build.py``) — the ``gen-linkml`` Makefile
    target, i.e. the entry point for a from-source checkout without an install.

Determinism: the Python generator emits a ``# Generation date:`` header line — we strip it so the generated modules are byte-reproducible across builds.
"""

from __future__ import annotations

from pathlib import Path

_NONDETERMINISTIC_PREFIX = "# Generation date:"


def generate_datamodel(root: str | Path) -> None:
    """Write the generated datamodel from ``schema/tvbo_datamodel.yaml``.

    * ``tvbo/datamodel/schema.py``                  — LinkML Python dataclasses,
    * ``tvbo/datamodel/pydantic.py``                — Pydantic models,
    * ``tvbo/datamodel/tvbo_datamodel.schema.json`` — JSON Schema for the
      ``tvbo validate`` CLI (checked with the lightweight ``jsonschema`` lib, so
      validation needs no runtime ``linkml``).
    """
    # Imported lazily so this module is importable without `linkml` (the heavy, build-time-only generator) — e.g. when hatchling merely inspects the hook.
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
    _copy_records(root)
    _write(out_dir / "schema.py", PythonGenerator(str(schema)).serialize() + _alias_support(schema))
    _write(out_dir / "pydantic.py", PydanticGenerator(str(schema)).serialize())

    # JSON Schema — relax `additionalProperties: false → true` everywhere so validation stays lenient, mirroring the previous `JsonschemaValidationPlugin(closed=False)`. `sort_keys` keeps it reproducible.
    js = json.loads(JsonSchemaGenerator(str(schema)).serialize())
    _relax_additional_properties(js)
    _drop_redundant_anyof_type(js)
    (out_dir / "tvbo_datamodel.schema.json").write_text(json.dumps(js, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# `boundaries` also implies `enforce: clamp`; yaml_loader._fold_state_variable_domains owns it.
_SEMANTIC_ALIASES = ("range", "boundaries")


def _alias_support(schema: Path) -> str:
    """Python appended to the generated ``schema.py`` so classes accept their aliases.

    LinkML treats ``aliases:`` as documentation — its loaders key on the canonical slot name, so a declared alias is inert and raises ``unexpected keyword argument``. Each class's ``__init__`` already receives exactly its own slots, which makes it the one place where an alias can be resolved without guessing whether a mapping is an instance or a keyed collection, and without a free-form key (a parameter named ``dt``) ever being mistaken for a slot. Every construction path — the LinkML loaders, ``cls(**data)``, nested and inlined members, subclasses — goes through it.

    It also applies LinkML's ``simple_dict_value`` annotation, which marks the slot a bare scalar stands for: ``omega: 0.0628`` means ``{value: 0.0628}`` and ``equation: "x+2"`` means ``{rhs: "x+2"}``. LinkML specifies this for keyed collections (``inlined_as_simple_dict``) but ``linkml_runtime``'s dataclass loader does not implement it, so it is applied here — and extended to single-valued inlined slots, which the spec does not cover. Slots explicitly marked ``inlined: false`` are references: their scalar is the target's *identifier*, not a value to wrap (``FreeParameter.parameter: ReducedWongWangEIB.J_i``), so they are skipped.
    """
    from linkml_runtime.utils.schemaview import SchemaView

    view = SchemaView(str(schema))

    shortcut_of: dict[str, str] = {}
    for cls_name in view.all_classes():
        # The annotation must be declared ON this class: induced slots inherit, and a subclass that redefines what a bare scalar means (DerivedParameter, whose scalar is an `equation`, not Parameter's `value`) must not silently take the parent's.
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
            if str(slot.range) in shortcut_of and slot.inlined is not False
        }
        if slot_lifts:
            lifts[cls_name] = slot_lifts

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
    return f"""

# {{class: {{slot: slot the scalar stands for}}}}, from `annotations.scalar_shortcut`: lets a value be written bare where the object has one obvious field.
_SCALAR_SHORTCUTS = {lifts!r}


_SCALARS = (str, int, float, bool)


def _is_literal(value):
    \"\"\"A bare value the shortcut may lift: a scalar, or a (nested) list of scalars.

    An array literal counts — a coordinate list to select, a coefficient matrix — because
    the slot it lifts into holds arrays as well as scalars. A list of MAPPINGS does not:
    that is the list spelling of a keyed collection, whose members lift individually.
    \"\"\"
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_literal(v) for v in value)
    return False


def _lift_scalar(value, target, multivalued, keyed=False):
    \"\"\"`0.0628` -> `{{'value': 0.0628}}`, leaving an already-written mapping alone.

    On a multivalued slot the members are lifted, not the collection: `{{omega: 0.0628}}`
    is a keyed collection of one Parameter, not a Parameter. A `keyed` collection's LIST
    spelling (`arguments: [v]`) is a list of member identifiers, not values, so its bare
    scalars are left for the loader to key on; only a non-keyed list (`additional_equations:
    ["x = -x"]` -> `[{{rhs: "x = -x"}}]`) lifts its elements.
    \"\"\"
    if not multivalued:
        return {{target: value}} if _is_literal(value) else value
    if isinstance(value, dict):
        return {{k: ({{target: v}} if _is_literal(v) else v) for k, v in value.items()}}
    if isinstance(value, list):
        if keyed:
            return value
        return [({{target: v}} if _is_literal(v) else v) for v in value]
    return value


# {{class: {{alias: canonical slot}}}} from the schema's `aliases:`, folded in __init__ where the kwargs are known to belong to this class.
_SLOT_ALIASES = {table!r}


def _install_slot_aliases() -> None:
    import warnings

    def _wrap(cls, amap):
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            for slot, (target, mv, keyed) in _SCALAR_SHORTCUTS.get(cls.__name__, {{}}).items():
                if slot in kwargs and kwargs[slot] is not None:
                    kwargs[slot] = _lift_scalar(kwargs[slot], target, mv, keyed)
            for alias, canonical in amap.items():
                if alias in kwargs:
                    value = kwargs.pop(alias)
                    if canonical in kwargs:
                        warnings.warn(
                            f"{{cls.__name__}} got both {{alias!r}} and its canonical "
                            f"slot {{canonical!r}}; ignoring {{alias!r}}.",
                            stacklevel=2,
                        )
                    else:
                        kwargs[canonical] = value
            original(self, *args, **kwargs)

        cls.__init__ = __init__

    for name in set(_SLOT_ALIASES) | set(_SCALAR_SHORTCUTS):
        cls = globals().get(name)
        if cls is not None:
            _wrap(cls, _SLOT_ALIASES.get(name, {{}}))


_install_slot_aliases()
"""


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

    ``JsonSchemaGenerator`` emits a slot's base range as a sibling ``type`` even when the slot declares ``any_of``. In JSON Schema a sibling ``type`` conjoins with ``anyOf``, so the base range silently *narrows* the union: ``n_parallel`` (``any_of: [integer, string]``, base range ``string`` from ``default_range``) rejects ``1`` with "1 is not of type 'string'". Only a *scalar* base-range stamp (``string``/``integer``/``number``/``boolean``) is this redundant, wrong sibling;
    a structural ``type: object``/``array`` beside ``anyOf`` (a class-level rule) is a real constraint, so it is left intact.
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


# Instance documents authored beside the schema that types them, materialized under `tvbo/` so a wheel and an editable install both find them at one import-relative path.
_RECORDS = {"study_layout.yaml": Path("tvbo") / "rules" / "study_layout.yaml"}


def _copy_records(root: Path) -> None:
    """Materialize the authored records from ``schema/`` into the package tree.

    A record is a LinkML *instance* (``tvbo_class: tvbo:StudyLayout``), so it is authored beside the schema that types it and never duplicated. The runtime reads the copy, which is gitignored and force-included in the wheel exactly as the generated datamodel is, so the single ground truth stays in ``schema/`` while a wheel-installed tvbo can still resolve it.
    """
    for source_name, rel_target in _RECORDS.items():
        source = root / "schema" / source_name
        if not source.is_file():
            raise FileNotFoundError(
                f"Cannot materialize the {source_name} record: {source} is missing. Ensure `schema/**` ships in the sdist."
            )
        target = root / rel_target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"# Generated from schema/{source_name} by hatch_build.py. Edit the source, not this copy.\n"
            + source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _write(target: Path, code: str) -> None:
    body = "".join(line for line in code.splitlines(keepends=True) if not line.startswith(_NONDETERMINISTIC_PREFIX))
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
            """Regenerate the datamodel before every wheel, sdist and editable build."""
            generate_datamodel(self.root)


if __name__ == "__main__":
    generate_datamodel(Path(__file__).resolve().parent)
