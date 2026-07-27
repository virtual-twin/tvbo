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
    _write(out_dir / "schema.py", PythonGenerator(str(schema)).serialize())
    _write(out_dir / "pydantic.py", PydanticGenerator(str(schema)).serialize())

    # JSON Schema — relax `additionalProperties: false → true` everywhere so
    # validation stays lenient, mirroring the previous
    # `JsonschemaValidationPlugin(closed=False)`. `sort_keys` keeps it reproducible.
    js = json.loads(JsonSchemaGenerator(str(schema)).serialize())
    _relax_additional_properties(js)
    _drop_redundant_anyof_type(js)
    (out_dir / "tvbo_datamodel.schema.json").write_text(
        json.dumps(js, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
