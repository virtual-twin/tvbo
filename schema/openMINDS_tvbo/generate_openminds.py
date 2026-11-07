#!/usr/bin/env python3
"""Generate openMINDS-compatible schemas from LinkML source of truth.

This script reads the TVBO LinkML datamodel and generates openMINDS JSON schema templates, dynamically mapping types to existing openMINDS namespaces where appropriate.

All type mappings are imported from tvbo.adapters.openminds to maintain a single source of truth.

Usage:
    python generate_openminds.py [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path to enable imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import all type mappings from the single source of truth
from tvbo.adapters.openminds import (
    EXTERNAL_TYPE_MAPPINGS,
    LINKML_TO_JSON_TYPE,
    OPENMINDS_CATEGORIES,
    OPENMINDS_EXTENDS,
    SKIP_CLASSES,
)


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def load_linkml_schema(path: Path) -> dict[str, Any]:
    """Load and parse a LinkML YAML schema."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_slot(
    slot_name: str,
    slots: dict[str, Any],
    class_def: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a slot definition, merging global slot with class-level usage."""
    global_slot = slots.get(slot_name, {}) or {}
    slot_usage = class_def.get("slot_usage", {}).get(slot_name, {}) or {}

    # Merge: slot_usage overrides global slot
    resolved = {**global_slot, **slot_usage}
    return resolved


def get_openminds_type(linkml_type: str) -> str:
    """Map a LinkML type to openMINDS type reference."""
    if linkml_type in EXTERNAL_TYPE_MAPPINGS:
        return EXTERNAL_TYPE_MAPPINGS[linkml_type]
    # Default: use tvbo namespace
    return f"tvbo:{linkml_type}"


def is_external_type(linkml_type: str) -> bool:
    """Check if a type maps to an external openMINDS namespace."""
    return linkml_type in EXTERNAL_TYPE_MAPPINGS


def linkml_range_to_openminds(
    range_type: str | None,
    is_multivalued: bool = False,
    is_inlined: bool = False,
    all_classes: set[str] | None = None,
) -> dict[str, Any]:
    """Convert LinkML range to openMINDS property definition.

    Returns a dict with type, _linkedTypes, or _embeddedTypes as appropriate.
    """
    all_classes = all_classes or set()
    result: dict[str, Any] = {}

    if range_type is None:
        result["type"] = "string"
    elif range_type in LINKML_TO_JSON_TYPE:
        result["type"] = LINKML_TO_JSON_TYPE[range_type]
    elif range_type in all_classes or range_type in EXTERNAL_TYPE_MAPPINGS:
        # It's a class reference
        openminds_type = get_openminds_type(range_type)
        if is_inlined:
            result["_embeddedTypes"] = [openminds_type]
        else:
            result["_linkedTypes"] = [openminds_type]
    else:
        # Assume string for unknown types (including enums)
        result["type"] = "string"

    if is_multivalued:
        result["type"] = "array"
        result["minItems"] = 1
        result["uniqueItems"] = True

    return result


def build_instruction(slot_name: str, slot_def: dict[str, Any]) -> str:
    """Generate an instruction string for a property."""
    description = slot_def.get("description", "")
    if description:
        return description

    # Generate generic instruction from slot name
    readable_name = slot_name.replace("_", " ")
    slot_def.get("range", "value")

    if slot_def.get("multivalued"):
        return f"Add all {readable_name}."
    return f"Enter the {readable_name}."


def inherited_chain(class_name: str, class_defs: dict[str, Any]) -> list[dict[str, Any]]:
    """The class and its ``is_a`` ancestors, most distant first.

    A subclass carries its parents' slots at runtime, so an export that emitted only a class's own declarations would silently drop them — a `PDESolver is_a Solver` would serialize without the step size and tolerances it actually has.
    """
    chain, seen, name = [], set(), class_name
    while name and name in class_defs and name not in seen:
        seen.add(name)
        chain.append(class_defs[name])
        name = (class_defs[name] or {}).get("is_a")
    return list(reversed(chain))


def convert_class_to_openminds(
    class_name: str,
    class_def: dict[str, Any],
    slots: dict[str, Any],
    all_classes: set[str],
    class_defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a LinkML class definition to openMINDS schema."""
    schema: dict[str, Any] = {
        "_type": get_openminds_type(class_name),
    }

    # Add extends if mapped
    if class_name in OPENMINDS_EXTENDS:
        schema["_extends"] = OPENMINDS_EXTENDS[class_name]

    # Add categories if mapped
    if class_name in OPENMINDS_CATEGORIES:
        schema["_categories"] = OPENMINDS_CATEGORIES[class_name]

    required_flags: dict[str, bool] = {}
    """Required flag per property name — a map, not an append-only list, so a subclass redeclaring an inherited slot as optional flips it off: inherited definitions run first and own last, and the last write wins."""

    # Properties
    properties: dict[str, Any] = {}

    # Inherited first, own last, so a subclass narrows what it redeclares.
    for definition in inherited_chain(class_name, class_defs or {class_name: class_def}):
        for slot_name in definition.get("slots", []) or []:
            slot_def = resolve_slot(slot_name, slots, definition)
            prop = convert_slot_to_property(slot_name, slot_def, all_classes)
            if prop:
                properties[slot_name] = prop
                required_flags[slot_name] = bool(slot_def.get("required"))
        for attr_name, attr_def in (definition.get("attributes", {}) or {}).items():
            if isinstance(attr_def, dict):
                prop = convert_slot_to_property(attr_name, attr_def, all_classes)
                if prop:
                    properties[attr_name] = prop
                    required_flags[attr_name] = bool(attr_def.get("required"))

    required = [name for name, is_required in required_flags.items() if is_required]
    if required:
        schema["required"] = required

    if properties:
        schema["properties"] = properties

    return schema


def convert_slot_to_property(
    slot_name: str,
    slot_def: dict[str, Any],
    all_classes: set[str],
) -> dict[str, Any] | None:
    """Convert a LinkML slot/attribute to openMINDS property."""
    if not slot_def:
        return {"type": "string", "_instruction": f"Enter the {slot_name}."}

    range_type = slot_def.get("range")
    is_multivalued = slot_def.get("multivalued", False)
    is_inlined = slot_def.get("inlined", False) or slot_def.get("inlined_as_list", False)

    prop = linkml_range_to_openminds(
        range_type,
        is_multivalued=is_multivalued,
        is_inlined=is_inlined,
        all_classes=all_classes,
    )

    # Add instruction prop["_instruction"] = build_instruction(slot_name, slot_def)

    # Handle enums
    enum_values = slot_def.get("enum")
    if enum_values:
        prop["enum"] = enum_values

    # Handle array items for simple types
    if is_multivalued and "type" in prop and prop["type"] == "array":
        if range_type in LINKML_TO_JSON_TYPE:
            prop["items"] = {"type": LINKML_TO_JSON_TYPE[range_type]}
        elif "_linkedTypes" not in prop and "_embeddedTypes" not in prop:
            prop["items"] = {"type": "string"}

    return prop


def generate_openminds_schemas(
    linkml_schema: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Generate openMINDS schema files from LinkML schema."""
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = linkml_schema.get("classes", {}) or {}
    slots = linkml_schema.get("slots", {}) or {}

    # Get all class names for type resolution
    all_classes = set(classes.keys())

    generated: list[str] = []

    for class_name, class_def in classes.items():
        # Skip classes that map to external types
        if is_external_type(class_name):
            continue

        # Skip helper/internal classes
        if class_name in SKIP_CLASSES:
            continue

        if class_def is None:
            class_def = {}

        # Convert to openMINDS
        schema = convert_class_to_openminds(class_name, class_def, slots, all_classes, classes)

        # Generate filename
        filename = f"{camel_to_snake(class_name).replace('_', '')}.schema.tpl.json"
        # Actually use camelCase for filenames to match openMINDS convention
        filename = f"{class_name[0].lower()}{class_name[1:]}.schema.tpl.json"

        # Write schema
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")

        generated.append(filename)

    return generated


def update_readme(output_dir: Path, generated_files: list[str]) -> None:
    """Update README with list of generated schemas."""
    output_dir.parent / "README.md"

    # Count schemas by category
    categories = {
        "Dynamics": [],
        "Network": [],
        "Simulation": [],
        "Mathematical": [],
        "Data": [],
        "Other": [],
    }

    category_keywords = {
        "Dynamics": ["dynamics", "statevariable", "derivedvariable", "coupling", "noise"],
        "Network": ["network", "node", "edge", "tractogram", "parcellation", "regionmapping"],
        "Simulation": ["simulation", "integrator", "monitor", "stimulus", "processing"],
        "Mathematical": ["equation", "parameter", "function", "range", "conditional", "distribution"],
        "Data": ["timeseries", "software", "environment"],
    }

    for filename in generated_files:
        lower_name = filename.lower()
        categorized = False
        for cat, keywords in category_keywords.items():
            if any(kw in lower_name for kw in keywords):
                categories[cat].append(filename.replace(".schema.tpl.json", ""))
                categorized = True
                break
        if not categorized:
            categories["Other"].append(filename.replace(".schema.tpl.json", ""))

    print(f"\nGenerated {len(generated_files)} schemas:")
    for cat, schemas in categories.items():
        if schemas:
            print(f"  {cat}: {len(schemas)}")
            for s in sorted(schemas):
                print(f"    - {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate openMINDS schemas from LinkML")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent.parent / "tvbo_datamodel.yaml",
        help="Path to LinkML schema YAML",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "schemas",
        help="Output directory for generated schemas",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    args = parser.parse_args()

    print(f"Loading LinkML schema from: {args.input}")
    schema = load_linkml_schema(args.input)

    if args.dry_run:
        classes = schema.get("classes", {}) or {}
        print(f"\nWould generate schemas for {len(classes)} classes:")
        for name in sorted(classes.keys()):
            if name in SKIP_CLASSES:
                print(f"  SKIP: {name}")
            elif is_external_type(name):
                print(f"  EXTERNAL: {name} -> {EXTERNAL_TYPE_MAPPINGS[name]}")
            else:
                print(f"  GEN: {name}")
        return

    print(f"Writing schemas to: {args.output}")
    generated = generate_openminds_schemas(schema, args.output)

    update_readme(args.output, generated)
    print(f"\nDone! Generated {len(generated)} schema files.")


if __name__ == "__main__":
    main()
