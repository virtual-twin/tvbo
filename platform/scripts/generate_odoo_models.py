#!/usr/bin/env python3
"""
Generate Odoo model definitions from TVBO YAML schemas.
Converts LinkML/YAML schema classes to Odoo model.Model classes.

Usage:
    python scripts/generate_odoo_models.py
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Set
import re


# Models that are manually defined in database_models.py (do not auto-generate)
MANUAL_MODELS: Set[str] = set()

# Reserved Odoo field names that must be renamed
RESERVED_FIELD_NAMES = {
    "id": "record_id",  # Odoo uses 'id' as the primary key
    "modified": "is_modified",  # Odoo uses 'modified()' as a method
}

# Mapping of LinkML types to Odoo field types
TYPE_MAPPING = {
    "string": "Char",
    "integer": "Integer",
    "float": "Float",
    "boolean": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
}

# Special field mappings
SPECIAL_FIELDS = {
    "name": ("Char", {"required": True, "index": True}),
    "label": ("Char", {"index": True}),
    "description": ("Text", {}),
    "acronym": ("Char", {}),
}


def get_display_fields_for_model(
    class_name: str,
    class_def: Dict,
    all_slots: Dict,
    all_classes: Dict = None,
    include_relations: bool = False,
) -> List[str]:
    """
    Determine which fields to display in list/form views based on schema definition.
    Prioritizes common display fields like name, label, description.
    Handles inheritance by looking at parent classes.

    Args:
        class_name: Name of the class
        class_def: Class definition dictionary
        all_slots: All slot definitions
        all_classes: All class definitions (for inheritance)
        include_relations: If True, include Many2one and Many2many fields (for form views)
    """
    display_fields = []

    # Priority order for common fields
    priority_fields = ["name", "label", "title", "description", "acronym"]

    # Get all slots for this class
    class_slots = list(class_def.get("slots", []))

    # Get attributes - handle both dict and list formats
    raw_attributes = class_def.get("attributes", {})
    if isinstance(raw_attributes, list):
        # Convert list of dicts to dict (SANDS schema format)
        class_attributes = {}
        for attr in raw_attributes:
            if isinstance(attr, dict):
                for k, v in attr.items():
                    class_attributes[k] = v if isinstance(v, dict) else {}
            elif isinstance(attr, str):
                class_attributes[attr] = {}
    else:
        class_attributes = raw_attributes or {}

    # Handle inheritance - get parent slots and attributes
    if all_classes and "is_a" in class_def:
        parent_name = class_def["is_a"]
        parent_def = all_classes.get(parent_name)
        if parent_def:
            parent_slots = parent_def.get("slots", [])
            raw_parent_attrs = parent_def.get("attributes", {})
            if isinstance(raw_parent_attrs, list):
                parent_attributes = {}
                for attr in raw_parent_attrs:
                    if isinstance(attr, dict):
                        for k, v in attr.items():
                            parent_attributes[k] = v if isinstance(v, dict) else {}
                    elif isinstance(attr, str):
                        parent_attributes[attr] = {}
            else:
                parent_attributes = raw_parent_attrs or {}
            # Add parent slots to the beginning
            class_slots = list(parent_slots) + class_slots
            # Merge parent attributes
            merged_attributes = dict(parent_attributes)
            merged_attributes.update(class_attributes)
            class_attributes = merged_attributes

    # Check for _rec_name to use as primary field
    rec_name = class_def.get("_rec_name")
    if rec_name and rec_name in class_slots:
        if rec_name not in display_fields:
            display_fields.append(rec_name)

    # Add priority fields if they exist
    for field in priority_fields:
        if field in class_slots or field in class_attributes:
            if field not in display_fields:
                display_fields.append(field)

    # Add other fields
    simple_types = {"string", "integer", "float", "boolean", "text", "date", "datetime"}

    # Add remaining slots
    for slot_name in class_slots:
        if slot_name in display_fields:
            continue
        if slot_name in RESERVED_FIELD_NAMES:
            continue
        slot_def = all_slots.get(slot_name)
        if slot_def:
            slot_range = slot_def.get("range", "string")
            # For form views with relations, include all fields
            # For list views without relations, only include simple types
            if (
                include_relations
                or slot_range in simple_types
                or slot_range in TYPE_MAPPING
            ):
                display_fields.append(slot_name)

    # Add attributes - only include simple types that will definitely exist in Odoo model
    for attr_name, attr_def in class_attributes.items():
        if attr_name in display_fields:
            continue
        if attr_name in RESERVED_FIELD_NAMES:
            continue
        if not isinstance(attr_def, dict):
            continue
        attr_range = attr_def.get("range", "string")
        # Only include simple types that definitely map to Odoo fields
        # Exclude complex types (other classes) as they may not be properly created
        if attr_range in simple_types or attr_range in TYPE_MAPPING:
            display_fields.append(attr_name)
        elif include_relations and attr_range and attr_range not in simple_types:
            # For relations, only include if it's a known class that likely exists
            # Skip nested/complex types that may fail
            pass  # Be conservative - skip relations in auto-generated views

    # Ensure we have at least one field
    if not display_fields:
        # Fallback to first available slot or attribute
        if class_slots:
            display_fields.append(class_slots[0])
        elif class_attributes:
            display_fields.append(list(class_attributes.keys())[0])

    return display_fields  # Return all available fields


def generate_database_views(
    module_dir: Path, all_classes: Dict, all_slots: Dict
) -> None:
    """Generate database_views.xml and menus.xml for key models dynamically from schemas."""

    # Models we want to create views for with their known fields
    # Only include fields that definitely exist in the Odoo models
    target_models = {
        "neural_mass_model": {
            "display_name": "Neural Mass Models",
            "list_fields": ["name", "label", "number_of_modes"],
            "form_fields": [
                "name", "label", "description", "number_of_modes",
                "parameters", "state_variables", "derived_parameters",
                "derived_variables", "coupling_inputs", "coupling_terms",
                "system_type", "functions", "stimulus", "modes",
                "source", "references", "iri", "has_reference"
            ],
        },
        "integrator": {
            "display_name": "Integrators",
            "list_fields": ["method", "step_size", "duration", "delayed"],
            "form_fields": [
                "method", "step_size", "duration", "steps",
                "time_scale", "unit", "noise", "state_wise_sigma",
                "transient_time", "scipy_ode_base", "number_of_stages",
                "intermediate_expressions", "update_expression",
                "delayed", "parameters"
            ],
        },
        "network": {
            "display_name": "Connectivity Networks",
            "list_fields": ["label", "number_of_regions", "tractogram", "parcellation"],
            "form_fields": [
                "label", "description", "number_of_regions", "number_of_nodes",
                "tractogram", "parcellation", "nodes", "edges", "coupling",
                "weights", "lengths", "normalization", "node_labels",
                "global_coupling_strength", "conduction_speed"
            ],
        },
        "simulation_study": {
            "display_name": "Research Studies",
            "list_fields": ["label", "title", "year", "doi"],
            "form_fields": [
                "label", "title", "description", "year", "doi", "key",
                "model", "sample", "simulation_experiments"
            ],
        },
        "coupling": {
            "display_name": "Coupling Functions",
            "list_fields": ["name", "label", "sparse", "delayed"],
            "form_fields": [
                "name", "label", "parameters", "coupling_function",
                "sparse", "pre_expression", "post_expression",
                "incoming_states", "local_states", "delayed",
                "inner_coupling", "region_mapping", "regional_connectivity",
                "aggregation", "distribution"
            ],
        },
        "brain_atlas": {
            "display_name": "Brain Atlases",
            "list_fields": ["name", "abbreviation", "versionIdentifier"],
            "form_fields": [
                "name", "abbreviation", "author", "versionIdentifier",
                "isVersionOf", "coordinateSpace"
            ],
        },
        "tractogram": {
            "display_name": "Tractograms",
            "list_fields": ["name", "label", "number_of_subjects"],
            "form_fields": [
                "name", "label", "description",
                "data_source", "number_of_subjects",
                "acquisition", "processing_pipeline", "reference"
            ],
        },
        "parcellation": {
            "display_name": "Parcellations",
            "list_fields": ["label", "atlas", "data_source"],
            "form_fields": [
                "label", "atlas", "data_source",
                "region_labels", "center_coordinates"
            ],
        },
        "simulation_experiment": {
            "display_name": "Simulation Experiments",
            "list_fields": ["label", "model", "network"],
            "form_fields": [
                "label", "description", "record_id",
                "model", "local_dynamics", "dynamics",
                "integration", "connectivity", "network", "coupling",
                "monitors", "stimulation", "field_dynamics",
                "modelfitting", "environment", "software",
                "additional_equations", "references"
            ],
        },
        "monitor": {
            "display_name": "Monitors",
            "list_fields": ["name", "label", "period"],
            "form_fields": [
                "name", "label", "acronym", "description",
                "time_scale", "parameters", "equation",
                "environment", "period", "imaging_modality"
            ],
        },
        "dynamics": {
            "display_name": "Dynamics",
            "list_fields": ["name", "label", "number_of_modes"],
            "form_fields": [
                "name", "label", "description", "number_of_modes",
                "parameters", "state_variables", "derived_parameters",
                "derived_variables", "coupling_inputs", "coupling_terms",
                "system_type", "functions", "stimulus", "modes",
                "source", "references", "iri", "has_reference",
                "is_modified", "output", "derived_from_model",
                "local_coupling_term"
            ],
        },
    }

    views_content = ['<?xml version="1.0" encoding="utf-8"?>', "<odoo>"]
    menu_content = ['<?xml version="1.0" encoding="utf-8"?>', "<odoo>"]

    # Main menu
    menu_content.extend(
        [
            "    <!-- Main TVBO Menu -->",
            '    <menuitem id="menu_tvbo_root"',
            '              name="TVBO Database"',
            '              sequence="10"',
            '              groups="base.group_user"',
            '              web_icon="tvbo,static/description/icon.png"/>',
            "",
            "    <!-- Database submenu -->",
            '    <menuitem id="menu_tvbo_database"',
            '              name="Browse Database"',
            '              parent="menu_tvbo_root"',
            '              sequence="10"/>',
            "",
        ]
    )

    sequence = 10
    for model_name, model_config in target_models.items():
        display_name = model_config["display_name"]
        list_fields = model_config["list_fields"]
        form_fields = model_config["form_fields"]

        model_tech = f"tvbo.{model_name}"
        action_id = f"action_{model_name}_list"

        # Generate list view (simple fields only)
        views_content.extend(
            [
                f"    <!-- {display_name} Views -->",
                f'    <record id="view_{model_name}_list" model="ir.ui.view">',
                f'        <field name="name">{model_tech}.list</field>',
                f'        <field name="model">{model_tech}</field>',
                '        <field name="arch" type="xml">',
                f'            <list string="{display_name}">',
            ]
        )

        for field in list_fields:  # Show simple fields in list view
            views_content.append(f'                <field name="{field}"/>')

        views_content.extend(
            ["            </list>", "        </field>", "    </record>", ""]
        )

        # Generate form view (ALL fields including relations)
        views_content.extend(
            [
                f'    <record id="view_{model_name}_form" model="ir.ui.view">',
                f'        <field name="name">{model_tech}.form</field>',
                f'        <field name="model">{model_tech}</field>',
                '        <field name="arch" type="xml">',
                f'            <form string="{display_name}">',
                "                <sheet>",
                "                    <group>",
            ]
        )

        for field in form_fields:  # ALL fields in form view
            views_content.append(f'                        <field name="{field}"/>')

        views_content.extend(
            [
                "                    </group>",
                "                </sheet>",
                "            </form>",
                "        </field>",
                "    </record>",
                "",
            ]
        )

        # Generate action
        views_content.extend(
            [
                f'    <record id="{action_id}" model="ir.actions.act_window">',
                f'        <field name="name">{display_name}</field>',
                f'        <field name="res_model">{model_tech}</field>',
                '        <field name="view_mode">list,form</field>',
                "    </record>",
                "",
            ]
        )

        # Generate menu item
        menu_content.extend(
            [
                f"    <!-- {display_name} -->",
                f'    <menuitem id="menu_{model_name}"',
                f'              name="{display_name}"',
                '              parent="menu_tvbo_database"',
                f'              action="{action_id}"',
                f'              sequence="{sequence}"/>',
                "",
            ]
        )

        sequence += 10

    views_content.append("</odoo>")
    menu_content.append("</odoo>")

    # Write files
    (module_dir / "views").mkdir(exist_ok=True)
    (module_dir / "views" / "database_views.xml").write_text("\n".join(views_content))
    (module_dir / "views" / "menus.xml").write_text("\n".join(menu_content))


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_odoo_field_type(
    linkml_type: str, is_multivalued: bool = False, range_class: str = None
) -> tuple:
    """
    Determine Odoo field type from LinkML type.
    Returns (field_type, field_options_dict)
    """
    if is_multivalued:
        if range_class:
            # Use Many2many instead of One2many to avoid inverse field requirements
            return (
                "Many2many",
                {"comodel_name": f"tvbo.{camel_to_snake(range_class)}"},
            )
        return ("Text", {})  # Fallback for arrays

    if range_class:
        # Reference to another model
        return ("Many2one", {"comodel_name": f"tvbo.{camel_to_snake(range_class)}"})

    odoo_type = TYPE_MAPPING.get(linkml_type, "Char")
    return (odoo_type, {})


def generate_field_definition(
    attr_name: str, attr_def: Dict[str, Any], model_name: str = ""
) -> str:
    """Generate a single Odoo field definition."""
    # Rename reserved field names
    original_name = attr_name
    if attr_name in RESERVED_FIELD_NAMES:
        attr_name = RESERVED_FIELD_NAMES[attr_name]

    # Check if this is a special field
    if original_name in SPECIAL_FIELDS:
        field_type, options = SPECIAL_FIELDS[original_name]
    else:
        linkml_range = attr_def.get("range", "string")
        is_multivalued = attr_def.get("multivalued", False)

        field_type, options = get_odoo_field_type(
            linkml_range,
            is_multivalued,
            linkml_range if linkml_range not in TYPE_MAPPING else None,
        )

        # For Many2many fields, add unique relation table name
        if field_type == "Many2many" and model_name:
            comodel = options["comodel_name"].replace("tvbo.", "")
            # Shorten to avoid PostgreSQL 63 char limit
            # Use abbreviated format: tvbo_<model>_<field>_rel
            relation_table = f"tvbo_{model_name}_{attr_name}_rel"
            # If still too long, truncate model and field names
            if len(relation_table) > 63:
                max_len = 63 - 10  # Reserve space for tvbo_ and _rel
                model_part = model_name[: max_len // 2]
                field_part = attr_name[: max_len // 2]
                relation_table = f"tvbo_{model_part}_{field_part}_rel"
            options["relation"] = relation_table

            # Handle self-referential Many2many (model references itself)
            if comodel == model_name:
                # Use different column names for self-referential relations
                options["column1"] = f"{model_name}_id"
                options["column2"] = f"{attr_name}_id"

    # Add description if available
    if "description" in attr_def:
        options["string"] = attr_def["description"].replace("\n", " ").strip()
    elif original_name != attr_name:
        # Add note about renaming for reserved fields
        options["string"] = f"{original_name.upper()} (renamed from '{original_name}')"

    # Add required constraint
    if attr_def.get("required"):
        options["required"] = True

    # Add default value
    if "ifabsent" in attr_def:
        default_val = attr_def["ifabsent"]
        if isinstance(default_val, str):
            if default_val.startswith("string("):
                options["default"] = default_val[7:-1]
            elif default_val.startswith("float("):
                options["default"] = float(default_val[6:-1])
            elif default_val.startswith("integer("):
                options["default"] = int(default_val[8:-1])

    # Format options
    options_str = ", ".join([f"{k}={repr(v)}" for k, v in options.items()])

    return f"    {attr_name} = fields.{field_type}({options_str})"


def is_enum_class(class_def: Dict[str, Any]) -> bool:
    """Check if a class definition is an enum (has permissible_values)."""
    return "permissible_values" in class_def


def generate_enum_model(class_name: str, class_def: Dict[str, Any]) -> str:
    """Generate Odoo model for an enum type."""
    model_name = camel_to_snake(class_name)
    description = class_def.get("description", class_name)

    lines = [
        f"class {class_name}(models.Model):",
        f"    _name = 'tvbo.{model_name}'",
        f"    _description = '{description}'",
        "    _rec_name = 'name'",
        "",
        "    name = fields.Char(required=True, index=True)",
        "    technical_name = fields.Char(required=True, index=True)",
        "    description = fields.Text()",
        "",
    ]

    return "\n".join(lines)


def generate_model_class(
    class_name: str,
    class_def: Dict[str, Any],
    schema_name: str,
    all_classes: Dict[str, Any] = {},
    all_slots: Dict[str, Any] = {},
) -> str:
    """Generate complete Odoo model class definition."""
    # Handle enum types
    if is_enum_class(class_def):
        return generate_enum_model(class_name, class_def)

    model_name = camel_to_snake(class_name)

    lines = [
        f"class {class_name}(models.Model):",
        f"    _name = 'tvbo.{model_name}'",
        f"    _description = '{class_def.get('description', class_name)}'",
        "",
    ]

    # Handle inheritance via is_a using _inherits (delegation inheritance)
    parent_class = class_def.get("is_a")
    parent_attributes = {}
    if parent_class:
        parent_model = camel_to_snake(parent_class)
        # Use _inherits for delegation inheritance - creates separate table
        lines.append(f"    _inherits = {{'tvbo.{parent_model}': '{parent_model}_id'}}")
        lines.append("")

        # Get parent class attributes to skip them
        if all_classes and parent_class in all_classes:
            parent_attributes = all_classes[parent_class].get("attributes", {})

    # Add _rec_name if 'name' or 'label' exists
    attributes = class_def.get("attributes", {})
    if isinstance(attributes, list):
        # Handle case where attributes is a list instead of dict
        attributes = {}
    slots = class_def.get("slots", [])

    # Only add _rec_name if not inheriting (parent will have it through _inherits)
    if not parent_class:
        if "name" in attributes or "name" in slots:
            lines.append("    _rec_name = 'name'")
        elif "label" in attributes or "label" in slots:
            lines.append("    _rec_name = 'label'")
        lines.append("")

    # Add parent reference field if using _inherits
    if parent_class:
        parent_model = camel_to_snake(parent_class)
        lines.append(
            f"    {parent_model}_id = fields.Many2one('tvbo.{parent_model}', required=True, ondelete='cascade')"
        )
        lines.append("")

    # Generate fields from slots (skip parent slots)
    for slot in slots:
        if slot in SPECIAL_FIELDS:
            field_type, options = SPECIAL_FIELDS[slot]
            options_str = ", ".join([f"{k}={repr(v)}" for k, v in options.items()])
            lines.append(f"    {slot} = fields.{field_type}({options_str})")
        elif slot in all_slots:
            # Process slot using its schema definition
            slot_def = all_slots[slot]
            if slot_def is not None and isinstance(slot_def, dict):
                lines.append(generate_field_definition(slot, slot_def, model_name))

    # Generate fields from attributes (skip parent attributes)
    for attr_name, attr_def in attributes.items():
        if (
            attr_name not in slots and attr_name not in parent_attributes
        ):  # Avoid duplicates and inherited fields
            lines.append(generate_field_definition(attr_name, attr_def, model_name))

    lines.append("")
    return "\n".join(lines)


def generate_enum_data_xml(class_name: str, class_def: Dict[str, Any]) -> str:
    """Generate XML data file for enum values."""
    model_name = camel_to_snake(class_name)
    permissible_values = class_def.get("permissible_values", {})

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<odoo>",
        f'    <data noupdate="1">',
        "",
    ]

    for idx, (value_name, value_def) in enumerate(permissible_values.items(), 1):
        record_id = f"{model_name}_{value_name}"
        description = (
            value_def.get("description", "") if isinstance(value_def, dict) else ""
        )

        lines.extend(
            [
                f'        <record id="{record_id}" model="tvbo.{model_name}">',
                f'            <field name="name">{value_name.replace("_", " ").title()}</field>',
                f'            <field name="technical_name">{value_name}</field>',
            ]
        )

        if description:
            lines.append(f'            <field name="description">{description}</field>')

        lines.append("        </record>")
        lines.append("")

    lines.extend(["    </data>", "</odoo>"])

    return "\n".join(lines)


def generate_odoo_module(schema_path: Path, output_dir: Path):
    """Generate complete Odoo module from YAML schema."""
    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    schema_name = schema.get("name", schema_path.stem)
    module_name = f"tvbo_{schema_name}"
    module_dir = output_dir / module_name

    # Create module structure
    module_dir.mkdir(exist_ok=True)
    (module_dir / "models").mkdir(exist_ok=True)
    (module_dir / "security").mkdir(exist_ok=True)
    (module_dir / "views").mkdir(exist_ok=True)

    # Generate __manifest__.py
    manifest = {
        "name": schema.get("title", schema_name),
        "version": "19.0.1.0.0",
        "category": "Research/Neuroscience",
        "summary": schema.get("description", ""),
        "author": "Charité Universitätsmedizin Berlin",
        "website": "https://github.com/virtual-twin/tvbo",
        "depends": ["base"],
        "data": [
            "security/ir.model.access.csv",
        ],
        "installable": True,
        "application": True,
        "auto_install": False,
    }

    manifest_content = "# -*- coding: utf-8 -*-\n" + str(manifest)
    (module_dir / "__manifest__.py").write_text(manifest_content)

    # Generate __init__.py files
    (module_dir / "__init__.py").write_text("from . import models\n")
    (module_dir / "models" / "__init__.py").write_text("from . import models\n")

    # Generate models.py
    models_content = [
        "# -*- coding: utf-8 -*-",
        f"# Generated from {schema_path.name}",
        "from odoo import models, fields, api",
        "",
        "",
    ]

    classes = schema.get("classes", {})
    for class_name, class_def in classes.items():
        models_content.append(generate_model_class(class_name, class_def, schema_name))
        models_content.append("")

    (module_dir / "models" / "models.py").write_text("\n".join(models_content))

    # Generate security file
    security_lines = [
        "id,name,model_id:id,group_id:id,perm_read,perm_write,perm_create,perm_unlink"
    ]

    for class_name in classes.keys():
        model_name = camel_to_snake(class_name)
        access_id = f"access_tvbo_{model_name}"
        model_id = f"model_tvbo_{model_name}"
        security_lines.append(
            f"{access_id},tvbo.{model_name},tvbo_{schema_name}.{model_id},base.group_user,1,1,1,1"
        )

    (module_dir / "security" / "ir.model.access.csv").write_text(
        "\n".join(security_lines)
    )

    print(f"✓ Generated Odoo module: {module_dir}")
    return module_dir


def main():
    """Main entry point."""
    # Resolve repository root (two levels above this script: platform/scripts -> platform -> repo)
    project_root = Path(__file__).resolve().parents[2]
    schema_dir = project_root / "schema"
    # Odoo module lives under platform/odoo-addons
    output_dir = project_root / "platform" / "odoo-addons"

    output_dir.mkdir(exist_ok=True)

    print("Generating unified Odoo module from TVBO schemas...")
    print()

    # Create single module structure
    module_name = "tvbo"
    module_dir = output_dir / module_name
    module_dir.mkdir(exist_ok=True)
    (module_dir / "models").mkdir(exist_ok=True)
    (module_dir / "security").mkdir(exist_ok=True)
    (module_dir / "views").mkdir(exist_ok=True)

    # Generate __manifest__.py
    manifest = {
        "name": "TVBO - The Virtual Brain Ontology",
        "version": "19.0.1.0.0",
        "category": "Research/Neuroscience",
        "summary": "Data models for brain simulation, DBS, and neuroanatomical atlases",
        "description": """
            Complete TVBO data models including:
            - Brain simulation dynamics and experiments
            - Deep Brain Stimulation (DBS) protocols
            - Neuroanatomical atlases and parcellations
        """,
        "author": "Charité Universitätsmedizin Berlin",
        "website": "https://github.com/virtual-twin/tvbo",
        "license": "LGPL-3",
        "depends": ["base", "website"],
        "data": [
            "security/ir.model.access.csv",
        ],
        "installable": True,
        "application": True,
        "auto_install": False,
    }

    manifest_content = "# -*- coding: utf-8 -*-\n" + str(manifest)
    (module_dir / "__manifest__.py").write_text(manifest_content)

    # Generate __init__.py files
    (module_dir / "__init__.py").write_text(
        "from . import models\nfrom . import controllers\n"
    )
    (module_dir / "models" / "__init__.py").write_text("from . import models\n")

    # Collect all classes from all schemas
    all_classes = {}
    all_enums = {}
    all_slots = {}  # Collect all slot definitions
    schemas = ["tvbo_datamodel.yaml", "tvb_dbs.yaml", "SANDS.yaml"]

    for schema_file in schemas:
        schema_path = schema_dir / schema_file
        if schema_path.exists():
            try:
                with open(schema_path) as f:
                    schema = yaml.safe_load(f)
                classes = schema.get("classes", {})
                enums = schema.get("enums", {})
                slots = schema.get("slots", {})
                all_classes.update(classes)
                all_enums.update(enums)
                all_slots.update(slots)
                print(
                    f"✓ Loaded {len(classes)} classes, {len(enums)} enums, and {len(slots)} slots from {schema_file}"
                )
            except Exception as e:
                print(f"✗ Error loading {schema_file}: {e}")

    # Filter out manually defined models
    auto_classes = {k: v for k, v in all_classes.items() if k not in MANUAL_MODELS}
    skipped = [k for k in all_classes.keys() if k in MANUAL_MODELS]

    if skipped:
        print(
            f"ℹ Skipping {len(skipped)} manually-defined models: {', '.join(skipped)}"
        )

    # Generate schema_models.py with auto-generated classes only
    models_content = [
        "# -*- coding: utf-8 -*-",
        "# Auto-generated from TVBO schemas - DO NOT EDIT MANUALLY",
        "# Re-run scripts/generate_odoo_models.py to update",
        "from odoo import models, fields, api",
        "",
        "",
    ]

    enum_classes = []
    regular_classes = []

    # Process enums from the 'enums' section
    for enum_name, enum_def in all_enums.items():
        enum_classes.append((enum_name, enum_def))

    # Process regular classes
    for class_name, class_def in auto_classes.items():
        regular_classes.append((class_name, class_def))

    # Generate enum models first
    for class_name, class_def in enum_classes:
        try:
            models_content.append(
                generate_model_class(
                    class_name, class_def, "tvbo", all_classes, all_slots
                )
            )
            models_content.append("")
        except Exception as e:
            print(f"✗ Error generating enum class {class_name}: {e}")

    # Then generate regular models
    for class_name, class_def in regular_classes:
        try:
            models_content.append(
                generate_model_class(
                    class_name, class_def, "tvbo", all_classes, all_slots
                )
            )
            models_content.append("")
        except Exception as e:
            print(f"✗ Error generating class {class_name}: {e}")

    (module_dir / "models" / "schema_models.py").write_text("\n".join(models_content))

    # Update models/__init__.py to import schema models only
    (module_dir / "models" / "__init__.py").write_text(
        "from . import schema_models\nfrom . import literature\n"
    )

    # Generate data XML files for enum values
    (module_dir / "data").mkdir(exist_ok=True)
    data_files = []

    for class_name, class_def in enum_classes:
        try:
            model_name = camel_to_snake(class_name)
            xml_filename = f"data_{model_name}.xml"
            xml_content = generate_enum_data_xml(class_name, class_def)
            (module_dir / "data" / xml_filename).write_text(xml_content)
            data_files.append(f"data/{xml_filename}")
        except Exception as e:
            print(f"✗ Error generating data for enum {class_name}: {e}")

    # Generate database views and menus
    generate_database_views(module_dir, all_classes, all_slots)

    # Check for database XML files
    database_files = []
    for db_file in [
        "database_neural_models.xml",
        "database_julia_dynamics.xml",
        "database_integrators.xml",
        "database_networks.xml",
        "database_studies.xml",
        "database_coupling_functions.xml",
    ]:
        if (module_dir / "data" / db_file).exists():
            database_files.append(f"data/{db_file}")

    # Check for website files
    website_files = []
    if (module_dir / "data" / "website_config.xml").exists():
        website_files.append("data/website_config.xml")
    if (module_dir / "views" / "website_templates.xml").exists():
        website_files.append("views/website_templates.xml")
    # Manually created template files - preserve these
    if (module_dir / "views" / "configurator_templates.xml").exists():
        website_files.append("views/configurator_templates.xml")

    # Check for view and menu files (auto-generated)
    view_files = []
    if (module_dir / "views" / "database_views.xml").exists():
        view_files.append("views/database_views.xml")
    if (module_dir / "views" / "menus.xml").exists():
        view_files.append("views/menus.xml")
    if (module_dir / "views" / "literature_views.xml").exists():
        view_files.append("views/literature_views.xml")

    # Update manifest with all data files
    manifest["data"] = (
        ["security/ir.model.access.csv"]
        + data_files
        + database_files
        + website_files
        + view_files
    )
    manifest_content = "# -*- coding: utf-8 -*-\n" + str(manifest)
    (module_dir / "__manifest__.py").write_text(manifest_content)

    # Generate security file for all auto-generated classes
    security_lines = [
        "id,name,model_id:id,group_id:id,perm_read,perm_write,perm_create,perm_unlink"
    ]

    # Add enum models
    for enum_name, _ in enum_classes:
        model_name = camel_to_snake(enum_name)
        access_id = f"access_tvbo_{model_name}"
        model_id = f"model_tvbo_{model_name}"
        security_lines.append(
            f"{access_id},tvbo.{model_name},{model_id},base.group_user,1,1,1,1"
        )

    # Add auto-generated models
    for class_name in auto_classes.keys():
        model_name = camel_to_snake(class_name)
        access_id = f"access_tvbo_{model_name}"
        model_id = f"model_tvbo_{model_name}"
        security_lines.append(
            f"{access_id},tvbo.{model_name},{model_id},base.group_user,1,1,1,1"
        )

    # Custom models not generated from schemas
    for model_name in ["mesh_term", "literature_reference"]:
        access_id = f"access_tvbo_{model_name}"
        model_id = f"model_tvbo_{model_name}"
        line = f"{access_id},tvbo.{model_name},{model_id},base.group_user,1,1,1,1"
        if line not in security_lines:
            security_lines.append(line)

    (module_dir / "security" / "ir.model.access.csv").write_text(
        "\n".join(security_lines)
    )

    print()
    print(f"✓ Generated unified Odoo module: {module_dir}")
    print(f"  Enum models: {len(enum_classes)}")
    print(f"  Regular models: {len(regular_classes)}")
    print(f"  Total models: {len(enum_classes) + len(regular_classes)}")
    print(f"  Enum data files: {len(data_files)}")
    print(f"  Database files: {len(database_files)}")
    print(f"  Website files: {len(website_files)}")
    print(
        f"  Total data files in manifest: {len(data_files) + len(database_files) + len(website_files) + 1}"
    )  # +1 for security
    print()
    print("To install in Odoo:")
    print("1. Module is already in ./odoo-addons/tvbo")
    print("2. Start with: docker-compose up -d")
    print("3. Access Odoo at http://localhost:8069")
    print("4. Update app list and install 'TVBO' module")


if __name__ == "__main__":
    main()
