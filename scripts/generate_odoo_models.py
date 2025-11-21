#!/usr/bin/env python3
"""
Generate Odoo model definitions from TVBO YAML schemas.
Converts LinkML/YAML schema classes to Odoo model.Model classes.

Usage:
    python scripts/generate_odoo_models.py
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
import re


# Mapping of LinkML types to Odoo field types
TYPE_MAPPING = {
    'string': 'Char',
    'integer': 'Integer',
    'float': 'Float',
    'boolean': 'Boolean',
    'date': 'Date',
    'datetime': 'Datetime',
}

# Special field mappings
SPECIAL_FIELDS = {
    'name': ('Char', {'required': True, 'index': True}),
    'label': ('Char', {'index': True}),
    'description': ('Text', {}),
    'acronym': ('Char', {}),
}


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def get_odoo_field_type(linkml_type: str, is_multivalued: bool = False,
                        range_class: str = None) -> tuple:
    """
    Determine Odoo field type from LinkML type.
    Returns (field_type, field_options_dict)
    """
    if is_multivalued:
        if range_class:
            return ('One2many', {'comodel_name': f'tvbo.{camel_to_snake(range_class)}'})
        return ('Text', {})  # Fallback for arrays

    if range_class:
        # Reference to another model
        return ('Many2one', {'comodel_name': f'tvbo.{camel_to_snake(range_class)}'})

    odoo_type = TYPE_MAPPING.get(linkml_type, 'Char')
    return (odoo_type, {})


def generate_field_definition(attr_name: str, attr_def: Dict[str, Any]) -> str:
    """Generate a single Odoo field definition."""
    # Check if this is a special field
    if attr_name in SPECIAL_FIELDS:
        field_type, options = SPECIAL_FIELDS[attr_name]
    else:
        linkml_range = attr_def.get('range', 'string')
        is_multivalued = attr_def.get('multivalued', False)

        field_type, options = get_odoo_field_type(
            linkml_range,
            is_multivalued,
            linkml_range if linkml_range not in TYPE_MAPPING else None
        )

    # Add description if available
    if 'description' in attr_def:
        options['string'] = attr_def['description'].replace('\n', ' ').strip()

    # Add required constraint
    if attr_def.get('required'):
        options['required'] = True

    # Add default value
    if 'ifabsent' in attr_def:
        default_val = attr_def['ifabsent']
        if isinstance(default_val, str):
            if default_val.startswith('string('):
                options['default'] = default_val[7:-1]
            elif default_val.startswith('float('):
                options['default'] = float(default_val[6:-1])
            elif default_val.startswith('integer('):
                options['default'] = int(default_val[8:-1])

    # Format options
    options_str = ', '.join([f"{k}={repr(v)}" for k, v in options.items()])

    return f"    {attr_name} = fields.{field_type}({options_str})"


def generate_model_class(class_name: str, class_def: Dict[str, Any],
                        schema_name: str) -> str:
    """Generate complete Odoo model class definition."""
    model_name = camel_to_snake(class_name)

    lines = [
        f"class {class_name}(models.Model):",
        f"    _name = 'tvbo.{model_name}'",
        f"    _description = '{class_def.get('description', class_name)}'",
        ""
    ]

    # Add _rec_name if 'name' or 'label' exists
    attributes = class_def.get('attributes', {})
    if isinstance(attributes, list):
        # Handle case where attributes is a list instead of dict
        attributes = {}
    slots = class_def.get('slots', [])

    if 'name' in attributes or 'name' in slots:
        lines.append("    _rec_name = 'name'")
    elif 'label' in attributes or 'label' in slots:
        lines.append("    _rec_name = 'label'")

    lines.append("")

    # Generate fields from slots
    for slot in slots:
        if slot in SPECIAL_FIELDS:
            field_type, options = SPECIAL_FIELDS[slot]
            options_str = ', '.join([f"{k}={repr(v)}" for k, v in options.items()])
            lines.append(f"    {slot} = fields.{field_type}({options_str})")

    # Generate fields from attributes
    for attr_name, attr_def in attributes.items():
        if attr_name not in slots:  # Avoid duplicates
            lines.append(generate_field_definition(attr_name, attr_def))

    lines.append("")
    return '\n'.join(lines)


def generate_odoo_module(schema_path: Path, output_dir: Path):
    """Generate complete Odoo module from YAML schema."""
    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    schema_name = schema.get('name', schema_path.stem)
    module_name = f"tvbo_{schema_name}"
    module_dir = output_dir / module_name

    # Create module structure
    module_dir.mkdir(exist_ok=True)
    (module_dir / 'models').mkdir(exist_ok=True)
    (module_dir / 'security').mkdir(exist_ok=True)
    (module_dir / 'views').mkdir(exist_ok=True)

    # Generate __manifest__.py
    manifest = {
        'name': schema.get('title', schema_name),
        'version': '19.0.1.0.0',
        'category': 'Research/Neuroscience',
        'summary': schema.get('description', ''),
        'author': 'Charité Universitätsmedizin Berlin',
        'website': 'https://github.com/virtual-twin/tvbo',
        'license': 'EUPL-1.2',
        'depends': ['base'],
        'data': [
            'security/ir.model.access.csv',
        ],
        'installable': True,
        'application': True,
        'auto_install': False,
    }

    manifest_content = "# -*- coding: utf-8 -*-\n" + str(manifest)
    (module_dir / '__manifest__.py').write_text(manifest_content)

    # Generate __init__.py files
    (module_dir / '__init__.py').write_text("from . import models\n")
    (module_dir / 'models' / '__init__.py').write_text("from . import models\n")

    # Generate models.py
    models_content = [
        "# -*- coding: utf-8 -*-",
        f"# Generated from {schema_path.name}",
        "from odoo import models, fields, api",
        "",
        ""
    ]

    classes = schema.get('classes', {})
    for class_name, class_def in classes.items():
        models_content.append(generate_model_class(class_name, class_def, schema_name))
        models_content.append("")

    (module_dir / 'models' / 'models.py').write_text('\n'.join(models_content))

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

    (module_dir / 'security' / 'ir.model.access.csv').write_text('\n'.join(security_lines))

    print(f"✓ Generated Odoo module: {module_dir}")
    return module_dir


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    schema_dir = project_root / 'schema'
    output_dir = project_root / 'odoo-addons'

    output_dir.mkdir(exist_ok=True)

    print("Generating unified Odoo module from TVBO schemas...")
    print()

    # Create single module structure
    module_name = "tvbo"
    module_dir = output_dir / module_name
    module_dir.mkdir(exist_ok=True)
    (module_dir / 'models').mkdir(exist_ok=True)
    (module_dir / 'security').mkdir(exist_ok=True)
    (module_dir / 'views').mkdir(exist_ok=True)

    # Generate __manifest__.py
    manifest = {
        'name': 'TVBO - The Virtual Brain Ontology',
        'version': '19.0.1.0.0',
        'category': 'Research/Neuroscience',
        'summary': 'Data models for brain simulation, DBS, and neuroanatomical atlases',
        'description': '''
            Complete TVBO data models including:
            - Brain simulation dynamics and experiments
            - Deep Brain Stimulation (DBS) protocols
            - Neuroanatomical atlases and parcellations
        ''',
        'author': 'Charité Universitätsmedizin Berlin',
        'website': 'https://github.com/virtual-twin/tvbo',
        'license': 'EUPL-1.2',
        'depends': ['base'],
        'data': [
            'security/ir.model.access.csv',
        ],
        'installable': True,
        'application': True,
        'auto_install': False,
    }

    manifest_content = "# -*- coding: utf-8 -*-\n" + str(manifest)
    (module_dir / '__manifest__.py').write_text(manifest_content)

    # Generate __init__.py files
    (module_dir / '__init__.py').write_text("from . import models\n")
    (module_dir / 'models' / '__init__.py').write_text("from . import models\n")

    # Collect all classes from all schemas
    all_classes = {}
    schemas = [
        'tvbo_datamodel.yaml',
        'tvb_dbs.yaml',
        'SANDS.yaml'
    ]

    for schema_file in schemas:
        schema_path = schema_dir / schema_file
        if schema_path.exists():
            try:
                with open(schema_path) as f:
                    schema = yaml.safe_load(f)
                classes = schema.get('classes', {})
                all_classes.update(classes)
                print(f"✓ Loaded {len(classes)} classes from {schema_file}")
            except Exception as e:
                print(f"✗ Error loading {schema_file}: {e}")

    # Generate single models.py with all classes
    models_content = [
        "# -*- coding: utf-8 -*-",
        "# Generated from TVBO schemas",
        "from odoo import models, fields, api",
        "",
        ""
    ]

    for class_name, class_def in all_classes.items():
        try:
            models_content.append(generate_model_class(class_name, class_def, 'tvbo'))
            models_content.append("")
        except Exception as e:
            print(f"✗ Error generating class {class_name}: {e}")

    (module_dir / 'models' / 'models.py').write_text('\n'.join(models_content))

    # Generate security file for all classes
    security_lines = [
        "id,name,model_id:id,group_id:id,perm_read,perm_write,perm_create,perm_unlink"
    ]

    for class_name in all_classes.keys():
        model_name = camel_to_snake(class_name)
        access_id = f"access_tvbo_{model_name}"
        model_id = f"model_tvbo_{model_name}"
        security_lines.append(
            f"{access_id},tvbo.{model_name},{model_id},base.group_user,1,1,1,1"
        )

    (module_dir / 'security' / 'ir.model.access.csv').write_text('\n'.join(security_lines))

    print()
    print(f"✓ Generated unified Odoo module: {module_dir}")
    print(f"  Total models: {len(all_classes)}")
    print()
    print("To install in Odoo:")
    print("1. Module is already in ./odoo-addons/tvbo")
    print("2. Start with: docker-compose up -d")
    print("3. Access Odoo at http://localhost:8069")
    print("4. Update app list and install 'TVBO' module")


if __name__ == '__main__':
    main()
