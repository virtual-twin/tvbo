#!/usr/bin/env python3
"""
Import TVBO database YAML files into Odoo XML data format.
Converts models, integrators, networks, and studies to Odoo data files.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")


def create_neural_model_data(yaml_files: List[Path]) -> ET.Element:
    """Create XML data for neural models."""
    odoo = ET.Element('odoo')
    data = ET.SubElement(odoo, 'data', noupdate="1")

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            model_data = yaml.safe_load(f)

        model_name = model_data.get('name', yaml_file.stem)
        model_id = f"neural_model_{model_name.lower().replace(' ', '_')}"

        # Create model record
        record = ET.SubElement(data, 'record', {
            'id': model_id,
            'model': 'tvbo.neural_model'
        })

        ET.SubElement(record, 'field', {'name': 'name'}).text = model_name

        description = model_data.get('description', '')
        if description:
            ET.SubElement(record, 'field', {'name': 'description'}).text = description

        num_modes = model_data.get('number_of_modes', 1)
        ET.SubElement(record, 'field', {'name': 'number_of_modes'}).text = str(num_modes)

        # Add parameters
        parameters = model_data.get('parameters', {})
        for param_name, param_def in parameters.items():
            param_record = ET.SubElement(data, 'record', {
                'id': f"{model_id}_param_{param_name}",
                'model': 'tvbo.model_parameter'
            })
            ET.SubElement(param_record, 'field', {'name': 'neural_model_id', 'ref': model_id})
            ET.SubElement(param_record, 'field', {'name': 'name'}).text = param_name

            if isinstance(param_def, dict):
                if 'value' in param_def:
                    ET.SubElement(param_record, 'field', {'name': 'value'}).text = str(param_def['value'])
                if 'description' in param_def:
                    ET.SubElement(param_record, 'field', {'name': 'description'}).text = param_def['description']
                if 'unit' in param_def:
                    ET.SubElement(param_record, 'field', {'name': 'unit'}).text = param_def['unit']

        # Add state variables
        state_vars = model_data.get('state_variables', {})
        for var_name, var_def in state_vars.items():
            var_record = ET.SubElement(data, 'record', {
                'id': f"{model_id}_var_{var_name}",
                'model': 'tvbo.state_variable'
            })
            ET.SubElement(var_record, 'field', {'name': 'neural_model_id', 'ref': model_id})
            ET.SubElement(var_record, 'field', {'name': 'name'}).text = var_name

            if isinstance(var_def, dict):
                if 'description' in var_def:
                    ET.SubElement(var_record, 'field', {'name': 'description'}).text = var_def['description']
                if 'initial_value' in var_def:
                    ET.SubElement(var_record, 'field', {'name': 'initial_value'}).text = str(var_def['initial_value'])

    return odoo


def create_integrator_data(yaml_files: List[Path]) -> ET.Element:
    """Create XML data for integrators."""
    odoo = ET.Element('odoo')
    data = ET.SubElement(odoo, 'data', noupdate="1")

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            integrator_data = yaml.safe_load(f)

        integrator_name = yaml_file.stem
        integrator_id = f"integrator_{integrator_name.lower()}"

        record = ET.SubElement(data, 'record', {
            'id': integrator_id,
            'model': 'tvbo.integrator'
        })

        ET.SubElement(record, 'field', {'name': 'name'}).text = integrator_name

        if 'method' in integrator_data:
            ET.SubElement(record, 'field', {'name': 'method'}).text = integrator_data['method']

        if 'step_size' in integrator_data:
            ET.SubElement(record, 'field', {'name': 'step_size'}).text = str(integrator_data['step_size'])

        if 'duration' in integrator_data:
            ET.SubElement(record, 'field', {'name': 'duration'}).text = str(integrator_data['duration'])

    return odoo


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    database_dir = project_root / 'database'
    output_dir = project_root / 'odoo-addons' / 'tvbo' / 'data'

    output_dir.mkdir(exist_ok=True)

    print("Generating Odoo data files from TVBO database...")

    # Process neural models
    models_dir = database_dir / 'models'
    if models_dir.exists():
        model_files = list(models_dir.glob('*.yaml'))
        print(f"Processing {len(model_files)} neural models...")

        models_xml = create_neural_model_data(model_files)
        output_file = output_dir / 'neural_models.xml'

        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(prettify_xml(models_xml))

        print(f"✓ Created {output_file}")

    # Process integrators
    integrators_dir = database_dir / 'integrators'
    if integrators_dir.exists():
        integrator_files = list(integrators_dir.glob('*.yaml'))
        print(f"Processing {len(integrator_files)} integrators...")

        integrators_xml = create_integrator_data(integrator_files)
        output_file = output_dir / 'integrators.xml'

        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(prettify_xml(integrators_xml))

        print(f"✓ Created {output_file}")

    print("\nData files generated successfully!")
    print("Restart Odoo and upgrade the TVBO module to load the data.")


if __name__ == '__main__':
    main()
