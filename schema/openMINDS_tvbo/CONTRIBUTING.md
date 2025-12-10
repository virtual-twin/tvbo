# Contributing to openMINDS_tvbo

Thank you for your interest in contributing to the openMINDS_tvbo schema module!

## Schema Format

All schemas follow the openMINDS JSON template format (`.schema.tpl.json`). Key conventions:

### Type Declaration
```json
{
  "_type": "tvbo:SchemaName",
  "_categories": ["category1", "category2"],
  "_extends": "path/to/base.schema.tpl.json"
}
```

### Property Types

**Linked Types** (references to other schemas):
```json
"propertyName": {
  "_instruction": "Description of what to add.",
  "_linkedTypes": [
    "tvbo:OtherSchema",
    "sands:BrainAtlasVersion",
    "core:File"
  ]
}
```

**Embedded Types** (inline objects):
```json
"propertyName": {
  "_embeddedTypes": [
    "tvbo:EmbeddedSchema"
  ]
}
```

**Primitive Types**:
```json
"propertyName": {
  "type": "string",
  "_instruction": "Description."
}
```

**Arrays**:
```json
"propertyName": {
  "type": "array",
  "minItems": 1,
  "uniqueItems": true,
  "_instruction": "Description.",
  "items": {
    "type": "string"
  }
}
```

**Enums**:
```json
"propertyName": {
  "type": "string",
  "enum": ["value1", "value2", "value3"],
  "_instruction": "Description."
}
```

## Namespace Prefixes

- `tvbo:` - TVBO schemas (this module)
- `sands:` - openMINDS_SANDS (brain atlases, parcellations)
- `core:` - openMINDS_core (files, actors, identifiers)
- `computation:` - openMINDS_computation (simulations, environments)
- `controlledTerms:` - openMINDS controlled vocabularies

## Adding New Schemas

1. Create a new `.schema.tpl.json` file in `/schemas/`
2. Follow the naming convention: `camelCase.schema.tpl.json`
3. Include required `_type` with `tvbo:` prefix
4. Add `_instruction` for every property
5. Use `_linkedTypes` for references to SANDS, core, or other schemas
6. Update `README.md` with the new schema

## SANDS Integration

When linking to brain atlas concepts, always use openMINDS_SANDS types:

| TVBO Concept | SANDS Type |
|--------------|------------|
| Brain atlas | `sands:BrainAtlas`, `sands:BrainAtlasVersion` |
| Parcellation | `sands:ParcellationTerminology`, `sands:ParcellationTerminologyVersion` |
| Brain region | `sands:ParcellationEntity`, `sands:ParcellationEntityVersion` |
| Coordinate space | `sands:CommonCoordinateSpace`, `sands:CommonCoordinateSpaceVersion` |
| Coordinates | `sands:CoordinatePoint` |

## Validation

Before submitting, ensure your schemas:

1. Are valid JSON
2. Follow the openMINDS template format
3. Have consistent naming with existing schemas
4. Include instructions for all properties
5. Use appropriate linked/embedded types

## Questions?

Open an issue in the repository or contact the TVBO maintainers.
