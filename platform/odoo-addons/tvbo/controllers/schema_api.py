# -*- coding: utf-8 -*-
from odoo import http
from tvbo.datamodel.tvbopydantic import (
    Monitor,
    ObservationModel,
    ProcessingStep,
    DataInjection,
    ArgumentMapping,
    ImagingModality,
    OperationType,
)
from pydantic.fields import FieldInfo
from typing import get_origin, get_args, Union

PYDANTIC_AVAILABLE = True


class SchemaAPIController(http.Controller):

    @http.route(
        "/tvbo/api/schema/model/<string:model_name>",
        type="json",
        auth="user",
        methods=["GET"],
    )
    def get_model_schema(self, model_name, **kwargs):
        """Get Pydantic schema for a given model class"""
        if not PYDANTIC_AVAILABLE:
            return {"error": "Pydantic models not available"}

        model_map = {
            "Monitor": Monitor,
            "ObservationModel": ObservationModel,
            "ProcessingStep": ProcessingStep,
            "DataInjection": DataInjection,
            "ArgumentMapping": ArgumentMapping,
        }

        model_class = model_map.get(model_name)
        if not model_class:
            return {"error": f"Model {model_name} not found"}

        return self._extract_field_schema(model_class)

    def _extract_field_schema(self, model_class):
        """Extract field information from a Pydantic model"""
        schema = {
            "name": model_class.__name__,
            "doc": model_class.__doc__,
            "fields": [],
        }

        # Get model fields
        for field_name, field_info in model_class.model_fields.items():
            field_schema = self._process_field(field_name, field_info)
            schema["fields"].append(field_schema)

        return schema

    def _process_field(self, field_name, field_info: FieldInfo):
        """Process a single Pydantic field"""
        field_type = field_info.annotation

        # Extract base type and check if optional
        is_optional = False
        is_list = False
        base_type = field_type

        # Handle Union types (Optional is Union[T, None])
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Check if it's Optional (Union with None)
            if type(None) in args:
                is_optional = True
                # Get the non-None type
                base_type = next((arg for arg in args if arg is not type(None)), str)
                origin = get_origin(base_type)

        # Handle list types
        if origin is list:
            is_list = True
            list_args = get_args(base_type)
            if list_args:
                base_type = list_args[0]

        # Determine field type category
        type_name = self._get_type_name(base_type)

        # Get enum values if applicable
        enum_values = None
        if hasattr(base_type, "__members__"):  # It's an enum
            enum_values = [{"value": v.value, "label": v.name} for v in base_type]

        # Get description
        description = field_info.description or ""

        # Get default value
        default = None
        if field_info.default is not None and field_info.default != ...:
            default = str(field_info.default)

        return {
            "name": field_name,
            "type": type_name,
            "is_optional": is_optional,
            "is_list": is_list,
            "required": field_info.is_required(),
            "description": description,
            "default": default,
            "enum_values": enum_values,
        }

    def _get_type_name(self, python_type):
        """Convert Python type to a simple string representation"""
        if python_type is str:
            return "string"
        elif python_type is int:
            return "integer"
        elif python_type is float:
            return "float"
        elif python_type is bool:
            return "boolean"
        elif hasattr(python_type, "__name__"):
            # Check if it's an enum
            if hasattr(python_type, "__members__"):
                return "enum"
            # Check if it's a Pydantic model
            if hasattr(python_type, "model_fields"):
                return "object"
            return python_type.__name__
        else:
            return "unknown"

    @http.route("/tvbo/api/schema/enums", type="json", auth="user", methods=["GET"])
    def get_enums(self, **kwargs):
        """Get all available enum types"""
        if not PYDANTIC_AVAILABLE:
            return {"error": "Pydantic models not available"}

        return {
            "ImagingModality": [
                {"value": v.value, "label": v.name, "doc": v.__doc__}
                for v in ImagingModality
            ],
            "OperationType": [
                {"value": v.value, "label": v.name} for v in OperationType
            ],
        }
