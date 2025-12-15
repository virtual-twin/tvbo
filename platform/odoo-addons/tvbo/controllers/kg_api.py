# -*- coding: utf-8 -*-
"""
Knowledge Graph API - Provides data and schema for the KG Browser.
Uses Pydantic models from tvbo.datamodel for consistent schema.
"""
import json
import logging

from odoo import http
from odoo.http import Response, request

from tvbo.datamodel.tvbopydantic import Coupling as PydanticCoupling
from tvbo.datamodel.tvbopydantic import Dynamics as PydanticDynamics
from tvbo.datamodel.tvbopydantic import Integrator as PydanticIntegrator
from tvbo.datamodel.tvbopydantic import Network as PydanticNetwork
from tvbo.datamodel.tvbopydantic import \
    SimulationExperiment as PydanticSimulationExperiment
from tvbo.datamodel.tvbopydantic import \
    SimulationStudy as PydanticSimulationStudy

_logger = logging.getLogger(__name__)
_logger.info("Pydantic models loaded successfully for KG API")


def safe_get(record, field, default=None):
    """Safely get field value from Odoo record."""
    val = getattr(record, field, None)
    if val is None or val is False:
        return default
    return val


def get_relation_value(record, field, sub_field='name'):
    """Get value from Many2one relation."""
    rel = getattr(record, field, None)
    if rel:
        return getattr(rel, sub_field, None)
    return None


class KnowledgeGraphAPI(http.Controller):
    """API endpoints for the Knowledge Graph Browser.
    
    Uses Pydantic models for serialization to ensure schema consistency.
    """

    # ===================
    # Schema endpoint
    # ===================

    @http.route('/tvbo/api/kg/schema', type='http', auth='public', methods=['GET'], csrf=False)
    def get_schema(self, **kw):
        """Get the search schema configuration for the browser."""
        schema = {
            "searchableFields": ["name", "label", "title", "description", "abstract", "doi", "method"],
            "facets": [
                {"field": "type", "label": "Type", "type": "string"},
                {"field": "system_type", "label": "System Type", "type": "string"},
                {"field": "year", "label": "Year", "type": "string"},
                {"field": "tags", "label": "Tags", "type": "array"},
            ]
        }
        return Response(
            json.dumps(schema),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    # ===================
    # Data endpoint
    # ===================

    @http.route('/tvbo/api/kg/data', type='http', auth='public', methods=['GET'], csrf=False)
    def get_all_data(self, **kw):
        """Get all knowledge graph data combined."""
        data = []

        try:
            # Neural Mass Models / Dynamics
            data.extend(self._get_dynamics())

            # Networks
            data.extend(self._get_networks())

            # Integrators
            data.extend(self._get_integrators())

            # Experiments
            data.extend(self._get_experiments())

            # Studies
            data.extend(self._get_studies())

            # Couplings
            data.extend(self._get_couplings())

            _logger.info(f"KG API: Returning {len(data)} items")

        except Exception as e:
            _logger.error(f"Error in get_all_data: {e}", exc_info=True)

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    # ===================
    # Dynamics
    # ===================

    def _get_dynamics(self):
        """Fetch and serialize all dynamics models."""
        results = []
        records = request.env['tvbo.dynamics'].sudo().search([])
        for r in records:
            results.append(self._serialize_dynamics(r))
        return results

    def _serialize_dynamics(self, record):
        """Serialize dynamics using Pydantic model."""
        name = safe_get(record, 'name', '')
        label = safe_get(record, 'label', '')
        description = safe_get(record, 'description', '')

        # Get system_type name
        system_type_name = get_relation_value(record, 'system_type', 'technical_name')
        if not system_type_name:
            system_type_name = get_relation_value(record, 'system_type', 'name')

        # Build tags
        tags = []
        if system_type_name:
            tags.append(system_type_name)

        # Add state variable labels as tags
        for sv in safe_get(record, 'state_variables', []):
            sv_label = safe_get(sv, 'label')
            if sv_label:
                tags.append(sv_label)

        pydantic_obj = PydanticDynamics(
            name=name or 'Unknown',
            label=label or None,
            description=description or None,
            source=safe_get(record, 'source'),
            iri=safe_get(record, 'iri'),
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields
        result['id'] = record.id
        result['type'] = 'dynamics'
        result['title'] = name or label or ''
        result['tags'] = tags
        result['system_type'] = system_type_name or ''

        return result

    # ===================
    # Networks
    # ===================

    def _get_networks(self):
        """Fetch and serialize all networks."""
        results = []
        records = request.env['tvbo.network'].sudo().search([])
        for r in records:
            results.append(self._serialize_network(r))
        return results

    def _serialize_network(self, record):
        """Serialize a network using Pydantic model."""
        label = safe_get(record, 'label', '')
        description = safe_get(record, 'description', '')
        number_of_regions = safe_get(record, 'number_of_regions', 0)
        number_of_nodes = safe_get(record, 'number_of_nodes', 0)

        # Build tags
        tags = []
        parcellation_label = get_relation_value(record, 'parcellation', 'label')
        if parcellation_label:
            tags.append(parcellation_label)

        pydantic_obj = PydanticNetwork(
            label=label or None,
            description=description or None,
            number_of_regions=number_of_regions or 1,
            number_of_nodes=number_of_nodes or 1,
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields
        result['id'] = record.id
        result['type'] = 'network'
        result['name'] = label or f'Network {record.id}'
        result['title'] = label or f'Network {record.id}'
        result['tags'] = tags

        return result

    # ===================
    # Integrators
    # ===================

    def _get_integrators(self):
        """Fetch and serialize all integrators."""
        results = []
        records = request.env['tvbo.integrator'].sudo().search([])
        for r in records:
            results.append(self._serialize_integrator(r))
        return results

    def _serialize_integrator(self, record):
        """Serialize an integrator using Pydantic model."""
        method = safe_get(record, 'method', '')
        step_size = safe_get(record, 'step_size', 0.01220703125)
        duration = safe_get(record, 'duration', 1000.0)
        time_scale = safe_get(record, 'time_scale', 'ms')

        # Build tags
        tags = []
        if method:
            tags.append(method)

        pydantic_obj = PydanticIntegrator(
            method=method or None,
            step_size=step_size,
            duration=duration,
            time_scale=time_scale,
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields (Integrator doesn't have 'name' in Pydantic)
        result['id'] = record.id
        result['type'] = 'integrator'
        result['name'] = method or f'Integrator {record.id}'
        result['label'] = method or f'Integrator {record.id}'
        result['title'] = method or f'Integrator {record.id}'
        result['description'] = f"Method: {method}, Step size: {step_size}, Duration: {duration}" if method else ''
        result['tags'] = tags

        return result

    # ===================
    # Experiments
    # ===================

    def _get_experiments(self):
        """Fetch and serialize all simulation experiments."""
        results = []
        records = request.env['tvbo.simulation_experiment'].sudo().search([])
        for r in records:
            results.append(self._serialize_experiment(r))
        return results

    def _serialize_experiment(self, record):
        """Serialize a simulation experiment using Pydantic model."""
        label = safe_get(record, 'label', '')
        description = safe_get(record, 'description', '')

        # Build tags
        tags = []
        dynamics_name = get_relation_value(record, 'local_dynamics', 'name')
        if dynamics_name:
            tags.append(dynamics_name)
        connectivity_label = get_relation_value(record, 'connectivity', 'label')
        if connectivity_label:
            tags.append(connectivity_label)
        network_label = get_relation_value(record, 'network', 'label')
        if network_label and network_label not in tags:
            tags.append(network_label)

        pydantic_obj = PydanticSimulationExperiment(
            label=label or None,
            description=description or None,
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields
        result['id'] = record.id
        result['type'] = 'experiment'
        result['name'] = label or f'Experiment {record.id}'
        result['title'] = label or f'Experiment {record.id}'
        result['abstract'] = description or ''
        result['tags'] = tags

        return result

    # ===================
    # Studies
    # ===================

    def _get_studies(self):
        """Fetch and serialize all simulation studies."""
        results = []
        records = request.env['tvbo.simulation_study'].sudo().search([])
        for r in records:
            results.append(self._serialize_study(r))
        return results

    def _serialize_study(self, record):
        """Serialize a simulation study using Pydantic model."""
        label = safe_get(record, 'label', '')
        description = safe_get(record, 'description', '')
        doi = safe_get(record, 'doi', '')
        year = safe_get(record, 'year', '')
        title = safe_get(record, 'title', '')

        # Build tags
        tags = []
        model_name = get_relation_value(record, 'model', 'name')
        if model_name:
            tags.append(model_name)

        pydantic_obj = PydanticSimulationStudy(
            label=label or None,
            description=description or None,
            doi=doi or None,
            title=title or None,
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields
        result['id'] = record.id
        result['type'] = 'study'
        result['name'] = title or label or f'Study {record.id}'
        result['title'] = title or label or f'Study {record.id}'
        result['abstract'] = description or ''
        result['year'] = str(year) if year else ''
        result['doi'] = doi or ''
        result['tags'] = tags

        return result

    # ===================
    # Couplings
    # ===================

    def _get_couplings(self):
        """Fetch and serialize all coupling functions."""
        results = []
        records = request.env['tvbo.coupling'].sudo().search([])
        for r in records:
            results.append(self._serialize_coupling(r))
        return results

    def _serialize_coupling(self, record):
        """Serialize a coupling function using Pydantic model."""
        name = safe_get(record, 'name', '')
        label = safe_get(record, 'label', '')
        delayed = safe_get(record, 'delayed', False)
        sparse = safe_get(record, 'sparse', False)

        # Get description from coupling function equation
        description = ''
        coupling_func = safe_get(record, 'coupling_function')
        if coupling_func:
            description = safe_get(coupling_func, 'definition', '')

        # Build tags
        tags = []
        if delayed:
            tags.append('delayed')
        if sparse:
            tags.append('sparse')

        pydantic_obj = PydanticCoupling(
            name=name or 'Linear',
            label=label or None,
            delayed=delayed,
            sparse=sparse,
        )
        result = pydantic_obj.model_dump(exclude_none=True)

        # Add browser-specific fields
        result['id'] = record.id
        result['type'] = 'coupling'
        result['title'] = name or label or f'Coupling {record.id}'
        result['description'] = description or ''
        result['tags'] = tags

        return result

    # ===================
    # Detail endpoints
    # ===================

    @http.route('/tvbo/api/kg/dynamics/<int:dynamics_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_dynamics_detail(self, dynamics_id, **kw):
        """Get detailed information about a specific dynamics model."""
        record = request.env['tvbo.dynamics'].sudo().browse(dynamics_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Dynamics not found"}),
                content_type='application/json',
                status=404
            )

        # Get base serialization
        data = self._serialize_dynamics(record)

        # Add detailed parameters
        data['parameters'] = []
        for p in safe_get(record, 'parameters', []):
            data['parameters'].append({
                "name": safe_get(p, 'name', ''),
                "label": safe_get(p, 'label', ''),
                "description": safe_get(p, 'description', ''),
            })

        # Add detailed state variables
        data['state_variables'] = []
        for sv in safe_get(record, 'state_variables', []):
            sv_data = {
                "name": safe_get(sv, 'name', ''),
                "label": safe_get(sv, 'label', ''),
                "description": safe_get(sv, 'description', ''),
            }
            equation = safe_get(sv, 'equation')
            if equation:
                sv_data['equation'] = {
                    "label": safe_get(equation, 'label', ''),
                    "definition": safe_get(equation, 'definition', ''),
                }
            data['state_variables'].append(sv_data)

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    @http.route('/tvbo/api/kg/network/<int:network_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_network_detail(self, network_id, **kw):
        """Get detailed information about a specific network."""
        record = request.env['tvbo.network'].sudo().browse(network_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Network not found"}),
                content_type='application/json',
                status=404
            )

        data = self._serialize_network(record)

        # Add detailed parcellation info
        parcellation = safe_get(record, 'parcellation')
        if parcellation:
            data['parcellation'] = {
                "label": safe_get(parcellation, 'label', ''),
                "data_source": safe_get(parcellation, 'data_source', ''),
            }

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    @http.route('/tvbo/api/kg/integrator/<int:integrator_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_integrator_detail(self, integrator_id, **kw):
        """Get detailed information about a specific integrator."""
        record = request.env['tvbo.integrator'].sudo().browse(integrator_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Integrator not found"}),
                content_type='application/json',
                status=404
            )

        data = self._serialize_integrator(record)

        # Add detailed parameters
        data['parameters'] = []
        for p in safe_get(record, 'parameters', []):
            data['parameters'].append({
                "name": safe_get(p, 'name', ''),
                "label": safe_get(p, 'label', ''),
                "value": safe_get(p, 'value', None),
            })

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    @http.route('/tvbo/api/kg/coupling/<int:coupling_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_coupling_detail(self, coupling_id, **kw):
        """Get detailed information about a specific coupling."""
        record = request.env['tvbo.coupling'].sudo().browse(coupling_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Coupling not found"}),
                content_type='application/json',
                status=404
            )

        data = self._serialize_coupling(record)

        # Add detailed coupling function
        coupling_func = safe_get(record, 'coupling_function')
        if coupling_func:
            data['coupling_function'] = {
                "label": safe_get(coupling_func, 'label', ''),
                "definition": safe_get(coupling_func, 'definition', ''),
            }

        # Add detailed parameters
        data['parameters'] = []
        for p in safe_get(record, 'parameters', []):
            data['parameters'].append({
                "name": safe_get(p, 'name', ''),
                "label": safe_get(p, 'label', ''),
            })

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    @http.route('/tvbo/api/kg/experiment/<int:experiment_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_experiment_detail(self, experiment_id, **kw):
        """Get detailed information about a specific experiment."""
        record = request.env['tvbo.simulation_experiment'].sudo().browse(experiment_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Experiment not found"}),
                content_type='application/json',
                status=404
            )

        data = self._serialize_experiment(record)

        # Add related entities
        local_dynamics = safe_get(record, 'local_dynamics')
        if local_dynamics:
            data['local_dynamics'] = {
                "id": local_dynamics.id,
                "name": safe_get(local_dynamics, 'name', ''),
            }

        integration = safe_get(record, 'integration')
        if integration:
            data['integration'] = {
                "id": integration.id,
                "method": safe_get(integration, 'method', ''),
            }

        connectivity = safe_get(record, 'connectivity')
        if connectivity:
            data['connectivity'] = {
                "id": connectivity.id,
                "label": safe_get(connectivity, 'label', ''),
            }

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    @http.route('/tvbo/api/kg/study/<int:study_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_study_detail(self, study_id, **kw):
        """Get detailed information about a specific study."""
        record = request.env['tvbo.simulation_study'].sudo().browse(study_id)
        if not record.exists():
            return Response(
                json.dumps({"error": "Study not found"}),
                content_type='application/json',
                status=404
            )

        data = self._serialize_study(record)

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )
