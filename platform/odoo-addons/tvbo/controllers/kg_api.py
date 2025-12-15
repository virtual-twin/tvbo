# -*- coding: utf-8 -*-
"""
Knowledge Graph API - Provides data and schema for the KG Browser.
Uses Pydantic models from tvbo.datamodel for consistent schema.
"""
import json
import logging
from odoo import http
from odoo.http import request, Response

_logger = logging.getLogger(__name__)

# Try to import Pydantic models
try:
    from tvbo.datamodel.tvbopydantic import (
        Dynamics,
        NeuralMassModel,
        Network,
        Coupling,
        Integrator,
        SimulationExperiment,
        SimulationStudy,
        Parameter,
        StateVariable,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    _logger.warning("tvbo.datamodel.tvbopydantic not available - using fallback serialization")
    PYDANTIC_AVAILABLE = False


class KnowledgeGraphAPI(http.Controller):
    """API endpoints for the Knowledge Graph Browser."""

    # ===================
    # Schema endpoint
    # ===================

    @http.route('/tvbo/api/kg/schema', type='http', auth='public', methods=['GET'], csrf=False)
    def get_schema(self, **kw):
        """Get the search schema configuration for the browser."""
        schema = {
            "searchableFields": ["name", "label", "title", "description", "abstract", "doi"],
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

        # Neural Mass Models
        data.extend(self._get_models())
        
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

        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    # ===================
    # Helper: Convert Odoo record to dict
    # ===================

    def _odoo_to_dict(self, record, record_type, extra_fields=None):
        """Convert Odoo record to a browser-compatible dict."""
        result = {
            "id": record.id,
            "type": record_type,
        }
        
        # Standard fields that might exist
        for field in ['name', 'label', 'description']:
            if hasattr(record, field):
                val = getattr(record, field)
                if val:
                    result[field] = val
        
        # Set title from name or label
        result['title'] = result.get('name') or result.get('label') or ''
        
        # Ensure name exists
        if 'name' not in result or not result['name']:
            result['name'] = result.get('label') or result.get('title') or ''
        
        # Extra fields
        if extra_fields:
            for field, default in extra_fields.items():
                if hasattr(record, field):
                    val = getattr(record, field)
                    if val:
                        # Handle Many2one relations
                        if hasattr(val, 'name'):
                            result[field] = val.name
                        elif hasattr(val, 'label'):
                            result[field] = val.label
                        else:
                            result[field] = val
                    else:
                        result[field] = default
                else:
                    result[field] = default
        
        # Ensure required fields have defaults
        result.setdefault('tags', [])
        result.setdefault('description', '')
        
        return result

    # ===================
    # Models
    # ===================

    def _get_models(self):
        """Fetch and serialize all neural mass models."""
        try:
            models = request.env['tvbo.neural_mass_model'].sudo().search([])
            return [self._serialize_model(m) for m in models]
        except Exception as e:
            _logger.error(f"Error fetching models: {e}")
            return []

    def _serialize_model(self, model):
        """Serialize a neural mass model."""
        result = self._odoo_to_dict(model, 'model', {
            'system_type': '',
            'source': '',
            'references': '',
            'iri': '',
        })
        
        # Get system type name if it's a relation
        if model.system_type:
            if hasattr(model.system_type, 'name'):
                result['system_type'] = model.system_type.name
            elif hasattr(model.system_type, 'technical_name'):
                result['system_type'] = model.system_type.technical_name
        
        # Build tags
        tags = []
        if result.get('system_type'):
            tags.append(result['system_type'])
        
        # Add state variable names as tags
        if hasattr(model, 'state_variables'):
            for sv in model.state_variables:
                if hasattr(sv, 'label') and sv.label:
                    tags.append(sv.label)
        
        result['tags'] = tags
        return result

    # ===================
    # Networks
    # ===================

    def _get_networks(self):
        """Fetch and serialize all networks."""
        try:
            networks = request.env['tvbo.network'].sudo().search([])
            return [self._serialize_network(n) for n in networks]
        except Exception as e:
            _logger.error(f"Error fetching networks: {e}")
            return []

    def _serialize_network(self, network):
        """Serialize a network."""
        result = self._odoo_to_dict(network, 'network', {
            'number_of_regions': 0,
            'number_of_nodes': 0,
        })
        
        # Build tags
        tags = []
        if hasattr(network, 'parcellation') and network.parcellation:
            if hasattr(network.parcellation, 'label') and network.parcellation.label:
                tags.append(network.parcellation.label)
        
        result['tags'] = tags
        return result

    # ===================
    # Integrators
    # ===================

    def _get_integrators(self):
        """Fetch and serialize all integrators."""
        try:
            integrators = request.env['tvbo.integrator'].sudo().search([])
            return [self._serialize_integrator(i) for i in integrators]
        except Exception as e:
            _logger.error(f"Error fetching integrators: {e}")
            return []

    def _serialize_integrator(self, integrator):
        """Serialize an integrator."""
        method = getattr(integrator, 'method', None) or ''
        step_size = getattr(integrator, 'step_size', None) or 0
        duration = getattr(integrator, 'duration', None) or 0
        
        return {
            "id": integrator.id,
            "type": "integrator",
            "name": method,
            "label": method,
            "title": method,
            "description": f"Step size: {step_size}, Duration: {duration}" if method else '',
            "step_size": step_size,
            "duration": duration,
            "tags": [method] if method else [],
        }

    # ===================
    # Experiments
    # ===================

    def _get_experiments(self):
        """Fetch and serialize all simulation experiments."""
        try:
            experiments = request.env['tvbo.simulation_experiment'].sudo().search([])
            return [self._serialize_experiment(e) for e in experiments]
        except Exception as e:
            _logger.error(f"Error fetching experiments: {e}")
            return []

    def _serialize_experiment(self, exp):
        """Serialize a simulation experiment."""
        result = self._odoo_to_dict(exp, 'experiment', {
            'references': '',
        })
        
        # Build tags
        tags = []
        if hasattr(exp, 'local_dynamics') and exp.local_dynamics:
            if hasattr(exp.local_dynamics, 'name') and exp.local_dynamics.name:
                tags.append(exp.local_dynamics.name)
        if hasattr(exp, 'connectivity') and exp.connectivity:
            if hasattr(exp.connectivity, 'label') and exp.connectivity.label:
                tags.append(exp.connectivity.label)
        
        result['tags'] = tags
        result['abstract'] = result.get('description', '')
        return result

    # ===================
    # Studies
    # ===================

    def _get_studies(self):
        """Fetch and serialize all simulation studies."""
        try:
            studies = request.env['tvbo.simulation_study'].sudo().search([])
            return [self._serialize_study(s) for s in studies]
        except Exception as e:
            _logger.error(f"Error fetching studies: {e}")
            return []

    def _serialize_study(self, study):
        """Serialize a simulation study."""
        result = self._odoo_to_dict(study, 'study', {
            'year': '',
            'doi': '',
            'title': '',
        })
        
        # Ensure year is string
        if result.get('year'):
            result['year'] = str(result['year'])
        
        # Build tags
        tags = []
        if hasattr(study, 'model') and study.model:
            if hasattr(study.model, 'name') and study.model.name:
                tags.append(study.model.name)
        
        result['tags'] = tags
        result['abstract'] = result.get('description', '')
        return result

    # ===================
    # Couplings
    # ===================

    def _get_couplings(self):
        """Fetch and serialize all coupling functions."""
        try:
            couplings = request.env['tvbo.coupling'].sudo().search([])
            return [self._serialize_coupling(c) for c in couplings]
        except Exception as e:
            _logger.error(f"Error fetching couplings: {e}")
            return []

    def _serialize_coupling(self, coupling):
        """Serialize a coupling function."""
        name = getattr(coupling, 'name', None) or ''
        label = getattr(coupling, 'label', None) or name
        
        # Build description from coupling function
        desc = ''
        if hasattr(coupling, 'coupling_function') and coupling.coupling_function:
            if hasattr(coupling.coupling_function, 'definition'):
                desc = coupling.coupling_function.definition or ''
        
        # Build tags
        tags = []
        if hasattr(coupling, 'delayed') and coupling.delayed:
            tags.append('delayed')
        if hasattr(coupling, 'sparse') and coupling.sparse:
            tags.append('sparse')
        
        return {
            "id": coupling.id,
            "type": "coupling",
            "name": name,
            "label": label,
            "title": name,
            "description": desc,
            "tags": tags,
        }

    # ===================
    # Detail endpoints
    # ===================

    @http.route('/tvbo/api/kg/model/<int:model_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_model_detail(self, model_id, **kw):
        """Get detailed information about a specific neural mass model."""
        model = request.env['tvbo.neural_mass_model'].sudo().browse(model_id)
        if not model.exists():
            return Response(
                json.dumps({"error": "Model not found"}),
                content_type='application/json',
                status=404
            )
        
        # Use Pydantic if available for schema-consistent output
        if PYDANTIC_AVAILABLE:
            try:
                pydantic_model = self._odoo_to_pydantic_dynamics(model)
                data = pydantic_model.model_dump(exclude_none=True)
                data['id'] = model.id
                data['type'] = 'model'
            except Exception as e:
                _logger.warning(f"Pydantic conversion failed: {e}, using fallback")
                data = self._serialize_model_detail(model)
        else:
            data = self._serialize_model_detail(model)
        
        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )

    def _odoo_to_pydantic_dynamics(self, model):
        """Convert Odoo neural mass model to Pydantic Dynamics."""
        params = {}
        if hasattr(model, 'parameters'):
            for p in model.parameters:
                if p.name:
                    params[p.name] = Parameter(
                        name=p.name,
                        label=p.label or None,
                        description=p.description or None,
                    )
        
        state_vars = {}
        if hasattr(model, 'state_variables'):
            for sv in model.state_variables:
                if sv.label:
                    state_vars[sv.label] = StateVariable(
                        name=sv.label,
                        label=sv.label,
                        description=sv.description or None,
                    )
        
        system_type = None
        if model.system_type:
            if hasattr(model.system_type, 'technical_name'):
                system_type = model.system_type.technical_name
        
        return Dynamics(
            name=model.name or 'Unknown',
            label=model.label or None,
            description=model.description or None,
            source=model.source or None,
            parameters=params if params else None,
            state_variables=state_vars if state_vars else None,
            system_type=system_type,
        )

    def _serialize_model_detail(self, model):
        """Fallback serialization for model details."""
        data = self._serialize_model(model)
        
        # Add parameters
        data['parameters'] = []
        if hasattr(model, 'parameters'):
            for p in model.parameters:
                data['parameters'].append({
                    "name": p.name or '',
                    "label": p.label or '',
                    "description": p.description or '',
                })
        
        # Add state variables
        data['state_variables'] = []
        if hasattr(model, 'state_variables'):
            for sv in model.state_variables:
                sv_data = {
                    "label": sv.label or '',
                    "description": sv.description or '',
                }
                if hasattr(sv, 'equation') and sv.equation:
                    sv_data['equation'] = {
                        "label": sv.equation.label or '',
                        "definition": sv.equation.definition or '',
                    }
                data['state_variables'].append(sv_data)
        
        return data

    @http.route('/tvbo/api/kg/network/<int:network_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_network_detail(self, network_id, **kw):
        """Get detailed information about a specific network."""
        network = request.env['tvbo.network'].sudo().browse(network_id)
        if not network.exists():
            return Response(
                json.dumps({"error": "Network not found"}),
                content_type='application/json',
                status=404
            )
        
        data = self._serialize_network(network)
        
        # Add parcellation info
        if hasattr(network, 'parcellation') and network.parcellation:
            data['parcellation'] = {
                "label": network.parcellation.label or '',
                "data_source": network.parcellation.data_source or '' if hasattr(network.parcellation, 'data_source') else '',
            }
        
        return Response(
            json.dumps(data),
            content_type='application/json',
            headers={'Access-Control-Allow-Origin': '*'}
        )
