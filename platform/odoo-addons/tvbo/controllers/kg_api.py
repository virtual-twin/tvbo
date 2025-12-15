# -*- coding: utf-8 -*-
"""
Knowledge Graph API - Clean MVP implementation.
No fallbacks. Fails fast when something unexpected happens.
"""
import json
import logging

from odoo import http
from odoo.http import Response, request

from tvbo.datamodel.tvbopydantic import Coupling as PydanticCoupling
from tvbo.datamodel.tvbopydantic import Dynamics as PydanticDynamics
from tvbo.datamodel.tvbopydantic import Integrator as PydanticIntegrator
from tvbo.datamodel.tvbopydantic import Network as PydanticNetwork
from tvbo.datamodel.tvbopydantic import SimulationExperiment as PydanticSimulationExperiment
from tvbo.datamodel.tvbopydantic import SimulationStudy as PydanticSimulationStudy
from tvbo.api.direct_ontology_api import get_direct_ontology_api

_logger = logging.getLogger(__name__)

# Singleton ontology API
_ontology_api = None


def get_ontology_api():
    """Get DirectOntologyAPI singleton. Fails if unavailable."""
    global _ontology_api
    if _ontology_api is None:
        _ontology_api = get_direct_ontology_api()
    return _ontology_api


def json_response(data, status=200):
    """Standard JSON response with CORS."""
    return Response(
        json.dumps(data),
        content_type='application/json',
        status=status,
        headers={'Access-Control-Allow-Origin': '*'}
    )


class KnowledgeGraphAPI(http.Controller):
    """Knowledge Graph API endpoints. Clean MVP."""

    # ===================
    # Schema
    # ===================

    @http.route('/tvbo/api/kg/schema', type='http', auth='public', methods=['GET'], csrf=False)
    def get_schema(self, **kw):
        """Get browser schema configuration."""
        return json_response({
            "searchableFields": ["name", "label", "title", "description", "abstract", "doi", "method", "definition", "symbol"],
            "facets": [
                {"field": "type", "label": "Type", "type": "string"},
                {"field": "ontology_type", "label": "Ontology Class", "type": "string"},
                {"field": "system_type", "label": "System Type", "type": "string"},
                {"field": "year", "label": "Year", "type": "string"},
                {"field": "tags", "label": "Tags", "type": "array"},
            ],
            "sources": ["database", "ontology"]
        })

    # ===================
    # Main data endpoint
    # ===================

    @http.route('/tvbo/api/kg/data', type='http', auth='public', methods=['GET'], csrf=False)
    def get_all_data(self, **kw):
        """Get all knowledge graph data."""
        include_ontology = kw.get('include_ontology', 'true').lower() != 'false'
        ontology_query = kw.get('ontology_query', '')

        data = []
        data.extend(self._get_dynamics())
        data.extend(self._get_networks())
        data.extend(self._get_integrators())
        data.extend(self._get_experiments())
        data.extend(self._get_studies())
        data.extend(self._get_couplings())

        if include_ontology:
            data.extend(self._get_ontology_concepts(ontology_query))

        return json_response(data)

    # ===================
    # Database serializers
    # ===================

    def _get_dynamics(self):
        records = request.env['tvbo.dynamics'].sudo().search([])
        return [self._serialize_dynamics(r) for r in records]

    def _serialize_dynamics(self, record):
        tags = []
        if record.system_type:
            tags.append(record.system_type.technical_name or record.system_type.name)
        for sv in record.state_variables:
            if sv.label:
                tags.append(sv.label)

        result = PydanticDynamics(
            name=record.name or 'Unknown',
            label=record.label or None,
            description=record.description or None,
            source=record.source or None,
            iri=record.iri or None,
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'dynamics',
            'title': record.name or record.label or '',
            'tags': tags,
            'system_type': record.system_type.technical_name if record.system_type else '',
        })

        return get_ontology_api().enrich_database_item(result, 'dynamics')

    def _get_networks(self):
        records = request.env['tvbo.network'].sudo().search([])
        return [self._serialize_network(r) for r in records]

    def _serialize_network(self, record):
        tags = []
        if record.parcellation:
            tags.append(record.parcellation.label)

        result = PydanticNetwork(
            label=record.label or None,
            description=record.description or None,
            number_of_regions=record.number_of_regions or 1,
            number_of_nodes=record.number_of_nodes or 1,
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'network',
            'name': record.label or f'Network {record.id}',
            'title': record.label or f'Network {record.id}',
            'tags': tags,
        })

        return get_ontology_api().enrich_database_item(result, 'network')

    def _get_integrators(self):
        records = request.env['tvbo.integrator'].sudo().search([])
        return [self._serialize_integrator(r) for r in records]

    def _serialize_integrator(self, record):
        tags = [record.method] if record.method else []

        result = PydanticIntegrator(
            method=record.method or None,
            step_size=record.step_size or 0.01220703125,
            duration=record.duration or 1000.0,
            time_scale=record.time_scale or 'ms',
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'integrator',
            'name': record.method or f'Integrator {record.id}',
            'title': record.method or f'Integrator {record.id}',
            'description': f"Method: {record.method}, Step: {record.step_size}, Duration: {record.duration}",
            'tags': tags,
        })

        return get_ontology_api().enrich_database_item(result, 'integrator')

    def _get_experiments(self):
        records = request.env['tvbo.simulation_experiment'].sudo().search([])
        return [self._serialize_experiment(r) for r in records]

    def _serialize_experiment(self, record):
        tags = []
        if record.local_dynamics:
            tags.append(record.local_dynamics.name)
        if record.connectivity:
            tags.append(record.connectivity.label)
        if record.network and record.network.label not in tags:
            tags.append(record.network.label)

        result = PydanticSimulationExperiment(
            label=record.label or None,
            description=record.description or None,
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'experiment',
            'name': record.label or f'Experiment {record.id}',
            'title': record.label or f'Experiment {record.id}',
            'abstract': record.description or '',
            'tags': tags,
        })

        return get_ontology_api().enrich_database_item(result, 'experiment')

    def _get_studies(self):
        records = request.env['tvbo.simulation_study'].sudo().search([])
        return [self._serialize_study(r) for r in records]

    def _serialize_study(self, record):
        tags = []
        if record.model:
            tags.append(record.model.name)

        result = PydanticSimulationStudy(
            label=record.label or None,
            description=record.description or None,
            doi=record.doi or None,
            title=record.title or None,
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'study',
            'name': record.title or record.label or f'Study {record.id}',
            'title': record.title or record.label or f'Study {record.id}',
            'abstract': record.description or '',
            'year': str(record.year) if record.year else '',
            'doi': record.doi or '',
            'tags': tags,
        })

        return get_ontology_api().enrich_database_item(result, 'study')

    def _get_couplings(self):
        records = request.env['tvbo.coupling'].sudo().search([])
        return [self._serialize_coupling(r) for r in records]

    def _serialize_coupling(self, record):
        tags = []
        if record.delayed:
            tags.append('delayed')
        if record.sparse:
            tags.append('sparse')

        result = PydanticCoupling(
            name=record.name or 'Linear',
            label=record.label or None,
            delayed=record.delayed,
            sparse=record.sparse,
        ).model_dump(exclude_none=True)

        result.update({
            'id': record.id,
            'type': 'coupling',
            'title': record.name or record.label or f'Coupling {record.id}',
            'description': record.coupling_function.definition if record.coupling_function else '',
            'tags': tags,
        })

        return get_ontology_api().enrich_database_item(result, 'coupling')

    # ===================
    # Ontology concepts
    # ===================

    def _get_ontology_concepts(self, query=''):
        api = get_ontology_api()
        concept_types = ['Model', 'NeuralMassModel', 'Coupling', 'IntegrationMethod',
                         'StateVariable', 'Parameter', 'BrainRegion', 'Parcellation',
                         'Tractogram', 'Monitor', 'Noise']

        results = []
        seen = set()

        for concept in concept_types:
            for node in api.search(concept, limit=50):
                storid = node.get('storid')
                if storid and storid not in seen:
                    seen.add(storid)
                    results.append(node)

        if query:
            for node in api.search(query, limit=50):
                storid = node.get('storid')
                if storid and storid not in seen:
                    seen.add(storid)
                    results.append(node)

        return results

    # ===================
    # Detail endpoints
    # ===================

    @http.route('/tvbo/api/kg/dynamics/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_dynamics_detail(self, record_id, **kw):
        record = request.env['tvbo.dynamics'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        data = self._serialize_dynamics(record)
        data['parameters'] = [{"name": p.name, "label": p.label, "description": p.description} for p in record.parameters]
        data['state_variables'] = [{
            "name": sv.name, "label": sv.label, "description": sv.description,
            "equation": {"label": sv.equation.label, "definition": sv.equation.definition} if sv.equation else None
        } for sv in record.state_variables]

        return json_response(data)

    @http.route('/tvbo/api/kg/network/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_network_detail(self, record_id, **kw):
        record = request.env['tvbo.network'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        data = self._serialize_network(record)
        if record.parcellation:
            data['parcellation'] = {"label": record.parcellation.label, "data_source": record.parcellation.data_source}

        return json_response(data)

    @http.route('/tvbo/api/kg/integrator/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_integrator_detail(self, record_id, **kw):
        record = request.env['tvbo.integrator'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        data = self._serialize_integrator(record)
        data['parameters'] = [{"name": p.name, "label": p.label, "value": p.value} for p in record.parameters]

        return json_response(data)

    @http.route('/tvbo/api/kg/coupling/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_coupling_detail(self, record_id, **kw):
        record = request.env['tvbo.coupling'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        data = self._serialize_coupling(record)
        if record.coupling_function:
            data['coupling_function'] = {"label": record.coupling_function.label, "definition": record.coupling_function.definition}
        data['parameters'] = [{"name": p.name, "label": p.label} for p in record.parameters]

        return json_response(data)

    @http.route('/tvbo/api/kg/experiment/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_experiment_detail(self, record_id, **kw):
        record = request.env['tvbo.simulation_experiment'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        data = self._serialize_experiment(record)
        if record.local_dynamics:
            data['local_dynamics'] = {"id": record.local_dynamics.id, "name": record.local_dynamics.name}
        if record.integration:
            data['integration'] = {"id": record.integration.id, "method": record.integration.method}
        if record.connectivity:
            data['connectivity'] = {"id": record.connectivity.id, "label": record.connectivity.label}

        return json_response(data)

    @http.route('/tvbo/api/kg/study/<int:record_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def get_study_detail(self, record_id, **kw):
        record = request.env['tvbo.simulation_study'].sudo().browse(record_id)
        if not record.exists():
            return json_response({"error": "Not found"}, 404)

        return json_response(self._serialize_study(record))

    # ===================
    # Ontology endpoints
    # ===================

    @http.route('/tvbo/api/kg/ontology/search', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_search(self, **kw):
        """Search ontology concepts."""
        query = kw.get('q', '')
        limit = int(kw.get('limit', 100))

        if not query:
            return json_response({"error": "Missing 'q' parameter"}, 400)

        api = get_ontology_api()
        results = api.search(query, limit=limit)

        # Build graph with relationships
        nodes = results
        links = []
        seen_storids = {n.get('storid') for n in nodes if n.get('storid')}

        for node in results[:10]:
            storid = node.get('storid')
            if storid:
                rels = api.get_relationships(storid)
                for link in rels.get('links', []):
                    if link.get('source') in seen_storids and link.get('target') in seen_storids:
                        links.append(link)

        return json_response({
            "results": results,
            "graph": {"nodes": nodes, "links": links},
            "query": query
        })

    @http.route('/tvbo/api/kg/ontology/node/<int:node_id>', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_node_detail(self, node_id, **kw):
        """Get ontology node by storid."""
        api = get_ontology_api()
        node_data = api.get_by_storid(node_id)

        if not node_data:
            return json_response({"error": "Node not found"}, 404)

        node_data['relationships'] = api.get_relationships(node_id)
        return json_response(node_data)

    @http.route('/tvbo/api/kg/ontology/node/<int:node_id>/children', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_node_children(self, node_id, **kw):
        """Get children of ontology node."""
        return json_response(get_ontology_api().get_children(node_id))

    @http.route('/tvbo/api/kg/ontology/node/<int:node_id>/parents', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_node_parents(self, node_id, **kw):
        """Get parents of ontology node."""
        return json_response(get_ontology_api().get_parents(node_id))

    @http.route('/tvbo/api/kg/ontology/graph', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_graph(self, **kw):
        """Get ontology class hierarchy."""
        return json_response(get_ontology_api().get_class_hierarchy())

    @http.route('/tvbo/api/kg/ontology/by-iri', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_by_iri(self, **kw):
        """Get ontology node by IRI."""
        iri = kw.get('iri', '')
        if not iri:
            return json_response({"error": "Missing 'iri' parameter"}, 400)

        api = get_ontology_api()
        node_data = api.get_by_iri(iri)

        if not node_data:
            return json_response({"error": "IRI not found"}, 404)

        return json_response(node_data)

    @http.route('/tvbo/api/kg/ontology/schema-link/<string:schema_class>', type='http', auth='public', methods=['GET'], csrf=False)
    def ontology_schema_link(self, schema_class, **kw):
        """Get ontology concept linked to schema class."""
        api = get_ontology_api()
        node_data = api.get_schema_ontology_link(schema_class)

        if not node_data:
            return json_response({"error": f"Schema class '{schema_class}' not linked"}, 404)

        if node_data.get('storid'):
            node_data['relationships'] = api.get_relationships(node_data['storid'])

        return json_response(node_data)

    @http.route('/tvbo/api/kg/ontology/sparql', type='http', auth='public', methods=['GET', 'POST'], csrf=False)
    def ontology_sparql(self, **kw):
        """Execute SPARQL query."""
        if request.httprequest.method == 'POST':
            query_string = request.httprequest.data.decode('utf-8')
        else:
            query_string = kw.get('query', '')

        if not query_string:
            return json_response({"error": "Missing 'query' parameter"}, 400)

        api = get_ontology_api()
        results = api.sparql(query_string, flatten=False)

        serialized = []
        for row in results:
            row_data = []
            for item in row:
                if hasattr(item, 'iri'):
                    label = getattr(item, 'label', [item.name])[0] if hasattr(item, 'label') else item.name
                    row_data.append({"iri": item.iri, "label": label})
                else:
                    row_data.append(str(item) if item is not None else None)
            serialized.append(row_data)

        return json_response({"results": serialized, "count": len(serialized)})

    # ===================
    # Graph visualization endpoint
    # ===================

    @http.route('/tvbo/api/kg/graph', type='http', auth='public', methods=['GET'], csrf=False)
    def get_knowledge_graph(self, **kw):
        """Get full knowledge graph: ontology + database items with relationships.

        Optional query params:
          - limit: max ontology classes to include (default: all)
          - db_only: if 'true', only include database items, not full ontology
        """
        api = get_ontology_api()
        db_only = kw.get('db_only', 'false').lower() == 'true'
        limit = int(kw.get('limit', 0)) if kw.get('limit') else 0

        nodes = []
        links = []

        # Get ontology class hierarchy (unless db_only)
        if not db_only:
            hierarchy = api.get_class_hierarchy()
            nodes = hierarchy.get('nodes', [])
            links = hierarchy.get('links', [])

            # Apply limit if specified
            if limit > 0 and len(nodes) > limit:
                nodes = nodes[:limit]
                # Filter links to only include nodes in our limited set
                node_ids = {n.get('storid') for n in nodes}
                links = [l for l in links if l.get('source') in node_ids and l.get('target') in node_ids]

        # Add database items as nodes linked to their ontology classes
        db_items = []
        db_items.extend(self._get_dynamics())
        db_items.extend(self._get_networks())
        db_items.extend(self._get_integrators())
        db_items.extend(self._get_experiments())
        db_items.extend(self._get_studies())
        db_items.extend(self._get_couplings())

        # Map database items to nodes and create links to ontology
        onto_storid_map = {n['storid']: n for n in nodes if n.get('storid')}

        for item in db_items:
            node_id = f"db_{item['type']}_{item['id']}"
            node = {
                'id': node_id,
                'storid': node_id,
                'label': item.get('name') or item.get('label') or item.get('title', ''),
                'type': 'instance',
                'db_type': item['type'],
                'title': item.get('title', ''),
                'iri': item.get('iri', ''),
            }
            nodes.append(node)

            # Link to ontology class if enriched
            if item.get('ontology_storid'):
                onto_storid = item['ontology_storid']
                if onto_storid in onto_storid_map:
                    links.append({
                        'source': node_id,
                        'target': onto_storid,
                        'type': 'instance_of',
                        'label': 'instance of',
                    })

        return json_response({
            "nodes": nodes,
            "links": links,
            "stats": {
                "ontology_classes": len([n for n in nodes if n.get('type') != 'instance']),
                "database_items": len(db_items),
                "total_nodes": len(nodes),
                "total_links": len(links),
            }
        })
