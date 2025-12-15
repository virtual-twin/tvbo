# -*- coding: utf-8 -*-
import json
from odoo import http
from odoo.http import request


class TVBOWebsite(http.Controller):

    @http.route('/', type='http', auth='public', website=True)
    def index(self, **kw):
        """TVBO Database homepage."""
        return request.render('tvbo.tvbo_homepage_view', {})

    @http.route('/tvbo/models', type='http', auth='public', website=True)
    def models(self, **kw):
        """List all neural models."""
        models = request.env['tvbo.neural_mass_model'].sudo().search([])
        return request.render('tvbo.tvbo_models_list', {
            'models': models,
        })

    @http.route('/tvbo/model/<int:model_id>', type='http', auth='public', website=True)
    def model_detail(self, model_id, **kw):
        """Show detailed neural model information."""
        model = request.env['tvbo.neural_mass_model'].sudo().browse(model_id)
        return request.render('tvbo.tvbo_model_detail', {
            'model': model,
        })

    @http.route('/tvbo/integrators', type='http', auth='public', website=True)
    def integrators(self, **kw):
        """List all integrators."""
        integrators = request.env['tvbo.integrator'].sudo().search([])
        return request.render('tvbo.tvbo_integrators_list', {
            'integrators': integrators,
        })

    @http.route('/tvbo/networks', type='http', auth='public', website=True)
    def networks(self, **kw):
        """List all connectivity networks."""
        networks = request.env['tvbo.network'].sudo().search([])
        return request.render('tvbo.tvbo_networks_list', {
            'networks': networks,
        })

    @http.route('/tvbo/studies', type='http', auth='public', website=True)
    def studies(self, **kw):
        """List all research studies."""
        studies = request.env['tvbo.simulation_experiment'].sudo().search([])
        return request.render('tvbo.tvbo_studies_list', {
            'studies': studies,
        })

    @http.route('/tvbo/kg', type='http', auth='public', website=True)
    def knowledge_graph(self, **kw):
        """Knowledge Graph Browser - API-powered search interface."""
        return request.render('tvbo.tvbo_kg_browser', {})
