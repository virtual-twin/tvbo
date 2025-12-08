# -*- coding: utf-8 -*-
from odoo import http
from odoo.http import request
from markupsafe import Markup
import json
import logging

_logger = logging.getLogger(__name__)


class ModelConfiguratorController(http.Controller):

    @http.route('/tvbo/configurator', type='http', auth='public', website=True)
    def model_configurator(self, **kwargs):
        """Main configurator page - loads existing models for browsing"""
        try:
            # Fetch all models, integrators, networks, coupling functions for selection
            models = request.env['tvbo.neural_mass_model'].sudo().search([])
            integrators = request.env['tvbo.integrator'].sudo().search([])
            couplings = request.env['tvbo.coupling'].sudo().search([])
            monitors = request.env['tvbo.monitor'].sudo().search([])
            networks = request.env['tvbo.network'].sudo().search([])
            _logger.info(f"Found {len(models)} neural mass models, {len(integrators)} integrators, {len(couplings)} couplings, {len(monitors)} monitors, {len(networks)} networks")

            # Prepare model data for JS
            model_data = []
            for model in models:
                try:
                    _logger.info(f"Processing model: {model.name}")
                    model_dict = {
                        'id': model.id,
                        'key': model.name,
                        'title': model.label or model.name,
                        'name': model.name,
                        'description': model.description or '',
                        'type': 'model',
                        'system_type': model.system_type.technical_name if model.system_type else 'continuous',
                        'parameters': [],
                        'derived_parameters': [],
                        'state_variables': [],
                        'derived_variables': [],
                        'output': [],
                        'functions': [],
                        'coupling_terms': [],
                    }

                    # Add parameters
                    try:
                        for param in model.parameters:
                            param_dict = {
                                'name': param.name,
                                'value': param.value,
                                'unit': param.unit or '',
                                'description': param.description or '',
                            }
                            if param.domain:
                                param_dict['domain'] = {
                                    'lo': param.domain.lo,
                                    'hi': param.domain.hi,
                                    'step': param.domain.step,
                                }
                            model_dict['parameters'].append(param_dict)
                        _logger.info(f"Added {len(model_dict['parameters'])} parameters")
                    except Exception as e:
                        _logger.error(f"Error loading parameters for {model.name}: {e}")

                    # Add derived_parameters
                    try:
                        if hasattr(model, 'derived_parameters'):
                            for dparam in model.derived_parameters:
                                dparam_dict = {
                                    'name': dparam.name,
                                    'unit': dparam.unit or '',
                                    'description': dparam.description or '',
                                }
                                if dparam.equation:
                                    dparam_dict['equation'] = {
                                        'lhs': dparam.equation.lefthandside or '',
                                        'rhs': dparam.equation.righthandside or '',
                                    }
                                model_dict['derived_parameters'].append(dparam_dict)
                            _logger.info(f"Added {len(model_dict['derived_parameters'])} derived parameters")
                    except Exception as e:
                        _logger.error(f"Error loading derived_parameters for {model.name}: {e}")

                    # Add state variables
                    try:
                        for sv in model.state_variables:
                            sv_dict = {
                                'name': sv.name,
                                'description': sv.description or '',
                                'initial_value': sv.initial_value,
                            }
                            if sv.equation:
                                sv_dict['equation'] = {
                                    'lhs': sv.equation.lefthandside or '',
                                    'rhs': sv.equation.righthandside or '',
                                }
                            if sv.domain:
                                sv_dict['domain'] = {
                                    'lo': sv.domain.lo,
                                    'hi': sv.domain.hi,
                                }
                            model_dict['state_variables'].append(sv_dict)
                        _logger.info(f"Added {len(model_dict['state_variables'])} state variables")
                    except Exception as e:
                        _logger.error(f"Error loading state variables for {model.name}: {e}")

                    # Add derived variables
                    try:
                        for dv in model.derived_variables:
                            dv_dict = {
                                'name': dv.name,
                                'description': dv.description or '',
                            }
                            if dv.equation:
                                dv_dict['equation'] = {
                                    'lhs': dv.equation.lefthandside or '',
                                    'rhs': dv.equation.righthandside or '',
                                }
                            model_dict['derived_variables'].append(dv_dict)
                    except Exception as e:
                        _logger.error(f"Error loading derived variables for {model.name}: {e}")

                    # Add output
                    try:
                        if hasattr(model, 'output'):
                            for ot in model.output:
                                ot_dict = {
                                    'name': ot.name,
                                    'description': ot.description or '',
                                    'unit': ot.unit or '',
                                }
                                if ot.equation:
                                    ot_dict['equation'] = {
                                        'lhs': ot.equation.lefthandside or '',
                                        'rhs': ot.equation.righthandside or '',
                                    }
                                model_dict['output'].append(ot_dict)
                            _logger.info(f"Added {len(model_dict['output'])} output transforms")
                    except Exception as e:
                        _logger.error(f"Error loading output for {model.name}: {e}")

                    # Add functions
                    try:
                        if hasattr(model, 'functions'):
                            for fn in model.functions:
                                fn_dict = {
                                    'name': fn.name,
                                    'description': fn.description or '',
                                }
                                if fn.equation:
                                    fn_dict['equation'] = {
                                        'lhs': fn.equation.lefthandside or '',
                                        'rhs': fn.equation.righthandside or '',
                                    }
                                model_dict['functions'].append(fn_dict)
                            _logger.info(f"Added {len(model_dict['functions'])} functions")
                    except Exception as e:
                        _logger.error(f"Error loading functions for {model.name}: {e}")

                    # Add coupling terms
                    try:
                        for ct in model.coupling_terms:
                            model_dict['coupling_terms'].append({
                                'name': ct.name,
                                'value': ct.value,
                            })
                    except Exception as e:
                        _logger.error(f"Error loading coupling terms for {model.name}: {e}")

                    model_data.append(model_dict)
                except Exception as e:
                    _logger.error(f"Error processing model {model.name}: {e}", exc_info=True)

            _logger.info(f"Returning {len(model_data)} models to template")

            # Prepare integrator data
            integrator_data = []
            for integrator in integrators:
                try:
                    integrator_dict = {
                        'id': integrator.id,
                        'name': integrator.method or f'Integrator_{integrator.id}',
                        'method': integrator.method or 'HeunDeterministic',
                        'step_size': integrator.step_size,
                        'duration': integrator.duration,
                    }
                    integrator_data.append(integrator_dict)
                except Exception as e:
                    _logger.error(f"Error processing integrator: {e}")

            # Prepare coupling data
            coupling_data = []
            for coupling in couplings:
                try:
                    coupling_dict = {
                        'id': coupling.id,
                        'name': coupling.name,
                        'label': coupling.label or coupling.name,
                    }
                    coupling_data.append(coupling_dict)
                except Exception as e:
                    _logger.error(f"Error processing coupling: {e}")

            # Prepare monitor data
            monitor_data = []
            for monitor in monitors:
                try:
                    monitor_dict = {
                        'id': monitor.id,
                        'name': monitor.name,
                        'label': monitor.label or monitor.name,
                        'period': monitor.period,
                    }
                    monitor_data.append(monitor_dict)
                except Exception as e:
                    _logger.error(f"Error processing monitor: {e}")

            # Prepare network data
            network_data = []
            tractograms_set = set()
            parcellations_set = set()
            for network in networks:
                try:
                    network_dict = {
                        'id': network.id,
                        'label': network.label or f'Network_{network.id}',
                        'number_of_regions': network.number_of_regions,
                        'tractogram': network.tractogram or '',
                        'parcellation': network.parcellation.label if network.parcellation else '',
                        'parcellation_id': network.parcellation.id if network.parcellation else None,
                    }
                    network_data.append(network_dict)

                    # Collect unique tractograms and parcellations
                    if network.tractogram:
                        tractograms_set.add(network.tractogram)
                    if network.parcellation and network.parcellation.label:
                        parcellations_set.add((network.parcellation.id, network.parcellation.label))
                except Exception as e:
                    _logger.error(f"Error processing network: {e}")

            # Convert to list for JSON
            tractograms_data = sorted(list(tractograms_set))
            parcellations_data = [{'id': p[0], 'label': p[1]} for p in sorted(parcellations_set, key=lambda x: x[1])]

            return request.render('tvbo.model_configurator_template', {
                'models_json': Markup(json.dumps(model_data)),
                'integrators_json': Markup(json.dumps(integrator_data)),
                'couplings_json': Markup(json.dumps(coupling_data)),
                'monitors_json': Markup(json.dumps(monitor_data)),
                'networks_json': Markup(json.dumps(network_data)),
                'tractograms_json': Markup(json.dumps(tractograms_data)),
                'parcellations_json': Markup(json.dumps(parcellations_data)),
            })
        except Exception as e:
            _logger.error(f"Error in model_configurator: {e}", exc_info=True)
            return request.render('tvbo.model_configurator_template', {
                'models_json': Markup('[]'),
                'integrators_json': Markup('[]'),
                'couplings_json': Markup('[]'),
                'monitors_json': Markup('[]'),
                'networks_json': Markup('[]'),
                'tractograms_json': Markup('[]'),
                'parcellations_json': Markup('[]'),
            })

    @http.route('/tvbo/configurator/save', type='jsonrpc', auth='user', website=True, csrf=True)
    def save_model(self, **kwargs):
        """Save a new neural mass model configuration to database"""
        try:
            data = kwargs.get('model_data')
            if not data:
                return {'success': False, 'error': 'No model data provided'}

            _logger.info(f"Saving model: {data.get('name')}")

            # Create the neural mass model
            model_vals = {
                'name': data.get('name'),
                'label': data.get('label') or data.get('name'),
                'description': data.get('description', ''),
                'number_of_modes': data.get('number_of_modes', 1),
            }

            # First create the Dynamics record (since NeuralMassModel inherits from it)
            dynamics = request.env['tvbo.dynamics'].sudo().create(model_vals)

            # Create parameters
            param_ids = []
            for param_data in data.get('parameters', []):
                # Create domain (Range) if provided
                domain_id = None
                if param_data.get('domain'):
                    domain_vals = {
                        'lo': param_data['domain'].get('lo'),
                        'hi': param_data['domain'].get('hi'),
                        'step': param_data['domain'].get('step'),
                    }
                    domain = request.env['tvbo.range'].sudo().create(domain_vals)
                    domain_id = domain.id

                param_vals = {
                    'name': param_data.get('name'),
                    'value': param_data.get('value'),
                    'unit': param_data.get('unit'),
                    'description': param_data.get('description'),
                    'domain': domain_id,
                }
                param = request.env['tvbo.parameter'].sudo().create(param_vals)
                param_ids.append(param.id)

            # Create state variables
            sv_ids = []
            for sv_data in data.get('state_variables', []):
                # Create equation if provided
                eq_id = None
                if sv_data.get('equation'):
                    eq_vals = {
                        'lefthandside': sv_data['equation'].get('lhs'),
                        'righthandside': sv_data['equation'].get('rhs'),
                    }
                    equation = request.env['tvbo.equation'].sudo().create(eq_vals)
                    eq_id = equation.id

                # Create domain if provided
                domain_id = None
                if sv_data.get('domain'):
                    domain_vals = {
                        'lo': sv_data['domain'].get('lo'),
                        'hi': sv_data['domain'].get('hi'),
                    }
                    domain = request.env['tvbo.range'].sudo().create(domain_vals)
                    domain_id = domain.id

                sv_vals = {
                    'name': sv_data.get('name'),
                    'description': sv_data.get('description'),
                    'initial_value': sv_data.get('initial_value', 0.1),
                    'equation': eq_id,
                    'domain': domain_id,
                }
                sv = request.env['tvbo.state_variable'].sudo().create(sv_vals)
                sv_ids.append(sv.id)

            # Create derived variables
            dv_ids = []
            for dv_data in data.get('derived_variables', []):
                eq_id = None
                if dv_data.get('equation'):
                    eq_vals = {
                        'lefthandside': dv_data['equation'].get('lhs'),
                        'righthandside': dv_data['equation'].get('rhs'),
                    }
                    equation = request.env['tvbo.equation'].sudo().create(eq_vals)
                    eq_id = equation.id

                dv_vals = {
                    'name': dv_data.get('name'),
                    'description': dv_data.get('description'),
                    'equation': eq_id,
                }
                dv = request.env['tvbo.derived_variable'].sudo().create(dv_vals)
                dv_ids.append(dv.id)

            # Create coupling terms
            ct_ids = []
            for ct_data in data.get('coupling_terms', []):
                ct_vals = {
                    'name': ct_data.get('name'),
                    'value': ct_data.get('value'),
                }
                ct = request.env['tvbo.parameter'].sudo().create(ct_vals)
                ct_ids.append(ct.id)

            # Update the dynamics record with all relationships
            dynamics.write({
                'parameters': [(6, 0, param_ids)],
                'state_variables': [(6, 0, sv_ids)],
                'derived_variables': [(6, 0, dv_ids)],
                'coupling_terms': [(6, 0, ct_ids)],
            })

            # Create the NeuralMassModel record linked to the Dynamics
            nmm = request.env['tvbo.neural_mass_model'].sudo().create({
                'dynamics_id': dynamics.id,
            })

            return {
                'success': True,
                'model_id': nmm.id,
                'message': f'Model "{data.get("name")}" created successfully!'
            }

        except Exception as e:
            _logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    @http.route('/tvbo/configurator/run', type='jsonrpc', auth='public', website=True, csrf=False)
    def run_simulation(self, **kwargs):
        """
        Run a simulation experiment by proxying to the TVBO API container.
        """
        import requests
        import os

        try:
            experiment_data = kwargs.get('experiment')
            duration = kwargs.get('duration')
            step_size = kwargs.get('step_size')
            backend = kwargs.get('backend')

            # MVP: Fail explicitly if required params missing
            if not experiment_data:
                return {'success': False, 'error': 'No experiment data provided'}
            if duration is None:
                return {'success': False, 'error': 'duration is required'}
            if step_size is None:
                return {'success': False, 'error': 'step_size is required'}
            if not backend:
                return {'success': False, 'error': 'backend is required'}

            _logger.info(f"Running simulation: duration={duration}ms, step_size={step_size}ms, backend={backend}")
            _logger.info(f"Experiment data: {experiment_data}")

            # Build the request payload for TVBO API
            payload = {
                'experiment': experiment_data,
                'duration': float(duration),
                'step_size': float(step_size),
                'backend': backend,
            }

            # Call the TVBO API container
            tvbo_api_url = os.environ.get('TVBO_API_URL', 'http://tvbo-api:8000')
            _logger.info(f"Calling TVBO API at {tvbo_api_url}/experiment/run")

            response = requests.post(
                f'{tvbo_api_url}/experiment/run',
                json=payload,
                timeout=300
            )

            _logger.info(f"TVBO API response status: {response.status_code}")
            result = response.json()
            _logger.info(f"TVBO API response keys: {list(result.keys())}")
            _logger.info(f"TVBO API success: {result.get('success')}")

            if response.status_code != 200:
                error_msg = result.get('detail', response.text)
                _logger.error(f"TVBO API error: {error_msg}")
                return {'success': False, 'error': f'TVBO API error: {error_msg}'}

            if not result.get('success'):
                error_msg = result.get('error', 'Unknown error from TVBO API')
                _logger.error(f"TVBO API returned failure: {error_msg}")
                return {'success': False, 'error': error_msg}

            # MVP: No fallbacks - pass through exactly what API returns
            _logger.info(f"Returning data with {len(result.get('data', []))} time points")
            return {
                'success': True,
                'data': result.get('data'),
                'time': result.get('time'),
                'state_variables': result.get('state_variables'),
                'region_labels': result.get('region_labels'),
                'sample_period': result.get('sample_period'),
                'message': 'Simulation completed successfully'
            }

        except requests.exceptions.ConnectionError:
            _logger.error("Cannot connect to TVBO API container")
            return {
                'success': False,
                'error': 'Cannot connect to TVBO API. Please ensure the tvbo-api container is running.'
            }
        except requests.exceptions.Timeout:
            _logger.error("TVBO API request timed out")
            return {
                'success': False,
                'error': 'Simulation timed out. Try reducing the duration or increasing step size.'
            }
        except Exception as e:
            _logger.error(f"Error running simulation: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
