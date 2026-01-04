# -*- coding: utf-8 -*-
#
# Module: utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
TVB-Optim Template Utilities
============================

Reusable Python functions for tvboptim Mako templates.
Import these in template blocks to avoid code duplication.

Usage in templates:
    <%
    from tvbo.templates.tvboptim.utils import (
        safe_name, as_list, is_network_observation,
        parse_loss_function, get_observation_refs
    )
    %>
"""
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# Basic Helpers
# =============================================================================

def safe_name(name: str) -> str:
    """Convert name to valid Python identifier."""
    return str(name).replace(' ', '_').replace('-', '_').lower()


def as_list(obj: Any) -> list:
    """Convert dict or list to list of values."""
    if obj is None:
        return []
    if hasattr(obj, 'values'):
        return list(obj.values())
    return list(obj)


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute access."""
    return getattr(obj, name, default) if obj else default


def to_numeric(val: Any) -> Union[int, float, Any]:
    """Convert string to numeric if possible."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return int(val) if '.' not in val else float(val)
        except ValueError:
            return val
    return val


# =============================================================================
# Observation Helpers
# =============================================================================

def is_network_observation(obs: Any) -> bool:
    """Check if observation is a network observation (static data from BIDS).
    
    Network observations have source starting with 'network.observations'.
    """
    if not obs:
        return False
    source = getattr(obs, 'source', None)
    if source and str(source).startswith('network.observations'):
        return True
    return False


def is_external_observation(obs: Any) -> bool:
    """Check if observation is external (has data_source or network.observations source)."""
    if not obs:
        return False
    # Explicit data_source
    if getattr(obs, 'data_source', None):
        return True
    # Source pointing to network.observations.*
    return is_network_observation(obs)


def obs_has_all_args(obs: Any) -> bool:
    """Check if observation has all required arguments satisfied.
    
    Returns True if all pipeline step arguments either have values
    or are implicitly satisfied by source.
    """
    pipeline = getattr(obs, 'pipeline', None) or []
    has_source = getattr(obs, 'source', None) or getattr(obs, 'source_observation', None)
    
    for step_idx, func in enumerate(pipeline):
        is_first_step = step_idx == 0
        args = getattr(func, 'arguments', None) or []
        for arg in args:
            if getattr(arg, 'name', None) and getattr(arg, 'value', None) is None:
                # First step's data-like args are satisfied by source
                if is_first_step and has_source and arg.name in ('data', 'X', 'x', 'input', 'timeseries', 'a'):
                    continue
                return False
    return True


def get_observation_refs(observations_dict: Dict[str, Any]) -> Tuple[Set[str], List[str]]:
    """Categorize observations into network vs simulation-derived.
    
    Returns:
        (network_observation_names, observation_names_with_all_args)
    """
    network_obs = set()
    valid_obs = []
    
    for name, obs in observations_dict.items():
        if is_network_observation(obs):
            network_obs.add(name)
        if obs_has_all_args(obs):
            valid_obs.append(name)
    
    return network_obs, valid_obs


# =============================================================================
# Loss Function Parsing
# =============================================================================

def parse_loss_arguments(loss_call: Any) -> Tuple[List[Dict], Set[str]]:
    """Parse loss function call arguments.
    
    Returns:
        (parsed_args, obs_refs) where:
        - parsed_args: list of dicts with 'name', 'type', and type-specific keys
        - obs_refs: set of observation names referenced
    """
    loss_args = getattr(loss_call, 'arguments', None) or []
    parsed_args = []
    obs_refs = set()
    
    for arg in loss_args:
        arg_name = getattr(arg, 'name', None)
        arg_value = getattr(arg, 'value', None)
        
        if not arg_name:
            continue
            
        if arg_value is not None:
            val_str = str(arg_value)
            
            # Check if numeric constant
            try:
                float(arg_value)
                parsed_args.append({
                    'name': arg_name,
                    'type': 'constant',
                    'value': arg_value,
                })
                continue
            except (ValueError, TypeError):
                pass
            
            # Parse observation references
            if val_str.startswith('observations.'):
                parts = val_str.split('.', 2)
                obs_name = parts[1] if len(parts) > 1 else None
                output_key = parts[2] if len(parts) > 2 else None
                if obs_name:
                    obs_refs.add(obs_name)
                    parsed_args.append({
                        'name': arg_name,
                        'type': 'observation',
                        'obs_name': obs_name,
                        'output_key': output_key,
                    })
            elif '.' in val_str:
                # Old-style obs_name.key
                obs_name, output_key = val_str.split('.', 1)
                obs_refs.add(obs_name)
                parsed_args.append({
                    'name': arg_name,
                    'type': 'observation',
                    'obs_name': obs_name,
                    'output_key': output_key,
                })
            else:
                # Just observation name
                obs_refs.add(val_str)
                parsed_args.append({
                    'name': arg_name,
                    'type': 'observation',
                    'obs_name': val_str,
                    'output_key': None,
                })
        else:
            # No value = runtime input
            parsed_args.append({
                'name': arg_name,
                'type': 'runtime',
                'kwarg_name': arg_name,
            })
    
    return parsed_args, obs_refs


def parse_loss_function(opt: Any) -> Optional[Dict]:
    """Parse optimization loss function specification.
    
    Returns dict with: opt_name, func_name, args, obs_refs, agg_over, agg_type
    or None if no loss defined.
    """
    loss_call = getattr(opt, 'loss', None)
    if not loss_call:
        return None
    
    # Determine function name
    func_ref = getattr(loss_call, 'function', None)
    callable_ref = getattr(loss_call, 'callable', None)
    
    if func_ref:
        func_name = str(func_ref) if isinstance(func_ref, str) else (
            getattr(func_ref, 'name', None) or str(func_ref)
        )
    elif callable_ref:
        func_name = getattr(callable_ref, 'name', None) or getattr(callable_ref, 'qualname', None) or 'loss'
    else:
        func_name = 'loss'
    
    # Parse aggregate specification
    aggregate = getattr(loss_call, 'aggregate', None)
    agg_over = None
    agg_type = None
    if aggregate:
        agg_over = str(getattr(aggregate, 'over', '')).split('.')[-1] or None
        agg_type = str(getattr(aggregate, 'type', 'mean')).split('.')[-1]
    
    # Parse arguments
    parsed_args, obs_refs = parse_loss_arguments(loss_call)
    
    return {
        'opt_name': getattr(opt, 'name', None) or 'loss',
        'func_name': func_name,
        'args': parsed_args,
        'obs_refs': obs_refs,
        'agg_over': agg_over,
        'agg_type': agg_type,
    }


# =============================================================================
# Parameter Parsing
# =============================================================================

def get_domain_bounds(param_name: str, model: Any, all_couplings: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Lookup domain bounds from model.parameters or coupling.parameters.
    
    Returns (lo, hi) tuple, where None means unbounded.
    """
    def extract_bounds(param):
        domain = getattr(param, 'domain', None)
        if domain:
            lo = getattr(domain, 'lo', None)
            hi = getattr(domain, 'hi', None)
            try:
                return (float(lo) if lo is not None else None,
                        float(hi) if hi is not None else None)
            except (TypeError, ValueError):
                pass
        return (None, None)
    
    # Check dynamics parameters
    if model and hasattr(model, 'parameters') and param_name in model.parameters:
        lo, hi = extract_bounds(model.parameters[param_name])
        if lo is not None or hi is not None:
            return (lo, hi)
    
    # Check coupling parameters
    for cobj in all_couplings.values():
        if hasattr(cobj, 'parameters') and cobj.parameters and param_name in cobj.parameters:
            return extract_bounds(cobj.parameters[param_name])
    
    return (None, None)


def parse_free_param(fp: Any, coupling_keys: Set[str], model: Any = None, all_couplings: Dict = None) -> Optional[Dict]:
    """Parse a free_parameter entry.
    
    Handles: str, dotted notation, stringified dict, dict, and Parameter objects.
    
    Returns dict with: name, heterogeneous, shape, coupling_key, dynamics_key,
                       lower_bound, upper_bound
    """
    all_couplings = all_couplings or {}
    result = None
    source_key = None
    is_coupling = False
    
    if isinstance(fp, str):
        stripped = fp.strip()
        
        # Check for stringified dict
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict) and 'name' in parsed:
                    param_name = str(parsed['name'])
                    if '.' in param_name:
                        prefix, param_name = param_name.rsplit('.', 1)
                        is_coupling = prefix in coupling_keys
                        source_key = prefix
                    result = {
                        'name': param_name,
                        'heterogeneous': bool(parsed.get('heterogeneous', False)),
                        'shape': parsed.get('shape'),
                        'coupling_key': source_key if is_coupling else None,
                        'dynamics_key': source_key if not is_coupling and source_key else None,
                    }
            except (ValueError, SyntaxError):
                pass
        
        if result is None:
            # Check for dotted notation
            if '.' in stripped:
                prefix, param_name = stripped.rsplit('.', 1)
                is_coupling = prefix in coupling_keys
                source_key = prefix
                result = {
                    'name': param_name,
                    'heterogeneous': False,
                    'shape': None,
                    'coupling_key': source_key if is_coupling else None,
                    'dynamics_key': source_key if not is_coupling else None,
                }
            else:
                result = {
                    'name': fp,
                    'heterogeneous': False,
                    'shape': None,
                    'coupling_key': None,
                    'dynamics_key': None,
                }
    
    elif isinstance(fp, dict) and 'name' in fp:
        param_name = str(fp['name'])
        if '.' in param_name:
            prefix, param_name = param_name.rsplit('.', 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        result = {
            'name': param_name,
            'heterogeneous': bool(fp.get('heterogeneous', False)),
            'shape': str(fp['shape']) if fp.get('shape') else None,
            'coupling_key': source_key if is_coupling else None,
            'dynamics_key': source_key if not is_coupling and source_key else None,
        }
        # Check for domain in dict
        domain = fp.get('domain', {})
        if isinstance(domain, dict):
            if 'lo' in domain:
                try:
                    result['lower_bound'] = float(domain['lo'])
                except (TypeError, ValueError):
                    pass
            if 'hi' in domain:
                try:
                    result['upper_bound'] = float(domain['hi'])
                except (TypeError, ValueError):
                    pass
    
    elif not isinstance(fp, (str, dict)):
        # Parameter object
        param_name = str(getattr(fp, 'name', ''))
        if '.' in param_name:
            prefix, param_name = param_name.rsplit('.', 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        result = {
            'name': param_name,
            'heterogeneous': bool(getattr(fp, 'heterogeneous', False)),
            'shape': str(fp.shape) if getattr(fp, 'shape', None) else None,
            'coupling_key': source_key if is_coupling else None,
            'dynamics_key': source_key if not is_coupling and source_key else None,
        }
        # Check domain on Parameter object
        domain = getattr(fp, 'domain', None)
        if domain:
            lo = getattr(domain, 'lo', None)
            hi = getattr(domain, 'hi', None)
            if lo is not None:
                try:
                    result['lower_bound'] = float(lo)
                except (TypeError, ValueError):
                    pass
            if hi is not None:
                try:
                    result['upper_bound'] = float(hi)
                except (TypeError, ValueError):
                    pass
    
    if result is None:
        return None
    
    # Set defaults
    result.setdefault('coupling_key', None)
    result.setdefault('dynamics_key', None)
    
    # Lookup bounds from model if not specified
    if 'lower_bound' not in result or 'upper_bound' not in result:
        if model or all_couplings:
            model_lo, model_hi = get_domain_bounds(result['name'], model, all_couplings)
            if 'lower_bound' not in result and model_lo is not None:
                result['lower_bound'] = model_lo
            if 'upper_bound' not in result and model_hi is not None:
                result['upper_bound'] = model_hi
    
    result.setdefault('lower_bound', None)
    result.setdefault('upper_bound', None)
    result.setdefault('shape', None)
    
    # Auto-detect coupling parameters
    if result.get('coupling_key') is None and model and all_couplings:
        param_name = result['name']
        is_dynamics = hasattr(model, 'parameters') and param_name in model.parameters
        if not is_dynamics:
            for ck, cobj in all_couplings.items():
                if hasattr(cobj, 'parameters') and cobj.parameters and param_name in cobj.parameters:
                    result['coupling_key'] = ck
                    break
    
    return result


# =============================================================================
# Exploration Parsing
# =============================================================================

def parse_exploration(expl: Any, all_couplings: Dict, get_pipeline_output_key_fn=None) -> Dict:
    """Parse exploration specification from YAML.
    
    Returns dict with: name, label, mode, n_parallel, axes, observable_*
    """
    exp_info = {
        'name': getattr(expl, 'name', ''),
        'label': getattr(expl, 'label', '') or '',
        'mode': getattr(expl, 'mode', None) or 'product',
        'n_parallel': int(getattr(expl, 'n_parallel', 1) or 1),
        'axes': [],
    }
    
    # Parse parameters
    params = getattr(expl, 'parameters', {})
    if hasattr(params, 'values'):
        params = params.values()
    
    for param in params:
        domain = getattr(param, 'domain', None)
        if not domain:
            continue
        
        pname = str(getattr(param, 'name', ''))
        source_key = None
        is_coupling_param = False
        
        if '.' in pname:
            prefix, pname = pname.rsplit('.', 1)
            is_coupling_param = prefix in all_couplings
            source_key = prefix
        
        exp_info['axes'].append({
            'name': pname,
            'lo': float(getattr(domain, 'lo', 0)),
            'hi': float(getattr(domain, 'hi', 1)),
            'n': int(getattr(domain, 'n', 10)),
            'is_coupling': is_coupling_param,
            'coupling_key': source_key if is_coupling_param else None,
            'dynamics_key': source_key if not is_coupling_param and source_key else None,
        })
    
    # Parse observable
    observable = getattr(expl, 'observable', None)
    if observable:
        func = getattr(observable, 'function', None)
        func_name = getattr(func, 'name', None) if hasattr(func, 'name') else str(func) if func else None
        args = getattr(observable, 'arguments', None) or []
        
        if args:
            exp_info['observable_type'] = 'function_call'
            exp_info['observable_func'] = func_name
            exp_info['observable_args'] = []
            for arg in args:
                arg_name = getattr(arg, 'name', None) or str(arg)
                arg_value = getattr(arg, 'value', None)
                if arg_value:
                    val_str = str(arg_value)
                    if '.' in val_str:
                        obs_ref, output_key = val_str.split('.', 1)
                        exp_info['observable_args'].append({'name': arg_name, 'obs': obs_ref, 'key': output_key})
                    else:
                        exp_info['observable_args'].append({'name': arg_name, 'obs': val_str, 'key': 'data'})
                else:
                    exp_info['observable_args'].append({'name': arg_name, 'obs': None, 'key': None})
        else:
            exp_info['observable_type'] = 'observation'
            exp_info['observable'] = func_name
            if get_pipeline_output_key_fn and func_name:
                exp_info['output_key'] = get_pipeline_output_key_fn(func_name)
            else:
                exp_info['output_key'] = None
    
    return exp_info


# =============================================================================
# Algorithm Helpers
# =============================================================================

def get_include_info(inc: Any) -> Tuple[str, Dict]:
    """Extract algorithm name and argument overrides from AlgorithmInclude.
    
    Returns (algo_name, {param_name: value}) tuple.
    """
    if hasattr(inc, 'algorithm'):
        algo = getattr(inc, 'algorithm', None)
        algo_name = getattr(algo, 'name', None) if hasattr(algo, 'name') else str(algo)
        args = {}
        for arg in as_list(getattr(inc, 'arguments', None)):
            name = getattr(arg, 'name', None)
            if name:
                args[str(name)] = getattr(arg, 'value', None)
        return algo_name, args
    return str(inc), {}


def get_all_observations_from_algo(algo: Any, algorithms_dict: Dict) -> List[str]:
    """Get all observation names including from included algorithms."""
    obs = []
    seen = set()
    
    # From included algorithms
    for inc in as_list(getattr(algo, 'includes', None)):
        inc_name, _ = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for o in as_list(getattr(inc_algo, 'observations', None)):
                o_str = str(o)
                if o_str not in seen:
                    obs.append(o_str)
                    seen.add(o_str)
    
    # This algorithm's observations
    for o in as_list(getattr(algo, 'observations', None)):
        o_str = str(o)
        if o_str not in seen:
            obs.append(o_str)
            seen.add(o_str)
    
    return obs


def get_all_hyperparams(algo: Any, algorithms_dict: Dict) -> Dict:
    """Get all hyperparameters including from included algorithms."""
    all_hp = {}
    
    for inc in as_list(getattr(algo, 'includes', None)):
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for hp in as_list(getattr(inc_algo, 'hyperparameters', None)):
                hp_name = str(getattr(hp, 'name', ''))
                if hp_name in arg_overrides:
                    all_hp[hp_name] = arg_overrides[hp_name]
                else:
                    all_hp[hp_name] = getattr(hp, 'value', None)
    
    for hp in as_list(getattr(algo, 'hyperparameters', None)):
        all_hp[str(getattr(hp, 'name', ''))] = getattr(hp, 'value', None)
    
    return all_hp
