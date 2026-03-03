# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Stimulus / Event Template
====================================

Generates AbstractExternalInput subclasses from experiment.events.

Each event with event_type == 'stimulus' produces a custom ExternalInput class
that evaluates the event's equation at time t, with spatial weighting (regions).

The event's `name` becomes the variable name available in dfun equations.
E.g., event name='P' → EXTERNAL_INPUTS = {'P': 1}, and `P` is available
as a variable in the dynamics method.

Context Variables:
- experiment: SimulationExperiment instance (required)
- stimulus_events: list of Event objects with event_type == 'stimulus' (set by parent template)

Output:
- One AbstractExternalInput subclass per stimulus event

Design:
- Maximally generic: works for ANY time-dependent event equation
- Spatial weighting: regions + weighting define per-node amplitude masks
- Parameters: event parameters become class DEFAULT_PARAMS
</%doc>
<%
from tvbo.export.code import render_expression

# Extract stimulus events from experiment context
assert 'experiment' in context.keys(), "experiment required for stimulus template"

events_list = list(experiment.events.values()) if experiment.events else []
stimulus_events = [ev for ev in events_list if 'stimulus' in str(getattr(ev, 'event_type', 'stimulus'))]

# Collect user-defined functions from model (for code rendering)
model = experiment.dynamics
_model_functions = getattr(model, 'functions', None) or {}
_exp_functions = getattr(experiment, 'functions', None) or {}
user_functions = {}
if hasattr(_model_functions, 'keys'):
    user_functions.update({str(fname): str(fname) for fname in _model_functions.keys()})
if hasattr(_exp_functions, 'keys'):
    user_functions.update({str(fname): str(fname) for fname in _exp_functions.keys()})

n_nodes = getattr(experiment.network, 'number_of_nodes', None) or getattr(experiment.network, 'number_of_regions', 1)

def stim_jaxcode(expr, param_names=None):
    """Render event equation to JAX code via SymPy parsing."""
    return render_expression(expr, format='jax', user_functions=user_functions,
                             parameters=param_names)
%>

% for event in stimulus_events:
<%
    ev_name = str(event.name)
    class_name = ev_name + 'Input'
    eq_rhs = str(event.equation.rhs) if event.equation else '0.0'
    ev_params = dict(event.parameters) if event.parameters else {}
    ev_regions = list(event.regions) if event.regions else []
    ev_weighting = list(event.weighting) if event.weighting else []

    # Build spatial mask: array of shape (n_nodes,) with weights per region
    has_spatial = bool(ev_regions)
%>

class ${class_name}(AbstractExternalInput):
    """External input: ${ev_name}(t).

    ${event.description or event.label or 'Time-dependent external input.'}

    Equation: ${eq_rhs}
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(
        % for pname, pobj in ev_params.items():
        ${pname}=${float(pobj.value) if pobj.value is not None else 0.0},
        % endfor
    )

    def prepare(self, network, dt: float):
        % if has_spatial:
        # Spatial weighting mask: stimulus applied to specific regions
        _mask = jnp.zeros(network.graph.n_nodes)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        return Bunch(mask=_mask), Bunch()
        % else:
        return Bunch(), Bunch()
        % endif

    def compute(self, t, state, input_data, input_state, params):
        # Unpack parameters
        % for pname in ev_params:
        ${pname} = params.${pname}
        % endfor

        # Evaluate event equation
        signal = ${stim_jaxcode(eq_rhs, param_names=list(ev_params.keys()) + ['t'])}

        % if has_spatial:
        # Apply spatial mask (broadcast to [1, n_nodes])
        return (signal * input_data.mask)[None, :]
        % else:
        # Global: broadcast to all nodes
        if jnp.ndim(signal) == 0:
            return jnp.full((1, state.shape[1]), signal)
        return signal[None, :]
        % endif

    def update_state(self, input_data, input_state, new_state):
        return input_state

% endfor
