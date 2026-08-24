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
from tvbo.codegen import render_expression

# Extract stimulus events from experiment context
assert 'experiment' in context.keys(), "experiment required for stimulus template"

from tvbo.templates.tvboptim.utils import active_stimulus_events
stimulus_events = active_stimulus_events(experiment)

# State index lookup for continuous-event conditions (map a state name to its
# row in the [n_states, n_nodes] state array).
_model = experiment.dynamics
_state_names = [str(sv.name) for sv in _model.state_variables.values()] if getattr(_model, 'state_variables', None) else []
_state_index = {nm: i for i, nm in enumerate(_state_names)}

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

# Integration timing — needed to size pre-generated per-step input arrays and
# to map an absolute time t to an integer step index (step = round(t / dt)).
# The native solver runs the main sim from t0=transient .. transient+duration,
# so absolute t spans the full [0, transient + duration] window; the array is
# sized to cover it and indexed by absolute t.
_dt = float(experiment.integration.step_size)
_inv_dt = 1.0 / _dt
_duration = float(experiment.integration.duration) if experiment.integration.duration else 0.0
_transient = float(experiment.integration.transient_time) if experiment.integration.transient_time else 0.0
_n_steps_total = int(round((_transient + _duration) / _dt)) + 2  # +2 for rounding safety

def stim_jaxcode(expr, param_names=None):
    """Render event equation to JAX code via SymPy parsing."""
    return render_expression(expr, format='jax', user_functions=user_functions,
                             parameters=param_names)

def _time_axis_distribution(event):
    """Return (param_name, info) for an event parameter declaring a per-step
    (``distribution.axis == 'time'``) stochastic draw, else (None, None).

    This is the iid-per-step driver path (Jaeger-2002 memory-capacity input):
    the signal is a fresh sample u[t] each integration step rather than a
    deterministic function of t. The array is pre-generated once with a fixed
    seed (reproducible, vmap/pmap-safe) and indexed per step. The SAME seed +
    bounds reproduce u[t] in Python for building delayed readout targets.
    """
    params = dict(event.parameters) if event.parameters else {}
    for pname, pobj in params.items():
        dist = getattr(pobj, 'distribution', None)
        if dist is None:
            continue
        axis = str(getattr(dist, 'axis', 'space'))
        if axis == 'time' or 'time' in axis:
            domain = getattr(dist, 'domain', None)
            return pname, {
                'dist': str(getattr(dist, 'name', 'Uniform')).lower(),
                'lo': float(getattr(domain, 'lo', -1.0)) if domain else -1.0,
                'hi': float(getattr(domain, 'hi', 1.0)) if domain else 1.0,
                'seed': int(getattr(dist, 'seed', None) or 0),
            }
    return None, None
%>

% for event in stimulus_events:
<%
    ev_name = str(event.name)
    class_name = ev_name + 'Input'
    ev_params = dict(event.parameters) if event.parameters else {}
    ev_regions = list(getattr(event, 'nodes', None) or getattr(event, 'regions', None) or [])
    ev_weighting = list(getattr(event, 'weights', None) or getattr(event, 'weighting', None) or [])

    # Build spatial mask: array of shape (n_nodes,) with weights per region
    has_spatial = bool(ev_regions)

    # `subset`: each trial stimulates round(fraction * pool) regions drawn without replacement; prepare() pre-samples every trial's mask and DEFAULT_PARAMS.trial selects the row.
    _wd = getattr(event, 'weight_distribution', None)
    is_subset = _wd is not None and str(getattr(_wd, 'name', '') or '').lower() == 'subset'
    if is_subset:
        if ev_weighting:
            raise ValueError(
                f"event {str(event.name)!r} declares both `weights` and a `subset` "
                f"weight_distribution; the per-trial mask is uniform over the drawn regions, "
                f"so the declared weighting would be discarded. Declare one or the other."
            )
        _wd_params = dict(getattr(_wd, 'parameters', None) or {})
        _frac_obj = _wd_params.get('fraction')
        _frac_val = getattr(_frac_obj, 'value', _frac_obj) if _frac_obj is not None else None
        subset_fraction = float(_frac_val) if _frac_val is not None else 0.1
        subset_seed = int(getattr(_wd, 'seed', None) or 0)
        subset_pool = [int(r) for r in ev_regions]  # empty -> all nodes
        _pool_n = len(subset_pool) if subset_pool else int(n_nodes)
        subset_k = max(1, int(round(subset_fraction * _pool_n)))
        subset_n_trials = 1
        for _expl in (getattr(experiment, 'explorations', None) or {}).values():
            _nt = getattr(_expl, 'n_trials', None)
            if _nt:
                subset_n_trials = max(subset_n_trials, int(_nt))
        has_spatial = False

    # Closed-loop 'continuous' event: onset is triggered when a state condition
    # crosses zero, then the affect waveform (a function of tau = t - t_trigger)
    # is emitted. A stateful ExternalInput carries armed/t_trigger/cond_prev.
    _ev_type = str(getattr(event, 'event_type', 'stimulus'))
    is_continuous = _ev_type in ('continuous', 'discrete')
    if is_continuous:
        cond_rhs = str(event.condition.rhs) if getattr(event, 'condition', None) else '0.0'
        _aff = getattr(event, 'affect', None) or getattr(event, 'equation', None)
        affect_rhs = str(_aff.rhs) if _aff else '0.0'
        cond_states = [str(s) for s in (getattr(event, 'condition_states', None) or [])]
        cond_idx = [_state_index.get(s, 0) for s in cond_states]
        _wlo = float(ev_params['window_lo'].value) if ('window_lo' in ev_params and ev_params['window_lo'].value is not None) else 0.0
        _whi = float(ev_params['window_hi'].value) if ('window_hi' in ev_params and ev_params['window_hi'].value is not None) else 1e30
        # +1 = trigger on upward zero crossing (default), -1 = downward
        _cdir = float(ev_params['crossing'].value) if ('crossing' in ev_params and ev_params['crossing'].value is not None) else 1.0
        wave_params = {k: v for k, v in ev_params.items() if k not in ('window_lo', 'window_hi', 'crossing')}

    # Data-driven stimulus: waveform read from a file and interpolated at time t,
    # instead of evaluating a symbolic equation.
    data_location = None if is_continuous else getattr(event, 'dataLocation', None)
    is_data = bool(data_location)
    if is_subset and (is_continuous or is_data):
        raise ValueError(
            f"event {ev_name!r} declares a `subset` weight_distribution, which is only "
            f"implemented for open-loop symbolic stimuli (event_type '{_ev_type}', "
            f"dataLocation={data_location!r})."
        )
    if is_data:
        sampling_rate = float(getattr(event, 'sampling_rate', None) or 1.0)
        interp_kind = str(getattr(event, 'interpolation', None) or 'linear')
        # optional onset (ms): shifts when the waveform starts playing
        onset = float(ev_params['onset'].value) if ('onset' in ev_params and ev_params['onset'].value is not None) else 0.0
    else:
        eq_rhs = str(event.equation.rhs) if event.equation else '0.0'
    # Per-step iid driver (symbolic branch only): an event parameter with
    # distribution.axis == 'time' pre-generates a fresh sample per integration
    # step (e.g. u ~ U(-1, 1)). Computed unconditionally; only consumed in the
    # non-data branch below.
    _stoch_pname, _stoch_info = (None, None) if is_continuous else _time_axis_distribution(event)
    is_stochastic = (not is_data) and (not is_continuous) and _stoch_pname is not None
    # The deterministic equation parameters exclude the stochastic one (it is
    # supplied by the pre-generated array, not a scalar DEFAULT_PARAM).
    det_params = {k: v for k, v in ev_params.items() if k != _stoch_pname}
%>

% if is_continuous:
class ${class_name}(AbstractExternalInput):
    """Closed-loop (state-triggered) external input: ${ev_name}.

    ${event.description or event.label or 'Condition-triggered input.'}

    Arms when the condition ``${cond_rhs}`` crosses zero (${'upward' if _cdir >= 0 else 'downward'})
    within the window [${_wlo}, ${_whi}] s, then emits the affect waveform as a
    function of ``tau`` = t - t_trigger:  ${affect_rhs}

    Triggering is detected by a per-step sign change of the condition (checked in
    update_state on the post-step state); there is no sub-step root polish, so the
    onset resolves to the integration step (dt). Per-node: each node arms
    independently on its own condition crossing.
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(
        % for pname, pobj in wave_params.items():
        ${pname}=${float(pobj.value) if pobj.value is not None else 0.0},
        % endfor
    )
    WINDOW_LO = ${_wlo}
    WINDOW_HI = ${_whi}
    CROSS_DIR = ${_cdir}
    COND_IDX = (${', '.join(str(i) for i in cond_idx)}${',' if len(cond_idx) == 1 else ''})

    def prepare(self, network, dt: float):
        _n = network.graph.n_nodes
        % if has_spatial:
        _mask = jnp.zeros(_n)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        % else:
        _mask = jnp.ones(_n)
        % endif
        input_data = Bunch(mask=_mask, dt=dt)
        # armed / trigger time / previous (direction-adjusted) condition / step
        # counter — per node. cond_prev starts at +inf so (cond_prev < 0) is False
        # on the first step: no false trigger regardless of crossing direction.
        input_state = Bunch(
            armed=jnp.zeros(_n),
            t_trigger=jnp.zeros(_n),
            cond_prev=jnp.full(_n, jnp.inf),
            step=jnp.array(0.0),
        )
        return input_data, input_state

    def compute(self, t, state, input_data, input_state, params):
        % for pname in wave_params:
        ${pname} = params.${pname}
        % endfor
        tau = t - input_state.t_trigger          # per-node time since each node's trigger
        _wave = ${stim_jaxcode(affect_rhs, param_names=list(wave_params.keys()) + ['tau'])}
        signal = jnp.where(input_state.armed > 0.5, _wave, 0.0)
        return (signal * input_data.mask)[None, :]

    def update_state(self, input_data, input_state, new_state):
        # Evaluate the condition on the post-step state (per node)
        % for s, i in zip(cond_states, cond_idx):
        ${s} = new_state[${i}]
        % endfor
        _cond = self.CROSS_DIR * (${stim_jaxcode(cond_rhs, param_names=cond_states) if cond_states else '0.0'})
        _t_now = input_state.step * input_data.dt
        _crossed = (input_state.cond_prev < 0.0) & (_cond >= 0.0)
        _in_win = (_t_now >= self.WINDOW_LO) & (_t_now < self.WINDOW_HI)
        _fire = _crossed & _in_win & (input_state.armed < 0.5)
        return Bunch(
            armed=jnp.where(_fire, 1.0, input_state.armed),
            t_trigger=jnp.where(_fire, _t_now, input_state.t_trigger),
            cond_prev=_cond,
            step=input_state.step + 1.0,
        )
% elif is_data:
class ${class_name}(AbstractExternalInput):
    """Data-driven external input: ${ev_name}(t) interpolated from a file.

    ${event.description or event.label or 'Data-driven stimulus.'}

    Source: ${data_location}  (sampling_rate=${sampling_rate}/ms, onset=${onset} ms, ${interp_kind})
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch()

    def prepare(self, network, dt: float):
        import numpy as _np
        _samples = jnp.asarray(_np.load(r"${data_location}"), dtype=jnp.float32).reshape(-1)
        _times = ${onset} + jnp.arange(_samples.shape[0], dtype=jnp.float32) / ${sampling_rate}
        % if has_spatial:
        _mask = jnp.zeros(network.graph.n_nodes)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        % else:
        _mask = jnp.ones(network.graph.n_nodes)
        % endif
        return Bunch(times=_times, signal=_samples, mask=_mask), Bunch()

    def compute(self, t, state, input_data, input_state, params):
        # interpolate the waveform at time t; zero outside the data window
        signal = jnp.interp(t, input_data.times, input_data.signal, left=0.0, right=0.0)
        return (signal * input_data.mask)[None, :]

    def update_state(self, input_data, input_state, new_state):
        return input_state
% else:
class ${class_name}(AbstractExternalInput):
    """External input: ${ev_name}(t).

    ${event.description or event.label or 'Time-dependent external input.'}

    Equation: ${eq_rhs}
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(
        % for pname, pobj in det_params.items():
        ${pname}=${float(pobj.value) if pobj.value is not None else 0.0},
        % endfor
        % if is_subset:
        trial=0.0,
        % endif
    )
    % if is_subset:

    # Keyed by fold_in(key(SUBSET_SEED), trial) so adding trials never perturbs the existing ones.
    SUBSET_SEED = ${subset_seed}
    SUBSET_K = ${subset_k}
    SUBSET_N_TRIALS = ${subset_n_trials}
    % endif
    % if is_stochastic:

    # iid per-step driver: u[t] = ${_stoch_info['dist'].capitalize()}(${_stoch_info['lo']}, ${_stoch_info['hi']}),
    # one fresh sample per integration step (axis='time'). Pre-generated once
    # with a fixed seed (reproducible, pure array → vmap/pmap-safe). The same
    # seed + bounds reproduce u[t] in Python for the delayed readout targets.
    STOCH_SEED = ${_stoch_info['seed']}
    STOCH_N_STEPS = ${_n_steps_total}
    STOCH_LO = ${_stoch_info['lo']}
    STOCH_HI = ${_stoch_info['hi']}
    INV_DT = ${_inv_dt}
    % endif

    def prepare(self, network, dt: float):
        % if is_stochastic:
        # Pre-generate the iid per-step sequence (indexed by step = round(t/dt)).
        _u = jax.random.uniform(
            jax.random.key(self.STOCH_SEED), (self.STOCH_N_STEPS,),
            minval=self.STOCH_LO, maxval=self.STOCH_HI,
        )
        % endif
        % if is_subset:
        # Pre-sample every trial's mask (pure array -> vmap/pmap-safe).
        % if subset_pool:
        _pool = jnp.asarray([${', '.join(str(r) for r in subset_pool)}])
        % else:
        _pool = jnp.arange(network.graph.n_nodes)
        % endif
        def _draw_mask(_k):
            _idx = jax.random.choice(_k, _pool, shape=(self.SUBSET_K,), replace=False)
            return jnp.zeros(network.graph.n_nodes).at[_idx].set(1.0)
        _keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(self.SUBSET_SEED), i))(jnp.arange(self.SUBSET_N_TRIALS))
        _masks = jax.vmap(_draw_mask)(_keys)
        % if is_stochastic:
        return Bunch(masks=_masks, u=_u), Bunch()
        % else:
        return Bunch(masks=_masks), Bunch()
        % endif
        % elif has_spatial:
        # Spatial weighting mask: stimulus applied to specific regions
        _mask = jnp.zeros(network.graph.n_nodes)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        % if is_stochastic:
        return Bunch(mask=_mask, u=_u), Bunch()
        % else:
        return Bunch(mask=_mask), Bunch()
        % endif
        % else:
        % if is_stochastic:
        return Bunch(u=_u), Bunch()
        % else:
        return Bunch(), Bunch()
        % endif
        % endif

    def compute(self, t, state, input_data, input_state, params):
        # Unpack parameters
        % for pname in det_params:
        ${pname} = params.${pname}
        % endfor
        % if is_stochastic:
        # Per-step iid sample: index the pre-generated sequence by step number.
        _step = jnp.int32(jnp.clip(t * self.INV_DT, 0, input_data.u.shape[0] - 1))
        ${_stoch_pname} = input_data.u[_step]
        % endif

        # Evaluate event equation
        signal = ${stim_jaxcode(eq_rhs, param_names=list(det_params.keys()) + ([_stoch_pname] if is_stochastic else []) + ['t'])}

        % if is_subset:
        # This trial's pre-sampled random-subset mask (params.trial is a state.external leaf).
        _mask = input_data.masks[jnp.int32(params.trial)]
        return (signal * _mask)[None, :]
        % elif has_spatial:
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
% endif

% endfor
