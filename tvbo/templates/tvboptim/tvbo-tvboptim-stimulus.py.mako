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

# Integration timing — needed to size pre-generated per-step input arrays and to map a time t to an integer step index. The scan runs from -transient to +duration on the measurement clock, so t spans [-transient, duration] and the step index counts from the scan start, not from t=0.
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
    # Sourced data-driven stimulus: the samples are another run's recorded output, named by a `data` parameter's `used:` DataRef and injected at run time (never inlined), so a per-subject run plays its own subject's recording.
    _data_param = None if is_continuous else ev_params.get('data')
    is_sourced = _data_param is not None and getattr(_data_param, 'used', None) is not None
    if _data_param is not None and not is_sourced:
        raise ValueError(
            f"event {ev_name!r} declares a `data` parameter without `used:`; a data-driven stimulus reads its "
            f"samples either from `dataLocation` (a file) or from a `data` parameter sourcing another run's output."
        )
    if is_sourced and is_data:
        raise ValueError(
            f"event {ev_name!r} declares both `dataLocation` and a sourced `data` parameter; the samples must come from one."
        )
    if is_subset and (is_continuous or is_data or is_sourced):
        raise ValueError(
            f"event {ev_name!r} declares a `subset` weight_distribution, which is only "
            f"implemented for open-loop symbolic stimuli (event_type '{_ev_type}', "
            f"dataLocation={data_location!r})."
        )
    if is_data:
        sampling_rate = float(getattr(event, 'sampling_rate', None) or 1.0)
        interp_kind = str(getattr(event, 'interpolation', None) or 'linear')
        if interp_kind not in ('linear', 'cubic'):
            raise ValueError(
                f"event {ev_name!r} declares interpolation {interp_kind!r}; the data-driven "
                f"stimulus is interpolated by tvboptim's DataInput, which implements 'linear' and 'cubic'."
            )
        # optional onset (ms): shifts when the waveform starts playing
        onset = float(ev_params['onset'].value) if ('onset' in ev_params and ev_params['onset'].value is not None) else 0.0
        # `onset` positions the waveform on the clock and is consumed here; `amplitude` becomes a live parameter of the emitted input. A data-driven waveform has no equation whose symbols anything else could bind to, so any further parameter would be accepted and never read.
        unsupported = sorted(k for k in ev_params if k not in ('onset', 'amplitude'))
        if unsupported:
            raise ValueError(
                f"event {ev_name!r} reads its signal from {data_location!r} and declares "
                f"parameter(s) {', '.join(unsupported)}, which nothing in a data-driven stimulus "
                f"evaluates. Such a stimulus takes `onset` (where the waveform starts) and "
                f"`amplitude` (its gain); a parameter the signal depends on belongs in an "
                f"`equation:` stimulus instead."
            )
        amplitude = float(ev_params['amplitude'].value) if ('amplitude' in ev_params and ev_params['amplitude'].value is not None) else 1.0
    elif is_sourced:
        sampling_rate = float(getattr(event, 'sampling_rate', None) or 1.0)
        interp_kind = str(getattr(event, 'interpolation', None) or 'linear')
        if interp_kind != 'linear':
            raise ValueError(
                f"event {ev_name!r} sources its samples from another run and declares interpolation {interp_kind!r}; "
                f"a sourced stimulus interpolates linearly between recorded samples, and no other kind is implemented for it."
            )
        onset = float(ev_params['onset'].value) if ('onset' in ev_params and ev_params['onset'].value is not None) else 0.0
        amplitude = float(ev_params['amplitude'].value) if ('amplitude' in ev_params and ev_params['amplitude'].value is not None) else 1.0
        # `channel` maps each node to the recorded column it plays (per node, so it can be swept); `trial` picks the recorded trial and is written by the random-seed axis, not declared.
        unsupported = sorted(k for k in ev_params if k not in ('data', 'onset', 'amplitude', 'channel'))
        if unsupported:
            raise ValueError(
                f"event {ev_name!r} plays another run's recording and declares parameter(s) {', '.join(unsupported)}, "
                f"which nothing in a sourced stimulus evaluates. It takes `data` (the recording), `channel` (the column "
                f"each node plays), `onset` and `amplitude`."
            )
        _ch = ev_params.get('channel')
        _ch_val = getattr(_ch, 'value', None) if _ch is not None else None
        channel_default = None if _ch_val is None else [float(v) for v in (_ch_val if isinstance(_ch_val, (list, tuple)) else [_ch_val])]
    else:
        eq_rhs = str(event.equation.rhs) if event.equation else '0.0'
    # Per-step iid driver (symbolic branch only): an event parameter with
    # distribution.axis == 'time' pre-generates a fresh sample per integration
    # step (e.g. u ~ U(-1, 1)). Computed unconditionally; only consumed in the
    # non-data branch below.
    _stoch_pname, _stoch_info = (None, None) if is_continuous else _time_axis_distribution(event)
    is_stochastic = (not is_data) and (not is_sourced) and (not is_continuous) and _stoch_pname is not None
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
% elif is_sourced:
class ${class_name}(AbstractExternalInput):
    """Data-driven external input played from another run's recording: ${ev_name}(t).

    ${event.description or event.label or 'Sourced data-driven stimulus.'}

    The samples are the run-time resolution of the `data` parameter's DataRef (sampling_rate=${sampling_rate}/ms, onset=${onset} ms, linear), injected by the caller rather than inlined, laid out (trial, sample, channel). Node k plays column `channel[k]` of trial `trial`; a random-seed axis writes `trial` per cell, so each seed of an ensemble plays its own recorded trial. `data`, `trial`, `channel` and `amplitude` are live config leaves. Outside the sampled span the stimulus is silent.
    """

    N_OUTPUT_DIMS = 1
    TRIAL_BANK = True
    ONSET = ${onset}
    SAMPLING_RATE = ${sampling_rate}

    def __init__(self, **kwargs):
        _samples = _stimulus_samples(${repr(ev_name)})
        _n_channels = _samples.shape[-1]
        % if channel_default is not None:
        _channel = jnp.asarray(${channel_default}, dtype=float)
        % else:
        _channel = None
        % endif
        self.DEFAULT_PARAMS = Bunch(data=_samples, trial=0.0, channel=_channel, amplitude=${amplitude})
        super().__init__(**kwargs)

    def prepare(self, network, dt: float):
        _n = network.graph.n_nodes
        _n_channels = self.params.data.shape[-1]
        # Undeclared, the columns are the nodes themselves, or one column is broadcast to all of them.
        if self.params.channel is None and _n_channels not in (1, _n):
            raise ValueError(
                "stimulus ${ev_name}: the recording has %d channels for %d nodes; declare `channel`, the column "
                "each node plays." % (_n_channels, _n)
            )
        % if has_spatial:
        _mask = jnp.zeros(_n)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        % else:
        _mask = jnp.ones(_n)
        % endif
        return Bunch(mask=_mask), Bunch()

    def compute(self, t, state, input_data, input_state, params):
        _data = params.data
        _n_samples = _data.shape[1]
        # A single recorded trial is played by every seed; a bank of them is indexed by the cell's trial.
        _row = _data[0] if _data.shape[0] == 1 else jnp.take(_data, jnp.asarray(params.trial).astype(jnp.int32), axis=0)
        _x = (t - self.ONSET) * self.SAMPLING_RATE
        _i0 = jnp.clip(jnp.floor(_x), 0, _n_samples - 2).astype(jnp.int32)
        _w = jnp.clip(_x - _i0, 0.0, 1.0)
        if params.channel is None:
            _ch = jnp.zeros(state.shape[1], dtype=jnp.int32) if _data.shape[-1] == 1 else jnp.arange(state.shape[1])
        else:
            _ch = jnp.asarray(params.channel).astype(jnp.int32)
        _v = _row[_i0, _ch] * (1.0 - _w) + _row[_i0 + 1, _ch] * _w
        _sounding = (_x >= 0.0) & (_x <= _n_samples - 1)
        return (jnp.where(_sounding, _v, 0.0) * params.amplitude * input_data.mask)[None, :]

    def update_state(self, input_data, input_state, new_state):
        return input_state
% elif is_data:
class ${class_name}(DataInput):
    """Data-driven external input: ${ev_name}(t), interpolated from a file.

    ${event.description or event.label or 'Data-driven stimulus.'}

    Source: ${data_location}  (sampling_rate=${sampling_rate}/ms, onset=${onset} ms, ${interp_kind})

    The samples go to tvboptim's DataInput, which interpolates them with diffrax inside the scan, so the declared interpolation is the one the solver runs. Outside the sampled span the stimulus is silent. `amplitude` (${amplitude}) scales it as a live config leaf, so a sweep or a gradient fit can write the drive's gain.

    The interpolation kind is construction metadata, not a parameter: the solve snapshots every entry of `params` into the config it jits, and a string is not a valid JAX type there. It lives on the class and is lent to the base class only where a tvboptim release reads it off `params`: prepare() in 0.4, compute() in 0.5. A DataInput that keeps the kind as its own attribute reads neither, which is why the class drops it from `params` only when it is there.
    """

    INTERPOLATION = "${interp_kind}"

    def __init__(self, **kwargs):
        import numpy as _np
        _samples = jnp.asarray(_np.load(r"${data_location}"), dtype=jnp.float32)
        _samples = _samples.reshape(-1) if _samples.ndim == 1 else _samples.reshape(_samples.shape[0], -1)
        _times = ${onset} + jnp.arange(_samples.shape[0], dtype=jnp.float32) / ${sampling_rate}
        super().__init__(times=_times, data=_samples, interpolation="${interp_kind}")
        # `amplitude` rides beside the samples as a live config leaf, which is what lets a sweep or a gradient fit write the drive's gain.
        _gain = Bunch(amplitude=${amplitude})
        _gain.update(kwargs)
        self.DEFAULT_PARAMS = Bunch(self.DEFAULT_PARAMS, **_gain)
        self.params = Bunch(self.params, **_gain)
        self.DEFAULT_PARAMS.pop("interpolation_type", None)
        self.params.pop("interpolation_type", None)

    def prepare(self, network, dt: float):
        """Lend the interpolation kind to `params` for the base class's prepare(), then take it back.

        tvboptim 0.4's DataInput reads it there to build its interpolator; left in `params`, it would be a string in the jitted solve config.
        """
        self.params.interpolation_type = self.INTERPOLATION
        try:
            input_data, input_state = super().prepare(network, dt)
        finally:
            self.params.pop("interpolation_type")
        % if has_spatial:
        _mask = jnp.zeros(network.graph.n_nodes)
        _regions = [${', '.join(str(r) for r in ev_regions)}]
        _weights = [${', '.join(str(float(w)) for w in ev_weighting) if ev_weighting else ', '.join('1.0' for _ in ev_regions)}]
        for _r, _w in zip(_regions, _weights):
            _mask = _mask.at[_r].set(_w)
        % else:
        _mask = jnp.ones(network.graph.n_nodes)
        % endif
        input_data.mask = _mask
        input_data.t_lo = self.params.times[0]
        input_data.t_hi = self.params.times[-1]
        return input_data, input_state

    def compute(self, t, state, input_data, input_state, params):
        # An interpolation holds its endpoint value beyond the sampled span; the stimulus is silent there.
        signal = super().compute(t, state, input_data, input_state, Bunch(params, interpolation_type=self.INTERPOLATION))
        _sounding = (t >= input_data.t_lo) & (t <= input_data.t_hi)
        return jnp.where(_sounding, signal, 0.0) * params.amplitude * input_data.mask
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
    SCAN_T0 = ${-_transient}   # the scan's own start on the measurement clock
    % endif

    def prepare(self, network, dt: float):
        % if is_stochastic:
        # Pre-generate the iid per-step sequence (indexed by step = round((t - scan_t0)/dt)).
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
        # Per-step iid sample: index the pre-generated sequence by step number, counted from the scan start so the settle draws its own samples rather than repeating the first.
        _step = jnp.int32(jnp.clip((t - self.SCAN_T0) * self.INV_DT, 0, input_data.u.shape[0] - 1))
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
