<%
"""
Generate TVB Monitor subclasses from observation pipeline metadata.

For each observation, generates a concrete class extending only the base
TVB classes (Monitor, Projection) — never importing leaf classes like
Raw, EEG, Bold.  The generated sample() method recreates the same
processing described by the observation's pipeline using TVB's per-step
stateful protocol.

Template context
----------------
experiment : object with .observations dict
    OR a simple namespace wrapping a single observation.
"""
import os as _os

observations = getattr(experiment, 'observations', None) or {}

# ── classification helpers ──────────────────────────────────────────

# Map class_reference names to monitor categories
_CATEGORY_BY_CR = {
    'Raw':           'raw',
    'RawVoi':        'raw_voi',
    'SubSample':     'subsample',
    'TemporalAverage':  'temporal_average',
    'SpatialAverage':   'spatial_average',
    'GlobalAverage':    'global_average',
    'EEG':              'projection_eeg',
    'MEG':              'projection_meg',
    'iEEG':             'projection_ieeg',
    'Bold':             'bold',
    'BoldRegionROI':    'bold_roi',
    'AfferentCoupling':               'afferent',
    'AfferentCouplingTemporalAverage': 'afferent_ta',
}

_PROJECTION_CATS = {'projection_eeg', 'projection_meg', 'projection_ieeg'}
_BOLD_CATS       = {'bold', 'bold_roi'}

# HRF kernel class names we know how to construct
_HRF_CLASSES = {
    'FirstOrderVolterra', 'Gamma', 'DoubleExponential', 'MixtureOfGammas',
}

# TVB sensor / projection class info  (class_name, module)
_SENSOR_INFO = {
    'projection_eeg':  ('SensorsEEG',      'tvb.datatypes.sensors'),
    'projection_meg':  ('SensorsMEG',      'tvb.datatypes.sensors'),
    'projection_ieeg': ('SensorsInternal',  'tvb.datatypes.sensors'),
}
_PROJ_INFO = {
    'projection_eeg':  ('ProjectionSurfaceEEG',  'tvb.datatypes.projections'),
    'projection_meg':  ('ProjectionSurfaceMEG',  'tvb.datatypes.projections'),
    'projection_ieeg': ('ProjectionSurfaceSEEG', 'tvb.datatypes.projections'),
}

def _pipeline_step_names(obs):
    """Return set of step names and function refs in the pipeline."""
    names = set()
    for step in (getattr(obs, 'pipeline', None) or []):
        n = getattr(step, 'name', None)
        if n:
            names.add(str(n))
        f = getattr(step, 'function', None)
        if f:
            names.add(str(f))
    return names

def _classify(obs):
    """Determine the monitor category for an observation."""
    cr = getattr(obs, 'class_reference', None)
    if cr is not None:
        cr_name = str(cr.name)
        if cr_name in _CATEGORY_BY_CR:
            return _CATEGORY_BY_CR[cr_name]

    # Infer from pipeline
    pnames = _pipeline_step_names(obs)
    if pnames & {'hemodynamic_response', 'hrf_kernel',
                 'HemodynamicResponseFunctionTVB', 'volterra_transform'}:
        return 'bold'
    if pnames & {'compute_gain', 'lead_field_projection'}:
        im = getattr(obs, 'imaging_modality', None)
        if im is not None:
            im = str(im).upper()
            if im == 'SEEG':
                return 'projection_ieeg'
            if im == 'MEG':
                return 'projection_meg'
        return 'projection_eeg'
    if pnames & {'spatial_average'}:
        return 'spatial_average'
    if pnames & {'global_mean'}:
        return 'global_average'
    if pnames & {'temporal_average'}:
        return 'temporal_average'
    if pnames & {'subsample'}:
        return 'subsample'

    # Source-based fallback
    src = getattr(obs, 'source', None)
    if src is not None and str(src) == 'coupling':
        period = getattr(obs, 'period', None)
        if period is not None:
            return 'afferent_ta'
        return 'afferent'

    period = getattr(obs, 'period', None)
    if period is not None:
        return 'subsample'

    return 'raw'

def _get_hrf_params(obs):
    """Extract HRF kernel class name, hrf_length, and equation params."""
    cr = getattr(obs, 'class_reference', None)
    hrf_class_name = 'FirstOrderVolterra'
    hrf_length = 20000.0
    hrf_length_from_cr = False
    hrf_eq_params = {}

    if cr is not None:
        for arg in (getattr(cr, 'constructor_args', None) or []):
            aname = str(arg.name)
            if aname == 'hrf_kernel' and arg.value is not None:
                hrf_class_name = str(arg.value)
            elif aname == 'hrf_length' and arg.value is not None:
                hrf_length = float(arg.value)
                hrf_length_from_cr = True

    for step in (getattr(obs, 'pipeline', None) or []):
        sn = str(getattr(step, 'name', '') or '')
        sf = str(getattr(step, 'function', '') or '')
        if sn in ('hemodynamic_response', 'hrf_kernel',
                   'HemodynamicResponseFunctionTVB') or sf == 'hrf_kernel':
            eq = getattr(step, 'equation', None)
            if eq:
                for pk, pv in (getattr(eq, 'parameters', {}) or {}).items():
                    val = getattr(pv, 'value', None)
                    if val is not None:
                        hrf_eq_params[str(pk)] = float(val)
            # Only use pipeline duration if constructor_args didn't set hrf_length
            if not hrf_length_from_cr:
                tr = getattr(step, 'time_range', None)
                if tr is not None:
                    hi = getattr(tr, 'hi', None)
                    if hi is not None:
                        try:
                            hrf_length = float(hi)
                        except (ValueError, TypeError):
                            pass
                args_coll = getattr(step, 'arguments', None) or {}
                for arg in (args_coll.values() if hasattr(args_coll, 'values') else args_coll):
                    aname = str(getattr(arg, 'name', ''))
                    if aname == 'duration' and getattr(arg, 'value', None) is not None:
                        try:
                            hrf_length = float(arg.value)
                        except (ValueError, TypeError):
                            pass

    return hrf_class_name, hrf_length, hrf_eq_params

def _get_projection_params(obs):
    """Extract sigma and reference from constructor_args and parameters."""
    sigma = 1.0
    reference = None
    cr = getattr(obs, 'class_reference', None)
    if cr is not None:
        for arg in (getattr(cr, 'constructor_args', None) or []):
            aname = str(arg.name)
            if aname == 'sigma' and arg.value is not None:
                sigma = float(arg.value)
            elif aname == 'reference' and arg.value is not None:
                reference = str(arg.value)
    obs_params = getattr(obs, 'parameters', None) or {}
    if hasattr(obs_params, 'get'):
        for pname in ('conductivity', 'sigma'):
            pv = obs_params.get(pname)
            if pv is not None:
                val = getattr(pv, 'value', None)
                if val is not None:
                    sigma = float(val)
        for pname in ('reference_electrode', 'reference'):
            pv = obs_params.get(pname)
            if pv is not None:
                val = getattr(pv, 'value', None)
                if val is not None:
                    reference = str(val)
    return sigma, reference

def _get_sensor_file(obs):
    """Extract sensor filename from data_source."""
    ds = getattr(obs, 'data_source', None)
    if ds is None:
        return None
    path = getattr(ds, 'path', None)
    if path:
        return _os.path.splitext(_os.path.basename(str(path)))[0]
    return None


# ── Classify all observations ──────────────────────────────────────

obs_list = []   # (obs_name, obs, category)

for obs_name, obs in (observations.items() if hasattr(observations, 'items') else []):
    cat = _classify(obs)
    # Skip observations that are not TVB monitor candidates
    if getattr(obs, 'aggregation', None) is not None and cat == 'raw':
        continue  # batch aggregation, not a per-step monitor
    if str(getattr(obs, 'source', '') or '').startswith('network.'):
        continue  # static data, not a monitor
    obs_list.append((str(obs_name), obs, cat))

# Default to raw passthrough if no monitor-capable observations
if not obs_list:
    obs_list.append(('raw', type('_RawObs', (), {'name': 'raw', 'period': None,
        'class_reference': None, 'pipeline': [], 'parameters': {},
        'source': None, 'aggregation': None, 'imaging_modality': None,
        'data_source': None})(), 'raw'))

# Determine which base-class imports are needed
needs_projection = any(c in _PROJECTION_CATS for _, _, c in obs_list)
needs_bold_eq     = any(c in _BOLD_CATS for _, _, c in obs_list)
needs_sensors = set()
needs_projections = set()
for _, obs, cat in obs_list:
    if cat in _SENSOR_INFO:
        needs_sensors.add(_SENSOR_INFO[cat])
        needs_projections.add(_PROJ_INFO[cat])
%>
##
## ── Imports ────────────────────────────────────────────────────────
import abc
import numpy
from tvb.simulator.monitors import Monitor
% if needs_projection:
from tvb.simulator.monitors import Projection
% endif
from tvb.basic.neotraits.api import Float, NArray, Attr
from tvb.simulator.backend.ref import ReferenceBackend
% if needs_bold_eq:
from tvb.datatypes import equations
% endif
% for cls_name, mod in sorted(needs_sensors):
from ${mod} import ${cls_name}
% endfor
% for cls_name, mod in sorted(needs_projections):
from ${mod} import ${cls_name}
% endfor
% if needs_projection:
from tvb.simulator import noise
from tvb.datatypes.region_mapping import RegionMapping
from tvb.simulator.common import numpy_add_at
% endif
##
## ── Generated monitor classes ──────────────────────────────────────
% for obs_name, obs, cat in obs_list:
<%
    period = getattr(obs, 'period', None)
    safe_name = obs_name.replace(' ', '_').replace('-', '_')
    class_name = safe_name[0].upper() + safe_name[1:] if safe_name else 'GeneratedMonitor'
%>

% if cat == 'raw':
class ${class_name}(Monitor):
    """Raw passthrough — records all variables at every step."""
    period = Float(default=0.0)

    def _config_vois(self, simulator):
        self.voi = numpy.arange(len(simulator.model.variables_of_interest))

    def _config_time(self, simulator):
        self.dt = simulator.integrator.dt
        self.period = self.dt
        self.istep = 1

    def sample(self, step, state):
        return [step * self.dt, state]

% elif cat == 'raw_voi':
class ${class_name}(Monitor):
    """Raw passthrough — records only selected VOI at every step."""
    period = Float(default=0.0)

    def _config_time(self, simulator):
        self.dt = simulator.integrator.dt
        self.period = self.dt
        self.istep = 1

    def sample(self, step, state):
        return [step * self.dt, state[self.voi]]

% elif cat == 'subsample':
class ${class_name}(Monitor):
    """Temporal decimation without averaging."""
% if period is not None:
    period = Float(default=${period})
% endif

    def sample(self, step, state):
        if step % self.istep == 0:
            return [step * self.dt, state[self.voi, :]]

% elif cat == 'temporal_average':
class ${class_name}(Monitor):
    """Running temporal mean over a sliding window."""
% if period is not None:
    period = Float(default=${period})
% endif

    def _config_time(self, simulator):
        super(${class_name}, self)._config_time(simulator)
        stock_size = (self.istep, self.voi.shape[0],
                      simulator.number_of_nodes,
                      simulator.model.number_of_modes)
        self._stock = numpy.zeros(stock_size)

    def sample(self, step, state):
        self._stock[((step % self.istep) - 1), :] = state[self.voi]
        if step % self.istep == 0:
            avg_stock = numpy.mean(self._stock, axis=0)
            time = (step - self.istep / 2.0) * self.dt
            return [time, avg_stock]

% elif cat == 'spatial_average':
class ${class_name}(Monitor):
    """Spatial averaging via region assignment matrix."""
% if period is not None:
    period = Float(default=${period})
% endif

    def config_for_sim(self, simulator):
        super(${class_name}, self).config_for_sim(simulator)
        n_nodes = simulator.number_of_nodes
        if simulator.surface is not None:
            spatial_mask = simulator.surface.region_mapping
        else:
            conn = simulator.connectivity
            if conn.cortical.size == 0:
                conn.cortical = numpy.array([True] * conn.weights.shape[0])
            spatial_mask = numpy.array([int(v) for v in conn.cortical])
        areas = numpy.unique(spatial_mask)
        n_areas = len(areas)
        spatial_sum = numpy.zeros((n_nodes, n_areas))
        spatial_sum[numpy.arange(n_nodes), spatial_mask] = 1
        spatial_sum = spatial_sum.T
        nodes_per_area = numpy.sum(spatial_sum, axis=1)[:, numpy.newaxis]
        self.spatial_mean = spatial_sum / nodes_per_area

    def sample(self, step, state):
        if step % self.istep == 0:
            time = step * self.dt
            monitored = numpy.dot(self.spatial_mean, state[self.voi, :])
            return [time, monitored.transpose((1, 0, 2))]

% elif cat == 'global_average':
class ${class_name}(Monitor):
    """Spatial mean across all network nodes."""
% if period is not None:
    period = Float(default=${period})
% endif

    def sample(self, step, state):
        if step % self.istep == 0:
            time = step * self.dt
            data = numpy.mean(state[self.voi, :], axis=1)[:, numpy.newaxis, :]
            return [time, data]

% elif cat in _PROJECTION_CATS:
<%
    sigma, reference = _get_projection_params(obs)
    sensor_file = _get_sensor_file(obs)
    sensor_cls, sensor_mod = _SENSOR_INFO[cat]
    proj_cls, proj_mod = _PROJ_INFO[cat]
%>
class ${class_name}(Projection):
    """Lead-field projection monitor (${cat.replace('projection_', '').upper()})."""
% if period is not None:
    period = Float(default=${period})
% endif
    sigma = Float(default=${sigma})
    sensors = Attr(${sensor_cls}, required=True)
    projection = Attr(${proj_cls}, default=None)
% if cat == 'projection_eeg':
    reference = Attr(str, required=False, default=${repr(reference) if reference else 'None'})
% endif

    def analytic(self, loc, ori):
        """Sarvas 1987 single-sphere approximation (Eq. 12)."""
        r_0, Q = loc, ori
        centre = numpy.mean(r_0, axis=0)[numpy.newaxis, :]
        radius = 1.05125 * max(numpy.sqrt(numpy.sum((r_0 - centre) ** 2, axis=1)))
        sen_loc = self.sensors.locations.copy()
        sen_dis = numpy.sqrt(numpy.sum(sen_loc ** 2, axis=1))
        sen_loc = sen_loc / sen_dis[:, numpy.newaxis] * radius + centre
        V_r = numpy.zeros((sen_loc.shape[0], r_0.shape[0]))
        for k in range(sen_loc.shape[0]):
            a = sen_loc[k, :] - r_0
            na = numpy.sqrt(numpy.sum(a ** 2, axis=1))[:, numpy.newaxis]
            V_r[k, :] = numpy.sum(Q * (a / na ** 3), axis=1) / (4.0 * numpy.pi * self.sigma)
        return V_r

% if cat == 'projection_eeg':
    def config_for_sim(self, simulator):
        super(${class_name}, self).config_for_sim(simulator)
        n_sensors = self.sensors.number_of_sensors
        self._ref_vec = numpy.zeros((n_sensors,))
        if self.reference:
            if self.reference.lower() != 'average':
                idx = self.sensors.labels.tolist().index(self.reference)
                self._ref_vec[idx] = 1.0
            else:
                self._ref_vec[:] = 1.0 / n_sensors
        self._ref_vec_mask = numpy.isfinite(self.gain).all(axis=1)
        self._ref_vec = self._ref_vec[self._ref_vec_mask]

    def sample(self, step, state):
        maybe = super(${class_name}, self).sample(step, state)
        if maybe is not None:
            time, sample = maybe
            sample -= self._ref_vec.dot(sample[:, self._ref_vec_mask])[:, numpy.newaxis]
            return time, sample.reshape((self.voi.size, -1, 1))
% endif

% elif cat in _BOLD_CATS:
<%
    hrf_class_name, hrf_length, hrf_eq_params = _get_hrf_params(obs)
    hrf_params_str = ', '.join(f'"{k}": {v}' for k, v in hrf_eq_params.items())
    # BOLD period: use obs.period, else parameters.TR, else 2000.0
    bold_period = period
    if bold_period is None:
        obs_params = getattr(obs, 'parameters', None) or {}
        tr_param = obs_params.get('TR') if hasattr(obs_params, 'get') else None
        if tr_param is not None:
            bold_period = getattr(tr_param, 'value', None)
    if bold_period is None:
        bold_period = 2000.0
    bold_period = float(bold_period)
%>
class ${class_name}(Monitor):
    """BOLD hemodynamic monitor — two-stage buffered HRF convolution."""
    period = Float(default=${bold_period})
    hrf_length = Float(default=${hrf_length})

    def _compute_hrf(self):
        self._stock_sample_rate = 2.0 ** -2
        required_len = self._stock_sample_rate * self.hrf_length
        self._stock_steps = int(numpy.ceil(required_len))
        stock_time_max = self.hrf_length / 1000.0
        stock_time_step = stock_time_max / self._stock_steps
        self._stock_time = numpy.arange(0.0, stock_time_max, stock_time_step)
% if hrf_params_str:
        hrf_kernel = equations.${hrf_class_name}(parameters={${hrf_params_str}})
% else:
        hrf_kernel = equations.${hrf_class_name}()
% endif
        self._hrf_kernel_eq = hrf_kernel
        G = hrf_kernel.evaluate(self._stock_time)
        G = G[::-1]
        self.hemodynamic_response_function = G[numpy.newaxis, :]
        self._interim_period = 1.0 / self._stock_sample_rate
        self._interim_istep = int(round(self._interim_period / self.dt))

    def _config_time(self, simulator):
        super(${class_name}, self)._config_time(simulator)
        self._compute_hrf()
        sample_shape = self.voi.shape[0], simulator.number_of_nodes, simulator.model.number_of_modes
        self._interim_stock = numpy.zeros((self._interim_istep,) + sample_shape)
        self._stock = numpy.zeros((self._stock_steps,) + sample_shape)

    def sample(self, step, state):
        # Accumulate into interim buffer at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi, :]
        # At interim period boundary, push temporal average into main circular buffer
        if step % self._interim_istep == 0:
            avg = numpy.mean(self._interim_stock, axis=0)
            self._stock[((step // self._interim_istep % self._stock_steps) - 1), :] = avg
        # At monitor period, convolve with HRF and return BOLD signal
        if step % self.istep == 0:
            time = step * self.dt
            hrf = numpy.roll(
                self.hemodynamic_response_function,
                ((step // self._interim_istep % self._stock_steps) - 1),
                axis=1)
% if hrf_class_name == 'FirstOrderVolterra':
            k1_V0 = self._hrf_kernel_eq.parameters["k_1"] * self._hrf_kernel_eq.parameters["V_0"]
            bold = (numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3))) - 1.0) * k1_V0
% else:
            bold = numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3)))
% endif
            bold = bold.reshape(self._stock.shape[1:])
            return [time, bold]

% if cat == 'bold_roi':
    # BoldRegionROI post-processing: average vertices within each region
    _inner_sample = sample

    def config_for_sim(self, simulator):
        super(${class_name}, self).config_for_sim(simulator)
        self._region_mapping = simulator.surface.region_mapping
        self._n_regions = simulator.surface.region_mapping_data.connectivity.number_of_regions

    def sample(self, step, state):
        result = self._inner_sample(step, state)
        if result:
            t, data = result
            data = data[self.voi, :]
            data = numpy.array([
                data[:, self._region_mapping == i, :].mean(axis=1)
                for i in range(self._n_regions)])
            data = numpy.swapaxes(data, 0, 1)
            return [t, data]
% endif

% elif cat == 'afferent':
class ${class_name}(Monitor):
    """Records afferent coupling input at every integration step."""
    period = Float(default=0.0)

    def _config_vois(self, simulator):
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]

    def _config_time(self, simulator):
        self.dt = simulator.integrator.dt
        self.period = self.dt
        self.istep = 1

    def sample(self, step, node_coupling):
        return [step * self.dt, node_coupling[self.voi]]

% elif cat == 'afferent_ta':
class ${class_name}(Monitor):
    """Temporally averaged afferent coupling input."""
% if period is not None:
    period = Float(default=${period})
% endif

    def _config_vois(self, simulator):
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]

    def _config_time(self, simulator):
        super(${class_name}, self)._config_time(simulator)
        stock_size = (self.istep, self.voi.shape[0],
                      simulator.number_of_nodes,
                      simulator.model.number_of_modes)
        self._stock = numpy.zeros(stock_size)

    def sample(self, step, node_coupling):
        self._stock[((step % self.istep) - 1), :] = node_coupling[self.voi]
        if step % self.istep == 0:
            avg = numpy.mean(self._stock, axis=0)
            time = (step - self.istep / 2.0) * self.dt
            return [time, avg]

% endif
% endfor
##
## ── monitors list ──────────────────────────────────────────────────
<%
    # Build constructor kwargs for each monitor
    monitor_items = []
    for obs_name, obs, cat in obs_list:
        safe = obs_name.replace(' ', '_').replace('-', '_')
        cname = safe[0].upper() + safe[1:] if safe else 'GeneratedMonitor'
        if cat in ('raw', 'raw_voi', 'afferent'):
            monitor_items.append(f'{cname}()')
        elif cat in _PROJECTION_CATS:
            # Projection monitors need sensors in constructor.
            # Always use TVB defaults — TVBO data_source paths reference TVBO
            # YAML files, not tvb_data sensor files.
            sensor_cls = _SENSOR_INFO[cat][0]
            proj_cls   = _PROJ_INFO[cat][0]
            parts = [
                f'sensors={sensor_cls}.from_file()',
                f'projection={proj_cls}.from_file()',
            ]
            monitor_items.append(f'{cname}({", ".join(parts)})')
        else:
            monitor_items.append(f'{cname}()')
%>
% if monitor_items:
monitors = [
% for item in monitor_items:
    ${item},
% endfor
]
% else:
monitors = []
% endif
