# -*- coding: utf-8 -*-
from collections import namedtuple

<%namespace name="utils" file="jax-utils.py.mako"/>

<%def name = "apply_monitor(i, monitor)" filter = "trim">
    <%
        from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, Projection, AfferentCoupling
    %>
    % if isinstance(monitor, AfferentCoupling):
    monitor_afferent_coupling_${i}(time_steps, node_coupling, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, RawVoi):
    monitor_raw_voi_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Raw):
    monitor_raw_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, TemporalAverage):
    monitor_temporal_average_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, SubSample):
    monitor_subsample_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Bold):
    monitor_bold_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, EEG):
    monitor_eeg_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Projection):
    monitor_projection_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % endif
</%def>

<%def name = "create_monitor(i, monitor, dt)" filter = "trim">
<%
    from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, Projection, AfferentCoupling
    from tvb.simulator.lab import equations

    import jax.numpy as jnp
    bold_fft_convolve = True
%>

### Raw monitor
% if isinstance(monitor, Raw) & (not isinstance(monitor, AfferentCoupling)):
def monitor_raw_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace, title = "Raw")
% endif

### Raw VOI monitor
% if isinstance(monitor, RawVoi):
def monitor_raw_voi_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    voi = ${utils.array_input(monitor.voi)}
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace[:, voi, :], title = "RawVoi")
% endif

### Afferent Coupling monitor
## There is an offset of -1 to the TVB affrent monitor...
% if isinstance(monitor, AfferentCoupling):
def monitor_afferent_coupling_${i}(time_steps, node_coupling, params, t_offset = 0):
    return monitor_raw_voi_${i}(time_steps, node_coupling, params, t_offset = t_offset)
% endif

### Temporal Average monitor
## Bold and Projection based monitors reuse the temporal average monitor in non replace_temporal_averaging mode
% if isinstance(monitor, TemporalAverage) or (not replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_temporal_average_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    voi = ${utils.array_input(monitor.voi)}
    % if isinstance(monitor, TemporalAverage):
    istep = ${monitor.istep}
    t_map = time_steps[::istep] - 1
    % elif isinstance(monitor, Projection):
    istep = ${monitor._period_in_steps}
    t_map = jnp.arange(0, time_steps.shape[0]-istep, istep)
    % else: ## Bold
    istep = ${monitor._interim_istep}
    t_map = time_steps[::istep] - 1
    % endif

    def op(ts):
        start_indices = (ts,) + (0,) * (trace.ndim - 1)
        slice_sizes = (istep,) + voi.shape + trace.shape[2:]
        return jnp.mean(jax.lax.dynamic_slice(trace[:, voi, :], start_indices, slice_sizes), axis=0)
    vmap_op = jax.vmap(op)
    trace_out = vmap_op(t_map)

    idxs = jnp.arange(((istep - 2) // 2), time_steps.shape[0], istep)
    return TimeSeries(time=(time_steps[idxs]) * dt, data=trace_out[0:idxs.shape[0], :, :], title = "TemporalAverage")
% endif

### SubSample monitor
## Bold and Projection based monitors reuse the temporal average monitor code in the case of replace_temporal_averaging = True the subsample monitor code is used (subsetting is less work than averaging)
% if isinstance(monitor, SubSample) or (replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_subsample_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    voi = ${utils.array_input(monitor.voi)}
    period = ${monitor.period} # sampling period in ms

    % if isinstance(monitor, SubSample):
    istep = ${monitor.istep}
    % elif isinstance(monitor, Projection):
    istep = ${monitor._period_in_steps}
    % else: ## Bold
    istep = ${monitor._interim_istep}
    % endif

    idxs = jnp.arange(istep-1, time_steps.shape[0], istep)
    return TimeSeries(time=(time_steps[idxs] + t_offset) * dt, data=trace[idxs[:, None], voi[None, :], :], title = "SubSample")
% endif

### BOLD monitor
% if isinstance(monitor, Bold):
import jax.scipy.signal as sig
exp, sin, sqrt = jnp.exp, jnp.sin, jnp.sqrt

def monitor_bold_${i}(time_steps, trace, params, t_offset = 0):
    # downsampling via temporal average / subsample
    dt = ${dt}
    voi = ${utils.array_input(monitor.voi)}
    period = ${monitor.period} # sampling period of the BOLD Monitor in ms
    ## period_int = ${monitor._interim_period} # sampling period in steps
    istep_int = ${monitor._interim_istep} # steps taken by the averaging/subsampling monitor to get an interim period of 4 ms
    istep = ${jnp.round(monitor.period / dt).astype(jnp.int32)}
    final_istep = ${(jnp.round(monitor.period / dt) / monitor._interim_istep).astype(jnp.int32)} # steps to take on the downsampled signal

    % if replace_temporal_averaging:
    res = monitor_subsample_${i}(time_steps, trace, None)
    %else:
    res = monitor_temporal_average_${i}(time_steps, trace, None)
    % endif
    time_steps_i = res.time
    trace_new = res.data

    time_steps_new = time_steps[jnp.arange(istep-1, time_steps.shape[0], istep)]

    # hemodynamic response function
% for par, _ in monitor.hrf_kernel.parameters.items():
    ${par} = params.${par}
% endfor
    stock = params.stock

    trace_new = jnp.vstack([stock, trace_new])

    op = lambda var: ${monitor.hrf_kernel.equation}
    stock_steps = ${jnp.ceil(monitor._stock_sample_rate * monitor.hrf_length).astype(jnp.int32)}
    stock_time_max = ${monitor.hrf_length / 1000.0} # stock time has to be in seconds
    stock_time_step = stock_time_max / stock_steps
    stock_time = jnp.arange(0.0, stock_time_max, stock_time_step)
    hrf = op(stock_time)

    ## Two versions to do the convolution via fft or via dot product
    # Convolution along time axis
    %if bold_fft_convolve:
    # via fft
    ## op1 = lambda x: sig.convolve(x, hrf, mode="full") ## much slower
    op1 = lambda x: sig.fftconvolve(x, hrf, mode="valid")
    op2 = lambda x: jax.vmap(op1, in_axes=(1), out_axes=(1))(x) # map over nodes
    op3 = lambda x: jax.vmap(op2, in_axes=(1), out_axes=(1))(x) # map over state variables
    bold = jax.vmap(op3, in_axes=(3), out_axes=(3))(trace_new) # map over modes
    %else:
    # via dot product
    hrf = hrf[::-1]
    def op_dot_slice(ts):
        start_indices = (ts,) + (0,) * (trace_new.ndim - 1)
        slice_sizes = (stock_steps,) + voi.shape + trace_new.shape[2:]
        trace_slice = jax.lax.dynamic_slice(trace_new, start_indices, slice_sizes)
        bold_slice = jnp.dot(jnp.transpose(trace_slice, (1,2,3,0)), hrf)
        return bold_slice
    tmap = jnp.arange(final_istep-1, time_steps_i.shape[0], final_istep)# - 1
    bold = jax.vmap(op_dot_slice)(tmap)
    %endif

    %if isinstance(monitor.hrf_kernel, equations.FirstOrderVolterra):
    bold = k_1 * V_0 * (bold - 1.0)
    %endif

    %if bold_fft_convolve:
    bold_idx = jnp.arange(final_istep-2, time_steps_i.shape[0], final_istep)[0:time_steps_new.shape[0]] + 1
    return TimeSeries(time=(time_steps_new + t_offset) * dt, data=bold[bold_idx, :, :], title = "BOLD")
    %else:
    return TimeSeries(time=(time_steps_new + t_offset) * dt, data=bold, title = "BOLD")
    %endif
% endif

### Projection monitor
% if isinstance(monitor, Projection):
def monitor_projection_${i}(time_steps, trace, params, t_offset = 0):
    voi = ${utils.array_input(monitor.voi)}
    dt = ${dt}

    gain = params.gain
    ## projection = jnp.matmul(trace[:, voi, :], gain.T)
    op = lambda x: jnp.matmul(x, gain.T)
    projection = jax.vmap(op, in_axes=(3), out_axes=(3))(trace)
    istep = ${monitor._period_in_steps}

     % if replace_temporal_averaging:
    res = monitor_subsample_${i}(time_steps, projection, None)
    t, proj_avg = res.time, res.data
    %else:
    res = monitor_temporal_average_${i}(time_steps, projection, None)

    ## Temporal Average
    ## istep = ${monitor._period_in_steps}
    ## def op(ts):
    ##     start_indices = (ts,) + (0,) * (trace.ndim - 1)
    ##     slice_sizes = (istep,) + voi.shape + projection.shape[2:]
    ##     return jnp.mean(jax.lax.dynamic_slice(projection[:, voi, :], start_indices, slice_sizes), axis=0)
    ## proj_avg = jax.lax.map(op, (time_steps[::istep] - 1))

    ## idxs = jnp.arange(((istep - 2) // 2), time_steps.shape[0], istep)
    ## t = (time_steps[idxs] + 0.5) * dt
    t, proj_avg = res.time, res.data
    t += 0.5 * dt
    % endif
    % if monitor.obsnoise is not None:
    proj_avg += params.obsnoise
    return TimeSeries(time=t, data=proj_avg, title = "Projection")
    %else:
    return TimeSeries(time=t, data=proj_avg, title = "Projection")
    % endif
% endif

### EEG monitor
% if isinstance(monitor, EEG):
def monitor_eeg_${i}(time_steps, trace, params, t_offset = 0):
    % if monitor.reference:
    res = monitor_projection_${i}(time_steps, trace, params, t_offset = t_offset)
    t, _eeg = res.time, res.data
    ref_vec = params.ref_vec
    return TimeSeries(time=t, data=_eeg - jnp.matmul(_eeg, ref_vec), title = "EEG")
    %else:
    return monitor_projection_${i}(time_steps, trace, params, t_offset = t_offset)
    % endif
% endif

</%def>
