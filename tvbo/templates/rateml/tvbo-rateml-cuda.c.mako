// -*- coding: utf-8 -*-
//
// RateML-style CUDA Kernel Template
// ==================================
//
// Generates GPU-accelerated simulation kernel from TVBO specification.
// Compatible with TVB's CUDA simulation framework.
//
<%doc>
Context Variables:
- model: Dynamics instance (required)
- experiment: SimulationExperiment (optional)
- coupling: Coupling instance (optional)
- swept_params: list of parameter names to sweep (optional, defaults to global_speed, global_coupling)

Output:
- CUDA kernel for parallel brain network simulation
</%doc>
<%
from tvbo.templates.rateml.utils import (
    cuda_code, has_boundaries, get_initial_value,
    get_domain_str, get_boundary_str
)

# Model name
model_name = model.name.replace(' ', '').replace('-', '')

# State variables
state_vars = list(model.state_variables.items())
n_states = len(state_vars)

# Parameters (constants)
params = list(model.parameters.items()) if model.parameters else []

# Derived parameters (computed from other params)
derived_params = list(model.derived_parameters.items()) if model.derived_parameters else []

# Derived variables (intermediate calculations)
derived_vars = list(model.derived_variables.items()) if model.derived_variables else []

# Coupling terms
coupling_terms = list(model.coupling_terms.keys()) if model.coupling_terms else ['c_glob']

# Swept parameters (for parameter space exploration)
if 'swept_params' not in context.keys():
    swept_params = ['global_speed', 'global_coupling']

# Check for noise
noisepresent = False
if 'experiment' in context.keys() and experiment:
    integration = getattr(experiment, 'integration', None)
    if integration and getattr(integration, 'noise', None):
        noisepresent = True

# Variables of interest (exposures)
exposures = [(name, sv) for name, sv in state_vars if getattr(sv, 'record', True)]
%>
#include <stdio.h>
#define PI_2 (2 * M_PI_F)
#define PI M_PI_F
#define INF INFINITY

// Buffer length defaults to the argument to the integrate kernel
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <curand_kernel.h>
#include <curand.h>
#include <stdbool.h>

// State variable boundary wrapping functions
% for sv_name, sv in state_vars:
% if get_boundary_str(sv):
__device__ float wrap_it_${sv_name}(float ${sv_name})
{
    float ${sv_name}dim[] = {${get_boundary_str(sv)}};
    if (${sv_name} < ${sv_name}dim[0]) ${sv_name} = ${sv_name}dim[0];
    else if (${sv_name} > ${sv_name}dim[1]) ${sv_name} = ${sv_name}dim[1];
    return ${sv_name};
}
% endif
% endfor

extern "C" __global__ void ${model_name}(
    // Config
    unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_work_items,
    float dt, float * __restrict__ weights, float * __restrict__ lengths,
    float * __restrict__ params_pwi,  // per work item swept params
    // State
    float * __restrict__ state_pwi,
    // Outputs
    float * __restrict__ tavg_pwi
)
{
    // Work id & size
    const unsigned int id = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int size = n_work_items;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * ${n_states} * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // Only process valid work items
    if (id >= size) return;

    // Unpack swept parameters
    % for i, sp in enumerate(swept_params):
    const float ${sp} = params(${i});
    % endfor

    // Model constants (parameters)
    % for p_name, param in params:
    const float ${p_name} = ${param.value if param.value is not None else 1.0}f;
    % endfor

    // Derived parameters
    const float rec_speed_dt = 1.0f / global_speed / dt;
    % for dp_name, dp in derived_params:
    const float ${dp_name} = ${cuda_code(dp.equation)};
    % endfor

    % if noisepresent:
    // Initialize random state for noise
    curandState crndst;
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);
    % endif

    // State variable declarations
    % for sv_name, sv in state_vars:
    float ${sv_name} = ${get_initial_value(sv)}f;
    % endfor

    // Derivative declarations
    % for sv_name, sv in state_vars:
    float d${sv_name} = 0.0f;
    % endfor

    // Derived variable declarations
    % for dv_name, dv in derived_vars:
    float ${dv_name} = 0.0f;
    % endfor

    // Coupling variable declarations
    % for ct in coupling_terms:
    float ${ct} = 0.0f;
    % endfor

    unsigned int dij_i = 0;
    float dij = 0.0f;
    float wij = 0.0f;

    // Initialize observables
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {
        tavg(i_node) = 0.0f;
        if (i_step == 0) {
            state(i_step, i_node) = 0.0f;
        }
    }

    // Time loop
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
        // Node loop
        for (int i_node = 0; i_node < n_node; i_node++)
        {
            // Reset coupling
            % for ct in coupling_terms:
            ${ct} = 0.0f;
            % endfor

            // Initialize tavg at start of integration window
            if (t == i_step) {
                % for i, (sv_name, sv) in enumerate(state_vars):
                tavg(i_node + ${i} * n_node) = 0;
                % endfor
            }

            // Load current state
            % for i, (sv_name, sv) in enumerate(state_vars):
            ${sv_name} = state((t) % nh, i_node + ${i} * n_node);
            % endfor

            // Coupling computation
            unsigned int i_n = i_node * n_node;
            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                float wij = weights[i_n + j_node];
                if (wij == 0.0f)
                    continue;

                // Get delay
                dij = lengths[i_n + j_node] * rec_speed_dt;
                dij = dij + 0.5f;
                dij_i = (int)dij;

                // Get delayed state from source node
                % for i, (sv_name, sv) in enumerate(state_vars):
                % if getattr(sv, 'coupling_variable', False):
                float ${sv_name}_j = state(((t - dij_i + nh) % nh), j_node + ${i} * n_node);
                % endif
                % endfor

                // Accumulate coupling (linear coupling: c_pop0 += wij * S_j)
                % for ct in coupling_terms:
                ${ct} += wij * global_coupling * ${state_vars[0][0]}_j;
                % endfor
            }

            // Compute derived variables
            % for dv_name, dv in derived_vars:
            ${dv_name} = ${cuda_code(dv.equation)};
            % endfor

            // Compute derivatives (forward Euler)
            % for i, (sv_name, sv) in enumerate(state_vars):
            d${sv_name} = dt * (${cuda_code(sv.equation)});
            % endfor

            // Update state
            % if noisepresent:
            // With noise
            % for sv_name, sv in state_vars:
            ${sv_name} += nsig * curand_normal(&crndst) + d${sv_name};
            % endfor
            % else:
            // Without noise
            % for sv_name, sv in state_vars:
            ${sv_name} += d${sv_name};
            % endfor
            % endif

            // Apply boundaries
            % for sv_name, sv in state_vars:
            % if get_boundary_str(sv):
            ${sv_name} = wrap_it_${sv_name}(${sv_name});
            % endif
            % endfor

            // Store updated state
            % for i, (sv_name, sv) in enumerate(state_vars):
            state((t + 1) % nh, i_node + ${i} * n_node) = ${sv_name};
            % endfor

            // Accumulate observable (time average)
            % for i, (sv_name, sv) in enumerate(exposures):
            tavg(i_node + ${i} * n_node) += ${sv_name} / n_step;
            % endfor

            __syncthreads();
        } // for i_node
    } // for t

#undef params
#undef state
#undef tavg
} // kernel ${model_name}
<%
# Check for BOLD observation and find its hemodynamic model among the network's dynamics
bold_obs = None
bold_model = None
if 'experiment' in context.keys() and experiment:
    obs = getattr(experiment, 'observations', None)
    if obs:
        for obs_name, obs_val in obs.items():
            modality = getattr(obs_val, 'imaging_modality', None)
            if modality and str(modality) == 'BOLD':
                bold_obs = obs_val
                break
    # `experiment.dynamics` is the ONE model this kernel integrates, not a library.
    dyn_dict = getattr(getattr(experiment, 'network', None), 'dynamics', None) or {}
    if hasattr(dyn_dict, 'items'):
        for dyn_name, dyn in dyn_dict.items():
            if 'balloon' in dyn_name.lower() or 'bold' in dyn_name.lower() or 'windkessel' in dyn_name.lower():
                bold_model = dyn
                break
%>
% if bold_model:
<%
# Extract from dynamics metadata - use names exactly as defined
bold_params = dict(bold_model.parameters.items()) if bold_model.parameters else {}
bold_states = list(bold_model.state_variables.items()) if bold_model.state_variables else []
bold_derived_params = dict(bold_model.derived_parameters.items()) if bold_model.derived_parameters else {}
bold_derived_vars = list(bold_model.derived_variables.items()) if bold_model.derived_variables else []

def get_bold_param_val(name, default):
    p = bold_params.get(name)
    if p is None:
        return default
    return p.value if hasattr(p, 'value') and p.value is not None else default
%>

// BOLD Hemodynamic Model (${bold_model.name})
// Generated from experiment.dynamics metadata

// Parameters (names exactly as defined in YAML)
% for pname, param in bold_params.items():
<% pval = get_bold_param_val(pname, 1.0) %>\
#define ${pname} ${pval}f
% endfor

// Derived constants
% for dp_name, dp in bold_derived_params.items():
#define ${dp_name} (${cuda_code(dp.equation)})
% endfor

extern "C" __global__ void bold_update(int n_node, float dt,
            // bold_state.shape = (${len(bold_states)}, n_nodes, n_threads)
            float * __restrict__ bold_state,
            // neural_state.shape = (n_nodes, n_threads)
            float * __restrict__ neural_state,
            // out.shape = (n_nodes, n_threads)
            float * __restrict__ out)
{
    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int nt = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    int var_stride = n_node * nt;
    for (int i_node=0; i_node < n_node; i_node++)
    {
        float *node_bold = bold_state + i_node * nt + it;

        // Load BOLD state variables
        % for i, (sv_name, sv) in enumerate(bold_states):
        float ${sv_name} = node_bold[${i} * var_stride];
        % endfor

        // Neural input (from dynamics model)
        float x = neural_state[i_node * nt + it];

        // Compute derivatives
        % for sv_name, sv in bold_states:
        float d${sv_name} = ${cuda_code(sv.equation)};
        % endfor

        // Integrate with forward Euler
        % for sv_name, sv in bold_states:
        ${sv_name} += dt * d${sv_name};
        % endfor

        // Store updated BOLD state
        % for i, (sv_name, sv) in enumerate(bold_states):
        node_bold[${i} * var_stride] = ${sv_name};
        % endfor

        // Compute BOLD signal output
        % for dv_name, dv in bold_derived_vars:
        out[i_node * nt + it] = ${cuda_code(dv.equation)};
        % endfor
    } // i_node
} // kernel bold_update
% endif
