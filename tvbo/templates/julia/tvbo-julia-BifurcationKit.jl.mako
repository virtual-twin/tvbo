using BifurcationKit
using OrdinaryDiffEq
<%
if 'model' in context.keys():
    model = context['model']

# Support schema-based ContinuationProblem or raw kwargs
cont = context.get('continuation', None)

# ---------------------------------------------------------------------------
# Helper: safely get nested attribute, falling back through chain
# ---------------------------------------------------------------------------
def _str(val):
    """Safely convert a value to string — handles LinkML PermissibleValue enums."""
    if val is None:
        return None
    if hasattr(val, 'text'):
        return val.text
    return str(val)

def _get(obj, path, default=None):
    """Get nested attribute: _get(cont, 'initial_state.duration', 2000.0)"""
    parts = path.split('.')
    cur = obj
    for p in parts:
        if cur is None:
            return default
        cur = getattr(cur, p, None)
    return cur if cur is not None else default

# ---------------------------------------------------------------------------
# Helper: get toolkit-specific setting from parameters slot
# ---------------------------------------------------------------------------
def _get_param(obj, param_name, default=None):
    """Look up a named parameter in obj.parameters list/dict (numeric values)."""
    if obj is None:
        return default
    params = getattr(obj, 'parameters', None)
    if params is None:
        return default
    if isinstance(params, dict):
        p = params.get(param_name)
        return p.value if p and p.value is not None else default
    if isinstance(params, (list, tuple)):
        for p in params:
            if getattr(p, 'name', None) == param_name:
                return p.value if p.value is not None else default
    return default

# ---------------------------------------------------------------------------
# Helper: get toolkit-specific option from options slot (string values)
# ---------------------------------------------------------------------------
def _get_option(obj, option_name, default=None):
    """Look up a named option in obj.options list/dict."""
    if obj is None:
        return default
    opts = getattr(obj, 'options', None)
    if opts is None:
        return default
    if isinstance(opts, dict):
        o = opts.get(option_name)
        return o.value if o and o.value is not None else default
    if isinstance(opts, (list, tuple)):
        for o in opts:
            if getattr(o, 'name', None) == option_name:
                return o.value if o.value is not None else default
    return default

# ---------------------------------------------------------------------------
# Free parameters: from schema (multivalued free_parameters) or raw kwargs
# ---------------------------------------------------------------------------
if cont and hasattr(cont, 'free_parameters') and cont.free_parameters:
    _fp_dict = cont.free_parameters
    if isinstance(_fp_dict, dict):
        _fp_first = next(iter(_fp_dict.values()))
        ICS = str(_fp_first.name)
    else:
        _fp_first = _fp_dict[0]
        ICS = str(_fp_first.name)
    if _fp_first.domain:
        p_min = float(_fp_first.domain.lo) if _fp_first.domain.lo else 0
        p_max = float(_fp_first.domain.hi) if _fp_first.domain.hi else 1
    elif model.parameters[ICS].domain:
        p_min = float(model.parameters[ICS].domain.lo) if model.parameters[ICS].domain.lo else 0
        p_max = float(model.parameters[ICS].domain.hi) if model.parameters[ICS].domain.hi else 1
    else:
        p_min = float(context.get('p_min', 0))
        p_max = float(context.get('p_max', 1))
else:
    ICS = context.get('ICS', None)
    p_min = float(context.get('p_min', model.parameters[ICS].domain.lo if model.parameters[ICS].domain else 0))
    p_max = float(context.get('p_max', model.parameters[ICS].domain.hi if model.parameters[ICS].domain else 1))

vois = {i: k for i, k in enumerate(model.state_variables.keys())}
p_default = float(model.parameters[ICS].value) if model.parameters[ICS].value is not None else 0.0

# ---------------------------------------------------------------------------
# Shorthand: schema → kwargs → default (for backward compat with raw kwargs)
# ---------------------------------------------------------------------------
def _val(schema_path, kwarg_name, default, cast=float):
    v = _get(cont, schema_path) if cont else None
    if v is not None:
        return cast(v)
    v = context.get(kwarg_name, None)
    if v is not None:
        return cast(v)
    return cast(default)

def _bval(sp, kw, d): return _val(sp, kw, d, cast=bool)
def _ival(sp, kw, d): return _val(sp, kw, d, cast=int)
def _fval(sp, kw, d): return _val(sp, kw, d, cast=float)

# ---------------------------------------------------------------------------
# Equilibrium: solver settings (flat on ContinuationProblem)
# ---------------------------------------------------------------------------
_ds          = _fval('ds', 'ds', 0.01)
_dsmin       = _fval('ds_min', 'dsmin', 1e-4)
_dsmax       = _fval('ds_max', 'dsmax', 0.1)
_max_steps   = _ival('max_steps', 'max_steps', 400)
_tol_stab    = _fval('tol_stability', 'tol_stability', 1e-10)
_nev         = _ival('nev', 'nev', 3)
_n_inv       = _ival('n_inversion', 'n_inversion', 2)
_max_bisect  = _ival('max_bisection_steps', 'max_bisection_steps', 25)
_detect_bif  = _ival('detect_bifurcation', 'detect_bifurcation', 3)

# Newton corrector (flat on ContinuationProblem)
_newton_tol  = _fval('newton_tol', 'newton_tol', 1e-12)
_newton_iter = _ival('newton_max_iterations', 'newton_max_iterations', 25)

# Algorithm
_alg = _str(_get(cont, 'algorithm', 'PALC')) if cont else context.get('algorithm', 'PALC')

# BifurcationKit-specific from options slot
_tangent = _str(_get_option(cont, 'tangent', None) or context.get('tangent_method', 'Secant'))

# ODE solver from initial_state.solver.method (Integrator alias)
_iss_solver = _str(_get(cont, 'initial_state.solver.method', None) or context.get('ode_solver', 'Tsit5'))

# Top-level
_bothside    = _bval('bothside', 'bothside', True)
quiet = context.get('quiet', True)

# Clamp starting value into [p_min, p_max]
p_start = max(p_min, min(p_max, p_default))

# ---------------------------------------------------------------------------
# Initial state (maps to InitialState)
# ---------------------------------------------------------------------------
_iss_method   = _str(_get(cont, 'initial_state.method', 'time_integration'))
_iss_duration = float(_get(cont, 'initial_state.duration', 2000.0))
_iss_atol     = float(_get(cont, 'initial_state.abs_tol', 1e-10))
_iss_rtol     = float(_get(cont, 'initial_state.rel_tol', 1e-10))

# ---------------------------------------------------------------------------
# Branches (BranchSwitch) — replaces periodic_orbits
# ---------------------------------------------------------------------------
_branches = []
if cont and hasattr(cont, 'branches') and cont.branches:
    _br_raw = cont.branches
    if isinstance(_br_raw, dict):
        _branches = list(_br_raw.values())
    else:
        _branches = list(_br_raw)
    _has_branches = len(_branches) > 0
elif context.get('periodic_orbits', False) or context.get('branches', False):
    _has_branches = True
else:
    _has_branches = False

# Branch helper: extract setting from branch sub-continuation, then parent cont, then kwargs, then default.
# This ensures values explicitly set on the parent (e.g. n_inversion: 8) propagate
# to PO branches unless the branch sub-continuation explicitly overrides them.
_br_cont = _get(_branches[0], 'continuation') if _branches else None
def _br_val(attr_path, kwarg, default, cast=float):
    # attr_path is like 'continuation.ds' — strip 'continuation.' prefix for sub-cont lookup
    sub_attr = attr_path.split('.', 1)[-1] if '.' in attr_path else attr_path
    # 1. Try branch sub-continuation (only if it was explicitly provided)
    if _br_cont is not None:
        v = _get(_br_cont, sub_attr)
        if v is not None:
            # Check if this differs from the schema default for that field
            # to distinguish user-set from auto-defaulted values
            _schema_defaults = {
                'ds': 0.01, 'ds_min': 1e-4, 'ds_max': 0.1, 'max_steps': 400,
                'newton_tol': 1e-12, 'newton_max_iterations': 25, 'nev': 3,
                'tol_stability': 1e-10, 'detect_bifurcation': 3,
                'n_inversion': 2, 'max_bisection_steps': 25,
            }
            schema_default = _schema_defaults.get(sub_attr)
            # If value differs from schema default, user explicitly set it → use it
            if schema_default is None or cast(v) != cast(schema_default):
                return cast(v)
            # If value matches schema default, fall through to parent
    # 2. Try parent continuation
    if cont is not None:
        v = _get(cont, sub_attr)
        if v is not None:
            return cast(v)
    # 3. Try explicit kwarg
    v = context.get(kwarg, None)
    if v is not None:
        return cast(v)
    return cast(default)

# PO solver overrides (via BranchSwitch.continuation → ContinuationProblem)
po_ds          = _br_val('continuation.ds', 'po_ds', 1e-4, float)
po_dsmin       = _br_val('continuation.ds_min', 'po_dsmin', 1e-6, float)
po_dsmax       = _br_val('continuation.ds_max', 'po_dsmax', 0.02, float)
po_max_steps   = _br_val('continuation.max_steps', 'po_max_steps', 400, int)
po_tol_stab    = _br_val('continuation.tol_stability', 'po_tol_stab', 1e-6, float)
po_detect_bif  = _br_val('continuation.detect_bifurcation', 'po_detect_bif', 3, int)
po_n_inv       = _br_val('continuation.n_inversion', 'po_n_inv', 8, int)
po_nev         = _br_val('continuation.nev', 'po_nev', 3, int)
po_newton_tol  = _br_val('continuation.newton_tol', 'po_newton_tol', 1e-10, float)
po_newton_iter = _br_val('continuation.newton_max_iterations', 'po_newton_iter', 20, int)

# PO Algorithm — BK-specific from continuation.options or branch.options
_po_cont = _get(_branches[0], 'continuation') if _branches else None
po_tangent = _str(_get_option(_po_cont, 'tangent', None) or context.get('po_tangent', 'Bordered'))

# Discretization settings
_po_disc = _get(_branches[0], 'discretization') if _branches else None
po_method = _str(_get(_po_disc, 'method', 'collocation'))  # collocation | trapezoid | shooting | poincare

# Collocation/Trapezoid parameters (from parameters slot)
po_mesh_intervals = int(_get_param(_po_disc, 'mesh_intervals', None) or context.get('po_mesh_intervals', 20))
po_degree         = int(_get_param(_po_disc, 'degree', None) or context.get('po_degree', 4))
po_meshadapt      = bool(_get_param(_po_disc, 'mesh_adaptation', None) or context.get('po_meshadapt', False))
po_jacobian       = _str(_get_option(_po_disc, 'jacobian', None) or context.get('po_jacobian', 'DenseAnalytical'))

# Shooting parameters
po_n_sections     = int(_get_param(_po_disc, 'n_sections', None) or context.get('po_n_sections', 15))
po_parallel       = bool(_get_param(_po_disc, 'parallel', None) or context.get('po_parallel', True))
po_ode_solver     = _str(_get_option(_po_disc, 'ode_solver', None) or context.get('po_ode_solver', 'Rodas5'))

# Branch-specific
po_delta_p        = _br_val('delta_p', 'po_delta_p', 0.01, float)
po_bothside       = _br_val('bothside', 'po_bothside', True, bool)

# BK-specific branch settings from options slot
_br0 = _branches[0] if _branches else None
# Linear solver: method-specific defaults, allow override
po_linear_solver_explicit = _str(_get_option(_br0, 'linear_solver', None) or context.get('po_linear_solver', None))
po_max_norm      = float(_get_param(_br0, 'max_norm_bound', None) or context.get('po_max_norm', 1e4))
_user_linear_solver = _str(_get_option(_br0, 'linear_solver', None) or context.get('po_linear_solver', None))
if _user_linear_solver:
    po_linear_solver = _user_linear_solver
else:
    po_linear_solver = 'COPBLS' if po_method == 'collocation' else 'MatrixBLS'
po_max_norm      = float(_get_param(_br0, 'max_norm_bound', None) or context.get('po_max_norm', 1e4))

# Determine which Hopf points to branch from
po_source = 'hopf:-1'
if _branches and hasattr(_branches[0], 'source_point') and _branches[0].source_point:
    po_source = _branches[0].source_point
po_all_hopf = po_source == 'hopf:all'

%>
##
<%include file="/tvbo-julia-model.jl.mako" args="model=model" />
##
# Override continuation parameter to start within [p_min, p_max]
p = merge(p, (${ICS} = ${float(p_start)},))

# Initial conditions from model defaults
x0 = [
        % for sv in model.state_variables.values():
        ${sv.initial_value if sv.initial_value != 0 else 0.1}, # Initial value for ${sv.name}
        % endfor
    ]

# Wrapper: BifurcationKit expects f!(du, x, p) (no explicit time argument)
function ${model.name}_vf!(du, x, p)
    ${model.name}!(du, x, p, 0.0)  # pass dummy time
    return du
end

# Find a steady state via time integration (more robust than raw Newton on x0)
function _find_steady_state(f!, x0, p; T=${_iss_duration})
    function ode_f!(du, u, _p, t)
        f!(du, u, p, t)
    end
    prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(ode_f!, x0, (0.0, T), nothing)
    sol = solve(prob_ode, ${_iss_solver}(); abstol=${_iss_atol}, reltol=${_iss_rtol}, save_everystep=false)
    return sol[:, end]
end

x0_eq = _find_steady_state(${model.name}!, x0, p)

################################################################################

# Record named state variables for each continuation step
record_from_sol = (x, p; k...) -> (${', '.join(f'{sv.name} = x[{i+1}]' for i, sv in enumerate(model.state_variables.values()))},)

# Bifurcation Problem
prob = BifurcationProblem(${model.name}_vf!, x0_eq, p, (@optic _.${ICS});
    record_from_solution = record_from_sol)

# ContinuationPar (shared struct — same for equilibrium & PO, PO overrides below)
opts_br = ContinuationPar(
    p_min=${float(p_min)}, p_max=${float(p_max)},
    ds = ${_ds},
    dsmin = ${_dsmin},
    dsmax = ${_dsmax},
    max_steps = ${_max_steps},
    tol_stability = ${_tol_stab},
    detect_bifurcation = ${_detect_bif},
    n_inversion = ${_n_inv},
    max_bisection_steps = ${_max_bisect},
    nev = ${_nev},
    newton_options = NewtonPar(tol = ${_newton_tol}, max_iterations = ${_newton_iter}))

% if quiet:
using Logging
prev_logger = current_logger()
global_logger(SimpleLogger(devnull, Logging.Error))
% endif

br = continuation(prob, ${_alg}(tangent = ${_tangent}()), opts_br; normC=norminf, bothside=${'true' if _bothside else 'false'})

% if quiet:
global_logger(prev_logger)
% endif

# Minimal export: pass raw continuation result only; Python side derives all arrays.
bifurcation_result = br

########################################################################################################################

## Branches (periodic orbits, codim-2, etc.)
% if _has_branches:

# Record PO envelope (max/min per state variable)
args_po = (	record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (
                % for i, sv in enumerate(model.state_variables.values()):
                max_${sv.name} = maximum(xtt[${i+1},:]),
				min_${sv.name} = minimum(xtt[${i+1},:]),
                % endfor
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		arg = (marker = :d, markersize = 1)
        ${'\n\t'.join([f"plot!(xtt.t, xtt[{i+1},:]; label = \"{sv.name}\", arg..., k...)" for i, sv in enumerate(model.state_variables.values())])}
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	normC = norminf)

# ContinuationPar for PO branch (override from parent)
opts_po_cont = ContinuationPar(
    opts_br,
    ds = ${po_ds},
    dsmin = ${po_dsmin},
    dsmax = ${po_dsmax},
    max_steps = ${po_max_steps},
    tol_stability = ${po_tol_stab},
    detect_bifurcation = ${po_detect_bif},
    n_inversion = ${po_n_inv},
    nev = ${po_nev},
    newton_options = NewtonPar(tol = ${po_newton_tol}, max_iterations = ${po_newton_iter}),
)

% if context.get('bif_point', None):
hopf_indices = [${context['bif_point']}]
% else:
hopf_indices = Int[]
for (i, sp) in enumerate(br.specialpoint)
    sp.type == :hopf && push!(hopf_indices, i)
end
% if not po_all_hopf:
if !isempty(hopf_indices)
    hopf_indices = [hopf_indices[end]]  # only last Hopf unless po_all_hopf requested
end
% endif
% endif

<%
# Determine linear solver per method (BifurcationKit.jl-specific defaults)
if po_linear_solver_explicit:
    # User explicitly specified - use it
    po_linear_solver = po_linear_solver_explicit
elif po_method == 'collocation':
    # COPBLS is optimal for collocation (handles COP structure)
    po_linear_solver = 'COPBLS'
elif po_method in ('shooting', 'poincare'):
    # MatrixBLS for shooting methods
    po_linear_solver = 'MatrixBLS'
else:
    # Trapezoid: omit linear_algo (use BK default)
    po_linear_solver = None
%>
po_branches = Any[]
for hopf_idx in hopf_indices
    try
% if po_method == 'collocation':
        # Orthogonal collocation at Gauss points
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitOCollProblem(${po_mesh_intervals}, ${po_degree};
                meshadapt = ${'true' if po_meshadapt else 'false'},
                jacobian = BifurcationKit.${po_jacobian}());
            plot = ${'true' if context.get('plot', False) else 'false'},
            args_po...,
            δp = ${po_delta_p},
            alg = PALC(tangent = ${po_tangent}()),
% if po_linear_solver:
            linear_algo = BifurcationKit.${po_linear_solver}(),
% endif
            verbosity = 0,
            bothside = ${'true' if po_bothside else 'false'},
            callback_newton = BifurcationKit.cbMaxNorm(${po_max_norm}),
        )
% elif po_method == 'trapezoid':
        # Trapezoid method (finite differences)
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitTrapProblem(M = ${po_mesh_intervals});
            plot = ${'true' if context.get('plot', False) else 'false'},
            args_po...,
            δp = ${po_delta_p},
            alg = PALC(tangent = ${po_tangent}()),
% if po_linear_solver:
            linear_algo = BifurcationKit.${po_linear_solver}(),
% endif
            verbosity = 0,
            bothside = ${'true' if po_bothside else 'false'},
            callback_newton = BifurcationKit.cbMaxNorm(${po_max_norm}),
        )
% elif po_method == 'shooting':
        # Standard shooting (multiple ODE integrations)
        prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(${model.name}_vf!, x0_eq, (0.0, 1.0), p)
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            ShootingProblem(${po_n_sections}, prob_ode, OrdinaryDiffEq.${po_ode_solver}(), parallel = ${'true' if po_parallel else 'false'});
            plot = ${'true' if context.get('plot', False) else 'false'},
            args_po...,
            δp = ${po_delta_p},
            alg = PALC(tangent = ${po_tangent}()),
% if po_linear_solver:
            linear_algo = ${po_linear_solver}(),
% endif
            verbosity = 0,
            bothside = ${'true' if po_bothside else 'false'},
            callback_newton = BifurcationKit.cbMaxNorm(${po_max_norm}),
        )
% elif po_method == 'poincare':
        # Poincaré shooting (hyperplane crossing)
        prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(${model.name}_vf!, x0_eq, (0.0, 1.0), p)
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PoincareShootingProblem(${po_n_sections}, prob_ode, OrdinaryDiffEq.${po_ode_solver}(), parallel = ${'true' if po_parallel else 'false'});
            plot = ${'true' if context.get('plot', False) else 'false'},
            args_po...,
            δp = ${po_delta_p},
            alg = PALC(tangent = ${po_tangent}()),
% if po_linear_solver:
            linear_algo = ${po_linear_solver}(),
% endif
            verbosity = 0,
            bothside = ${'true' if po_bothside else 'false'},
            callback_newton = BifurcationKit.cbMaxNorm(${po_max_norm}),
        )
% endif
        push!(po_branches, br_po)
    catch e
        @warn "PO continuation from Hopf $hopf_idx failed" exception=(e, catch_backtrace())
    end
end

po_results = (hopf_indices = hopf_indices, branches = po_branches)

% endif
