using BifurcationKit
using OrdinaryDiffEq
<%!
from tvbo.adapters.julia_model import build_model_context
%>
<%
## All variables are pre-computed by BifurcationKitAdapter._prepare_context()
## Template only places values — no processing.
svs = list(model.state_variables.values())
mc = build_model_context(model, network, constraints=constraints)
n_nodes = mc.get("n_nodes", 1)
# A single node records each state variable directly; a network reduces derived observables across nodes below.
_rec = ", ".join(f"{sv.name} = x[{i+1}]" for i, sv in enumerate(svs))
# The record argument supplies the continuation parameter, so exclude it from the closure's destructure.
_destr_no_ics = ", ".join(n for n in (nm.strip() for nm in mc["destructure"].split(",")) if n and n != ICS)
%>
##
<%include file="/tvbo-julia-model.jl.mako" args="mc=mc" />
##
# Override continuation parameter to start within [p_min, p_max]
p = merge(p, (${ICS} = ${float(p_start)},))

# Initial conditions from model defaults (network: n_nodes blocks per state var)
x0 = [
        % for v in mc['u0']:
        ${v if v is not None else 0.1},
        % endfor
    ]

# Wrapper: BifurcationKit expects f!(du, x, p) (no explicit time argument)
function ${model.name}_vf!(du, x, p)
    ${model.name}!(du, x, p, 0.0)  # pass dummy time
    return du
end

# Find a steady state via time integration (more robust than raw Newton on x0)
function _find_steady_state(f!, x0, p; T=${iss_duration})
    function ode_f!(du, u, _p, t)
        f!(du, u, p, t)
    end
    prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(ode_f!, x0, (0.0, T), p)
<%
    solve_kwargs = ['save_everystep=false']
    if iss_atol is not None:
        solve_kwargs.insert(0, f'abstol={iss_atol}')
    if iss_rtol is not None:
        solve_kwargs.insert(-1 if iss_atol else 0, f'reltol={iss_rtol}')
    solve_kw_str = ', '.join(solve_kwargs)
%>\
% if iss_solver:
    sol = solve(prob_ode, ${iss_solver}(); ${solve_kw_str})
% else:
    sol = solve(prob_ode; ${solve_kw_str})
% endif
    return sol[:, end]
end

x0_eq = _find_steady_state(${model.name}!, x0, p)

################################################################################

# Record observables for each continuation step
% if network:
## Recompute derived observables per node, reusing the model's equation emission, and reduce to max and mean.
record_from_sol = (${mc['arg_x']}, ${ICS}; k...) -> begin
    (; ${_destr_no_ics}) = p
    N = ${n_nodes}
% for name, rhs in mc['derived_params']:
    ${name} = ${rhs}
% endfor
% for line in mc['coupling_pre']:
    ${line}
% endfor
% for name in mc['record_obs']:
    _${name}_max = -Inf; _${name}_sum = 0.0
% endfor
    @inbounds for i in 1:N
% for line in mc['unpack']:
        ${line}
% endfor
% for line in mc.get('pernode_gather', []):
        ${line}
% endfor
% for line in mc['coupling_body']:
        ${line}
% endfor
% for name, rhs in mc['derived_vars']:
        ${name} = ${rhs}
% endfor
% for name in mc['record_obs']:
        _${name}_max = max(_${name}_max, ${name}); _${name}_sum += ${name}
% endfor
    end
    (${', '.join(f'{name}_max = _{name}_max, {name}_mean = _{name}_sum / N' for name in mc['record_obs'])},)
end
% else:
record_from_sol = (x, p; k...) -> (${_rec},)
% endif

# Bifurcation Problem
prob = BifurcationProblem(${model.name}_vf!, x0_eq, p, (@optic _.${ICS});
    record_from_solution = record_from_sol)

# ContinuationPar
opts_br = ContinuationPar(${cp_args_str})

% if quiet:
using Logging
prev_logger = current_logger()
global_logger(SimpleLogger(devnull, Logging.Error))
% endif

br = continuation(prob, ${alg_str}, opts_br; ${cont_call_kwargs_str})

% if quiet:
global_logger(prev_logger)
% endif

bifurcation_result = br

########################################################################################################################

% if branches:
<%
br0 = branches[0]
%>
## Branches (periodic orbits, codim-2, etc.)

# Record PO envelope (max/min per state variable)
args_po = (	record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (
                % for i, sv in enumerate(svs):
                max_${sv.name} = maximum(xtt[${i+1},:]),
				min_${sv.name} = minimum(xtt[${i+1},:]),
                % endfor
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		arg = (marker = :d, markersize = 1)
        ${'\n\t'.join([f"plot!(xtt.t, xtt[{i+1},:]; label = \"{sv.name}\", arg..., k...)" for i, sv in enumerate(svs)])}
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	normC = norminf)

## PO ContinuationPar
% if br0['po_cp_args_str']:
opts_po_cont = ContinuationPar(opts_br, ${br0['po_cp_args_str']})
% else:
opts_po_cont = opts_br
% endif

## Source point selection
hopf_indices = Int[]
for (i, sp) in enumerate(br.specialpoint)
    sp.type == :hopf && push!(hopf_indices, i)
end
% if br0['all_hopf']:
# Using all Hopf points
% elif br0['hopf_idx'] is not None:
if !isempty(hopf_indices)
% if br0['hopf_idx'] < 0:
    hopf_indices = [hopf_indices[end${'+' + str(br0['hopf_idx'] + 1) if br0['hopf_idx'] != -1 else ''}]]
% else:
    hopf_indices = [hopf_indices[${br0['hopf_idx']}]]
% endif
end
% else:
if !isempty(hopf_indices)
    hopf_indices = [hopf_indices[end]]
end
% endif

## PO continuation
po_branches = Any[]
for hopf_idx in hopf_indices
    try
% if br0['method'] == 'collocation':
<%
    coll_kwargs = []
    if br0['meshadapt'] is not None:
        coll_kwargs.append(f"meshadapt = {'true' if br0['meshadapt'] else 'false'}")
    if br0['jacobian'] is not None:
        coll_kwargs.append(f"jacobian = BifurcationKit.{br0['jacobian']}()")
    coll_kw_str = ', ' + ', '.join(coll_kwargs) if coll_kwargs else ''
%>\
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitOCollProblem(${br0['mesh_intervals']}, ${br0['degree']}${coll_kw_str});
            ${br0['po_kwargs_str']},
        )
% elif br0['method'] == 'trapezoid':
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitTrapProblem(M = ${br0['mesh_intervals']});
            ${br0['po_kwargs_str']},
        )
% elif br0['method'] == 'shooting':
<%
    ode_kwargs = []
    if br0['ode_abstol'] is not None:
        ode_kwargs.append(f"abstol = {br0['ode_abstol']}")
    if br0['ode_reltol'] is not None:
        ode_kwargs.append(f"reltol = {br0['ode_reltol']}")
    ode_kw_str = '; ' + ', '.join(ode_kwargs) if ode_kwargs else ''
    shoot_kwargs = []
    if br0['parallel'] is not None:
        shoot_kwargs.append(f"parallel = {'true' if br0['parallel'] else 'false'}")
    shoot_kw_str = ', ' + ', '.join(shoot_kwargs) if shoot_kwargs else ''
%>\
        prob_ode = ODEProblem(${model.name}!, copy(x0), (0.0, ${br0['ode_time_span']}), p${ode_kw_str})
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            ShootingProblem(${br0['n_sections']}, prob_ode, OrdinaryDiffEq.${br0['ode_solver']}()${shoot_kw_str});
            ${br0['po_kwargs_str']},
        )
% elif br0['method'] == 'poincare':
<%
    ode_kwargs_p = []
    if br0['ode_abstol'] is not None:
        ode_kwargs_p.append(f"abstol = {br0['ode_abstol']}")
    if br0['ode_reltol'] is not None:
        ode_kwargs_p.append(f"reltol = {br0['ode_reltol']}")
    ode_kw_str_p = '; ' + ', '.join(ode_kwargs_p) if ode_kwargs_p else ''
    poinc_kwargs = []
    if br0['parallel'] is not None:
        poinc_kwargs.append(f"parallel = {'true' if br0['parallel'] else 'false'}")
    poinc_kw_str = ', ' + ', '.join(poinc_kwargs) if poinc_kwargs else ''
%>\
        prob_ode = ODEProblem(${model.name}!, copy(x0), (0.0, ${br0['ode_time_span']}), p${ode_kw_str_p})
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PoincareShootingProblem(${br0['n_sections']}, prob_ode, OrdinaryDiffEq.${br0['ode_solver']}()${poinc_kw_str});
            ${br0['po_kwargs_str']},
        )
% endif
        push!(po_branches, br_po)
    catch e
        @warn "PO continuation from Hopf $hopf_idx failed" exception=(e, catch_backtrace())
    end
end

# BifurcationKit records only amplitude and period, so phase-resample each orbit to NPROF points to keep the waveform recoverable.
NPROF = 400
NVARS = ${len(svs)}
_PHASE_GRID = LinRange(0.0, 1.0, NPROF)

# Depends only on the orbit's time axis, so it is computed once per orbit and reused for every state variable.
function _phase_brackets(tn)
    map(_PHASE_GRID) do g
        k = searchsortedfirst(tn, g)
        k <= 1        ? (1, 1, 0.0) :
        k > length(tn) ? (length(tn), length(tn), 0.0) :
        (k - 1, k, (g - tn[k-1]) / max(tn[k] - tn[k-1], eps()))
    end
end

po_profiles = Any[]
for (bi, br_po) in enumerate(po_branches)
    try
        # Rows are the branch steps that `po_results.branches` serialises, not the saved solutions, which can be a coarser stride.
        nrows = length(br_po.branch)
        profs = fill(NaN, nrows, NPROF, NVARS)
        for (si, _s) in enumerate(br_po.sol)
            _row = hasproperty(_s, :step) ? Int(_s.step) + 1 : si
            (1 <= _row <= nrows) || continue
            xtt = get_periodic_orbit(br_po.prob, _s.x, _s.p)
            _t = xtt.t
            _tn = (_t .- _t[1]) ./ max(_t[end] - _t[1], eps())
            _ord = sortperm(_tn); _tn = _tn[_ord]           # ensure ascending for interpolation
            _br = _phase_brackets(_tn)
            for vi in 1:NVARS
                _yv = xtt[vi, _ord]
                for (pj, (lo, hi, w)) in enumerate(_br)
                    profs[_row, pj, vi] = (1 - w) * _yv[lo] + w * _yv[hi]
                end
            end
        end
        push!(po_profiles, profs)
    catch e
        @warn "PO branch $bi: orbit-profile extraction failed; this branch's waveforms \
               (E(t), x(t), … over one period) will be MISSING from the result" exception=(e, catch_backtrace())
        push!(po_profiles, nothing)
    end
end

po_results = (hopf_indices = hopf_indices, branches = po_branches, profiles = po_profiles)

% endif

% if codim2_branches:
########################################################################################################################
## Codim-2 continuation
codim2_results = Any[]

% for c2 in codim2_branches:
<%
src_type = c2['source_type']
## In BifurcationKit.jl, fold points can be :bp or :fold.
## Hopf points are :hopf.
is_fold = src_type in ('fold', 'branch_point', 'bp')
%>\
# Codim-2 branch: ${c2['name']} (${src_type} → ${c2['ICS2']})
begin
    local _bif_indices = Int[]
    for (i, sp) in enumerate(br.specialpoint)
% if is_fold:
        (sp.type == :bp || sp.type == :fold) && push!(_bif_indices, i)
% else:
        sp.type == :${src_type} && push!(_bif_indices, i)
% endif
    end

% if c2['all_source']:
    # All ${src_type} points
% elif c2['source_idx_jl'] is not None:
    if !isempty(_bif_indices)
        _bif_indices = [_bif_indices[${c2['source_idx_jl']}]]
    end
% else:
    if !isempty(_bif_indices)
        _bif_indices = [_bif_indices[end]]
    end
% endif

    local _opts_c2 = ContinuationPar(${c2['codim2_cp_str']})

    for _bif_idx in _bif_indices
        try
            local _br_c2 = continuation(
                br, _bif_idx, (@optic _.${c2['ICS2']}),
                _opts_c2;
                ${c2['codim2_kwargs_str']},
            )
            push!(codim2_results, _br_c2)
        catch e
            @warn "Codim-2 continuation from ${src_type} $_bif_idx failed" exception=(e, catch_backtrace())
        end
    end
end

% endfor
% endif
