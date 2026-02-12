using BifurcationKit
using OrdinaryDiffEq
<%
## All variables are pre-computed by BifurcationKitAdapter._prepare_context()
## Template only places values — no processing.
svs = list(model.state_variables.values())
%>
##
<%include file="/tvbo-julia-model.jl.mako" args="model=model" />
##
# Override continuation parameter to start within [p_min, p_max]
p = merge(p, (${ICS} = ${float(p_start)},))

# Initial conditions from model defaults
x0 = [
        % for sv in svs:
        ${sv.initial_value if sv.initial_value != 0 else 0.1}, # Initial value for ${sv.name}
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

# Record named state variables for each continuation step
record_from_sol = (x, p; k...) -> (${', '.join(f'{sv.name} = x[{i+1}]' for i, sv in enumerate(svs))},)

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
    coll_kw_str = ',\n                ' + ',\n                '.join(coll_kwargs) if coll_kwargs else ''
%>\
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitOCollProblem(${br0['mesh_intervals']}, ${br0['degree']}${coll_kw_str});
            ${br0['po_kwargs_str']},
        )
% elif br0['method'] == 'trapezoid':
% if br0['mesh_intervals'] is not None:
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitTrapProblem(M = ${br0['mesh_intervals']});
            ${br0['po_kwargs_str']},
        )
% else:
        br_po = continuation(
            br, hopf_idx, opts_po_cont,
            PeriodicOrbitTrapProblem();
            ${br0['po_kwargs_str']},
        )
% endif
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
        prob_ode = ODEProblem(${model.name}!, copy(x0), (0.0, 1.0), p${ode_kw_str})
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
        prob_ode = ODEProblem(${model.name}!, copy(x0), (0.0, 1.0), p${ode_kw_str_p})
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

po_results = (hopf_indices = hopf_indices, branches = po_branches)

% endif
