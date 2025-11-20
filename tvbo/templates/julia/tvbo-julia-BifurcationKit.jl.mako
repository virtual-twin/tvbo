using BifurcationKit
<%
if 'model' in context.keys():
    model = context['model']

vois = {i: k for i, k in enumerate(model.state_variables.keys())}
p_min = context.get('p_min', model.parameters[ICS].domain.lo if model.parameters[ICS].domain else 0)
p_max = context.get('p_max', model.parameters[ICS].domain.hi if model.parameters[ICS].domain else 1)
# Periodic orbit tuning (lightweight defaults overridable from Python kwargs)
po_all_hopf = context.get('po_all_hopf', False)
po_mesh_intervals = context.get('po_mesh_intervals', 50)
po_degree = context.get('po_degree', 4)
po_meshadapt = context.get('po_meshadapt', True)
po_max_steps = context.get('po_max_steps', 120)
po_ds = context.get('po_ds', ds if ds else 0.001)
po_dsmin = context.get('po_dsmin', dsmin if dsmin else 1e-4)
po_dsmax = context.get('po_dsmax', dsmax if dsmax else 0.1)
quiet = context.get('quiet', True)

%>
##
<%include file="/tvbo-julia-model.jl.mako" args="model=model" />
##
# Initial conditions
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

################################################################################

# Bifurcation Problem (deterministic; ignores any stochastic noise definitions)
prob = BifurcationProblem(${model.name}_vf!, x0, p, (@optic _.${ICS}))


# continuation options
opts_br = ContinuationPar(
    p_min=${float(p_min)}, p_max=${float(p_max)},
    ds = ${float(ds) if ds else 0.002},
    dsmin = ${float(dsmin) if dsmin else 5e-5},
    dsmax = ${float(dsmax) if dsmax else 0.05},
    max_steps = ${max_steps if max_steps else 120},
    tol_stability=${tol_stability if tol_stability else 1e-8},
    n_inversion=8, max_bisection_steps=60, nev=1)

% if quiet:
using Logging
prev_logger = current_logger()
global_logger(SimpleLogger(devnull, Logging.Error))
% endif

br = continuation(prob, PALC(), opts_br; normC=norminf, bothside=true)

# Minimal export: pass raw continuation result only; Python side derives all arrays.
bifurcation_result = br

########################################################################################################################

## Periodic Orbits
% if periodic_orbits:

# Branch of periodic orbits:
args_po = (	record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (
                % for i, sv in enumerate(model.metadata.state_variables.values()):
                max_${sv.name} = maximum(xtt[${i+1},:]),
				min_${sv.name} = minimum(xtt[${i+1},:]),
                % endfor
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		arg = (marker = :d, markersize = 1)
        ${'\n\t'.join([f"plot!(xtt.t, xtt[{i+1},:]; label = \"{sv.name}\", arg..., k...)" for i, sv in enumerate(model.metadata.state_variables.values())])}
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	# we use the supremum norm
	normC = norminf)

# continuation parameters
opts_po_cont = ContinuationPar(
    opts_br,
    dsmin = ${po_dsmin},
    ds = ${po_ds},
    dsmax = ${po_dsmax},
    max_steps = ${po_max_steps},
    tol_stability = ${tol_stability if tol_stability else 1e-5},
)

% if bif_point:
hopf_indices = [${bif_point}]
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

po_branches = Any[]  # store each periodic orbit branch
for hopf_idx in hopf_indices
    br_po = continuation(
        br, hopf_idx, opts_po_cont,
        PeriodicOrbitOCollProblem(${po_mesh_intervals}, ${po_degree}; meshadapt = ${'true' if po_meshadapt else 'false'});
        plot = ${'true' if plot else 'false'},
        args_po...,
        bothside = false,
        verbosity = ${verbosity if verbosity else 0}
    )
    push!(po_branches, br_po)
end

po_results = (hopf_indices = hopf_indices, branches = po_branches)

% if quiet:
global_logger(prev_logger)
% endif


% endif
