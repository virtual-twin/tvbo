ENV["LC_ALL"] = "en_US.UTF-8"
ENV["LANG"] = "en_US.UTF-8"
ENV["LANGUAGE"] = "en_US.UTF-8"

using Revise, Plots
using BifurcationKit
using JLD2
using JSON

const BK = BifurcationKit

function TMvf!(dz, z, p, t = 0)
    (;${", ".join([p.name for p in model.parameters.values()] + [p.name for p in model.coupling_terms.values()])}) = p
    ${", ".join([sv.name for sv in model.state_variables.values()])} = z

    local_coupling = 0

    ${"\n    ".join([f"{dp.name} = {dp.equation.rhs.replace('**', '^')}" for dp in model.derived_parameters.values()])}

    ${"\n    ".join([f"{dv.name} = {dv.equation.rhs.replace('**', '^')}" for dv in model.derived_variables.values()])}

    ${"\n    ".join([f"dz[{i+1}] = {sv.equation.rhs.replace('**', '^')}" for i, sv in enumerate(model.state_variables.values())])}
    dz
end

# parameter values
par_tm = (${", ".join([f"{p.name} = {p.value}" for p in model.parameters.values()] + [f"{p.name} = 0.0" for p in model.coupling_terms.values()])})

# initial condition
z0 = ${[0.1 for i in model.state_variables] if not random_initial_conditions else (f"rand({len(model.state_variables)})" if not initial_conditions else initial_conditions)}

# Bifurcation Problem
prob = BifurcationProblem(TMvf!, z0, par_tm, (@optic _.${ICS});

	record_from_solution = (x, p; k...) -> (${", ".join([f"{sv.name} = x[{i+1}]" for i, sv in enumerate(model.state_variables.values())])}),)


# continuation options
opts_br = ContinuationPar(p_min=${float(p_min)}, p_max=${float(p_max)},
    ds = ${float(ds) if ds else 0.001},
    dsmin = ${float(dsmin) if dsmin else 1e-6},
    dsmax = ${float(dsmax) if dsmax else 0.1},
    max_steps = ${max_steps if max_steps else 110},
    ## detect_bifurcation=3,
    tol_stability=1e-12,
    # Optional: bisection options for locating bifurcations
    n_inversion=16, max_bisection_steps=200, nev=3,
    save_to_file=${'true' if filename else 'false'})

# continuation of equilibria

br = continuation(
    prob, PALC(${"tangent=Bordered()" if (not codim2 or bordered==False) else ""}), opts_br;
    normC=norminf,
    filename="${filename}",
    bothside=true,
    plot = true)

scene = plot(br, plotfold=false, markersize=4, legend=:topleft)
${f'savefig(scene, "{filename}.svg")' if filename else ""}

hopf_indices = []

sp = br.specialpoint  # Access the special points

if isa(sp, Vector)  # If sp is a vector of points
    for (i, point) in enumerate(sp)
        if point.type == :hopf
            println("Hopf bifurcation detected at index $i, parameter: ", point.param)
            push!(hopf_indices, i)  # Store the index of the Hopf bifurcation
        end
    end
else
    if sp.type == :hopf
        println("Hopf bifurcation detected at parameter value: ", sp.param)
        push!(hopf_indices, 1)  # Store index 1 if only one point
    else
        println("No Hopf bifurcation detected.")
    end
end

################################################################################
################################################################################
## Periodic Orbits
% if periodic_orbits:

# Branch of periodic orbits:
args_po = (	record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		arg = (marker = :d, markersize = 1)
        ${'\n\t'.join([f"plot!(xtt.t, xtt[{i+1},:]; label = \"{sv.name}\", arg..., k...)" for i, sv in enumerate(model.state_variables.values())])}
		plot!(br; subplot = 1, putspecialptlegend = false)
		end,
	# we use the supremum norm
	normC = norminf)

# continuation parameters
opts_po_cont = ContinuationPar(
    opts_br,
    dsmin = ${float(dsmin) if dsmin else 1e-4},
    ds = ${float(ds) if ds else 0.001},
    dsmax = ${float(dsmax) if dsmax else 0.1},
    max_steps = ${max_steps if max_steps else 110},
    ## detect_bifurcation = ${detect_bifurcation if detect_bifurcation else 3},
    tol_stability = ${tol_stability if tol_stability else 1e-5},
    save_to_file = ${'true' if filename else 'false'}
)

br_potrap = @time continuation(
	br, ${bif_point}, opts_po_cont,
	PeriodicOrbitOCollProblem(50, 4; meshadapt = true);
	plot = true,
	args_po...,
    bothside = true,
    filename = "${filename}_po_${bif_point}",
    verbosity = ${verbosity if verbosity else 0}
	)

# Assuming br_potrap.param, br_potrap.min, and br_potrap.max are arrays or values
param = br_potrap.param
min_val = br_potrap.min
max_val = br_potrap.max

# Create a dictionary to store the values
data = Dict("param" => param, "min" => min_val, "max" => max_val)

% if filename:
# Save the dictionary as a JSON file
println("${filename}_po_${bif_point}.json")
open("${filename}_po_${bif_point}.json", "w") do f
    write(f, JSON.json(data))
end
% endif

% endif
################################################################################################################################################################
% if codim2:
# Codimension 2 bifurcation
sn_codim2 = continuation(br, ${fold_point}, (@optic _.${ICS2}),
	ContinuationPar(opts_br, p_min=${float(p_min)}, p_max=${float(p_max)}, ds = -0.001, dsmax = 0.05, save_to_file=${'true' if filename else 'false'});
	normC = norminf,
	# compute both sides of the initial condition
	bothside = true,
	# detection of codim 2 bifurcations
	detect_codim2_bifurcation = 2,
    filename="${filename}"
	)



hp_codim2 = continuation(br, ${hopf_point}, (@optic _.${ICS2}),
	ContinuationPar(opts_br, p_max = 2.8, ds = -0.001, dsmax = 0.025) ;
	normC = norminf,
	# detection of codim 2 bifurcations
	detect_codim2_bifurcation = 2,
	# compute both sides of the initial condition
	bothside = true,
	)


# plotting
scene = plot(sn_codim2, vars = (:${ICS}, :${VOI}), branchlabel = "Fold")
plot!(scene, hp_codim2, vars = (:${ICS2}, :${VOI}), branchlabel = "Hopf")
plot!(scene, br)

% endif
## ${f'savefig(scene, "{filename}.svg")' if filename else ""}
################################################################################################################################################################
