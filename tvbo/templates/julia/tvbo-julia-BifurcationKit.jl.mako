using BifurcationKit
using OrdinaryDiffEq
<%namespace name="model_fn" file="/tvbo-julia-model.jl.mako"/>
<%
## All variables are pre-computed by BifurcationKitAdapter._prepare_context(); the template only places them.
svs = list(model.state_variables.values())
%>
##
<%include file="/tvbo-julia-model.jl.mako" args="mc=mc" />
##
# Override continuation parameter to start within [p_min, p_max]
p = merge(p, (${start_params},))

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

% if integrate_to_start:
# The continuation starts from the state the model settles to from x0.
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
% else:
# The continuation starts from the model's initial state, which BifurcationKit's first Newton step corrects onto the branch.
x0_eq = x0
% endif

################################################################################

# Record observables for each continuation step
% if network:
## Recompute derived observables per node, reusing the model's equation emission, and reduce to max and mean.
record_from_sol = (${mc['arg_x']}, ${ICS}; k...) -> begin
    (; ${record_destructure}) = p
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
${model_fn.node_locals(mc, '        ')}\
% for name in mc['record_obs']:
        _${name}_max = max(_${name}_max, ${name}); _${name}_sum += ${name}
% endfor
    end
    (${record_fields},)
end
% else:
record_from_sol = (x, p; k...) -> (${record_fields},)
% endif

"""The vector field's Jacobian *J*, raising where an entry is not finite, so a continuation that would build its branch through it fails instead."""
function _finite_jacobian(J)
    all(isfinite, J) || error("the vector field's Jacobian has a non-finite entry at this iterate")
    return J
end

# Bifurcation Problem: BifurcationKit's own ForwardDiff Jacobian, checked by every continuation below.
prob = BifurcationProblem(${model.name}_vf!, x0_eq, p, (@optic _.${ICS});
    record_from_solution = record_from_sol)
prob = let vf = prob.VF
    re_make(prob; J = (x, p) -> _finite_jacobian(BifurcationKit.jacobian(vf, x, p)))
end

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
## Periodic-orbit branches

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

hopf_indices = Int[]
for (i, sp) in enumerate(br.specialpoint)
    sp.type == :hopf && push!(hopf_indices, i)
end
po_branches = Any[]
% for br0 in branches:

# Periodic-orbit branch: ${br0['name']}
% if br0['po_cp_args_str']:
opts_po_cont = ContinuationPar(opts_br, ${br0['po_cp_args_str']})
% else:
opts_po_cont = opts_br
% endif
% if br0['hopf_idx_jl'] is None:
_po_sources = hopf_indices
% else:
_po_sources = isempty(hopf_indices) ? Int[] : [hopf_indices[${br0['hopf_idx_jl']}]]
% endif
for hopf_idx in _po_sources
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
        # An orbit's period is positive, so a branch that reaches a non-positive one is not an orbit.
        minimum(br_po.branch.period) > 0 || error("the branch's period reaches $(minimum(br_po.branch.period))")
        push!(po_branches, br_po)
    catch e
        @warn "PO continuation from Hopf $hopf_idx failed" exception=(e, catch_backtrace())
    end
end
% endfor

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

_describe(points) = join(["$(sp.type) ($(sp.status)) at step $(sp.step), $(sp.printsol)" for sp in points], "; ")

"""Whether special points `a` and `b` of curve `br` are one point: the same type, and the same second parameter and primary `ics` within the two C-intervals they were located to together, carried to the primary by the curve's slope at `a`."""
function _coincide(a, b, br, ics)
    rows = (br.branch[max(a.idx - 1, 1)], br.branch[min(a.idx + 1, length(br))])
    slope = abs((rows[2][ics] - rows[1][ics]) / (rows[2].param - rows[1].param))
    δ = a.precision + b.precision
    return a.type == b.type && abs(a.param - b.param) <= δ && abs(getproperty(a.printsol, ics) - getproperty(b.printsol, ics)) <= δ * slope
end

"""The special points codim-2 curve `br` reports, by their location and values, whatever BifurcationKit labels them: a point whose bisection did not converge is warned about and dropped, wherever it sits; on a curve that records the Hopf frequency ωₕ, a converged point is a Bogdanov-Takens point exactly when its |ωₕ| is below 1e-3 of the curve's median |ωₕ|, and a `bt` at or above that is warned about and dropped; points that `_coincide` are one point, the one located most precisely. `source` names the point the curve started from."""
function _report_codim2_points!(br, ics, source)
    unlocated = findall(sp -> sp.status != :converged, br.specialpoint)
    isempty(unlocated) || @warn "Codim-2 curve from $source (steps $(first(br.branch.step)) to $(last(br.branch.step))): not reporting $(_describe(br.specialpoint[unlocated])), whose bisection did not converge"
    deleteat!(br.specialpoint, unlocated)
    if hasproperty(first(br.branch), :ωₕ)
        ω = sort(abs.(br.branch.ωₕ))
        ω_bt = 1e-3 * ω[cld(length(ω), 2)]
        not_bt = findall(sp -> sp.type == :bt && abs(sp.printsol.ωₕ) >= ω_bt, br.specialpoint)
        isempty(not_bt) || @warn "Codim-2 curve from $source: not reporting $(_describe(br.specialpoint[not_bt])) as a Bogdanov-Takens point, its |ωₕ| at or above $ω_bt"
        deleteat!(br.specialpoint, not_bt)
        for (i, sp) in pairs(br.specialpoint)
            sp.type != :endpoint && abs(sp.printsol.ωₕ) < ω_bt && (br.specialpoint[i] = setproperties(sp; type = :bt))
        end
    end
    order = sortperm(br.specialpoint; by = sp -> sp.precision)
    copies = Int[]
    for (n, i) in enumerate(order), j in order[1:(n - 1)]
        j in copies || i in copies || _coincide(br.specialpoint[i], br.specialpoint[j], br, ics) && push!(copies, i)
    end
    deleteat!(br.specialpoint, sort(copies))
    return br
end

% for c2 in codim2_branches:
<%
src_type = c2['source_type']
%>\
# Codim-2 branch: ${c2['name']} (${src_type} → ${c2['ICS2']})
begin
    local _bif_indices = Int[]
    for (i, sp) in enumerate(br.specialpoint)
% if c2['is_fold']:
        (sp.type == :bp || sp.type == :fold) && push!(_bif_indices, i)
% else:
        sp.type == :${src_type} && push!(_bif_indices, i)
% endif
    end

% if c2['source_idx_jl'] is None:
    # All ${src_type} points
% else:
    if !isempty(_bif_indices)
        _bif_indices = [_bif_indices[${c2['source_idx_jl']}]]
    end
% endif

    local _opts_c2 = ContinuationPar(${c2['codim2_cp_str']})

    for _bif_idx in _bif_indices
        try
% if c2['is_fold']:
            # Detection without bisection stores the step past the fold, so the curve starts from the fold newton_fold locates from there.
            local _fold = BifurcationKit.newton_fold(br, _bif_idx; normN = norminf, options = _opts_c2.newton_options, start_with_eigen = true, bdlinsolver = MatrixBLS(), usehessian = false)
            _fold.converged || error("newton_fold did not locate the fold from special point $_bif_idx")
            local _source = @set br.specialpoint[_bif_idx] = setproperties(br.specialpoint[_bif_idx]; x = _fold.u.u, param = _fold.u.p)
% else:
            local _source = br
% endif
            local _br_c2 = continuation(
                _source, _bif_idx, (@optic _.${c2['ICS2']}),
                _opts_c2;
                ${c2['codim2_kwargs_str']},
            )
            push!(codim2_results, _report_codim2_points!(_br_c2, :${ICS}, "${src_type} $_bif_idx"))
        catch e
            @warn "Codim-2 continuation from ${src_type} $_bif_idx failed" exception=(e, catch_backtrace())
        end
    end
end

% endfor
% endif
