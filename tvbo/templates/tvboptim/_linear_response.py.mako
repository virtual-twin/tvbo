## Linear-response codegen partials (JAX/tvboptim backend).
##
## Resolution — the symbolic per-node RHS / Jacobians and the state/coupling/parameter
## layout — comes from tvbo.analysis.linear_response.linear_response_context (Python).
## These <%def>s emit only the code STRUCTURE, rendering each symbolic entry via
## render_expression, so the same metadata prints to any backend (a Julia partial would
## emit Julia). No Python string-emit. `ctx` is the linear_response_context dict.
##
## Per-node unpack: state from `s`, coupling from `c`, parameters from the bunch `p_`
## (heterogeneous per-node params gathered by the traced node index `_i`).
<%def name="lr_node_unpack(ctx)">\
% for i, v in enumerate(ctx['svs']):
    ${v} = s[${i}]
% endfor
% for i, cpl in enumerate(ctx['net_cpls']):
    ${cpl} = c[${i}]
% endfor
% for p in ctx['pnames']:
% if p in ctx['pernode']:
    ${p} = jnp.asarray(getattr(p_, '${p}'))[_i]
% else:
    ${p} = getattr(p_, '${p}')
% endif
% endfor
</%def>\
##
## Deterministic network vector field dy/dt = f(y): per-node RHS vmapped over nodes,
## long-range coupling = connectome matvec. Settling it gives the operating point.
<%def name="lr_vf(ctx, name='_lr_vf')">\
<%
    from tvbo.codegen import render_expression
    _jc = lambda e: render_expression(e, format='jax', parameters=ctx['syms'])
%>\
def ${name}_node(s, c, p_, _i):
${self.lr_node_unpack(ctx)}\
    return jnp.array([${', '.join(_jc(e) for e in ctx['rhs'])}])

def ${name}(x, weights, p):
    _W = jnp.asarray(weights); _X = jnp.asarray(x); _N = _W.shape[0]
% if ctx['n_cpl']:
    _C = jnp.stack([_W @ _X[${ctx['src_k']}] for _ in range(${ctx['n_cpl']})])
% else:
    _C = jnp.zeros((0, _N))
% endif
    _F = jax.vmap(lambda i: ${name}_node(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))
    return _F.T
</%def>\
##
## Network Jacobian A at an operating point x: per-node local block ∂f/∂state (Jloc)
## on the block-diagonal + coupling block ∂f/∂coupling (Jcpl) scattered by the connectome.
<%def name="lr_jacobian(ctx, name='_lr_jacobian')">\
<%
    from tvbo.codegen import render_expression
    _jc = lambda e: render_expression(e, format='jax', parameters=ctx['syms'])
    n_sv, n_cpl, src_k = ctx['n_sv'], ctx['n_cpl'], ctx['src_k']
    _mat = lambda M, ncol: '[' + ', '.join('[' + ', '.join(_jc(M[k, l]) for l in range(ncol)) + ']' for k in range(n_sv)) + ']'
%>\
def ${name}_jloc(s, c, p_, _i):
${self.lr_node_unpack(ctx)}\
    return jnp.array(${_mat(ctx['Jloc'], n_sv)})
% if n_cpl:

def ${name}_jcpl(s, c, p_, _i):
${self.lr_node_unpack(ctx)}\
    return jnp.array(${_mat(ctx['Jcpl'], n_cpl)})
% endif

def ${name}(x, weights, p):
    _W = jnp.asarray(weights); _X = jnp.asarray(x); _N = _W.shape[0]
% if n_cpl:
    _C = jnp.stack([_W @ _X[${src_k}] for _ in range(${n_cpl})])
% else:
    _C = jnp.zeros((0, _N))
% endif
    _Jl = jax.vmap(lambda i: ${name}_jloc(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))
% if n_cpl:
    _Jc = jax.vmap(lambda i: ${name}_jcpl(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))
% endif
    _A = jnp.zeros((${n_sv} * _N, ${n_sv} * _N))
% for k in range(n_sv):
% for l in range(n_sv):
    _A = _A.at[${k}*_N:${k+1}*_N, ${l}*_N:${l+1}*_N].add(jnp.diag(_Jl[:, ${k}, ${l}]))
% endfor
% for cix in range(n_cpl):
    _A = _A.at[${k}*_N:${k+1}*_N, ${src_k}*_N:${src_k+1}*_N].add(_Jc[:, ${k}, ${cix}][:, None] * _W)
% endfor
% endfor
    return _A
</%def>\
##
## Operating point: settle the noise-free vector field to the deterministic fixed point and
## linearise (Jacobian A). Emitted ONCE — the covariance/psd/fisher observables below are all
## linear-algebra solves on this shared A, so the FP settle and eig assembly run a single time.
## Binds _lr_fp (fixed point) and _lr_A (Jacobian) using _lr_weights/_lr_params/_lr_x0 (set by
## the caller). Reuses the module-level ${vf}/${jac}. The Jacobian carries the model's native
## rate units (per-ms by convention); rescaling it to per-second HERE (once) makes every
## downstream quantity — stationary covariance, power spectrum (Hz), Fisher information —
## physical, rather than each observable re-deriving the unit conversion. `time_scale` is
## seconds per model time unit (from integration.unit), so this is metadata-driven, not hardcoded.
<%def name="lr_operating_point(ctx, dt=0.1, n_settle=200000, n_newton=8, vf='_lr_vf', jac='_lr_jacobian', time_scale=1.0e-3)">\
def _lr_settle(_x, _):
    return _x + ${dt} * ${vf}(_x, _lr_weights, _lr_params), None
_lr_fp = jax.lax.scan(_lr_settle, _lr_x0, None, length=${n_settle})[0]
# Newton-polish the settled point to the EXACT fixed point (residual -> machine precision),
# reusing the symbolic Jacobian. The settle reaches the right basin; Newton removes the O(dt)
# residual so the operating point matches a Newton/fsolve solve — critical near bifurcations
# and under stimulus, where the covariance/Fisher are sensitive to sub-percent FP error.
def _lr_newton(_x, _):
    _f = ${vf}(_x, _lr_weights, _lr_params)
    _J = ${jac}(_x, _lr_weights, _lr_params)
    return _x - jnp.linalg.solve(_J, _f.reshape(-1)).reshape(_x.shape), None
_lr_fp = jax.lax.scan(_lr_newton, _lr_fp, None, length=${n_newton})[0]
_lr_A = ${jac}(_lr_fp, _lr_weights, _lr_params) / ${time_scale}   # per-(model time unit) -> per-second
</%def>\
##
## A derived-variable expression (e.g. the FIC constraint variable I_E) as a per-node function of
## (state, network coupling, params) — same structure/symbols as the vector field, but returning
## the scalar quantity per node. Used to form the constraint residual of a constraint-defined
## operating point. `expr` is the unfolded sympy expression from linear_response.constraint_expr.
<%def name="lr_constraint_fn(ctx, name, expr)">\
<%
    from tvbo.codegen import render_expression
    _jc = lambda e: render_expression(e, format='jax', parameters=ctx['syms'])
%>\
def ${name}_node(s, c, p_, _i):
${self.lr_node_unpack(ctx)}\
    return ${_jc(expr)}

def ${name}(x, weights, p):
    _W = jnp.asarray(weights); _X = jnp.asarray(x); _N = _W.shape[0]
% if ctx['n_cpl']:
    _C = jnp.stack([_W @ _X[${ctx['src_k']}] for _ in range(${ctx['n_cpl']})])
% else:
    _C = jnp.zeros((0, _N))
% endif
    return jax.vmap(lambda i: ${name}_node(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))
</%def>\
##
## Constraint-defined operating point (Deco FIC — the paper's fsolve on the steady-state, NOT a
## stochastic tuning loop). Solve the EXTENDED system for (state, free_param) simultaneously:
##   f(state; theta) = 0                       (n_sv·N deterministic fixed-point equations)
##   constraint(state; theta) - target = 0     (N equations, e.g. I_E = target)
## by Newton with an autodiff Jacobian (exact, backend-agnostic) — reusing ${vf}/${jac}/${constraint_fn}.
## Warm-start the state from a noise-off settle at the default free-param value. Binds _lr_fp (state)
## and rebinds _lr_params with the solved free_param, so the covariance/psd/fisher below linearise at
## the tuned operating point with NO change. Vmaps across a parameter sweep (each cell solves in ms).
<%def name="lr_constrained_operating_point(ctx, free_param, constraint_fn, target, dt=0.1, n_settle=200000, n_newton=15, vf='_lr_vf', jac='_lr_jacobian', time_scale=1.0e-3)">\
<% n_sv = ctx['n_sv'] %>\
def _lr_csettle(_x, _):
    return _x + ${dt} * ${vf}(_x, _lr_weights, _lr_params), None
_lr_s0 = jax.lax.scan(_lr_csettle, _lr_x0, None, length=${n_settle})[0]
_lr_N = _lr_weights.shape[0]
_lr_theta0 = jnp.broadcast_to(jnp.asarray(getattr(_lr_params, '${free_param}')), (_lr_N,))
def _lr_cresidual(_z):
    _st = _z[:${n_sv} * _lr_N].reshape(${n_sv}, _lr_N)
    _th = _z[${n_sv} * _lr_N:]
    _p = Bunch({**_lr_params, '${free_param}': _th})
    _f = ${vf}(_st, _lr_weights, _p).reshape(-1)                       # f(state; theta) = 0
    _g = ${constraint_fn}(_st, _lr_weights, _p) - ${target}           # constraint(state; theta) = 0
    return jnp.concatenate([_f, _g])
_lr_z0 = jnp.concatenate([_lr_s0.reshape(-1), _lr_theta0])
def _lr_cnewton(_z, _):
    return _z - jnp.linalg.solve(jax.jacobian(_lr_cresidual)(_z), _lr_cresidual(_z)), None
_lr_z = jax.lax.scan(_lr_cnewton, _lr_z0, None, length=${n_newton})[0]
_lr_fp = _lr_z[:${n_sv} * _lr_N].reshape(${n_sv}, _lr_N)
_lr_params = Bunch({**_lr_params, '${free_param}': _lr_z[${n_sv} * _lr_N:]})   # solved free parameter
_lr_A = ${jac}(_lr_fp, _lr_weights, _lr_params) / ${time_scale}
</%def>\
##
## Continuous Lyapunov Σ solve on the shared A (Deco 2014 Fig 5, Eq 24): A Σ + Σ Aᵀ + Q = 0,
## Q = σ² I, by eigendecomposition Σ = V M Vᴴ, M = -(V⁻¹QV⁻ᴴ)/(λᵢ+λ̄ⱼ) — backend-independent
## (jnp.linalg.eig/inv), no scipy. Returns the excitatory-block covariance P[:N,:N]. A stationary
## covariance exists only when A is Hurwitz (all eigenvalues in the left half-plane); past a
## bifurcation the operating point is unstable, so the result is masked to NaN (jittable guard,
## survives vmap over a grid) rather than returning a meaningless non-PSD matrix.
<%def name="lr_covariance(ctx, name, sigma, return_='covariance')">\
def ${name}(A):
    _n = A.shape[0]; _N = _n // ${ctx['n_sv']}
    _Q = (${sigma} ** 2) * jnp.eye(_n)
    _lam, _V = jnp.linalg.eig(A)
    _Vi = jnp.linalg.inv(_V)
    _M = -(_Vi @ _Q.astype(_V.dtype) @ _Vi.conj().T) / (_lam[:, None] + jnp.conj(_lam)[None, :])
    _P = (_V @ _M @ _V.conj().T).real[:_N, :_N]
    _P = jnp.where(jnp.max(_lam.real) < 0.0, _P, jnp.nan)   # no stationary covariance if A not Hurwitz
% if return_ == 'correlation':
    _d = jnp.sqrt(jnp.diag(_P))          # Pearson correlation of the excitatory gating (Deco 'Q')
    return _P / jnp.outer(_d, _d)
% else:
    return _P
% endif
</%def>\
##
## Analytic power spectrum on the shared A (Deco 2014 Fig 5, Eq 28): per excitatory node,
## Φ_k(ω) = σ² Σ_l |(iωI − A)⁻¹_{kl}|², over a log-frequency grid (Hz). Returns [n_freq, N].
## A is already per-second (rescaled at the operating point), so ω = 2πf with f in Hz is consistent.
<%def name="lr_psd(ctx, name, sigma, f_lo=0.1, f_hi=50.0, n_freq=128)">\
def ${name}(A):
    _n = A.shape[0]; _N = _n // ${ctx['n_sv']}
    _freqs = jnp.geomspace(${f_lo}, ${f_hi}, ${n_freq})
    _A = A.astype(jnp.complex128)
    _I = jnp.eye(_n, dtype=jnp.complex128)
    def _phi(_f):
        _M = jnp.linalg.inv(1j * (2.0 * jnp.pi * _f) * _I - _A)
        return (${sigma} ** 2) * jnp.sum(jnp.abs(_M[:_N]) ** 2, axis=1)
    _psd = jax.vmap(_phi)(_freqs)
    # The linear-response spectrum is defined only at a stable operating point; mask past a
    # bifurcation (A not Hurwitz), consistent with the covariance guard above.
    return jnp.where(jnp.max(jnp.linalg.eigvals(A).real) < 0.0, _psd, jnp.nan)
</%def>\
##
## Fisher information vs stimulus intensity (Deco 2014 Fig 6f, Eqs 33-34). Sweeps the stimulus
## amplitude ΔI on `nodes` (the event's target regions), RE-SETTLING the operating point at each
## intensity — the one piece the shared covariance/psd do not need, since the stimulus shifts the
## fixed point. Then FI(ΔI) = μ'ᵀP⁻¹μ' + ½Tr[(P'P⁻¹)²] with μ = the excitatory-mean FP block and
## P = the excitatory stationary covariance, by finite differences over ΔI. ${vf}/${jac} are the
## Fisher-specific vector field / Jacobian (the stimulated variable is heterogeneous per-node).
<%def name="lr_fisher(ctx, name, stim_var, nodes, sigma, dI_lo, dI_hi, dI_step, vf, jac, time_scale=1.0e-3, dt=0.1, n_settle=200000, n_newton=8)">\
def ${name}():
    _N = _lr_weights.shape[0]
    _dIs = jnp.arange(${dI_lo}, ${dI_hi} + 0.5 * ${dI_step}, ${dI_step})
    _mask = jnp.zeros(_N).at[jnp.asarray(${nodes})].set(1.0)             # stimulus target nodes
    _base = jnp.broadcast_to(jnp.asarray(getattr(_lr_params, '${stim_var}')), (_N,))
    def _moments(_dI):
        _p = Bunch({**_lr_params, '${stim_var}': _base + _dI * _mask})   # stimulus as per-node input
        def _settle(_x, _):
            return _x + ${dt} * ${vf}(_x, _lr_weights, _p), None
        _fp = jax.lax.scan(_settle, _lr_x0, None, length=${n_settle})[0]
        def _newton(_x, _):                                             # polish to the exact stimulated FP
            _f = ${vf}(_x, _lr_weights, _p)
            _J = ${jac}(_x, _lr_weights, _p)
            return _x - jnp.linalg.solve(_J, _f.reshape(-1)).reshape(_x.shape), None
        _fp = jax.lax.scan(_newton, _fp, None, length=${n_newton})[0]
        _A = ${jac}(_fp, _lr_weights, _p) / ${time_scale}               # per-second Jacobian at ΔI
        _lam, _V = jnp.linalg.eig(_A); _Vi = jnp.linalg.inv(_V)
        _Qn = (${sigma} ** 2) * jnp.eye(_A.shape[0])
        _M = -(_Vi @ _Qn.astype(_V.dtype) @ _Vi.conj().T) / (_lam[:, None] + jnp.conj(_lam)[None, :])
        _P = (_V @ _M @ _V.conj().T).real[:_N, :_N]                      # excitatory covariance
        return _fp[0], _P                                               # μ = S_e block, P
    _mus, _Ps = jax.lax.map(_moments, _dIs)                             # each ΔI re-settles (sequential)
    _step = _dIs[1] - _dIs[0]
    def _fi(_k):
        _dmu = (_mus[_k + 1] - _mus[_k]) / _step
        _dP = (_Ps[_k + 1] - _Ps[_k]) / _step
        _Pinv = jnp.linalg.pinv(_Ps[_k])
        return _dmu @ _Pinv @ _dmu + 0.5 * jnp.trace((_dP @ _Pinv) @ (_dP @ _Pinv))
    return jax.vmap(_fi)(jnp.arange(_dIs.shape[0] - 1))
</%def>\
##
## Orchestrator: emit the WHOLE linear-response analysis block from a resolved spec. This keeps the
## code structure AND its orchestration (which partials, in what order, the operating-point choice,
## the per-observable loop) in the template — Python only RESOLVES the spec (symbols, params,
## stimulus, constraint) via linear_response._lr_analysis_spec. `spec` fields: ctx, op_constraint,
## constraint_expr, time_scale, obs (list of per-observable dicts). Emitted once; every covariance/
## psd/fisher observable shares the single operating point (_lr_A / _lr_params).
<%def name="lr_analysis_block(spec)">\
<% ctx = spec['ctx'] %>\
_lr_weights = jnp.asarray(network.graph.weights)
_lr_params = state.dynamics
_lr_x0 = jnp.broadcast_to(jnp.reshape(jnp.asarray(state.initial_state.dynamics), (${ctx['n_sv']}, -1)), (${ctx['n_sv']}, _lr_weights.shape[0]))
${self.lr_vf(ctx)}\
${self.lr_jacobian(ctx)}\
% if spec['op_constraint']:
## Deco FIC: the paper's fsolve on the steady state — solve [state, free_param] under the constraint,
## deterministically, instead of the stochastic tuning loop. Rebinds _lr_params with the solved param.
${self.lr_constraint_fn(ctx, '_lr_constraint', spec['constraint_expr'])}\
${self.lr_constrained_operating_point(ctx, spec['op_constraint']['free_parameter'], '_lr_constraint', spec['op_constraint']['target'], time_scale=spec['time_scale'])}\
% else:
${self.lr_operating_point(ctx, time_scale=spec['time_scale'])}\
% endif
% for o in spec['obs']:
% if o['type'] == 'covariance':
${self.lr_covariance(ctx, '_cov_' + o['name'], o['sigma'], o['return'])}\
obs.${o['name']} = _cov_${o['name']}(_lr_A)     # excitatory-block stationary ${o['return']} (Lyapunov, Eq 24)
% elif o['type'] == 'psd':
${self.lr_psd(ctx, '_psd_' + o['name'], o['sigma'], o['f_lo'], o['f_hi'], o['n_freq'])}\
obs.${o['name']} = _psd_${o['name']}(_lr_A)     # analytic power spectrum per excitatory node (Eq 28)
% elif o['type'] == 'fisher':
## Fisher needs the stimulated variable per-node -> its own vector field / Jacobian from fisher_ctx.
${self.lr_vf(o['fisher_ctx'], name='_fvf_' + o['name'])}\
${self.lr_jacobian(o['fisher_ctx'], name='_fjac_' + o['name'])}\
${self.lr_fisher(ctx, '_fisher_' + o['name'], o['stim_var'], o['nodes'], o['sigma'], o['dI_lo'], o['dI_hi'], o['dI_step'], '_fvf_' + o['name'], '_fjac_' + o['name'], time_scale=spec['time_scale'])}\
obs.${o['name']} = _fisher_${o['name']}()     # Fisher information over the ΔI sweep (Eqs 33-34)
% endif
% endfor
</%def>\
