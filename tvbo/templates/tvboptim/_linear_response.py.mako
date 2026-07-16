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
## the caller). Reuses the module-level ${vf}/${jac}.
<%def name="lr_operating_point(ctx, dt=0.1, n_settle=200000, vf='_lr_vf', jac='_lr_jacobian')">\
def _lr_settle(_x, _):
    return _x + ${dt} * ${vf}(_x, _lr_weights, _lr_params), None
_lr_fp = jax.lax.scan(_lr_settle, _lr_x0, None, length=${n_settle})[0]
_lr_A = ${jac}(_lr_fp, _lr_weights, _lr_params)
</%def>\
##
## Continuous Lyapunov Σ solve on the shared A (Deco 2014 Fig 5, Eq 24): A Σ + Σ Aᵀ + Q = 0,
## Q = σ² I, by eigendecomposition Σ = V M Vᴴ, M = -(V⁻¹QV⁻ᴴ)/(λᵢ+λ̄ⱼ) — backend-independent
## (jnp.linalg.eig/inv), no scipy. Returns the excitatory-block covariance P[:N,:N].
<%def name="lr_covariance(ctx, name, sigma, return_='covariance')">\
def ${name}(A):
    _n = A.shape[0]; _N = _n // ${ctx['n_sv']}
    _Q = (${sigma} ** 2) * jnp.eye(_n)
    _lam, _V = jnp.linalg.eig(A)
    _Vi = jnp.linalg.inv(_V)
    _M = -(_Vi @ _Q.astype(_V.dtype) @ _Vi.conj().T) / (_lam[:, None] + jnp.conj(_lam)[None, :])
    _P = (_V @ _M @ _V.conj().T).real[:_N, :_N]
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
<%def name="lr_psd(ctx, name, sigma, f_lo=0.1, f_hi=50.0, n_freq=128)">\
def ${name}(A):
    _n = A.shape[0]; _N = _n // ${ctx['n_sv']}
    _freqs = jnp.geomspace(${f_lo}, ${f_hi}, ${n_freq})
    _I = jnp.eye(_n, dtype=jnp.complex128)
    def _phi(_f):
        _M = jnp.linalg.inv(1j * (2.0 * jnp.pi * _f) * _I - A.astype(jnp.complex128))
        return (${sigma} ** 2) * jnp.sum(jnp.abs(_M[:_N]) ** 2, axis=1)
    return jax.vmap(_phi)(_freqs)
</%def>\
