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
