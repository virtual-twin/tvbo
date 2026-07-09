<%doc>
Warm-started sweep codegen partials (tvboptim backend)
======================================================

Reusable Mako `<%def>`s for the warm-started Exploration path
(`sweep_seeding == 'from_previous'`) — a quasi-static parameter sweep that carries
each settled state forward into the next value. Inserted only when a warm-started
exploration is present:

    <%namespace name="sweep" file="tvbo-tvboptim-sweep.py.mako"/>
    ...
    % elif expl['sweep_seeding'] == 'from_previous':
    ${sweep.warmstart_sweep_body(expl, solver_class, dt)}

Both variants delegate to the ONE tvboptim primitive `_adiabatic_scan` (ramp the swept
parameter, carrying the settled state; reduce each rollout via observe/statistics):

- **envelope preset** (`expl['adiabatic']`, from `strategy: adiabatic_scan`): the
  oscillation envelope of a signal → `env_lo`/`env_hi`/`env_mean`.
- **record path** (`expl['warmstart_records']`): the exploration's declared
  trajectory-reduction observations, each recorded as a statistic on its source state
  variable's settled rollout.

Resolution (axis→state-path, envelope signal → observe code, record → callable + source
var index, segment/transient times, up/down/bidirectional → bothways) is done in the
experiment template's context blocks; these defs only lay out code from that clean
metadata. `_adiabatic_scan` is imported once at module top, gated by `has_warmstart`;
`ExplorationResult`/`Bunch`/`jnp` are already in scope.
</%doc>\
##
<%def name="warmstart_sweep_body(expl, solver_class, dt, solver_kwargs='')">\
<%
    a = expl.get('adiabatic')
    axis = a['axis'] if a else expl['axes'][0]
    name = axis['name']
    label = axis.get('label', name)  # dotted axis name for the ExplorationResult (== space key)
    path = ("coupling.%s.%s" % (axis['coupling_key'], name)) if axis.get('is_coupling') \
           else ("dynamics.%s" % name)
    bothways = expl['sweep_direction'] == 'bidirectional'
%>\
% if a:
    # -- Adiabatic bifurcation scan (delegates to tvboptim adiabatic_scan) --
    # Ramp the swept parameter up then back down, carrying the settled state; record the
    # oscillation envelope (per-node temporal min/max averaged across nodes, plus the mean)
    # of the observed signal at each value. The up/down branches expose any hysteresis.
    def _adia_observe(_r):
        return ${a['signal_code']}
    _adia_stats = {"mean": lambda _a: _a.mean(), "lo": lambda _a: _a.min(axis=0).mean(), "hi": lambda _a: _a.max(axis=0).mean()}
    _adia_res = _adiabatic_scan(
        _network, ${solver_class}(${solver_kwargs}),
        accessor=lambda _c: _c.${path},
        low=${axis['lo']}, high=${axis['hi']}, n=kwargs.get('n_${name}', ${axis['n']}),
        t=${a['segment_time']}, skip=${a['skip']}, dt=${dt}, bothways=${bothways},
        observe=_adia_observe, statistics=_adia_stats,
    )
    _adia_p = jnp.asarray(_adia_res.p)
    return ExplorationResult(
        name='${expl['name']}',
        axes=[Bunch(name='${label}', explored_values=_adia_p, n=int(_adia_p.shape[0]), is_coupling=${bool(axis.get('is_coupling'))}, coupling_key=${repr(axis.get('coupling_key'))})],
        observations={'env_lo': jnp.asarray(_adia_res.stats['lo']), 'env_hi': jnp.asarray(_adia_res.stats['hi']), 'env_mean': jnp.asarray(_adia_res.stats['mean'])},
        observable='adiabatic', dt=${dt}, n_up=int(_adia_res.n_up), strategy='adiabatic_scan',
    )
% else:
    # -- Warm-started record sweep (delegates to tvboptim adiabatic_scan) --
    # Ramp the swept parameter, carrying each settled state into the next value; at every
    # value record the declared trajectory-reduction observations of the settled rollout.
    # bidirectional (up then back down) exposes hysteresis / multistability.
    def _ws_observe(_r):
        return _r.ys
    _ws_stats = {
% for r in expl['warmstart_records']:
        '${r['name']}': (lambda _a, _f=${r['call']}, _i=${r['var_idx']}: _f(_a[:, _i, :])),
% endfor
    }
    _ws_res = _adiabatic_scan(
        _network, ${solver_class}(${solver_kwargs}),
        accessor=lambda _c: _c.${path},
        low=${axis['lo']}, high=${axis['hi']}, n=kwargs.get('n_${name}', ${axis['n']}),
        t=${expl['warmstart_segment']}, skip=${expl['warmstart_skip']}, dt=${dt}, bothways=${bothways},
        observe=_ws_observe, statistics=_ws_stats,
    )
    _ws_p = jnp.asarray(_ws_res.p)
    return ExplorationResult(
        name='${expl['name']}',
        axes=[Bunch(name='${label}', explored_values=_ws_p, n=int(_ws_p.shape[0]), is_coupling=${bool(axis.get('is_coupling'))}, coupling_key=${repr(axis.get('coupling_key'))})],
        observations={${', '.join("'%s': jnp.asarray(_ws_res.stats['%s'])" % (r['name'], r['name']) for r in expl['warmstart_records'])}},
        observable='warmstart', dt=${dt}, n_up=int(_ws_res.n_up), strategy='warmstart',
    )
% endif
</%def>\
