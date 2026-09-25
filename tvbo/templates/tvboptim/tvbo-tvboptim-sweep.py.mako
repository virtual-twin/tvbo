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
<%!
from tvbo.templates.tvboptim.utils import axis_keypath
%>
##
<%def name="warmstart_sweep_body(expl, solver_class, dt, solver_kwargs='')">\
<%
    a = expl.get('adiabatic')
    axis = a['axis'] if a else expl['axes'][0]
    name = axis['name']
    label = axis.get('label', name)  # dotted axis name for the ExplorationResult (== space key)
    path = axis_keypath(axis)
    bothways = expl['sweep_direction'] == 'bidirectional'
    # Scan points for the ticker: n per branch, doubled when bidirectional, resolved as the scan call does.
    n_total_expr = ("2 * " if bothways else "") + ("kwargs.get('n_%s', %s)" % (name, axis['n']))
%>\
% if a:
    # Ramp the parameter up then down carrying the settled state, recording the envelope at each value so hysteresis shows.
    def _adia_observe(_r):
        return ${a['signal_code']}
    # observe() runs once per swept value, so wrapping it ticks per point instead of going silent until the scan returns.
    _adia_observe = progress_ticker(${n_total_expr}, label="${expl['name']}")(_adia_observe)
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
<%
    ws_analysis = [a for a in (expl.get('warmstart_analysis') or []) if a['type'] == 'lyapunov']
    obs_pairs = ["'%s': jnp.asarray(_ws_res.stats['%s'])" % (r['name'], r['name']) for r in expl['warmstart_records']]
    for a in ws_analysis:
        obs_pairs += ["'%s': %s_lam" % (a['name'], a['name']), "'%s_xi': %s_xi" % (a['name'], a['name'])]
%>\
    # Carry each settled state into the next value and record the declared observations of the settled rollout.
    def _ws_observe(_r):
        return _r.ys
    # observe() is called once per swept value, so the ticker fires as each point settles.
    _ws_observe = progress_ticker(${n_total_expr}, label="${expl['name']}")(_ws_observe)
    _ws_stats = {
% for r in expl['warmstart_records']:
        '${r['name']}': (lambda _a, _f=${r['call']}, _i=${r['var_idx']}: _f(_a[:, _i, :])),
% endfor
% if ws_analysis:
        # The settled state at each value, the same carry the scan feeds forward, seeds the analysis pass below.
        '_seed_state': (lambda _a: _a[-1]),
% endif
    }
    _ws_res = _adiabatic_scan(
        _network, ${solver_class}(${solver_kwargs}),
        accessor=lambda _c: _c.${path},
        low=${axis['lo']}, high=${axis['hi']}, n=kwargs.get('n_${name}', ${axis['n']}),
        t=${expl['warmstart_segment']}, skip=${expl['warmstart_skip']}, dt=${dt}, bothways=${bothways},
        observe=_ws_observe, statistics=_ws_stats,
    )
    _ws_p = jnp.asarray(_ws_res.p)
% if ws_analysis:
    # Independent per value, so a memory-bounded lax.map runs one analysis per point.
    _ws_seed_states = jnp.asarray(_ws_res.stats['_seed_state'])   # (n_values, n_states, n_nodes)
% for a in ws_analysis:
${lyapunov_map(a, path, dt, solver_class, solver_kwargs, '_ws_p', '_ws_seed_states')}\
% endfor
% endif
    return ExplorationResult(
        name='${expl['name']}',
        axes=[Bunch(name='${label}', explored_values=_ws_p, n=int(_ws_p.shape[0]), is_coupling=${bool(axis.get('is_coupling'))}, coupling_key=${repr(axis.get('coupling_key'))})],
        observations={${', '.join(obs_pairs)}},
        observable='warmstart', dt=${dt}, n_up=int(_ws_res.n_up), strategy='warmstart',
    )
% endif
</%def>\
##
<%def name="lyapunov_map(a, path, dt, solver_class, solver_kwargs, values_expr, seeds_expr)">\
    # Re-seeded from each value's settled state, so ${a['name']} tracks the continued branch rather than a cold start.
    _le_solve_${a['name']}, _le_cfg_${a['name']} = prepare(_network, ${solver_class}(${solver_kwargs}), t0=0.0, t1=${a['segment_time']}, dt=${dt})
    def _lyap_at_${a['name']}(_carry):
        _val, _seed = _carry
        _cfg = eqx.tree_at(lambda _c: _c.${path}, _le_cfg_${a['name']}, _val)
        _cfg = eqx.tree_at(lambda _c: _c.initial_state.dynamics, _cfg, _seed)
        return benettin_spectrum_and_vectors(_le_solve_${a['name']}, _cfg, t=${a['segment_time']}, n=${a['n_steps']}, k=${a['n_exponents']})
    _lyap_pairs_${a['name']} = (${values_expr}, ${seeds_expr})
    # Ticks as each independent Lyapunov point completes, so a long branch does not go silent.
    _exps_${a['name']}, ${a['name']}_xi = jax.lax.map(
        progress_ticker(len(_lyap_pairs_${a['name']}[0]), label="${a['name']}")(_lyap_at_${a['name']}),
        _lyap_pairs_${a['name']})
    ${a['name']}_lam = _exps_${a['name']}[:, 0]   # leading exponent lambda_1(value)
</%def>\
##
<%doc>
Branch-restart analysis body (from_experiment, source_point='branch')
====================================================================
An independent exploration that restarts a per-point analysis over another experiment's
recorded branch. Each cell replays one branch point — its swept-parameter value and settled
state, loaded at runtime as `_BRANCH_SEED` — and records the analysis (Lyapunov λ₁ / ξ_i).
Because the cells are independent (unlike the sequential scan that produced the branch), HPC
array tasks shard it by slicing the paired (value, seed) arrays: `shard=(i, N)` keeps cells j
where j % N == i. Reuses the exact per-point Lyapunov map (`lyapunov_map`) as the warm-start
scan's post-scan pass — one renderer, one runtime primitive — just sourcing the pairs from
the loaded branch instead of an in-process scan.
</%doc>\
<%def name="branch_analysis_body(expl, solver_class, dt, solver_kwargs='')">\
<%
    axis = expl['axes'][0]
    name = axis['name']
    label = axis.get('label', name)
    path = axis_keypath(axis)
    analyses = expl.get('warmstart_analysis') or []
    obs_pairs = []
    for a in analyses:
        obs_pairs += ["'%s': %s_lam" % (a['name'], a['name']), "'%s_xi': %s_xi" % (a['name'], a['name'])]
%>\
    # -- Analysis restart over a sibling experiment's recorded branch (from_experiment: branch) --
    assert _BRANCH_SEED is not None, (
        "exploration '${expl['name']}' needs a from_experiment branch seed; run its "
        "source_experiment first so its recorded branch is available")
    _branch_p = jnp.asarray(_BRANCH_SEED['axis_values'])          # (n_cells,) swept values
    # Keyed by name, never positional: sort the loaded states by their model row index.
    _branch_seeds = _BRANCH_SEED['seeds']
    _rows = sorted(_branch_seeds, key=lambda _s: _STATE_INDEX[_s])
    _branch_ic = jnp.stack([jnp.asarray(_branch_seeds[_s]) for _s in _rows], axis=1)  # (n_cells, n_states, n_nodes)
    # A bidirectional branch repeats its swept values, so the position index is what keeps the two hysteresis branches separable.
    _branch_idx = jnp.arange(_branch_p.shape[0])
    # HPC shard: keep this array task's slice of the branch (cells j where j % N == i).
    _shard = kwargs.get('shard')
    if _shard is not None:
        _si, _sN = _shard
        _branch_p = _branch_p[_si::_sN]
        _branch_ic = _branch_ic[_si::_sN]
        _branch_idx = _branch_idx[_si::_sN]
% for a in analyses:
${lyapunov_map(a, path, dt, solver_class, solver_kwargs, '_branch_p', '_branch_ic')}\
% endfor
    return ExplorationResult(
        name='${expl['name']}',
        # A flat ordered sequence rather than a grid, so indexing by branch position reassembles shards losslessly where a value repeats.
        axes=[Bunch(name='branch_point', explored_values=_branch_idx, n=int(_branch_idx.shape[0]))],
        observations={${', '.join(obs_pairs)}},
        cell_coords={'branch_point': _branch_idx},
        # Declared, not inferred: `n` above is the sliced length, so a shard looks complete.
        is_shard=_shard is not None,
        observable='branch', dt=${dt}, strategy='branch',
    )
</%def>\
