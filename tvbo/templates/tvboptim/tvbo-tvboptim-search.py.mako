<%doc>
Search-family codegen partials (tvboptim backend)
=================================================

Reusable Mako `<%def>`s for the adaptive Exploration strategies. Kept out of the
monolithic experiment template and inserted only when the strategy is present:

    <%namespace name="search" file="tvbo-tvboptim-search.py.mako"/>
    ...
    % if expl['strategy'] == 'nsga2':
    ${search.nsga2_body(expl)}
    % endif

Resolution (axis→state-path, log10 decode, free-param bounds, metric observations)
is done in the experiment template's context blocks (`exp_info['nsga2_axes']`,
`refine_infos`); these defs only lay out the code from that clean metadata. The
heavy imports (`_np`, `_Problem`, `_NSGA2`, `_pymoo_minimize`, `_HV`) live at the top
of the generated module gated by `has_nsga2`; `DataAxis`/`Space`/`ParallelExecution`
(has_explorations) and `Parameter`/`BoundedParameter` (has_optimization) are reused.
</%doc>
<%!
def obs_acc(oa, name):
    """Emit a `.data`-unwrapping access of observation `name` off a results Bunch `oa`."""
    return ("(getattr(%s, '%s').data if hasattr(getattr(%s, '%s', None), 'data') "
            "else getattr(%s, '%s'))") % (oa, name, oa, name, oa, name)

def axis_value(expr, transform, col):
    """Render the decoded decision value for column `col` (log10 → 10**x)."""
    idx = "%s[:, %d]" % (expr, col)
    return "10.0 ** %s" % idx if transform == 'log10' else idx
%>\
##
<%def name="nsga2_body(expl)">\
<%
    ga = expl['ga']
    axes = expl['nsga2_axes']
    objs = expl['objectives']
    obj_bunch = ', '.join('%s=%s' % (o, obs_acc('_oa', o)) for o in objs)
%>\
    # ── NSGA-II multi-objective search (pymoo) over the decision space ──
    # Each candidate is one model(state) run from the shared base warm-up
    # (model_fn/state/result_transient) — the same path as a bare simulation — so the
    # objective values are byte-identical to the reference workflow's ga_evaluate.
    _xl = _np.array(${[a['lo'] for a in axes]})
    _xu = _np.array(${[a['hi'] for a in axes]})
    _obj_names = ${objs}
    _ref_point = _np.array(${list(ga['reference_point'])})
    _all_evals = []
    _hv_log = []
    def _nsga_eval(_X):
        _batch = copy.deepcopy(state)
% for i, ax in enumerate(axes):
        _batch.${ax['path']} = DataAxis(jnp.asarray(${axis_value('_X', ax['transform'], i)}))
% endfor
        _space = Space(_batch, mode='zip')
        @jax.jit
        def _obj_fn(_s):
            _oa = compute_all_observations(model_fn(_s), _s, result_transient)
            return Bunch(${obj_bunch})
        _results = list(ParallelExecution(_obj_fn, _space, n_pmap=${expl['n_workers']}).run())
        _F = _np.empty((len(_results), ${len(objs)}))
        for _i, _r in enumerate(_results):
            _F[_i] = [float(_np.asarray(getattr(_r, _n))) for _n in _obj_names]
            _all_evals.append({'x': [float(v) for v in _np.asarray(_X[_i])], 'F': [float(v) for v in _F[_i]]})
        return _np.nan_to_num(_F, nan=1e6, posinf=1e6, neginf=1e6)
    class _MOOProblem(_Problem):
        def __init__(self):
            super().__init__(n_var=${len(axes)}, n_obj=${len(objs)}, xl=_xl, xu=_xu)
        def _evaluate(self, _X, _out, *_a, **_k):
            _out['F'] = _nsga_eval(_X)
    def _hv_cb(_algorithm):
        _Fp = _algorithm.pop.get('F')
        try:
            _hv = float(_HV(ref_point=_ref_point)(_Fp))
        except Exception:
            _hv = float('nan')
        _hv_log.append(_hv)
    _res = _pymoo_minimize(_MOOProblem(), _NSGA2(pop_size=${int(ga['population_size'])}), ('n_gen', ${int(ga['num_generations'])}), callback=_hv_cb, seed=${int(ga['seed'])}, verbose=False)
    return ExplorationResult(name='${expl['name']}', pareto_X=_np.atleast_2d(_np.asarray(_res.X)), pareto_F=_np.atleast_2d(_np.asarray(_res.F)), pareto_objectives=list(_obj_names), hv_per_gen=_np.asarray(_hv_log), all_evals=_all_evals, strategy='nsga2')
</%def>\
##
<%def name="refine_body(refine)">\
<%
    n = refine['n_nodes']
    fc = obs_acc('_oa', refine['fc_obs'])
    fg = obs_acc('_oa', refine['freq_obs'])
    rets = ['final_fc_corr=%s' % fc, 'final_freq_corr=%s' % fg, 'final_loss=(-(%s) - (%s))' % (fc, fg)]
    # Final values of the upstream exploration's objectives (e.g. psd_mae, one_minus_fc).
    rets += ['final_%s=%s' % (o, obs_acc('_oa', o)) for o in refine.get('objectives', [])]
    rets += ['%s_final=jnp.asarray(_final.%s)' % (fp['name'], fp['path']) for fp in refine['free_params']]
%>\
            # ── Parallel gradient refinement from every Pareto seed (${refine['exploration']}) ──
            # One full gradient optimization per non-dominated solution, run in parallel;
            # each seed rides a DataAxis on its decision variables (decoded per axis
            # transform), then the free parameters are marked before OptaxOptimizer.run.
            _pareto = results['explorations']['${refine['exploration']}']
            _pareto_X = _np.atleast_2d(_np.asarray(_pareto.pareto_X))
            # Bare optimizer (no print/save callbacks — they format the loss, which is a
            # BatchTracer under the per-seed ParallelExecution). Matches the reference's
            # OptaxOptimizer(loss, optax.adam(LR)).
            _refine_opt = OptaxOptimizer(loss_fn, optax.${refine['optimizer']}(${refine['learning_rate']}))
            _seed = copy.deepcopy(state)
% for ax in refine['seed_axes']:
            _seed.${ax['path']} = DataAxis(jnp.asarray(${axis_value('_pareto_X', ax['transform'], ax['col'])}))
% endfor
            def _optimize_from_seed(_s):
% for fp in refine['free_params']:
<%
    val = 'jnp.asarray(_s.%s)' % fp['path']
    if fp['hetero'] and fp['seeded']:
        val = '%s * jnp.ones(%d)' % (val, n)
    bounded = fp['lo'] is not None or fp['hi'] is not None
    lo = fp['lo'] if fp['lo'] is not None else '-jnp.inf'
    hi = fp['hi'] if fp['hi'] is not None else 'jnp.inf'
%>\
% if bounded:
                _s.${fp['path']} = BoundedParameter(${val}, ${lo}, ${hi})
% else:
                _s.${fp['path']} = Parameter(${val})
% endif
% endfor
                _final, _ = _refine_opt.run(_s, max_steps=${refine['max_steps']}, chunk_size=${refine['max_steps']}, mode='${refine['opt_mode']}')
                _oa = compute_all_observations(model_fn(_final), _final, transient)
                return Bunch(${', '.join(rets)})
            _refine_rows = list(ParallelExecution(_optimize_from_seed, Space(_seed, mode='zip'), n_pmap=${refine['n_workers']}).run())
            stage_results['${refine['name']}'] = Bunch(seeds=_refine_rows, pareto=_pareto, pareto_X=_pareto_X)
</%def>\
