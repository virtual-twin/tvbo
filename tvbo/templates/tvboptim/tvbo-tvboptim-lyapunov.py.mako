## Benettin-with-Q Lyapunov analysis, emitted into the experiment script when an
## observation declares `analysis.type: lyapunov`. The code STRUCTURE lives in this
## <%def> partial (driven by the resolved analysis metadata) rather than a hardcoded
## tvbo runtime helper, so the generated script stays self-contained. The adapter
## (render_analysis_observations) resolves the metadata and emits only the call.
<%def name="benettin_function()">\
def benettin_spectrum_and_vectors(solve_fn, config, t, n=10, k=None):
    """Benettin QR Lyapunov spectrum + leading Lyapunov vector profile.

    Tangent-space propagation (``jax.linearize``) with QR renormalisation every
    segment — as in tvboptim's ``_lyapunov_spectrum_jvp`` — additionally keeping
    the time-averaged per-node norm of the orthonormalised leading tangent vector,
    ``xi_i`` (Taher et al. 2019): the per-node energy the most-unstable direction
    carries. tvboptim's own routine forms the identical Q frame and discards it.

    Args:
        solve_fn, config: a one-segment ``prepare(...)`` pair; the flow is re-seeded
            from ``config.initial_state.dynamics`` each segment.
        t: segment duration (integration time between renormalisations).
        n: number of rescaling segments (total tangent time = ``n * t``).
        k: number of exponents/vectors to track (``None`` for the full 2N spectrum).

    Returns:
        exponents (jnp.ndarray, (k,)): Lyapunov exponents, descending.
        xi (jnp.ndarray, (n_nodes,)): time-averaged per-node leading-vector norm.
    """
    u0 = config.initial_state.dynamics          # (n_states, n_nodes)
    n_states, n_nodes = u0.shape
    D = u0.size
    if k is None:
        k = D

    def _flow(u_flat):
        u = u_flat.reshape(n_states, n_nodes)
        cfg = Bunch({**config, "initial_state": Bunch({**config.initial_state, "dynamics": u})})
        return solve_fn(cfg).ys[-1].reshape(-1)

    def _step(carry, _):
        u_flat, Q, log_sum, xi_sum = carry
        new_u, jvp = jax.linearize(_flow, u_flat)        # linearise the flow at u
        tangents = jax.vmap(jvp)(Q.T)                    # propagate Q columns -> (k, D)
        Q_new, R = jnp.linalg.qr(tangents.T)             # renormalise -> (D, k), (k, k)
        log_sum = log_sum + jnp.log(jnp.abs(jnp.diag(R)))
        lead = Q_new[:, 0].reshape(n_states, n_nodes)    # unit leading tangent vector
        xi_sum = xi_sum + jnp.sqrt(jnp.sum(lead ** 2, axis=0))   # per-node norm
        return (new_u, Q_new, log_sum, xi_sum), None

    init = (u0.reshape(-1), jnp.eye(D)[:, :k], jnp.zeros(k), jnp.zeros(n_nodes))
    (_, _, log_sum, xi_sum), _ = jax.lax.scan(_step, init, None, length=n)
    exponents = jnp.sort(log_sum / (n * t))[::-1]
    return exponents, xi_sum / n
</%def>
