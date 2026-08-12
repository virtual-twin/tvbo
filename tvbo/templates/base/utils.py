# -*- coding: utf-8 -*-
"""
Utilities for base templates.

Extracts Python logic from Mako templates for cleaner, testable code.
"""


def get_coupling_terms(model):
    """Extract coupling inputs from model, separating global from local.

    Returns:
        tuple: (all_terms, global_terms, has_local_coupling)
    """
    # Prefer coupling_inputs; fall back to coupling_terms for backward compat
    ci = getattr(model, "coupling_inputs", None)
    ct = getattr(model, "coupling_terms", None)
    if ci:
        all_terms = list(ci.keys())
    elif ct:
        all_terms = list(ct.keys())
    else:
        all_terms = []
    global_terms = [t for t in all_terms if t != "local_coupling"]
    has_local = "local_coupling" in all_terms
    return all_terms, global_terms, has_local


def gathered_states(model, coupling=None):
    """Names of the state variables transmitted to connected nodes, in gather order.

    The coupling's ``incoming_states`` is the authority: the schema defines
    ``coupling_variable`` as the model's default, "used when omitted", and says the coupling "may override this via its incoming_states attribute". JansenRit1995 is
    the case that needs the override — it marks only ``y4`` a coupling variable, yet the inter-column coupling transmits the PSP difference ``y1 - y2``.

    The delay history, the ``x_j`` gather and the history write-back must all agree on this one order, so all three read it from here. Disagreeing would not raise: a
    delayed source would silently be read from another state's row.

    Args:
        model: The :class:`~tvbo.classes.dynamics.Dynamics` declaring the states.
        coupling: The coupling whose ``incoming_states`` overrides the default, or
            ``None`` to use the model's ``coupling_variable`` states.

    Returns:
        list[str]: State-variable names, ordered as the gathered rows are.
    """
    names = list(model.state_variables.keys())
    declared = getattr(coupling, "incoming_states", None) or [] if coupling is not None else []
    if isinstance(declared, str):
        declared = [declared]
    declared = [s if isinstance(s, str) else getattr(s, "name", str(s)) for s in declared]
    declared = [s for s in dict.fromkeys(declared) if s in names]
    if declared:
        return declared
    return [n for n, sv in model.state_variables.items() if sv.coupling_variable]


def gathered_state_indices(model, coupling=None):
    """Positions in ``model.state_variables`` of :func:`gathered_states`, same order."""
    order = list(model.state_variables.keys())
    return [order.index(name) for name in gathered_states(model, coupling)]


def referenced(names, *expressions):
    """The subset of *names* that any of *expressions* refers to, order preserved.

    Used to unpack only what a generated function body reads. A coupling's ``post()`` and ``pre()`` share one parameter list but each uses part of it, so unpacking the
    whole list leaves the rest dead — ``post()`` returning ``G * gx`` was still binding
    ``cmax``, ``cmin``, ``midpoint`` and ``r``.

    Matching is on word boundaries, so ``r`` is not found inside ``r_max``.

    Args:
        names: Candidate names, in the order they should be emitted.
        *expressions: Expression sources to search; ``None`` entries are skipped.
    """
    import re

    text = "\n".join(str(e) for e in expressions if e is not None)
    return [n for n in names if re.search(rf"\b{re.escape(str(n))}\b", text)]


def time_series_inputs(candidates, body):
    """Which of *candidates* the function *body* reads as its incoming samples.

    A pipeline function names the samples either ``X`` (the generator's own convention) or by an argument the spec declared without a default — ``data`` in a body written
    as ``jnp.mean(data[...])``. Both must resolve to ``ts.data``; binding only ``X`` left the spec's name pointing at an unfilled positional parameter, so calling the
    function raised :class:`TypeError`.
    """
    return referenced(list(dict.fromkeys(candidates)), body)


def retime(body, inputs):
    """Rewrite *body* to read the time axis, substituting ``t_<name>`` for each input.

    ``apply_on_dimension: time`` applies the same expression to the time vector, so the sample symbols swap for their time counterparts.

    An attribute of the same name is left alone. ``\\b`` matches immediately after a dot, so a plain word-boundary substitution rewrote ``ts.data`` into ``ts.t_data``
    and the generated function raised ``AttributeError`` on the TimeSeries. The lookbehind requires the name to start a reference, not continue one.
    """
    import re

    out = body
    for name in inputs:
        out = re.sub(rf"(?<![\w.]){re.escape(name)}\b", f"t_{name}", out)
    return out


def get_source_code(func):
    """Backend-ready source text a function supplies directly, or ``None``.

    A ``Function`` states its body either as an ``equation`` (symbolic, printed per backend) or as ``source_code`` (already written in the target language). A generator
    that reads only the first emits nothing for the second — which is how the JAX observation for ``kuramoto_order`` came out as ``data = 0.0``.
    """
    if getattr(func, "source_code", None):
        return func.source_code
    equation = getattr(func, "equation", None)
    if equation is not None and getattr(equation, "pycode", None):
        return equation.pycode
    return None


def model_expressions(model):
    """Every right-hand side a generated dfun body will contain, as one searchable string.

    Pair with :func:`referenced` to unpack only the parameters a model actually uses. The four groups must all be included: a parameter can reach the derivatives indirectly,
    through a derived parameter (``C1 = C``), a derived variable, or a function body (``Sigm`` reads ``e0``, ``r`` and ``v0`` and nothing else does), and dropping its
    unpack on the strength of the state equations alone would emit an unbound name.
    """
    parts = []
    for group in ("state_variables", "derived_variables", "derived_parameters", "functions"):
        for obj in (getattr(model, group, None) or {}).values():
            rhs = getattr(getattr(obj, "equation", None), "rhs", None)
            if rhs is not None:
                parts.append(str(rhs))
    return "\n".join(parts)


def coupling_bindings(model, coupling, incoming=(), local=()):
    """Resolve which names a coupling's ``pre``/``post`` bodies must bind, and to what.

    A coupling expression may name a source state three ways, and they are distinct references, not spellings of one:

    ``u``
        the bare name — the gathered source row, ``x_j[k]``;
    ``u_j``
        the same row written explicitly, used when the target's value also appears;
    ``u_i``
        the *target* node's own current value, ``current_state[i]``.

    A difference coupling (``u_j - u_i``) needs the last two together, which is why the bare name cannot stand in for either. Word-boundary matching keeps the three apart:
    ``\\bu\\b`` does not match the ``u`` inside ``u_j``, so a pre-expression written only in subscripts binds no unread bare name. This mirrors ``state_aliases_j`` /
    ``state_aliases_i`` in :func:`tvbo.templates.tvboptim.utils.resolve_coupling_spec`.

    Args:
        model: The dynamics declaring the state variables.
        coupling: The coupling whose expressions are being emitted.
        incoming: Declared source-state names.
        local: Declared target-local state names.

    Returns:
        dict: ``cvar_names`` (gather order), ``cvar_index``, ``sv_index``, ``bare``
        (states read by bare name), ``pre_j``/``pre_i``/``post_i`` (alias, index) pairs.

    Raises:
        ValueError: a local state the model does not declare, or a source state that is
            read but never transmitted — both of which would emit an unbound name.
    """
    import re

    states = list(model.state_variables.keys())
    sv_index = {name: i for i, name in enumerate(states)}
    cvar_names = gathered_states(model, coupling)
    cvar_index = {name: i for i, name in enumerate(cvar_names)}
    vec_states = list(dict.fromkeys(list(incoming) + list(local)))

    unknown = [s for s in local if s not in sv_index]
    if unknown:
        raise ValueError(
            f"coupling `local_states` names {unknown}, which the model does not declare as "
            f"state variables ({states}); a local state is the target node's own state, so "
            "it must be one of them."
        )

    pre_rhs = str(coupling.pre_expression.rhs) if coupling.pre_expression else ""
    post_rhs = str(coupling.post_expression.rhs) if coupling.post_expression else ""

    def bare(name, text):
        return re.search(rf"\b{re.escape(name)}\b", text) is not None

    ungathered = [s for s in vec_states if s not in cvar_index and s not in local and bare(s, pre_rhs)]
    if ungathered:
        raise ValueError(
            f"coupling `pre_expression` reads {ungathered} as a source state, but only "
            f"{cvar_names} are transmitted, so there is no gathered row for them. List them "
            "in the coupling's `incoming_states` (or mark them `coupling_variable: true` on "
            f"the model), or read the target's own value as {[s + '_i' for s in ungathered]}."
        )

    return {
        "cvar_names": cvar_names,
        "cvar_index": cvar_index,
        "sv_index": sv_index,
        "vec_states": vec_states,
        "bare": [s for s in vec_states if s in cvar_index and bare(s, pre_rhs)],
        "pre_j": [(f"{s}_j", cvar_index[s]) for s in vec_states if f"{s}_j" in pre_rhs and s in cvar_index],
        "pre_i": [(f"{s}_i", sv_index[s]) for s in vec_states if f"{s}_i" in pre_rhs and s in sv_index],
        "post_i": [(f"{s}_i", sv_index[s]) for s in vec_states if f"{s}_i" in post_rhs and s in sv_index],
    }


def get_func_name(model, override=None):
    """Get function name from model, with optional override."""
    if override:
        return override
    if hasattr(model, "name") and model.name:
        return model.name.replace(" ", "").replace("-", "")
    return "dfun"


def get_func_args(f):
    """Extract argument names from a function object.

    Handles both dict-like (f.arguments.values()) and list-like arguments.
    """
    args = f.arguments
    if hasattr(args, "values"):
        return [arg.name if hasattr(arg, "name") else str(arg) for arg in args.values()]
    return [arg.name if hasattr(arg, "name") else str(arg) for arg in args]


def np_module(fmt):
    """Get numpy module name for format."""
    return "jnp" if fmt == "jax" else "np"


# Special functions that require scipy.special (numpy) or jax.scipy.special (jax)
SCIPY_SPECIAL_FUNCTIONS = {"erfc", "erf", "gamma", "gammaln", "bessel", "beta"}


def needs_scipy_special(model, fmt):
    """Check if model equations use scipy.special functions.

    Renders derived variables and state equations to detect scipy.special usage.
    Returns True if any equation contains scipy.special (for numpy) or jsp.special (for jax).
    """
    search_str = "scipy.special" if fmt in ("numpy", "scipy") else "jsp.special"

    # Check derived variables
    for dv in (model.derived_variables or {}).values():
        code = model.render_equation(dv, format=fmt)
        if search_str in code:
            return True

    # Check state variable equations
    for sv in model.state_variables.values():
        code = model.render_equation(sv, format=fmt)
        if search_str in code:
            return True

    return False


# ── Distribution utilities ───────────────────────────────────────────────────
# Reusable across all backends (Julia/NetworkDynamics, JAX, NumPy, PyRates, …)


def collect_sv_distributions(model):
    """Collect state variables that have a distribution for random ICs.

    Returns list of (index, StateVariable, Distribution).
    """
    result = []
    for i, sv in enumerate(model.state_variables.values()):
        d = getattr(sv, "distribution", None)
        if d and getattr(d, "domain", None):
            result.append((i, sv, d))
    return result


def collect_param_distributions(model):
    """Collect parameters that have a distribution for heterogeneous sampling.

    Returns list of (param_name, Parameter, Distribution).
    """
    result = []
    for p_name, p in (model.parameters or {}).items():
        d = getattr(p, "distribution", None)
        if d and getattr(d, "domain", None):
            result.append((p_name, p, d))
    return result


def has_distributions(model):
    """Check if model has any distributions (SV or parameter)."""
    return bool(collect_sv_distributions(model) or collect_param_distributions(model))


def get_distribution_seed(model, default=0):
    """Find the first explicit distribution `seed`, or *default* (the experiment-global seed).

    Mirrors the seed-resolution rule: a distribution's own `seed` wins; absent one, the caller passes ``execution.random_seed`` (default 0) so the draw inherits the experiment-global seed.
    """
    for _, _, d in collect_sv_distributions(model) + collect_param_distributions(model):
        s = getattr(d, "seed", None)
        if s is not None:
            return int(s)
    return default


def _dist_params(dist):
    """Extract mean/std from distribution parameters dict."""
    params = getattr(dist, "parameters", None) or {}
    mu = sigma = None
    for p in params.values() if hasattr(params, "values") else params:
        name = str(getattr(p, "name", ""))
        val = getattr(p, "value", None)
        if name == "mean" and val is not None:
            mu = float(val)
        elif name == "std" and val is not None:
            sigma = float(val)
    return mu, sigma


def sample_expression(dist, backend="julia"):
    """Generate a sampling expression string for a Distribution.

    Args:
        dist: Distribution object with .name, .domain, .parameters
        backend: 'julia', 'numpy', or 'jax'

    Returns:
        str: code expression that samples one value from the distribution.
    """
    name = str(dist.name) if dist.name else "Uniform"
    lo = dist.domain.lo
    hi = dist.domain.hi

    if backend == "julia":
        return _sample_julia(name, lo, hi, dist)
    elif backend in ("numpy", "jax"):
        mod = "jnp" if backend == "jax" else "np"
        return _sample_numpy(name, lo, hi, dist, mod)
    else:
        return _sample_numpy(name, lo, hi, dist, "np")


def _sample_julia(name, lo, hi, dist):
    if name == "Uniform":
        return f"{lo} .+ ({hi} - {lo}) .* rand(rng)"
    elif name == "Gaussian":
        mu, sigma = _dist_params(dist)
        if mu is None:
            mu = f"({lo} + {hi}) / 2"
        if sigma is None:
            sigma = f"({hi} - {lo}) / 6"
        return f"{mu} .+ {sigma} .* randn(rng)"
    return f"{lo} .+ ({hi} - {lo}) .* rand(rng)"


def _sample_numpy(name, lo, hi, dist, mod="np"):
    if name == "Uniform":
        return f"rng.uniform({lo}, {hi}, size=n_nodes)"
    elif name == "Gaussian":
        mu, sigma = _dist_params(dist)
        if mu is None:
            mu = f"({lo} + {hi}) / 2"
        if sigma is None:
            sigma = f"({hi} - {lo}) / 6"
        return f"rng.normal({mu}, {sigma}, size=n_nodes)"
    return f"rng.uniform({lo}, {hi}, size=n_nodes)"


# ── Graph generator utilities ────────────────────────────────────────────────
# Database-driven dispatch: each GraphGenerator type lives as a YAML file in tvbo/database/graph_generators/<Type>.yaml, with its per-backend bindings declared there. No hard-coded Python tables — adding a new generator is a new YAML file, possibly plus a Python materialiser in tvbo.graph_generators.


def _get_gen_params(gen):
    """Extract parameter values from a GraphGenerator as a dict."""
    params = getattr(gen, "parameters", None) or {}
    result = {}
    items = params.values() if hasattr(params, "values") else params
    for p in items:
        name = str(getattr(p, "name", ""))
        val = getattr(p, "value", None)
        if name and val is not None:
            try:
                result[name] = int(float(val)) if float(val) == int(float(val)) else float(val)
            except (ValueError, TypeError):
                result[name] = val
    return result


# Cache of generator-type → bindings dict, loaded lazily from the database to avoid YAML parse overhead on every codegen invocation.
_BINDINGS_CACHE: dict[str, dict] = {}


def _load_bindings(gtype: str) -> dict:
    """Load (and cache) the `bindings:` block of a generator's database entry.

    Returns a dict mapping backend name → {callable, args, ...}. Raises
    ValueError if the generator type is not found in the database.
    """
    if gtype in _BINDINGS_CACHE:
        return _BINDINGS_CACHE[gtype]

    import yaml

    from tvbo.data.registry import resolve

    try:
        path = resolve("GraphGenerator", gtype)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Unknown graph generator type {gtype!r}. Add a YAML file under tvbo/database/graph_generators/ to register it."
        ) from e

    with open(path) as f:
        entry = yaml.safe_load(f) or {}
    bindings = entry.get("bindings", {}) or {}
    _BINDINGS_CACHE[gtype] = bindings
    return bindings


def graph_generator_call(gen, n_nodes, backend="julia"):
    """Generate a graph constructor call string from a GraphGenerator object.

    Args:
        gen: GraphGenerator with .type, .parameters, .seed, .directed
        n_nodes: number of nodes (from Network.number_of_nodes)
        backend: 'julia', 'networkx', 'python', or any other key declared
                 in the generator's `bindings:` block in the database.

    Returns:
        str: constructor call, e.g. 'barabasi_albert(20, 4)'
    """
    gtype = str(gen.type)
    params = _get_gen_params(gen)
    params["n"] = n_nodes

    bindings = _load_bindings(gtype)
    binding = bindings.get(backend)
    if not binding:
        available = sorted(bindings.keys())
        raise ValueError(
            f"GraphGenerator {gtype!r} has no binding for backend {backend!r}. "
            f"Available backends in its database entry: {available}"
        )

    callable_name = binding.get("callable")
    arg_order = binding.get("args", [])
    if not callable_name:
        raise ValueError(f"Binding for {gtype}/{backend} is missing the `callable:` field.")

    args = [str(params[a]) for a in arg_order if a in params]
    return f"{callable_name}({', '.join(args)})"
