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


def get_distribution_seed(model, default=42):
    """Find the first explicit seed from any distribution, or return default."""
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
# Database-driven dispatch: each GraphGenerator type lives as a YAML file in
# tvbo/database/graph_generators/<Type>.yaml, with its per-backend bindings
# declared there. No hard-coded Python tables — adding a new generator is a
# new YAML file, possibly plus a Python materialiser in tvbo.graph_generators.


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


# Cache of generator-type → bindings dict, loaded lazily from the database
# to avoid YAML parse overhead on every codegen invocation.
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
            f"Unknown graph generator type {gtype!r}. Add a YAML file under "
            f"tvbo/database/graph_generators/ to register it."
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
