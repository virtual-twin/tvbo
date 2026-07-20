"""Resolve a ``Procedural`` GraphGenerator's typed DAG into backend-independent expressions.

A ``Procedural`` generator describes a network construction as an ordered DAG of *typed*
steps — a distance kernel, a stochastic connection mask, a Gaussian field, an axis
normalisation — instead of per-generator Python. This module turns that metadata into
SymPy expressions, which the printers in ``tvbo/codegen/code.py`` render natively for
every backend.

Two properties make it declarative rather than a lowering dialect:

**Nothing is built from expression strings.** Each step's options are schema *fields*, and
this module constructs the SymPy tree directly (``Function("pairwise_distance")(pos)``).
That is not a stylistic choice: SymPy's parser cannot represent keyword arguments at all,
silently collapses ``M[M != 0]`` to ``True``, and turns an unregistered head into implicit
multiplication (``eigvals(M)`` -> ``M*eigvals``) with no error. Building nodes directly
makes all three unreachable. The single exception is the ``equation`` step, whose ``rhs``
is author-written algebra — which is what the parser is actually for.

**The deterministic/stochastic split is inferred, not declared.** ``partition`` computes
which steps transitively depend on the generator's seed. A backend evaluates the
deterministic prefix once and traces only the stochastic suffix per realisation, so
sweeping N network realisations costs N x (the seeded tail), not N x (the whole
construction). The boundary is a property of the DAG, so every backend gets the same
answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import sympy as sp

from tvbo.parse.expression import parse_eq

# Sampler head per Distribution.name, and the order its parameters are passed.
# Mirrors the backend sampler table in dev/GenericProcedureEngine.md §2.3; the
# vocabulary is numpyro / Distributions.jl aligned.
# Each entry maps a family to its sampler head and its parameters IN CALL ORDER, each
# with the value of that family's standard form. A parameter a spec omits falls back to
# the standard form (Normal(0, 1), Uniform(0, 1), ...) rather than erroring, because that
# is what the family means without further qualification — and because omitting one is
# how these have always been written. Typos do NOT fall back: `_sampler` rejects any
# parameter name the family does not define, so `mena: 0.5` is an error rather than a
# silent mean of 0.
_SAMPLERS: Dict[str, Tuple[str, Tuple[Tuple[str, float], ...]]] = {
    "normal": ("sample_normal", (("mean", 0.0), ("std", 1.0))),
    # `Gaussian` is the same family under the name `distribution_pdf` already accepts;
    # registering it here keeps one spelling from resolving on one step type and failing
    # on another.
    "gaussian": ("sample_normal", (("mean", 0.0), ("std", 1.0))),
    "uniform": ("sample_uniform", (("lo", 0.0), ("hi", 1.0))),
    "lognormal": ("sample_lognormal", (("mu", 0.0), ("sigma", 1.0))),
    "beta": ("sample_beta", (("a", 1.0), ("b", 1.0))),
    "exponential": ("sample_exponential", (("scale", 1.0),)),
}

_COMPARISONS = {"le": sp.Le, "lt": sp.Lt, "ge": sp.Ge, "gt": sp.Gt}

# The symbol a step's seeded draw is keyed by. A backend binds it to whatever carries
# PRNG state there (a base seed for numpy, a jax PRNG key); the DAG only says *that* the
# step is seeded, never how the backend represents randomness.
#
# The name is underscore-prefixed and RESERVED. Seededness is detected by looking for this
# symbol in a step's expression, and the binding is written into the evaluation namespace,
# so an ordinary generator parameter sharing the name would both be silently overwritten by
# the seed (wrong values, no error) and make every step look seeded, disabling the
# deterministic-prefix hoisting. `build` rejects the collision rather than allowing either.
KEY = sp.Symbol("_prng_key")

# Number of nodes; bound by the layout so a step never hard-codes a size.
N_NODES = sp.Symbol("n_nodes")


def _host_env() -> Dict[str, Any]:
    """Operations available during eager materialisation that are NOT array primitives.

    Deliberately tiny, and deliberately not a parallel array vocabulary. Everything that
    *is* array algebra — sampling, reductions, linear algebra, normalisation — is defined
    once per backend in the printer tables and reached through them, so eager numpy and
    emitted JAX cannot drift. What remains here is the residue that no printer should
    ever emit:

    * ``load_matrix`` — resolves a Network IRI and reads a matrix from storage. I/O.

    Runtime *validation* is intentionally absent: a guard like
    ``require(preserve == 'binary_mask')`` is a constraint on a parameter's allowed
    values, which belongs in the schema (permissible values) where it is checked once and
    documented, not re-implemented as a procedure step carrying a string literal that no
    expression language can represent.
    """
    from tvbo.graph_generators.catalog import load_matrix

    return {"load_matrix": load_matrix}


class ProceduralError(ValueError):
    """A Procedural generator's DAG is malformed (unknown step type, bad reference, ...)."""


def _ref(name: Any, env: Mapping[str, sp.Expr], step: str, field: str) -> sp.Expr:
    """Resolve a field that names a previously-defined intermediate."""
    key = str(name)
    if key not in env:
        known = ", ".join(sorted(env)) or "<none>"
        raise ProceduralError(
            f"step {step!r}: field {field!r} references {key!r}, which is not a "
            f"previously-defined intermediate (available: {known}). Steps are ordered; "
            f"a step may only reference names defined before it."
        )
    return env[key]


def _number(value: Any, step: str, field: str) -> sp.Expr:
    """Coerce a scalar field to a SymPy number, accepting `inf` for an open bound."""
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("inf", "+inf", "infinity"):
            return sp.oo
        if token in ("-inf", "-infinity"):
            return -sp.oo
    try:
        return sp.sympify(value)
    except (TypeError, ValueError) as exc:
        raise ProceduralError(
            f"step {step!r}: field {field!r} must be a number, got {value!r}."
        ) from exc


def _dist_name(spec: Any) -> str:
    return str(_get(spec, "name", "") or "").lower()


def _get(spec: Any, field: str, default: Any = None) -> Any:
    """Read a field from a Distribution, tolerating a datamodel object or a plain dict."""
    if isinstance(spec, Mapping):
        return spec.get(field, default)
    return getattr(spec, field, default)


def _steps(spec: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """``Procedure.steps`` as ordered ``(name, fields)`` pairs.

    ``steps`` is a keyed collection: the key is the step's name, and insertion order is
    evaluation order. A sequence is not an accepted serialisation — the schema rejects it
    — so it is reported here by name rather than surfacing later as an ``AttributeError``
    from inside a step builder.
    """
    steps = spec.get("steps")
    if steps is None:
        return []
    if not isinstance(steps, Mapping):
        raise ProceduralError(
            f"`steps` must be a mapping keyed by step name, got {type(steps).__name__}."
        )
    return list(steps.items())


def _dist_param(spec: Any, name: str) -> Any:
    """A Distribution parameter's value, unwrapping the Parameter object if present."""
    params = _get(spec, "parameters") or {}
    entry = params.get(name) if isinstance(params, Mapping) else getattr(params, name, None)
    if entry is None:
        return None
    value = _get(entry, "value", entry)
    return entry if value is None else value


def _sampler(
    spec: Any, step: str, shape: Sequence[sp.Expr], substream: Any = 0
) -> sp.Expr:
    """Build a draw: ``sample_<d>(key, substream, *params, *shape)``.

    ``substream`` is written in the spec (``seed_offset``) rather than inferred from step
    order, so the sub-stream a step draws from is a stated property of the generator that
    every backend derives identically — and reordering or inserting steps cannot silently
    change an existing generator's output.
    """
    name = _dist_name(spec)
    if name not in _SAMPLERS:
        raise ProceduralError(
            f"step {step!r}: distribution {_get(spec, 'name')!r} has no sampler "
            f"(known: {', '.join(sorted(_SAMPLERS))})."
        )
    head, defaults = _SAMPLERS[name]
    known = {pname for pname, _ in defaults}
    declared = _get(spec, "parameters") or {}
    unknown = sorted(set(declared) - known) if isinstance(declared, Mapping) else []
    if unknown:
        # A name the family does not define is a typo, not a preference: without this the
        # value is dropped and the parameter it was meant for silently takes its standard
        # form, so the draw comes from a different distribution than the spec describes.
        raise ProceduralError(
            f"step {step!r}: distribution {name!r} has no parameter(s) "
            f"{', '.join(repr(u) for u in unknown)} (defines: {', '.join(sorted(known))})."
        )
    params = []
    for pname, fallback in defaults:
        value = _dist_param(spec, pname)
        params.append(_number(fallback if value is None else value, step, pname))
    return sp.Function(head)(KEY, sp.Integer(int(substream)), *params, *shape)


# --------------------------------------------------------------------------- #
# Step builders — one per `type`, each (fields, env, ctx) -> SymPy expression.  #
# --------------------------------------------------------------------------- #
def _step_equation(fields, env, ctx, step):
    equation = _get(fields, "equation")
    rhs = _get(equation, "rhs") if equation is not None else None
    if rhs is None:
        raise ProceduralError(f"step {step!r}: an `equation` step requires `equation.rhs`.")
    # The one place a string is parsed: author-written algebra over named intermediates.
    return parse_eq(str(rhs), parameters=list(env) + list(ctx["parameters"]))


def _step_pairwise_distance(fields, env, ctx, step):
    pos = _ref(_get(fields, "of") or "layout", env, step, "of")
    expr = sp.Function("pairwise_distance")(pos)
    diagonal = _get(fields, "diagonal")
    if diagonal is not None:
        expr = sp.Function("fill_diagonal")(expr, _number(diagonal, step, "diagonal"))
    return expr


def _step_distribution_pdf(fields, env, ctx, step):
    """Evaluate a Distribution's density at the positions named by ``of``.

    A vector ``mean`` with a scalar ``cov`` denotes the isotropic multivariate form of
    the named family — the same distribution, evaluated over coordinates rather than a
    scalar, which is what a spatial field is.
    """
    pos = _ref(_get(fields, "of") or "layout", env, step, "of")
    dist = _get(fields, "distribution")
    if dist is None:
        raise ProceduralError(f"step {step!r}: `distribution` is required.")
    if _dist_name(dist) not in ("normal", "gaussian"):
        raise ProceduralError(
            f"step {step!r}: distribution_pdf is defined for Normal, got "
            f"{_get(dist, 'name')!r}."
        )
    mean = _dist_param(dist, "mean")
    cov = _dist_param(dist, "cov")
    if mean is None or cov is None:
        raise ProceduralError(
            f"step {step!r}: a Normal field requires `mean` and `cov` parameters."
        )
    if not isinstance(mean, (list, tuple)):
        raise ProceduralError(f"step {step!r}: `mean` must be a coordinate list.")
    mean_t = sp.Tuple(*[_number(m, step, "mean") for m in mean])
    return sp.Function("gaussian_pdf")(pos, mean_t, _number(cov, step, "cov"))


def _step_normalize(fields, env, ctx, step):
    of = _ref(_get(fields, "of"), env, step, "of")
    return sp.Function("normalize")(of, sp.Integer(int(_get(fields, "axis") or 0)))


def _step_minmax_rescale(fields, env, ctx, step):
    of = _ref(_get(fields, "of"), env, step, "of")
    target = _get(fields, "target_range")
    if target is None:
        raise ProceduralError(f"step {step!r}: `target_range` is required.")
    lo, hi = _get(target, "lo"), _get(target, "hi")
    if lo is None or hi is None:
        raise ProceduralError(f"step {step!r}: `target_range` needs both `lo` and `hi`.")
    return sp.Function("minmax_rescale")(
        of, _number(lo, step, "target_range.lo"), _number(hi, step, "target_range.hi")
    )


def _step_sample(fields, env, ctx, step):
    """An [n_nodes, n_nodes] draw from ``distribution`` — a random weight matrix.

    Distinct from ``stochastic_mask``, which compares a draw against something and yields
    a boolean. This yields the draw itself (the substrate of a random reservoir).

    ``of`` names a Distribution-valued generator parameter. A ``sample`` step consumes no
    array, so ``of`` carries its only input here — which is what lets a curated generator
    expose its randomness as a *choice* (RandomReservoir's ``weight_distribution``) while
    the inline ``distribution`` states the family it defaults to.
    """
    of = _get(fields, "of")
    inline = _get(fields, "distribution")
    dist = inline
    if of is not None:
        name = str(of)
        if name not in ctx["parameters"] and name in env:
            # On every other step type `of` names an intermediate, so reaching for one
            # here is the natural mistake. Falling back to the default distribution would
            # build a plausible network from the wrong family without saying so.
            raise ProceduralError(
                f"step {step!r}: `of` names the intermediate {name!r}, but a `sample` "
                f"step draws from a Distribution-valued parameter, not from an array."
            )
        supplied = ctx["parameters"].get(name)
        # An unset optional parameter is absent or None, and a *declared* one (the curated
        # entry's `{datatype: Distribution, ...}` interface block) carries no family name.
        # Neither is a distribution to draw from, so both fall through to the default.
        if supplied is not None and _dist_name(supplied):
            dist = supplied
        elif inline is None:
            raise ProceduralError(
                f"step {step!r}: `of` names parameter {name!r}, which is not set to a "
                f"Distribution, and the step declares no inline `distribution` to fall "
                f"back to."
            )
    if dist is None:
        raise ProceduralError(f"step {step!r}: `distribution` is required.")
    return _sampler(dist, step, (N_NODES, N_NODES), _get(fields, "seed_offset") or 0)


def _step_stochastic_mask(fields, env, ctx, step):
    """A distance/similarity threshold against a random draw — a connection mask.

    Written as fields (`of`, `comparison`, `distribution`) rather than a condition string,
    so nothing about the comparison or the draw has to survive a parser.
    """
    of = _ref(_get(fields, "of"), env, step, "of")
    comparison = str(_get(fields, "comparison") or "le").lower()
    if comparison not in _COMPARISONS:
        raise ProceduralError(
            f"step {step!r}: comparison {comparison!r} unknown "
            f"(known: {', '.join(sorted(_COMPARISONS))})."
        )
    dist = _get(fields, "distribution")
    if dist is None:
        raise ProceduralError(f"step {step!r}: `distribution` is required.")
    draw = _sampler(dist, step, (N_NODES, N_NODES), _get(fields, "seed_offset") or 0)
    return _COMPARISONS[comparison](of, draw)


_STEP_BUILDERS = {
    "equation": _step_equation,
    "pairwise_distance": _step_pairwise_distance,
    "distribution_pdf": _step_distribution_pdf,
    "normalize": _step_normalize,
    "minmax_rescale": _step_minmax_rescale,
    "sample": _step_sample,
    "stochastic_mask": _step_stochastic_mask,
}

# Step types whose result depends on the generator's PRNG state. A step is also seeded
# when it references a seeded intermediate; `partition` takes that closure.
_SEEDED_TYPES = {"stochastic_mask"}


def build(spec: Mapping[str, Any]) -> List[Tuple[str, sp.Expr]]:
    """Resolve a ``Procedure``'s ``steps`` DAG to ordered ``(name, expression)`` pairs.

    ``spec`` is a ``Procedure`` block — ``steps`` (an ordered mapping of typed
    ``ProcedureStep``s, keyed by name) and ``output`` — plus the generator's
    ``parameters``. Node positions are not a special slot: a layout is an ordinary step.
    """
    parameters = dict(spec.get("parameters") or {})
    ctx = {"parameters": parameters}
    if KEY.name in parameters:
        raise ProceduralError(
            f"parameter {KEY.name!r} is reserved: it names the generator's PRNG state, "
            f"which the resolver binds itself. Rename the parameter."
        )
    env: Dict[str, sp.Expr] = {name: sp.Symbol(name) for name in parameters}
    # `n_nodes` comes from Network.number_of_nodes, so it is bound for every generator.
    # Node POSITIONS are not pre-bound: a layout is an ordinary step (produced by e.g.
    # `grid_positions`), so referencing one that was never defined must fail like any
    # other dangling reference rather than resolve to a free symbol.
    env["n_nodes"] = N_NODES

    resolved: List[Tuple[str, sp.Expr]] = []
    for name, fields in _steps(spec):
        step_type = str(_get(fields, "type") or "equation")
        builder = _STEP_BUILDERS.get(step_type)
        if builder is None:
            raise ProceduralError(
                f"step {name!r}: unknown type {step_type!r} "
                f"(known: {', '.join(sorted(_STEP_BUILDERS))})."
            )
        expr = builder(fields, env, ctx, name)
        resolved.append((name, expr))
        env[name] = sp.Symbol(name)
    return resolved


def seeded_steps(spec: Mapping[str, Any], resolved=None) -> set:
    """Names of steps that depend on the generator's PRNG state, transitively.

    Seededness is read off the RESOLVED EXPRESSION, not the declared step type: a step is
    seeded if its expression draws randomness anywhere (it carries the PRNG symbol) or if
    it references a seeded intermediate. Trusting the declared type instead would miss an
    `equation` step that calls a sampler head directly — and a missed draw is the worst
    possible failure here, because the step would be hoisted into the deterministic prefix
    and every "independent" realisation would silently share one draw.
    """
    sampler_heads = {head for head, _ in _SAMPLERS.values()}
    seeded: set = set()
    for name, expr in (resolved if resolved is not None else build(spec)):
        # Primary signal: does the expression actually DRAW? Keying off the presence of a
        # sampler head rather than the PRNG symbol's name means an `equation` step that
        # calls a sampler is caught however its key argument is spelled — a step wrongly
        # judged deterministic gets hoisted, and every "independent" realisation then
        # silently shares one draw.
        draws = any(f.func.__name__ in sampler_heads for f in expr.atoms(sp.Function))
        symbols = {str(sym) for sym in expr.free_symbols}
        if draws or KEY.name in symbols or (symbols & seeded):
            seeded.add(name)
    return seeded


def materialize(
    spec: Mapping[str, Any],
    params: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a Procedure's DAG eagerly, in numpy, and return its ``output`` values.

    The evaluation goes through the SAME primitive tables the other backends emit from:
    each step is rendered by the numpy printer and evaluated. There is deliberately no
    second numpy implementation of `sample`, `eigvals`, `normalize` and friends — one
    definition per backend is what keeps eager materialisation and emitted JAX from
    drifting apart, which a parallel evaluator cannot guarantee.

    Only genuinely non-expression operations stay host-side (see ``_host_env``): resolving
    a Network IRI is I/O, not array algebra, and no printer should ever emit it.
    """
    import numpy as np

    from tvbo.codegen.code import render_expression

    # Supplied values win over the spec's own, and the merge happens BEFORE `build` so a
    # step that selects its distribution by parameter name (`of`) sees the override. The
    # resolver and the evaluator must read one parameter set, not two.
    merged = {**(spec.get("parameters") or {}), **(params or {})}
    spec = {**spec, "parameters": merged}

    host = _host_env()
    env: Dict[str, Any] = dict(host)
    env["np"] = np
    env.update(merged)
    # A generator's PRNG state. numpy's Generator is the host analogue of the pure key a
    # traced backend threads; the DAG only ever says *that* a step is seeded.
    env[KEY.name] = 0 if seed is None else int(seed)
    if env.get("n_nodes") is None:
        raise ProceduralError(
            "materialize: `n_nodes` must be supplied (from Network.number_of_nodes)."
        )

    symbol_names: List[str] = [k for k in env if k not in host and k != "np"]
    for name, expr in build(spec):
        symbol_names.append(name)
        source = render_expression(expr, format="numpy")
        try:
            env[name] = eval(source, {"__builtins__": {}}, env)  # noqa: S307 — rendered by us
        except Exception as exc:  # noqa: BLE001
            raise ProceduralError(
                f"step {name!r} failed to evaluate: {type(exc).__name__}: {exc}\n"
                f"  rendered: {source}"
            ) from exc

    outputs: Dict[str, Any] = {}
    for key, entry in (spec.get("output") or {}).items():
        rhs = _get(_get(entry, "equation"), "rhs") if entry is not None else None
        if rhs is None:
            raise ProceduralError(f"output {key!r}: requires `equation.rhs`.")
        expr = parse_eq(str(rhs), parameters=symbol_names)
        source = render_expression(expr, format="numpy")
        outputs[key] = eval(source, {"__builtins__": {}}, env)  # noqa: S307
    if "weights" in outputs:
        outputs["weights"] = np.asarray(outputs["weights"], dtype=float)
    for key, value in outputs.items():
        _reject_nan(value, key)
    return outputs


def _reject_nan(value: Any, key: str) -> None:
    """A generator must not emit a NaN connectome.

    Degenerate constructions divide by a quantity the DAG itself produced — a spectral
    radius of 0 when every sampled edge was masked out — and the result is a NaN adjacency
    matrix that simulates happily and yields NaN trajectories far from here. Checking the
    outputs catches that for every generator, where a per-generator guard step could only
    cover one construction at a time.

    NaN specifically, not every non-finite value: ``pairwise_distance`` documents
    ``diagonal: inf`` as the way to suppress self-connections, so an infinity in a
    distance-like output is a declared intent rather than a failed divide. A degenerate
    divide still surfaces here, because scaling a matrix that has any zero entry by an
    infinite factor produces NaN at those entries.
    """
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind not in "fc" or not np.isnan(array).any():
        return
    raise ProceduralError(
        f"output {key!r} contains {int(np.isnan(array).sum())} NaN of {array.size} values. "
        f"A step divided by a degenerate quantity — for a sparse random construction this "
        f"is typically a spectral radius or normalisation sum of 0, meaning the draw "
        f"produced no surviving edges; increase the connection density or the number of "
        f"nodes."
    )


def draw(
    distribution: Any,
    shape: Sequence[int],
    seed: Optional[int] = None,
    substream: int = 0,
) -> Any:
    """Sample ``shape`` values from ``distribution``, through the same printer sampler.

    A construction that needs one array of draws rather than a whole DAG — a per-unit
    downward projection, say — still goes through the printer table, so it cannot drift
    from what a `sample` step produces for the same distribution and seed.
    """
    from tvbo.codegen.code import render_expression

    dims = tuple(sp.Integer(int(s)) for s in shape)
    expr = _sampler(distribution, "draw", dims, substream)
    source = render_expression(expr, format="numpy")
    import numpy as np

    return eval(  # noqa: S307 — rendered by the numpy printer
        source, {"__builtins__": {}}, {"np": np, KEY.name: 0 if seed is None else int(seed)}
    )


def partition(spec: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Split the DAG into (deterministic prefix, stochastic suffix), in DAG order.

    The prefix is evaluated once; the suffix is what a backend re-evaluates per network
    realisation. For a generator whose only randomness is a connection mask, the suffix is
    a handful of array ops while the geometry stays a constant.
    """
    # Resolve once and share: `build` re-parses every `equation` rhs through SymPy, and
    # partition/seeded_steps/materialize would otherwise each pay that cost separately.
    resolved = build(spec)
    seeded = seeded_steps(spec, resolved=resolved)
    names = [name for name, _ in resolved]
    return ([n for n in names if n not in seeded], [n for n in names if n in seeded])
