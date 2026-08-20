"""Resolve ``field_dynamics`` into the assembly plan the FEM template emits.

The PDE backend used to read only ``operators[].coefficient``, sum them into one scalar and assemble ``M + dt*D*K`` — so the ``equation.rhs`` an author wrote was decorative, and a reader could not tell which of the two descriptions ran. Everything here exists to make the declared equation *be* the executed operator.

A field system is lowered to the block form

    M du/dt = A u + M f(u, t)

with one block row per state variable. ``A`` collects every term that is LINEAR in the state variables (mass and stiffness blocks, assembled once); ``f`` collects whatever remains, evaluated at the current state each step. That split is what lets an implicit scheme run: the linear part is solved implicitly and is unconditionally stable, while a nonlinear reaction term stays explicit (IMEX). Callers get told which terms went where.

Recognised operators inside an RHS: ``laplacian(v)``, ``div(c*grad(v))`` and bare state variables. A coefficient may be a scalar parameter or a per-vertex field, which is what makes a spatially varying propagation scale expressible.
"""

from __future__ import annotations

import sympy as sp

_OPERATOR_NAMES = ("laplacian", "div", "grad", "lap")

_TOP_OPERATORS = ("laplacian", "lap", "div")
"""Operators that may head a term. ``grad`` only ever appears inside ``div``."""

_METHOD_ALIASES = {
    "implicit euler": "implicit-euler",
    "backward euler": "implicit-euler",
    "implicit-euler": "implicit-euler",
    "be": "implicit-euler",
    "crank-nicolson": "crank-nicolson",
    "crank nicolson": "crank-nicolson",
    "cn": "crank-nicolson",
    "trapezoidal": "crank-nicolson",
}


class FieldPlanError(ValueError):
    """A field system that cannot be lowered to the block form above."""


def _rhs_text(equation) -> str:
    """The right-hand side of an ``Equation``, however it is spelled."""
    for attr in ("righthandside", "rhs"):
        val = getattr(equation, attr, None)
        if val:
            return str(val)
    return str(equation or "").strip()


def _scalar(value):
    """A float from a Parameter, a raw number, or None when it is not scalar."""
    val = getattr(value, "value", value)
    if val is None or isinstance(val, (list, tuple)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parameter_table(field_dynamics) -> dict:
    """``{name: scalar}`` for every declared parameter that has a usable value.

    Non-scalar parameters (a per-vertex field) stay out of the table on purpose: they are resolved at run time against the mesh, and baking one into generated source would inline an array the size of the surface.
    """
    params = getattr(field_dynamics, "parameters", None) or {}
    items = params.values() if hasattr(params, "values") else params
    table = {}
    for p in items:
        name = getattr(p, "name", None)
        if not name:
            continue
        val = _scalar(p)
        if val is not None:
            table[str(name)] = val
    return table


def _field_parameters(field_dynamics) -> set:
    """Names of parameters that are per-vertex fields rather than scalars."""
    params = getattr(field_dynamics, "parameters", None) or {}
    items = params.values() if hasattr(params, "values") else params
    out = set()
    for p in items:
        name = getattr(p, "name", None)
        if not name:
            continue
        if _scalar(p) is None:
            out.add(str(name))
    return out


def _mesh_path(mesh, experiment) -> str:
    """The mesh file, resolved against the declaring spec's directory when relative.

    Same rule as every other file a spec names (``Network.data_file``, a sourced parameter):
    a recipe means the same mesh wherever it is loaded from, so the run does not depend on the working directory it was launched in. A ``format:`` prefix is preserved for the template to strip, and an unresolvable path is passed through so the failure names the file rather than an empty string.
    """
    from pathlib import Path

    from tvbo.data.param_io import _resolve_path

    raw = str(getattr(mesh, "mesh_file", None) or getattr(mesh, "dataLocation", None) or "")
    if not raw:
        return raw
    prefix, separator, tail = raw.partition(":")
    path, prefix = (tail, prefix + separator) if separator and prefix.isalpha() and len(prefix) > 1 else (raw, "")
    source_file = getattr(experiment, "_source_file", None)
    resolved = _resolve_path(path, Path(source_file).parent if source_file else None)
    return prefix + (str(resolved) if resolved is not None else path)


def _field_sources(field_dynamics, experiment) -> dict:
    """``{name: [path, key]}`` for every per-vertex coefficient that declares its own provenance.

    A field parameter carrying ``source:`` or ``producer:`` is materialised here, at codegen time, to the same content-addressed artifact every other backend reads. That is what lets the generated module assemble itself: without it ``build()`` has no value for the coefficient and the caller must hand one in, so a declared experiment could not run unattended. The array is never inlined — a cortical surface's mask is 32,492 values.
    """
    from pathlib import Path

    from tvbo.data import param_io

    params = getattr(field_dynamics, "parameters", None) or {}
    items = params.values() if hasattr(params, "values") else params
    source_file = getattr(experiment, "_source_file", None)
    source_dir = Path(source_file).parent if source_file else None

    out = {}
    for p in items:
        name = str(getattr(p, "name", "") or "")
        if not name or _scalar(p) is not None or not param_io.is_lazy(p):
            continue
        path, key = param_io.materialise(p, source_dir=source_dir, context=experiment)
        out[name] = [str(path), key]
    return out


def _event_table(experiment, parameters: dict) -> dict:
    """Stimulus events as ``{name: expression}``, with their parameters merged into ``parameters`` in place.

    A stimulus is a declared function of time, so it is substituted into the equation that names it rather than carried through as an opaque symbol. What the solver evaluates each step is then the RHS the recipe prints with the stimulus written out, and there is no second description of the drive that could disagree with the first.
    """
    events = getattr(experiment, "events", None) or {}
    items = events.values() if hasattr(events, "values") else events
    table = {}
    for event in items:
        name = str(getattr(event, "name", "") or "")
        text = _rhs_text(getattr(event, "equation", None))
        if not name or not text:
            continue
        own = _parameter_table(event)
        clashing = [k for k, v in own.items() if parameters.get(k, v) != v]
        if clashing:
            raise FieldPlanError(
                f"event {name!r} redefines parameter(s) {clashing} with different values; "
                f"rename them or declare them once on field_dynamics"
            )
        parameters.update(own)
        table[name] = _sympify(text, [], set(parameters) | {"t"})
    return table


def _sympify(text: str, variables, parameters):
    """Parse an RHS with operators kept as unevaluated functions."""
    local = {name: sp.Function(name) for name in _OPERATOR_NAMES}
    local.update({v: sp.Symbol(v) for v in variables})
    local.update({p: sp.Symbol(p) for p in parameters})
    try:
        return sp.sympify(text, locals=local)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise FieldPlanError(f"cannot parse field equation {text!r}: {exc}") from exc


def _operator_names(expr) -> set:
    """Differential-operator names appearing anywhere in *expr*."""
    return {getattr(a.func, "__name__", "") for a in expr.atoms(sp.Function)} & set(_OPERATOR_NAMES)


def _operand(expr, variables):
    """Classify one multiplicative term.

    Returns ``(kind, variable, coefficient_field)`` where kind is ``mass``, ``stiffness`` or ``None`` when the term carries no state variable at all (a pure source).
    """
    funcs = [a for a in expr.atoms(sp.Function) if getattr(a.func, "__name__", "") in _TOP_OPERATORS]
    if not funcs:
        syms = {str(s) for s in expr.free_symbols} & set(variables)
        if not syms:
            return None, None, None
        if len(syms) > 1:
            raise FieldPlanError(
                f"term {expr} mixes state variables {sorted(syms)}; the implicit solve "
                f"needs each term linear in exactly one variable"
            )
        return "mass", syms.pop(), None
    if len(funcs) > 1:
        raise FieldPlanError(f"term {expr} applies more than one differential operator")

    call = funcs[0]
    name = call.func.__name__
    arg = call.args[0]

    if name in ("laplacian", "lap"):
        inner, weight = arg, None
    elif name == "div":
        grads = [a for a in arg.atoms(sp.Function) if getattr(a.func, "__name__", "") == "grad"]
        if len(grads) != 1:
            raise FieldPlanError(f"div(...) must wrap exactly one grad(...): {expr}")
        inner = grads[0].args[0]
        weight = sp.simplify(arg / grads[0])
    else:
        raise FieldPlanError(f"operator {name}(...) is not supported outside div(grad(.))")

    syms = {str(s) for s in inner.free_symbols} & set(variables)
    if len(syms) != 1:
        raise FieldPlanError(f"operator argument {inner} must be exactly one state variable")
    return "stiffness", syms.pop(), (None if weight is None else str(weight))


def _linear_blocks(rhs, row, variables, parameters, field_params):
    """Split one equation's RHS into implicit blocks and an explicit remainder.

    A term the block form cannot take is deferred to the explicit remainder — that is the IMEX split, and a nonlinear reaction such as ``u*v`` belongs there. A term carrying a differential operator does not: nothing downstream can evaluate ``laplacian(...)``, so deferring it would surface as a ``NameError`` inside the generated solver rather than as a plan error here.
    """
    blocks, explicit = [], []
    for term in sp.Add.make_args(sp.expand(rhs)):
        try:
            kind, var, weight = _operand(term, variables)
        except FieldPlanError:
            if _operator_names(term):
                raise
            explicit.append(term)
            continue
        if kind is None:
            explicit.append(term)
            continue

        call = [a for a in term.atoms(sp.Function) if getattr(a.func, "__name__", "") in _TOP_OPERATORS]
        factor = sp.simplify(term / (call[0] if call else sp.Symbol(var)))
        if {str(s) for s in factor.free_symbols} & set(variables):
            explicit.append(term)  # coefficient depends on the state -> nonlinear
            continue
        # A per-vertex coefficient cannot collapse to a number; it rides as a named field.
        coeff_fields = sorted({str(s) for s in factor.free_symbols} & field_params)
        if len(coeff_fields) > 1:
            raise FieldPlanError(f"term {term} multiplies more than one field parameter")
        try:
            scalar = float(factor.subs(parameters)) if not coeff_fields else None
        except (TypeError, ValueError) as exc:
            raise FieldPlanError(f"coefficient {factor} of term {term} is not numeric: {exc}") from exc

        if kind == "stiffness" and coeff_fields and not weight:
            raise FieldPlanError(
                f"term {term} multiplies a Laplacian by the field {coeff_fields[0]!r}. "
                f"c(x)*laplacian(u) has no exact FEM assembly (its weak form needs grad(c) "
                f"and is not self-adjoint); write it in divergence form as "
                f"div({coeff_fields[0]}*grad({var})) instead"
            )
        blocks.append(
            {
                "row": row,
                "col": variables.index(var),
                "kind": kind,
                "scalar": scalar,
                "coefficient_field": coeff_fields[0] if coeff_fields else None,
                "weight_field": weight,
                "term": str(term),
            }
        )
    if explicit:
        stranded = sorted(_operator_names(sp.Add(*explicit)))
        if stranded:
            raise FieldPlanError(
                f"the remainder {sp.Add(*explicit)} still applies {stranded}, which the explicit "
                f"path cannot evaluate; only laplacian(v) and div(c*grad(v)) over a single state "
                f"variable are assembled"
            )
    return blocks, explicit


def field_assembly_plan(experiment) -> dict:
    """Everything the FEM template needs, resolved from the experiment datamodel.

    Raises:
        FieldPlanError: when the declared system cannot be lowered — an explicit failure
            in place of the old behaviour, which silently ignored the equation entirely.
    """
    fd = getattr(experiment, "field_dynamics", None)
    if fd is None:
        raise FieldPlanError("experiment has no field_dynamics")

    svars = list(getattr(fd, "state_variables", None) or [])
    if not svars:
        raise FieldPlanError("field_dynamics declares no state_variables")

    names = [str(getattr(v, "name", None) or getattr(v, "label", f"u{i}")) for i, v in enumerate(svars)]
    if len(set(names)) != len(names):
        raise FieldPlanError(f"duplicate field state-variable names: {names}")

    parameters = _parameter_table(fd)
    field_params = _field_parameters(fd)
    events = _event_table(experiment, parameters)

    blocks, explicit_rows, legacy = [], {}, False
    for row, var in enumerate(svars):
        text = _rhs_text(getattr(var, "equation", None))
        if not text:
            raise FieldPlanError(f"state variable {names[row]!r} declares no equation")
        rhs = _sympify(text, names, set(parameters) | field_params | set(events))
        if events:
            rhs = rhs.subs({sp.Symbol(name): expr for name, expr in events.items()})
        rows, explicit = _linear_blocks(rhs, row, names, parameters, field_params)
        blocks.extend(rows)
        if explicit:
            explicit_rows[names[row]] = str(sp.Add(*explicit))

    if not blocks:
        coeffs = []
        for op in getattr(fd, "operators", None) or []:
            c = op.coefficient
            val = _scalar(c)
            if val is None:
                val = parameters.get(str(getattr(c, "name", None) or c), 1.0)
            coeffs.append(val)
        if not coeffs:
            raise FieldPlanError(
                "no linear operator found: the equations produced no assembleable term and no `operators:` were declared"
            )
        blocks = [
            {
                "row": 0,
                "col": 0,
                "kind": "stiffness",
                "scalar": sum(coeffs),
                "coefficient_field": None,
                "weight_field": None,
                "term": "legacy operators",
            }
        ]
        legacy = True

    solver = getattr(fd, "solver", None)
    dt = float(getattr(solver, "step_size", None) or 1.0)
    raw_method = str(getattr(solver, "method", "") or "").strip().lower()
    method = _METHOD_ALIASES.get(raw_method, "crank-nicolson" if not raw_method else None)
    if method is None:
        raise FieldPlanError(
            f"unknown time-integration method {raw_method!r}; expected one of {sorted(set(_METHOD_ALIASES.values()))}"
        )
    duration = float(getattr(getattr(experiment, "integration", None), "duration", 0.0) or 0.0)

    mesh = getattr(fd, "mesh", None) or getattr(svars[0], "mesh", None)
    if mesh is None:
        raise FieldPlanError("no mesh declared on field_dynamics or its state variables")

    return {
        "variables": names,
        "labels": [str(getattr(v, "label", None) or n) for v, n in zip(svars, names, strict=True)],
        "initial_values": [float(getattr(v, "initial_value", 0.0) or 0.0) for v in svars],
        "blocks": blocks,
        "explicit": explicit_rows,
        "events": sorted(events),
        "parameters": parameters,
        "field_parameters": sorted(field_params),
        "field_sources": _field_sources(fd, experiment),
        "boundary_conditions": _boundary_plan(fd, svars, names),
        "method": method,
        "dt": dt,
        "steps": int(round(duration / dt)) if dt > 0 else 0,
        "duration": duration,
        "mesh": {
            "element_type": str(getattr(mesh, "element_type", "") or ""),
            "path": _mesh_path(mesh, experiment),
            "format": str(getattr(mesh, "mesh_format", None) or ""),
        },
        "legacy_operators": legacy,
    }


def _boundary_plan(field_dynamics, svars, names) -> list:
    """Dirichlet constraints per state variable.

    Only Dirichlet is lowered. A declared Neumann/Robin/Periodic condition raises rather than being silently dropped: on a closed surface (the cortical case) there is no boundary at all and the distinction never arises, but on a bounded domain quietly ignoring it changes the solution. Every condition is checked before one is taken, so a Neumann/Robin declared beside a Dirichlet raises instead of being skipped by an early exit.
    """
    out = []
    for row, var in enumerate(svars):
        bcs = list(getattr(var, "boundary_conditions", None) or []) or list(
            getattr(field_dynamics, "boundary_conditions", None) or []
        )
        for bc in bcs:
            kind = str(getattr(bc, "bc_type", "") or "").lower()
            if kind != "dirichlet":
                raise FieldPlanError(
                    f"boundary condition {kind!r} on {names[row]!r} is declared but not implemented; only Dirichlet is lowered"
                )
        for bc in bcs[:1]:
            # `equation` is the canonical slot and `value` its alias, and the dialect folds an alias into the canonical slot at construction — so reading `value` first would find nothing and silently hold every boundary at 0.0.
            declared = getattr(bc, "equation", None)
            if declared is None:
                declared = getattr(bc, "value", None)
            try:
                value = float(_rhs_text(declared) or 0.0)
            except ValueError:
                raise FieldPlanError(f"non-constant Dirichlet value on {names[row]!r} is not supported") from None
            out.append({"row": row, "value": value, "on_region": getattr(bc, "on_region", None) or None})
    return out
