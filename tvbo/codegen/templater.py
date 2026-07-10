#
# Module: templater.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Generate TVBO Python classes from ontology definitions via Mako templates.

Provides helpers that read model, parameter, state-variable, equation,
coupling and integrator metadata from the ontology and render it into
executable TVBO/TVB source using the templates in `tvbo.templates`.
"""
import logging
import re
from os.path import join
from typing import Any

import black
import numpy as np
import sympy as sp
from mako.template import Template
from sympy import pycode

from tvbo import templates
from tvbo.ontology import owl as ontology
from tvbo.classes import equation as equations, coupling

logger = logging.getLogger(__name__)


def _log_source(rendered_code: str) -> None:
    """Emit rendered code with line numbers when ``print_source`` is requested."""
    numbered = "\n".join(
        f"{i}\t{line}" for i, line in enumerate(rendered_code.split("\n"), start=1)
    )
    logger.info("rendered source:\n%s", numbered)

exec_globals = {}
TEMPLATES = templates.root


def is_derived(obs: Any, experiment: Any) -> bool:
    """Return True if ``obs`` derives from other observations in ``experiment``.

    An Observation is derived when any item in its multivalued ``source``
    slot names ANOTHER observation in the same experiment. Source entries
    may be bare strings, objects with a ``name`` attribute, or inlined
    Observation/StateVariable instances.

    A SELF-reference (an observation whose ``source`` names itself — e.g. an
    observation ``r_A`` with ``source: [r_A]`` that simply observes the model
    variable ``r_A``) is NOT derived: an observation cannot derive from itself.
    Without this exclusion such observations are mis-routed to the derived path,
    where they have no pipeline and are never computed, so the generated
    ``observations.r_A = _all_obs.r_A`` extraction raises AttributeError.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return False
    self_name = getattr(obs, "name", None)
    for s in (getattr(obs, "source", None) or []):
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            return True
    return False


def source_observations(obs: Any, experiment: Any) -> list:
    """Return the source names of ``obs`` that resolve to other observations.

    A filtered view of ``obs.source`` keeping only entries whose name
    matches a key in ``experiment.observations``.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return []
    self_name = getattr(obs, "name", None)
    out = []
    for s in (getattr(obs, "source", None) or []):
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            out.append(name)
    return out


def format_code(code: str, format: str = "python", use_black: bool = True, **kwargs: Any) -> str:
    """Format code using black for Python variants.

    Args:
        code: Source code string to format
        format: Language/variant (python, jax, numpy, scipy, tvboptim)
        use_black: Whether to apply black formatting (default True)
        **kwargs: Additional black.FileMode options (line_length, etc.)
    """
    if format in ["tvb", "python", "autodiff", "jax", "numpy", "scipy", "tvboptim"]:
        if use_black:
            code = black.format_str(code, mode=black.FileMode(**kwargs))
    return code


def get_statevariable_equations(model):
    """Map each state variable to its symbolic time-derivative equation.

    For every state variable of `model`, look up the matching `TimeDerivative`
    expression among the model's symbolic differential equations, keyed by the
    derivative's label with the model acronym suffix stripped.

    Args:
        model: An ontology model class, or a model name to resolve.

    Returns:
        A dict mapping state-variable name to its differential-equation
        expression.
    """
    acr = ontology.get_model_suffix(model)
    diff_eqs = equations.symbolic_differential_equations(model)
    state_variable_dfuns = {
        k: diff_eqs[
            (
                ontology.intersection(v.subclasses(), ontology.onto.TimeDerivative.descendants())[0]
                .label.first()
                .replace(acr, "")
                if ontology.intersection(v.subclasses(), ontology.onto.TimeDerivative.descendants())
                else v.has_derivative[0].label.first().replace(acr, "")
            )
        ]
        for k, v in ontology.get_model_statevariables(model).items()
    }
    return state_variable_dfuns


def get_model_info(model):
    """Collect the codegen-relevant metadata for a model into a dict.

    Resolve `model` from the ontology if given as a name, then gather its
    parameters and constants, coupling-variable indices, coupling terms,
    non-integrated variables (functions and conditionals) with their
    dependency-sorted symbolic expressions, state variables, state-variable
    differential equations, and variables of interest.

    Args:
        model: An ontology model class, or a model name to resolve.

    Returns:
        A dict with keys `parameters`, `cvars`, `coupling_terms`,
        `non_integrated_variables`, `ninvar_dfuns`, `state_variables`,
        `state_variable_dfuns` and `vois`.
    """
    if isinstance(model, str):
        model = ontology.get_model(model)

    svs = ontology.get_model_statevariables(model)
    cvars = [list(svs.keys()).index(c) for c in ontology.get_model_cvars(model).keys()]

    parameters = ontology.get_model_parameters(model)
    parameters.update(ontology.get_model_constants(model))

    non_integrated_variables = ontology.get_model_functions(model)
    non_integrated_variables.update(ontology.get_model_conditionals(model))
    ninvar_dfuns = equations.symbolic_model_functions(model)
    ninvar_dfuns.update(equations.symbolic_conditions(model))
    ninvar_dfuns = equations.sort_equations_by_dependencies(ninvar_dfuns)

    # TODO: Remove hack that replaces 'e' with 'E' in equations created by pycode
    ninvar_dfuns = {
        var: re.sub(r"(?<=[( ])e(?= )", "E", pycode(eq, fully_qualified_modules=False)) for var, eq in ninvar_dfuns.items()
    }
    for var, eq in ninvar_dfuns.items():
        if "if" in eq and "else" in eq:
            ninvar_dfuns[var] = equations.convert_ifelse_to_np_where(eq)

    model_info = dict(
        parameters=parameters,
        cvars=cvars,
        coupling_terms=list(ontology.get_model_coupling_terms(model).keys()),
        non_integrated_variables=non_integrated_variables,
        ninvar_dfuns=ninvar_dfuns,
        state_variables=svs,
        state_variable_dfuns=get_statevariable_equations(model),
        vois=ontology.get_model_vois(model),
    )
    return model_info


def get_param_info(param_class):
    """Extract label, symbol, default, range, definition and dependencies of a parameter.

    Fall back from the ontology `defaultValue` to `value` when no default is
    set.

    Args:
        param_class: Ontology parameter class to read metadata from.

    Returns:
        A dict describing the parameter.
    """
    return dict(
        label=param_class.label.first(),
        symbol=param_class.symbol.first(),
        default=(
            param_class.defaultValue.first()
            if not isinstance(param_class.defaultValue.first(), type(None))
            else param_class.value.first()
        ),
        range=ontology.get_range(param_class),
        definition=param_class.definition.first(),
        dependencies=param_class.has_dependency,
    )


def get_sv_info(sv_class):
    """Extract label, symbol, default, range, boundaries and definition of a state variable.

    Parse the ontology `stateVariableRange` and `stateVariableBoundaries`
    strings, defaulting to an effectively unbounded range and open boundaries
    when unset.

    Args:
        sv_class: Ontology state-variable class to read metadata from.

    Returns:
        A dict describing the state variable.
    """
    range = sv_class.stateVariableRange.first().split(",") if sv_class.stateVariableRange else [-1e100, 1e100]

    if sv_class.stateVariableBoundaries:
        boundaries = [svb.strip() for svb in sv_class.stateVariableBoundaries.first().split(",")]

    else:
        boundaries = (None, None)

    return dict(
        label=sv_class.label.first(),
        symbol=sv_class.symbol.first(),
        default=sv_class.defaultValue.first(),
        range=range,
        boundaries=boundaries,
        definition=sv_class.definition.first(),
    )


def boolean2bitwise(code_str):
    """Rewrite Python boolean operators as their bitwise equivalents.

    Replace `and`/`or`/`not` with `&`/`|`/`~` so element-wise array
    expressions evaluate correctly.

    Args:
        code_str: Source expression using boolean keywords.

    Returns:
        The expression with boolean operators replaced by bitwise operators.
    """
    return code_str.replace("and", "&").replace("or", "|").replace("not", "~")


def equation2class(EQ, fout=None, print_source=False, **kwargs):
    """Generate a TVBO equation class from an ontology equation.

    Sympify the equation, render it (with its Python and LaTeX forms and
    default parameter values) through the `_tvbo-tvb-equation.py.mako`
    template, then either write the source, print it, or execute it and return
    an instantiated equation object.

    Args:
        EQ: Ontology equation class to convert.
        fout: Optional path; when given, write the rendered source there instead
            of instantiating.
        print_source: When True, print the rendered source with line numbers.
        **kwargs: Overrides for the equation's default parameter values.

    Returns:
        The instantiated equation object, or `None` when `fout` is given.
    """
    var = sp.symbols("var")
    eq_name = EQ.label.first()
    definition = EQ.definition.first()
    eq_type = ", ".join([c.name for c in ontology.intersection(EQ.is_a, ontology.onto.Equation.descendants())])

    eq = equations.sympify_value(EQ)
    code_str = boolean2bitwise(
        equations.convert_ifelse_to_np_where(pycode(eq.subs({"t": var}), fully_qualified_modules=False))
    )
    latex_str = sp.latex(eq, mul_symbol="dot")

    values = ontology.get_default_values(EQ)
    for k, v in kwargs.items():
        if k in values:
            values[k] = v

    local_vars = {}
    template = templates.lookup.get_template("_tvbo-tvb-equation.py.mako")
    rendered_code = template.render(
        eq_name=eq_name,
        definition=definition,
        code_str=code_str,
        latex_str=latex_str,
        parameters=values,
        eq_type=eq_type,
    )
    if print_source:
        _log_source(rendered_code)
    if fout:
        with open(fout, "w") as f:
            f.write(rendered_code)
    else:
        exec(rendered_code, exec_globals, local_vars)
        eq_kwargs = {k: v for k, v in values.items() if k in ["equation", "parameters", "gid"]}
        tvbo_equation = local_vars[f"{eq_name}"](**eq_kwargs)
        return tvbo_equation


def coupling2class(CF, fout=None, print_source=False, **kwargs):
    """Generate a TVBO coupling class from an ontology coupling function.

    Resolve `CF` from the ontology if given as a name, build the pre- and
    post-summation expressions and parameters, and render them through the
    `_tvbo-tvb-coupling.py.mako` template. Then either write the source, print
    it, or execute it and return an instantiated coupling object. Keyword
    arguments matching coupling attributes are passed as array-coerced
    constructor values.

    Args:
        CF: Ontology coupling-function class, or a name to resolve.
        fout: Optional path; when given, write the rendered source there instead
            of instantiating.
        print_source: When True, print the rendered source with line numbers.
        **kwargs: Constructor overrides for recognised coupling attributes.

    Returns:
        The instantiated coupling object, or `None` when `fout` is given.
    """
    from tvbo.datamodel import schema as tvbo_datamodel

    if isinstance(CF, str):
        CF = ontology.get_coupling_function(CF)

    sparse = ontology.onto.SparseCoupling in CF.is_a
    coupling_name = CF.label.first()
    eqs = equations.get_symbolic_coupling(CF)
    fpre = pycode(eqs["pre"], fully_qualified_modules=False)
    fpost = pycode(eqs["post"], fully_qualified_modules=False)

    local_vars = {}
    template = templates.lookup.get_template("_tvbo-tvb-coupling.py.mako")
    parameters = coupling.get_parameters(CF)

    rendered_code = template.render(
        class_name=coupling_name,
        pre_expr=fpre,
        post_expr=fpost,
        parameters={v["label"]: tvbo_datamodel.Parameter(**v) for k, v in parameters.items()},
        sparse=sparse,
    )
    if print_source:
        _log_source(rendered_code)
    if fout:
        with open(fout, "w") as f:
            f.write(rendered_code)
    else:
        exec(rendered_code, exec_globals, local_vars)
        tvbo_coupling = local_vars[f"{coupling_name}"]
        class_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(tvbo_coupling, k):
                if not isinstance(v, np.ndarray):
                    v = np.array([v])
                class_kwargs[k] = v
        return tvbo_coupling(**class_kwargs)


def formulate_dependency_imports(dependencies):
    """Build `from ... import ...` statements for dotted dependency paths.

    Only dependencies containing a dotted path produce an import; the final
    segment is imported from the preceding module path.

    Args:
        dependencies: Iterable of dependency strings (e.g. `"pkg.mod.name"`).

    Returns:
        A generator of import statements, one per dotted dependency.
    """
    for d in dependencies:
        if len(d.split(".")) > 1:
            yield f"from {'.'.join(d.split('.')[0:-1])} import {d.split('.')[-1]}"


def model2class(
    model,
    fout=None,
    print_source=False,
    split_nonintegrated_variables=False,
    return_instance=True,
    sub_ninvars=False,
    bifmodel=False,
    **kwargs,
):
    """Generate a TVBO model class from an ontology model.

    Resolve `model` from the ontology if given as a name, assemble its state
    variables, differential equations, non-integrated variables, parameters and
    dependency imports via `get_model_info`, and render them through the
    `_tvbo-tvb-model_old.py.mako` template. Then either write the source, print
    it, or execute it and return the model class or an instance.

    Args:
        model: An ontology model class, or a model name to resolve.
        fout: Optional path; when given, write the rendered source there instead
            of instantiating.
        print_source: When True, print the rendered source with line numbers.
        split_nonintegrated_variables: When True, treat non-integrated variables
            as additional state variables (forced off when `sub_ninvars` is set).
        return_instance: When True, return an instance; otherwise return the
            class.
        sub_ninvars: When True, substitute non-integrated function expressions
            into the state equations instead of splitting them out.
        bifmodel: Reserved flag (currently unused in rendering).
        **kwargs: Constructor overrides for recognised model attributes.

    Returns:
        The instantiated model, the model class, or `None` when `fout` is given.

    Raises:
        ValueError: If a model name cannot be resolved in the ontology.
    """
    if isinstance(model, str):
        model = ontology.get_model(model)

        if not model:
            raise ValueError(f"Model {model} not found in the ontology. Available models:" + "\n".join(ontology.get_models()))

    if sub_ninvars:
        split_nonintegrated_variables = False

    model_name = model.label.first()
    local_vars = {}
    model_info = get_model_info(model)

    state_variables = model_info["state_variables"]

    if split_nonintegrated_variables:
        state_variables.update(model_info["non_integrated_variables"])
        nivars = list(model_info["ninvar_dfuns"].keys())
    else:
        nivars = None

    for k, sv in state_variables.items():
        state_variables[k] = get_sv_info(sv)
    sv_dfuns = model_info["state_variable_dfuns"]
    # if split_nonintegrated_variables:
    sv_dfuns.update(model_info["ninvar_dfuns"])

    if sub_ninvars:
        sv_dfuns = equations.substitute_function_in_state_equations(sv_dfuns, model_info["ninvar_dfuns"])

    parameters = dict()
    for k, param in model_info["parameters"].items():
        parameters[k] = get_param_info(param)

    dependencies = list()
    for v in model.descendants():
        if len(v.has_dependency) > 0:
            dependencies.extend(v.has_dependency)
    dependencies = list(set(dependencies))
    import_statements = list(formulate_dependency_imports(dependencies))

    template = Template(filename=join(TEMPLATES, "_tvbo-tvb-model_old.py.mako"))

    rendered_code = template.render(
        model_name=model_name,
        coupling_terms=model_info["coupling_terms"],
        non_integrated_variables=nivars if nivars and len(nivars) > 0 else None,
        state_variables=state_variables,
        state_variable_dfuns=sv_dfuns,
        parameters=parameters,
        spatial_parameter_names=[],
        cvars=model_info["cvars"],
        ninvar_dfuns=model_info["ninvar_dfuns"],
        vois=model_info["vois"],
        import_statements=import_statements,
    )
    if print_source:
        _log_source(rendered_code)
    if fout:
        with open(fout, "w") as f:
            f.write(rendered_code)
    else:
        exec(rendered_code, exec_globals, local_vars)
        tvbo_model = local_vars[f"{model_name}"]
        class_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(tvbo_model, k):
                if isinstance(v, float):
                    v = np.array([v])
                class_kwargs[k] = v
        if return_instance:
            return tvbo_model(**class_kwargs)
        else:
            return tvbo_model


### Integrator ###
def get_integrator_info(integrator):
    """Collect scheme metadata for an integrator into a dict.

    Args:
        integrator: Ontology integrator class describing the scheme.

    Returns:
        A dict with the integrator `class_name`, the number of derivative stages
        `n_dx`, its `intermediate_steps`, and the `dX_expr` update expression
        (`None` when unset).
    """
    n_dx = len(integrator.intermediate_steps) + 1
    intermediade_steps = integrator.intermediate_steps if n_dx > 1 else []
    dX = integrator.dX.first() if integrator.dX else None

    info = {
        "class_name": integrator.name,
        "n_dx": n_dx,
        "intermediate_steps": intermediade_steps,
        "dX_expr": dX,
    }
    return info


def integrator2class(integrator, return_instance=True, **kwargs):
    """Resolve an ontology integrator to a TVB integrator class or instance.

    Look up the matching `tvb.simulator.integrators` class, selecting the
    stochastic variant when a `noise` keyword is supplied and the deterministic
    variant otherwise.

    Args:
        integrator: Ontology integrator class, or a name to search for.
        return_instance: When True, return an instance built from `kwargs`;
            otherwise return the class.
        **kwargs: Integrator constructor arguments; a `noise` entry selects the
            stochastic variant.

    Returns:
        The TVB integrator instance or class.
    """
    from tvb.simulator import integrators

    if isinstance(integrator, str):
        integrator = ontology.search_all(integrator)

    Integrator = getattr(
        integrators,
        integrator.name + ("Stochastic" if "noise" in kwargs else "Deterministic"),
    )

    if return_instance:
        return Integrator(**kwargs)
    return Integrator
