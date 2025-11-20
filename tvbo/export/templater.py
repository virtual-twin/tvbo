#
# Module: templater.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import re
from os.path import join

import autopep8
import black
import numpy as np
import sympy as sp
from mako.template import Template
from sympy import pycode

from tvbo import templates
from tvbo.knowledge import ontology
from tvbo.knowledge.simulation import equations, network

exec_globals = {}

# exec_globals = {
#     # TVB Imports (Classes)
#     "Attr": import_module("tvb.basic.neotraits.api").Attr,
#     "Final": import_module("tvb.basic.neotraits.api").Final,
#     "HasTraits": import_module("tvb.basic.neotraits.api").HasTraits,
#     "List": import_module("tvb.basic.neotraits.api").List,
#     "NArray": import_module("tvb.basic.neotraits.api").NArray,
#     "Range": import_module("tvb.basic.neotraits.api").Range,
#     "Coupling": import_module("tvb.simulator.coupling").Coupling,
#     "SparseCoupling": import_module("tvb.simulator.coupling").SparseCoupling,
#     "SparseHistory": import_module("tvb.simulator.history").SparseHistory,
#     "Model": import_module("tvb.simulator.models.base").Model,
#     "Integrator": import_module("tvb.simulator.integrators").Integrator,
#     "IntegratorStochastic": import_module(
#         "tvb.simulator.integrators"
#     ).IntegratorStochastic,
#     "Simulator": import_module("tvb.simulator.simulator").Simulator,
#     "SciPyODE": import_module("tvb.simulator.integrators").SciPyODE,
#     "SciPySDE": import_module("tvb.simulator.integrators").SciPySDE,
#     "Additive": import_module("tvb.simulator.noise").Additive,
#     "Multiplicative": import_module("tvb.simulator.noise").Multiplicative,
#     # Numba Imports (Functions)
#     "float64": import_module("numba").float64,
#     "f64": import_module("numba").float64,
#     "guvectorize": import_module("numba").guvectorize,
#     # SymPy Imports (Functions)
#     "pycode": import_module("sympy").pycode,
#     # Mako Imports (Classes)
#     "Context": import_module("mako.runtime").Context,
#     "Template": import_module("mako.template").Template,
#     # TVB Imports (Functions)
#     "simple_gen_astr": import_module("tvb.simulator.common").simple_gen_astr,
#     # NumPy Imports (Modules)
#     "numpy": import_module("numpy"),
#     "exp": import_module("numpy").exp,
#     "where": import_module("numpy").where,
#     "log": import_module("numpy").log,
#     "sqrt": import_module("numpy").sqrt,
#     "sin": import_module("numpy").sin,
#     "cos": import_module("numpy").cos,
#     "tanh": import_module("numpy").tanh,
#     "pi": import_module("numpy").pi,
#     "inf": import_module("numpy").inf,
#     "np": import_module("numpy"),
#     # SymPy Imports (Modules)
#     "sp": import_module("sympy"),
# }

TEMPLATES = templates.root


def format_code(code, format="python", use_black=True, **kwargs):
    if format in ["tvb", "python", "autodiff"]:
        code = autopep8.fix_code(code)
        if use_black:
            code = black.format_str(code, mode=black.FileMode(**kwargs))
    return code


def get_statevariable_equations(model):
    acr = ontology.get_model_suffix(model)
    diff_eqs = equations.symbolic_differential_equations(model)
    state_variable_dfuns = {
        k: diff_eqs[
            (
                ontology.intersection(
                    v.subclasses(), ontology.onto.TimeDerivative.descendants()
                )[0]
                .label.first()
                .replace(acr, "")
                if ontology.intersection(
                    v.subclasses(), ontology.onto.TimeDerivative.descendants()
                )
                else v.has_derivative[0].label.first().replace(acr, "")
            )
        ]
        for k, v in ontology.get_model_statevariables(model).items()
    }
    return state_variable_dfuns


def get_model_info(model):
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
        var: re.sub(r"(?<=[( ])e(?= )", "E", pycode(eq, fully_qualified_modules=False))
        for var, eq in ninvar_dfuns.items()
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
    range = (
        sv_class.stateVariableRange.first().split(",")
        if sv_class.stateVariableRange
        else [-1e100, 1e100]
    )

    if sv_class.stateVariableBoundaries:
        boundaries = [
            svb.strip() for svb in sv_class.stateVariableBoundaries.first().split(",")
        ]

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
    return code_str.replace("and", "&").replace("or", "|").replace("not", "~")


def equation2class(EQ, fout=None, print_source=False, **kwargs):
    var = sp.symbols("var")
    eq_name = EQ.label.first()
    definition = EQ.definition.first()
    eq_type = ", ".join(
        [
            c.name
            for c in ontology.intersection(
                EQ.is_a, ontology.onto.Equation.descendants()
            )
        ]
    )

    eq = equations.sympify_value(EQ)
    code_str = boolean2bitwise(
        equations.convert_ifelse_to_np_where(
            pycode(eq.subs({"t": var}), fully_qualified_modules=False)
        )
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
        # Print each line with its line number
        for i, line in enumerate(rendered_code.split("\n"), start=1):
            print(f"{i}\t{line}")
    if fout:
        with open(fout, "w") as f:
            f.write(rendered_code)
    else:
        exec(rendered_code, exec_globals, local_vars)
        eq_kwargs = {
            k: v for k, v in values.items() if k in ["equation", "parameters", "gid"]
        }
        tvbo_equation = local_vars[f"{eq_name}"](**eq_kwargs)
        return tvbo_equation


def coupling2class(CF, fout=None, print_source=False, **kwargs):
    from tvbo.datamodel import tvbo_datamodel

    if isinstance(CF, str):
        CF = ontology.get_coupling_function(CF)

    sparse = ontology.onto.SparseCoupling in CF.is_a
    coupling_name = CF.label.first()
    eqs = equations.get_symbolic_coupling(CF)
    fpre = pycode(eqs["pre"], fully_qualified_modules=False)
    fpost = pycode(eqs["post"], fully_qualified_modules=False)

    local_vars = {}
    template = templates.lookup.get_template("_tvbo-tvb-coupling.py.mako")
    parameters = network.get_parameters(CF)

    rendered_code = template.render(
        class_name=coupling_name,
        pre_expr=fpre,
        post_expr=fpost,
        parameters={
            v["label"]: tvbo_datamodel.Parameter(**v) for k, v in parameters.items()
        },
        sparse=sparse,
    )
    if print_source:
        # Print each line with its line number
        for i, line in enumerate(rendered_code.split("\n"), start=1):
            print(f"{i}\t{line}")
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
    if isinstance(model, str):
        model = ontology.get_model(model)

        if not model:
            raise ValueError(
                f"Model {model} not found in the ontology. Available models:"
                + "\n".join(ontology.get_models())
            )

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
        sv_dfuns = equations.substitute_function_in_state_equations(
            sv_dfuns, model_info["ninvar_dfuns"]
        )

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
        # Print each line with its line number
        for i, line in enumerate(rendered_code.split("\n"), start=1):
            print(f"{i}\t{line}")
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
