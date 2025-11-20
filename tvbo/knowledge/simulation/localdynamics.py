#
# Module: localdynamics.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

import os
import re
import tempfile
from os.path import basename, dirname, join, splitext
from tvbo.knowledge.simulation.perturbation import Stimulus
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import owlready2
from linkml_runtime.loaders import yaml_loader
from matplotlib import colormaps
from sympy import (
    Derivative,
    Eq,
    Function,
    Symbol,
    latex,
    pycode,
    symbols,
)
from tvbo.parse.expression import parse_eq

from tvbo import templates
from tvbo.data.types import TimeSeries
from tvbo.datamodel import tvbo_datamodel
from tvbo.datamodel import tvbopydantic as _pdm
from tvbo.datamodel.tvbo_datamodel import Case, DerivedVariable, Equation
from tvbo.export import templater, report
from tvbo.knowledge import ontology, query, simulation
from tvbo.knowledge.simulation import equations
from tvbo.parse import metadata
from tvbo.analysis import BifurcationResult

TEMPLATES = templates.root
available_neural_mass_models = set(ontology.get_models().values())


## BifurcationResult moved to tvbo.analysis.bifurcation


def clean_code(code):
    cleaned_code = re.sub(r"∞", "inf", code)
    return cleaned_code


def order_by_equations(derived_variables, dependent_equations):
    """
    Orders the `derived_variables` dictionary based on the key order of the `dependent_equations` dictionary.

    Parameters:
    derived_variables (dict): Dictionary to be ordered.
    dependent_equations (dict): Dictionary providing the key order for sorting.

    Returns:
    dict: A new dictionary ordered by the key order from `dependent_equations`.
    """
    dependency = {k.replace("dot", ""): v for k, v in dependent_equations.items()}
    # Order derived_variables based on the order in dependent_equations
    ordered_dict = {
        k: derived_variables[k] for k in dependency if k in derived_variables
    }

    return ordered_dict


def class2metadata(ontoclass, metadata):
    metadata.description = ontology.get_def(ontoclass, mode="long")
    dependent_equations = equations.sort_equations_by_dependencies(
        equations.symbolic_model_equations(ontoclass)
    )
    state_variables = order_by_equations(
        ontology.get_model_statevariables(ontoclass), dependent_equations
    )
    state_variables = ontology.get_model_statevariables(ontoclass)

    functions = order_by_equations(
        ontology.get_model_functions(ontoclass), dependent_equations
    )
    update_parameters(metadata, ontoclass)

    for k, v in state_variables.items():
        range = ontology.get_range(v)

        if v.stateVariableBoundaries:
            boundary = ontology.get_range(v.stateVariableBoundaries.first())
            boundaries = tvbo_datamodel.Range(lo=boundary[0], hi=boundary[1])
        else:
            boundaries = None

        td = v.has_derivative.first()
        if k not in metadata.state_variables:
            metadata.state_variables.update(
                {
                    k: tvbo_datamodel.StateVariable(
                        name=k,
                        equation=tvbo_datamodel.Equation(
                            lhs=td.symbol.first(),
                            rhs=td.value.first()
                            .replace("numpy.", "")
                            .replace("np.", ""),
                        ),
                        description=ontology.get_def(v),
                        domain=tvbo_datamodel.Range(lo=range[0], hi=range[1]),
                        boundaries=boundaries,
                        coupling_variable=v in ontoclass.has_cvar,
                    )
                }
            )
        elif k in metadata.state_variables:
            state_var = metadata.state_variables[k]
            updates = {
                "equation": state_var.equation
                or tvbo_datamodel.Equation(
                    lhs=td.symbol.first(),
                    rhs=td.value.first().replace("numpy.", "").replace("np.", ""),
                ),
                "description": state_var.description or ontology.get_def(v),
                "domain": state_var.domain
                or tvbo_datamodel.Range(lo=range[0], hi=range[1]),
                "boundaries": state_var.boundaries or boundaries,
                "coupling_variable": state_var.coupling_variable
                or (v in ontoclass.has_cvar),
            }

            for attr, value in updates.items():
                setattr(state_var, attr, value)

    for k, v in functions.items():
        if k not in metadata.derived_variables and k not in metadata.derived_parameters:
            metadata.derived_variables.update(
                {
                    k: tvbo_datamodel.DerivedVariable(
                        name=k,
                        equation=tvbo_datamodel.Equation(
                            lhs=v.symbol.first(),
                            rhs=v.value.first()
                            .replace("numpy.", "")
                            .replace("np.", ""),
                        ),
                        description=v.definition.first(),
                    )
                }
            )

    for condpar in ontology.get_model_conditionals(ontoclass).values():
        val = simulation.equations.sympify_value(condpar)
        name = ontology.replace_suffix(condpar)
        metadata.derived_variables.update(
            {
                name: DerivedVariable(
                    name=name,
                    symbol=condpar.symbol.first(),
                    conditional=True,
                    cases=[
                        Case(condition=condition, equation=Equation(lhs=name, rhs=expr))
                        for expr, condition in val.args
                    ],
                    equation=tvbo_datamodel.Equation(
                        lhs=name,
                        conditionals=[
                            tvbo_datamodel.ConditionalBlock(
                                condition=condtion, expression=expr
                            )
                            for expr, condtion in val.args
                        ],
                    ),
                )
            }
        )

    for k, v in ontology.get_model_coupling_terms(ontoclass).items():
        if k not in metadata.coupling_terms:
            metadata.coupling_terms.update(
                {
                    k: tvbo_datamodel.Parameter(
                        name=k,
                    ),
                }
            )

    for r in ontoclass.has_reference:
        if r.name not in metadata.references:
            metadata.references.append(r.name)


def update_parameters(metadata, ontoclass, verbose=0, **kwargs):
    for k, v in ontology.get_default_values(ontoclass, class_as_key=True).items():
        label = ontology.replace_suffix(k.label.first())
        if range := ontology.get_range(k):
            lo, hi, step = range
            domain = tvbo_datamodel.Range(lo=lo, hi=hi, step=step)
        else:
            domain = None

        if label not in metadata.parameters and not any(
            synonym in metadata.parameters for synonym in k.synonym + k.symbol
        ):
            if verbose > 0:
                print(f"using parameter {label} from the ontology")
            metadata.parameters.update(
                {
                    label: tvbo_datamodel.Parameter(
                        name=label,
                        value=kwargs.get(k, v),
                        description=ontology.get_def(k, mode="short").replace(
                            "\n", " "
                        ),
                        domain=domain,
                        definition=k.definition.first(),
                    )
                }
            )

        if label in metadata.parameters:
            if metadata.parameters[label].description is None:
                metadata.parameters[label].description = ontology.get_def(
                    k, mode="short"
                ).replace("\n", " ")

            if metadata.parameters[label].unit is None:
                metadata.parameters[label].unit = (
                    k.has_unit.first().name if k.has_unit else k.unit.first()
                )

            if metadata.parameters[label].value is None:
                metadata.parameters[label].value = k.defaultValue.first()


def update_equations(model):
    from sympy import Function, diff

    substitutions = {}

    t = symbols("t")
    equations = model.get_equations()

    for k, eq in model.get_equations().items():
        k_orig = k.replace("_dot", "").replace("dot", "")

        if "dot" in k:
            k = k.replace("_dot", "").replace("dot", "")
            # k = rf"\dot{{{k}}}"
            # k = Function(k)(t)
            # k = diff(k, t)
            k = Derivative(symbols(k), t)
        else:
            k = symbols(k)

        # Always coerce entries to sympy.Eq so downstream code can rely on .lhs/.rhs
        # Previously we only wrapped missing keys, which left existing items as raw
        # expressions (e.g., Mul) without lhs/rhs and caused AttributeError later.
        equations[k_orig] = eq if isinstance(eq, Eq) else Eq(k, eq)

        missing_symbols = [
            s
            for s in eq.free_symbols
            if str(s) not in model.parameters
            and str(s) not in model.state_variables
            and (model.derived_variables and str(s) not in model.derived_variables)
        ]

        if missing_symbols:
            for s in missing_symbols:
                labelsearch = query.label_search(
                    str(s),
                    root_class=model.ontology,
                    exact_match=["symbol", "synonym", "tvbSourceVariable"],
                    case_sensitive=True,
                )
                if len(labelsearch) > 1:
                    # print(labelsearch)
                    labelsearch = query.label_search(
                        str(s),
                        root_class=model.ontology,
                        exact_match="all",
                        case_sensitive=True,
                    )
                    # print(labelsearch)

                if not labelsearch:
                    # if str(s) != "t":
                    #     print(str(s))
                    #     print("for equation:", k, s, "not found in ontology")
                    continue

                if len(labelsearch) > 1:
                    labelsearch = list(
                        np.array(labelsearch)[
                            [ontology.replace_suffix(l) == str(s) for l in labelsearch]
                        ]
                    )

                synonyms = labelsearch[0].synonym + labelsearch[0].symbol

                match = next(
                    (syn for syn in synonyms if str(syn) in model.parameters),
                    None,
                )

                if match:
                    substitutions.update({s: Symbol(match)})

    def substitute_equations(
        metadata_dict, substitutions, equations, time_derivative=False
    ):
        for variable_key, v in metadata_dict.items():
            if (
                isinstance(v.equation, type(None))
                and str(variable_key) in equations.keys()
            ):
                eq = tvbo_datamodel.Equation(rhs=equations[str(variable_key)])
            elif str(variable_key) in equations.keys():
                eq = equations[str(variable_key)]
            else:
                if not isinstance(v.equation, type(None)):
                    eq = v.equation
                else:
                    raise ValueError(f"{v}, {equations.keys()}")

            # Use model-scoped symbolic elements for parsing instead of global clash
            eq = parse_eq(eq, local_dict=model.get_symbolic_elements(), evaluate=False)

            eq_sub = eq.subs(substitutions)
            rhs_substitution = pycode(
                eq_sub,
                fully_qualified_modules=False,
                user_functions={k: k for k in model.functions.keys()},
            )

            if "euqation" in metadata_dict[variable_key]:
                metadata_dict[variable_key].equation.rhs = rhs_substitution

            if time_derivative:
                lhs = Derivative(Symbol(variable_key), t)
            else:
                lhs = Symbol(variable_key)

            equations[variable_key] = Eq(lhs, eq_sub)

    if substitutions != {}:
        substitute_equations(
            model.state_variables,
            substitutions,
            equations,
            time_derivative=True,
        )
        substitute_equations(model.derived_variables, substitutions, equations)

    return equations


def sort_equations(model, variable_type):
    # sort equations (compute dependency tree on the fly; avoid stored state)
    G_dep = model.get_dependency_tree()
    if isinstance(G_dep, tuple):
        G_dep = G_dep[0]
    sorted_variables = []
    for tg in nx.dag.topological_generations(G_dep):
        sorted_variables.extend(sorted(tg, key=lambda x: str(x)))

    original_metadata = model[variable_type].copy()

    sorted_variables_metadata = {}
    for var_name in sorted_variables:
        if str(var_name) in model[variable_type]:
            sorted_variables_metadata[str(var_name)] = original_metadata.pop(
                str(var_name)
            )

    for missing_key in original_metadata:
        sorted_variables_metadata = {
            missing_key: original_metadata[missing_key],
            **sorted_variables_metadata,
        }

    # Update the original dictionary in-place
    model[variable_type].clear()
    model[variable_type].update(sorted_variables_metadata)


class Dynamics(tvbo_datamodel.Dynamics):
    def __init__(self, name="Dynamics", **kwargs):
        if name is not None:
            kwargs["name"] = str(name)

        # Initialize datamodel (base class sets up empty containers)
        super().__init__(**kwargs)

        # Auto-populate only when a name was provided; keep default Dynamics() empty
        if name != "Dynamics":
            # Backfill from ontology by name when available
            self._populate_from_ontology_if_available()

            # Finalize metadata/equations
            self.update_metadata()
            self.calculate_derived_parameters()

    # Factory constructors
    @classmethod
    def from_datamodel(cls, model_meta: tvbo_datamodel.Dynamics):
        inst = cls(**model_meta._as_dict)
        # Ensure ontology backfill happened (in __init__), but double-check
        inst._populate_from_ontology_if_available()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_ontology(cls, ontoclass: owlready2.ThingClass | str, **kwargs):
        # Construct with name and then populate from ontology
        if isinstance(ontoclass, str):
            ontoclass = query.label_search(
                ontoclass, root_class="NeuralMassModel", exact_match=["label"]
            )[0]
        inst = cls(name=ontoclass.name, **kwargs)
        inst._populate_from_ontology(ontoclass, **kwargs)
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_file(cls, path: str) -> "Dynamics":
        inst = yaml_loader.load(path, cls)
        inst._populate_from_ontology_if_available()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    # Internal helpers
    def _populate_from_ontology_if_available(self):
        oc = self.ontology
        if oc is not None:
            self._populate_from_ontology(oc)
        else:
            # YAML-based import hook (let errors surface instead of swallowing)
            oc = metadata.import_yaml_model(self)
            if oc is not None:
                self._populate_from_ontology(oc)

    def _populate_from_ontology(self, oc, **kwargs):
        # Fill schema fields from ontology, without persisting runtime-only state
        class2metadata(oc, self)
        update_parameters(self, oc, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name} - {len(self.parameters)} parameters and {len(self.state_variables)} state variables"

    def to_yaml(self, filepath: str | None = None):
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    # ---- Runtime convenience properties (no extra attributes) ----
    @property
    def metadata(self):
        # Alias: the datamodel instance itself holds the schema fields
        return self

    @property
    def ontology(self):
        name = getattr(self, "name", None)
        if not name:
            return None
        cl = ontology.onto.search_one(label=str(name))
        return cl if cl in available_neural_mass_models else None

    def search_ontology(self, search_str: str, **kwargs):
        return ontology.search_in_model(search_str, self.ontology, **kwargs)

    @property
    def keyed_parameters(self):
        return {
            Symbol(p.name): p.value for p in getattr(self, "parameters", {}).values()
        }

    def get_symbolic_elements(self, include_time_symbol: bool = True):
        """Build a unified local_dict for parsing model expressions.

        Includes symbols for parameters, coupling terms, derived parameters, derived
        variables, output transforms, state variables, function names, and (optionally)
        the time symbol 't'.

        Returns
        -------
        dict
            Mapping of names to SymPy objects suitable for parse_eq(local_dict=...).
        """
        scope: dict[str, object] = {}

        # Time symbol
        if include_time_symbol:
            scope["t"] = Symbol("t")

        # Parameters as Symbols
        for p in getattr(self, "parameters", {}).values():
            scope[str(p.name)] = Symbol(str(p.name))

        # Coupling terms and default local_coupling
        for ct in getattr(self, "coupling_terms", {}).keys():
            scope[str(ct)] = Symbol(str(ct))
        if "local_coupling" not in scope:
            scope["local_coupling"] = Symbol("local_coupling")

        # Derived parameters / variables / output transforms as Symbols
        for name in getattr(self, "derived_parameters", {}).keys():
            scope[str(name)] = Symbol(str(name))
        for name in getattr(self, "derived_variables", {}).keys():
            scope[str(name)] = Symbol(str(name))
        for name in getattr(self, "output_transforms", {}).keys():
            scope[str(name)] = Symbol(str(name))

        # State variables as Symbols
        for name in getattr(self, "state_variables", {}).keys():
            scope[str(name)] = Symbol(str(name))

        # Functions: undefined function heads; also add their argument symbols
        for fname, f in getattr(self, "functions", {}).items():
            scope[str(fname)] = Function(str(fname))
            for arg in getattr(f, "arguments", {}).values():
                scope[str(arg.name)] = Symbol(str(arg.name))

        if "e" not in scope:
            from sympy import E

            scope["e"] = E

        return scope

    def update_metadata(self):
        for v, eq in update_equations(self).items():
            equation = tvbo_datamodel.Equation(lhs=str(eq.lhs), rhs=str(eq.rhs))
            if v in self.state_variables:
                self.state_variables[v].equation = equation
            elif v in self.derived_variables:
                self.derived_variables[v].equation = equation
        # Build dependency order without storing state
        _ = self.get_dependency_tree()
        sort_equations(self, "derived_parameters")
        sort_equations(self, "derived_variables")
        sort_equations(self, "output_transforms")
        # sort_equations(self, "state_variables") #TODO: Test if sorting is really not necessary

    # -----------------------
    # Fluent builder helpers and setters
    # -----------------------
    @staticmethod
    def _coerce_range(domain):
        """Accept None | Range | (lo, hi[, step]) and return tvbo_datamodel.Range."""
        if domain is None or isinstance(domain, tvbo_datamodel.Range):
            return domain
        if isinstance(domain, (list, tuple)):
            if len(domain) == 2:
                lo, hi = domain
                return tvbo_datamodel.Range(lo=float(lo), hi=float(hi))
            if len(domain) == 3:
                lo, hi, step = domain
                return tvbo_datamodel.Range(
                    lo=float(lo), hi=float(hi), step=float(step)
                )
        return domain

    def _coerce_equation(self, expr, lhs: str | None = None):
        """Coerce str | sympy.Eq | sympy.Expr | tvbo_datamodel.Equation into Equation.

        Also normalizes common implicit multiplication patterns in string expressions, e.g.:
        - a(Y - X) -> a*(Y - X) when 'a' is not a known function
        - XY -> X*Y, cZ -> c*Z, 2X -> 2*X, X2 -> X*2
        """
        if isinstance(expr, tvbo_datamodel.Equation):
            return expr
        # Accept pydantic Equation as well
        if isinstance(expr, getattr(_pdm, "Equation", object)):
            return tvbo_datamodel.Equation(**expr.model_dump())
        # Accept sympy Eq/Expr
        if isinstance(expr, Eq):
            return tvbo_datamodel.Equation(lhs=str(expr.lhs), rhs=str(expr.rhs))
        return tvbo_datamodel.Equation(lhs=lhs, rhs=expr)

    # Interop with Pydantic models
    def to_pydantic(self):
        """Return a tvbopydantic.Dynamics validated instance for this model."""
        from tvbo.datamodel import tvbopydantic as _pdm

        return _pdm.Dynamics.model_validate(self._as_dict)

    @classmethod
    def from_pydantic(cls, pyd_obj):
        """Create a Dynamics from a tvbopydantic.Dynamics (or dict-like)."""
        data = pyd_obj.model_dump() if hasattr(pyd_obj, "model_dump") else dict(pyd_obj)
        inst = cls(**data)
        # Keep behavior consistent with other constructors
        inst._populate_from_ontology_if_available()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    # Parameters
    def add_parameter(
        self,
        name: str,
        value: float | None = None,
        unit: str | None = None,
        description: str | None = None,
        domain=None,
        definition: str | None = None,
        symbol: str | None = None,
    ):
        rng = self._coerce_range(domain)
        self.parameters[str(name)] = tvbo_datamodel.Parameter(
            name=str(name),
            value=value,
            unit=unit,
            description=description,
            domain=rng,
            definition=definition,
            symbol=symbol,
        )
        return self

    def update_parameters_from_equations(
        self, default_value: float = 1.0, overwrite: bool = False
    ):
        """Scan all equations and add any free symbols as parameters (default value if missing).

        - Skips symbols that are known state variables, derived variables, or function arguments
        - Skips the time symbol 't'
        - Removes any previously added parameters that later become known entities
        - Returns the list of parameter names that were added (or updated if overwrite=True)
        """
        # Gather equations as sympy Eq objects
        eqs = self.get_equations(format="dict")
        all_eqs = []
        for key in [
            "derived-parameters",
            "functions",
            "derived-variables",
            "state-equations",
            "output-transformations",
        ]:
            all_eqs.extend(eqs.get(key, []) or [])

        # Known non-parameter entities: states, derived vars, output transforms, derived parameters, function arguments, and 't'
        nonparam_known = set(map(str, self.state_variables.keys()))
        nonparam_known |= set(map(str, self.derived_variables.keys()))
        nonparam_known |= set(map(str, self.output_transforms.keys()))
        nonparam_known |= set(map(str, self.derived_parameters.keys()))
        for f in self.functions.values():
            nonparam_known |= {str(arg.name) for arg in f.arguments.values()}
        nonparam_known.add("t")

        # If any existing parameters clash with known entities, remove them (they were falsely inferred earlier)
        to_remove = [
            pname
            for pname in list(self.parameters.keys())
            if str(pname) in nonparam_known
        ]
        for pname in to_remove:
            del self.parameters[pname]

        # Known symbols also include existing parameters (after clean-up above)
        known = set(map(str, self.parameters.keys())) | nonparam_known

        # Collect all free symbols appearing in RHS (and LHS just in case)
        found = set()
        for eq in all_eqs:
            for s in getattr(eq.rhs, "free_symbols", set()):
                found.add(str(s))
            for s in getattr(eq.lhs, "free_symbols", set()):
                found.add(str(s))

        added = []
        for s in sorted(found):
            if s in known:
                continue
            if overwrite or s not in self.parameters:
                self.parameters[s] = tvbo_datamodel.Parameter(
                    name=s, value=float(default_value)
                )
                added.append(s)

        return added

    # (No generic setters/removers; callers can mutate self.parameters[...] if needed)

    # State variables
    def add_state_variable(
        self,
        name: str,
        equation=None,
        *,
        description: str | None = None,
        domain=None,
        boundaries=None,
        initial_value: float | None = 0.1,
        unit: str | None = None,
        coupling_variable: bool = False,
        stimulation_variable: bool | None = None,
        symbol: str | None = None,
    ):
        eq = (
            self._coerce_equation(equation, lhs=str(name))
            if equation is not None
            else None
        )
        self.state_variables[str(name)] = tvbo_datamodel.StateVariable(
            name=str(name),
            equation=eq,
            description=description,
            domain=self._coerce_range(domain),
            boundaries=self._coerce_range(boundaries),
            initial_value=initial_value if initial_value is not None else None,
            unit=unit,
            coupling_variable=coupling_variable,
            stimulation_variable=stimulation_variable,
            symbol=symbol,
        )
        # Automatically infer and add missing parameters referenced in the new equation
        self.update_parameters_from_equations(default_value=1.0, overwrite=False)
        return self

    # (No generic removers; callers can del self.state_variables[name] if needed)

    # Derived variables
    def add_derived_variable(
        self,
        name: str,
        expression=None,
        *,
        conditionals: list[tuple[object, object]] | None = None,
        unit: str | None = None,
        description: str | None = None,
        symbol: str | None = None,
    ):
        eq = (
            self._coerce_equation(expression, lhs=str(name))
            if expression is not None
            else tvbo_datamodel.Equation(lhs=str(name))
        )
        cases = []
        cond_blocks = []
        if conditionals:
            for expr, cond in conditionals:
                cases.append(
                    tvbo_datamodel.Case(
                        condition=str(cond),
                        equation=tvbo_datamodel.Equation(lhs=str(name), rhs=str(expr)),
                    )
                )
                cond_blocks.append(
                    tvbo_datamodel.ConditionalBlock(
                        condition=str(cond), expression=str(expr)
                    )
                )
        if cond_blocks:
            eq.conditionals = cond_blocks
        self.derived_variables[str(name)] = tvbo_datamodel.DerivedVariable(
            name=str(name),
            equation=eq,
            unit=unit,
            description=description,
            conditional=bool(cond_blocks),
            cases=cases,
            symbol=symbol,
        )
        return self

    # (No generic removers; callers can del self.derived_variables[name] if needed)

    # Functions
    def add_function(
        self,
        name: str,
        expression=None,
        *,
        arguments=(),
        description: str | None = None,
        definition: str | None = None,
    ):
        # Normalize arguments into a dict[str, Parameter]
        if isinstance(arguments, dict):
            args_dict = {
                str(k): (
                    v
                    if isinstance(v, tvbo_datamodel.Parameter)
                    else tvbo_datamodel.Parameter(name=str(k))
                )
                for k, v in arguments.items()
            }
        else:
            args = list(arguments) if isinstance(arguments, (list, tuple)) else []
            args_dict = {str(a): tvbo_datamodel.Parameter(name=str(a)) for a in args}
        eq = (
            self._coerce_equation(expression, lhs=str(name))
            if expression is not None
            else None
        )
        self.functions[str(name)] = tvbo_datamodel.Function(
            name=str(name),
            equation=eq,
            arguments=args_dict,
            description=description,
            definition=definition,
        )
        return self

    # Coupling and Output transforms
    def add_coupling_term(
        self, name: str, description: str | None = None, unit: str | None = None
    ):
        key = str(name)
        self.coupling_terms[key] = tvbo_datamodel.Parameter(
            name=key, description=description, unit=unit
        )
        # Keep parameters clean: remove any parameter with same name
        if key in self.parameters:
            del self.parameters[key]
        # Auto-mark coupling target(s): any state variable whose RHS uses this symbol

        c_sym = Symbol(key)
        eqs_state = self.get_equations(format="state-equations")
        for sv_name, eq in (eqs_state or {}).items():
            rhs = getattr(eq, "rhs", None)
            if rhs is not None and c_sym in getattr(rhs, "free_symbols", set()):
                if sv_name in self.state_variables:
                    self.state_variables[sv_name].coupling_variable = True

        return self

    def add_output_transform(
        self,
        name: str,
        expression=None,
        *,
        unit: str | None = None,
        description: str | None = None,
    ):
        eq = (
            self._coerce_equation(expression, lhs=str(name))
            if expression is not None
            else None
        )
        self.output_transforms[str(name)] = tvbo_datamodel.DerivedVariable(
            name=str(name), equation=eq, unit=unit, description=description
        )
        return self

    # Derived parameters
    def add_derived_parameter(
        self,
        name: str,
        expression=None,
        *,
        unit: str | None = None,
        description: str | None = None,
        symbol: str | None = None,
    ):
        eq = (
            self._coerce_equation(expression, lhs=str(name))
            if expression is not None
            else None
        )
        self.derived_parameters[str(name)] = tvbo_datamodel.DerivedParameter(
            name=str(name),
            equation=eq,
            unit=unit,
            description=description,
            symbol=symbol,
        )
        return self

    def plot_ontology(self, **kwargs):
        from tvbo.plot import network

        return network.plot_model(self.ontology, **kwargs)

    def render_equation(self, obj, format="latex"):
        from tvbo.export.code import render_equation

        scope = self.get_symbolic_elements()
        # Tell the printer which names are functions so it emits f(x) cleanly
        uf = {str(name): str(name) for name in getattr(self, "functions", {}).keys()}
        return render_equation(
            obj.equation, local_dict=scope, format=format, user_functions=uf
        )

    def get_equations(self, format="metadata"):
        # if format == "sympy":
        #     return simulation.equations.symbolic_model_equations(self.ontology)
        # elif format == "latex":
        #     return simulation.equations.render_latex_equations(self.ontology)

        scope = self.get_symbolic_elements()
        equations = {}
        # Determine system type (default to continuous)
        discrete = getattr(self, "system_type", "continuous") == "discrete"

        equations["derived-parameters"] = []
        for k, dp in self.derived_parameters.items():
            equations["derived-parameters"].append(
                Eq(lhs=Symbol(k), rhs=parse_eq(dp.equation, local_dict=scope))
            )

        equations["functions"] = []
        for k, f in self.functions.items():
            arguments = [Symbol(arg.name) for arg in f.arguments.values()]
            k = Function(k)(*arguments)
            equations["functions"].append(
                Eq(lhs=k, rhs=parse_eq(f.equation, local_dict=scope))
            )

        equations["derived-variables"] = []
        for k, dv in self.derived_variables.items():
            # Prefer conditionals on the Equation if present; fallback to rhs parsing
            has_conditionals = bool(getattr(dv.equation, "conditionals", None)) and (
                len(getattr(dv.equation, "conditionals", [])) > 0
            )
            if getattr(dv, "conditional", False) or has_conditionals:
                expression = simulation.equations.conditionals2piecewise(dv.equation)
            else:
                expression = parse_eq(dv.equation, local_dict=scope)

            equations["derived-variables"].append(Eq(lhs=Symbol(k), rhs=expression))

        equations["state-equations"] = []
        for k, sv in self.state_variables.items():
            t = Symbol("t")
            sv_symbol = Symbol(k)
            # Prefer conditionals on the Equation if present; fallback to rhs parsing
            has_conditionals = bool(getattr(sv.equation, "conditionals", None)) and (
                len(getattr(sv.equation, "conditionals", [])) > 0
            )
            if has_conditionals:
                expression = simulation.equations.conditionals2piecewise(sv.equation)
            else:
                expression = parse_eq(sv.equation, local_dict=scope)

            lhs_expr = sv_symbol if discrete else Derivative(sv_symbol, t)
            equations["state-equations"].append(Eq(lhs=lhs_expr, rhs=expression))

        if format == "state-equations":

            def _sv_name(_eq):
                return (
                    _eq.lhs.args[0].name
                    if isinstance(_eq.lhs, Derivative)
                    else _eq.lhs.name
                )

            return {_sv_name(_eq): _eq for _eq in equations["state-equations"]}

        equations["output-transformations"] = []
        for k, ot in self.output_transforms.items():
            equations["output-transformations"].append(
                Eq(lhs=Symbol(k), rhs=parse_eq(ot.equation, local_dict=scope))
            )
        # self.keyed_equations = equations
        if format == "dict":
            return equations

        return {
            (
                eq.lhs.name
                if isinstance(eq.lhs, Function)
                else (
                    eq.lhs.args[0].name
                    if isinstance(eq.lhs, Derivative)
                    else eq.lhs.name
                )
            ): eq
            for eq in equations["derived-parameters"]
            + equations["functions"]
            + equations["derived-variables"]
            + equations["state-equations"]
            + equations["output-transformations"]
        }

    def fill_in_equations(self, **kwargs):
        # Substitute parameters and defaults into equations (useful for fixed-point search)
        sub = self.keyed_parameters
        sub.update(kwargs)
        sub.update({"c_pop0": 0, "local_coupling": 0})
        return [eq.subs(sub) for eq in self.get_equations().values()]

    def calculate_derived_parameters(self):
        if self.derived_parameters is not None:
            for k, dp in self.derived_parameters.items():
                eq = parse_eq(
                    dp.equation,
                    local_dict=self.get_symbolic_elements(),
                    evaluate=False,
                ).subs({Symbol(p.name): p.value for p in self.parameters.values()})
                sol = eq.evalf()
                self.derived_parameters[k].value = sol

            return {
                k: self.derived_parameters[k].value for k in self.derived_parameters
            }
        else:
            return None

    def get_dependency_tree(self, ontomapping=False, include_state_equations=False):
        import sympy

        # Build dependency graph primarily for sorting derived quantities.
        # Exclude state-equations by default to avoid cycles in discrete systems
        # (e.g., algebraic dv depending on states and states depending on dv at same step).
        eqs = self.get_equations(format="dict")
        eq_list = []
        for key in ["derived-parameters", "functions", "derived-variables"]:
            eq_list.extend(eqs.get(key, []))
        if include_state_equations:
            eq_list.extend(eqs.get("state-equations", []))

        G = equations.dependency_tree(eq_list)

        if not ontomapping:
            return G

        symbol_onto_mapping = {}
        onto_symbol_mapping = {}
        for n in G.nodes:
            suffix = (
                ontology.get_model_suffix(self.ontology or self.name)
                if str(n)
                not in ["c_pop0", "c_pop1", "local_coupling", "c_loc", "c_glob"]
                else ""
            )
            if isinstance(n, sympy.core.function.Derivative):
                searchstr = f"{n.args[0]}dot{suffix}"
            else:
                searchstr = f"{n}{suffix}"
            search = ontology.intersection(
                ontology.onto.search(label=searchstr),
                ontology.onto[self.name].descendants(),
            )
            if len(search) == 1:
                ontoclass = search[0]
                symbol_onto_mapping[n] = ontoclass

        for s, onto_cls in symbol_onto_mapping.items():
            onto_symbol_mapping[onto_cls] = s

        G_onto = nx.MultiDiGraph()
        for n in G.nodes():
            if n in symbol_onto_mapping:
                G_onto.add_node(symbol_onto_mapping[n])
        for u, v in G.edges():
            if u in symbol_onto_mapping and v in symbol_onto_mapping:
                G_onto.add_edge(symbol_onto_mapping[u], symbol_onto_mapping[v])

        return G, G_onto, symbol_onto_mapping, onto_symbol_mapping

    def plot_dependency_tree(
        self,
        ax=None,
        edgecolor="#426665",
        color_nodes_by=None,
        pos="graphviz",
        edgekwargs={"connectionstyle": "arc3,rad=0", "edge_color": "grey"},
        **kwargs,
    ):

        import sympy

        from tvbo.plot import network

        if not ax:
            fig, ax = plt.subplots(figsize=(12, 2))
            return_fig = True
        else:
            return_fig = False

        G = self.get_dependency_tree(include_state_equations=True)
        if isinstance(G, tuple):
            G = G[0]

        if color_nodes_by is not None:
            G, G_onto, symbol_onto_mapping, onto_symbol_mapping = (
                self.get_dependency_tree(ontomapping=True, include_state_equations=True)
            )
            edgecolor = None
            cat_dict, categories = network.get_node_color_mapping(
                G_onto, color_nodes_by, return_categories=True
            )
            kwargs.update(
                {
                    "node_colors": [
                        cat_dict[categories[symbol_onto_mapping[n]]] for n in G.nodes
                    ]
                }
            )

        G = nx.relabel_nodes(
            G,
            {
                Symbol("local_coupling"): Symbol("c_loc"),
                Symbol("c_pop0"): Symbol("c_glob"),
            },
        )

        if pos == "graphviz":
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

            min_y = min(
                (y for key, (x, y) in pos.items() if isinstance(key, sympy.Derivative)),
                default=None,
            )
            if min_y is not None:
                pos = {
                    key: (x, min_y) if isinstance(key, sympy.Derivative) else (x, y)
                    for key, (x, y) in pos.items()
                }
        else:
            pos = nx.kamada_kawai_layout(G)

        network.draw_custom_nodes(
            G,
            pos,
            labels={node: f"${sympy.latex(node)}$" for node in G.nodes},
            ax=ax,
            alpha=1,
            facecolor="white",
            edgecolor=edgecolor,
            **kwargs,
        )

        edges = nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, **edgekwargs)
        for e in edges:
            e.set_zorder(0)
        ax.axis("off")
        ax.set_xlim([1.01 * l for l in ax.get_xlim()])

        if return_fig:
            for ax in fig.axes:
                ax.set_box_aspect(1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.tight_layout()

            plt.close()
            return fig

    def render_code(self, format="tvb", alt_label=None, **kwargs):
        self.update_metadata()

        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-model.py.mako")

        elif format in ["python", "jax-python", "python-jax"]:
            template = templates.lookup.get_template("tvbo-python-model.py.mako")

        elif format == "python-network":
            template = templates.lookup.get_template("tvbo-python-model.py.mako")
            kwargs.update({"coupling_as_argument": True})

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("tvbo-jax-dfuns.py.mako")

        elif format == "julia":
            template = templates.lookup.get_template(
                "tvbo-julia-DifferentialEquations.jl.mako"
            )
        elif format == "bifurcation-julia":
            template = templates.lookup.get_template(
                "tvbo-julia-BifurcationKit.jl.mako"
            )
        elif format == "bifurcation-numcont":
            template = templates.lookup.get_template("tvbo-numcont.py.mako")
        elif format == "bifurcation-auto7p":
            template = templates.lookup.get_template("tvbo-auto7p.py.mako")
        elif format in ["pde-fem", "pde-python", "pde"]:
            # Generic Python FEM (scikit-fem) template
            template = templates.lookup.get_template("tvbo-pde-fem.py.mako")
        else:
            raise ValueError(f"Format {format} not supported.")

        rendered_code = template.render(model=self, jax="jax" in format, **kwargs)
        return templater.format_code(rendered_code, format=format)

    def display_markdown(self, format="tvb", **kwargs):
        from IPython.display import Markdown, display

        code = templater.format_code(
            self.render_code(format=format, **kwargs), format=format
        )
        return Markdown(
            f"```{'python' if format in ['tvb', 'python', 'jax', 'autodiff'] else format}\n{code}\n```"
        )

    def execute(self, format="tvb", **kwargs):

        local_vars = {}
        if format == "tvb":
            rendered_code = clean_code(self.render_code(format=format, **kwargs))
            exec(
                rendered_code,
                templater.exec_globals,
                local_vars,
            )
            tvb_obj = local_vars[self.name](**kwargs)
            tvb_obj.title = self.label
            tvb_obj.configure()
            return tvb_obj

        elif format.lower() in ["c", "sympy2c"]:
            try:
                import importlib

                _sympy2c = importlib.import_module("sympy2c")
                Module = getattr(_sympy2c, "Module")
                OdeFast = getattr(_sympy2c, "OdeFast")
            except Exception as e:
                raise RuntimeError(
                    "sympy2c is not installed. Install it to use format='c' or 'sympy2c'."
                ) from e

            params = self.keyed_parameters
            params.update({Symbol(str(cterm)): 0.0 for cterm in self.coupling_terms})
            params.update({Symbol("local_coupling"): 0.0})

            scope = self.get_symbolic_elements()
            derived_variables = {
                Symbol(k): parse_eq(v.equation, local_dict=scope)
                for k, v in self.derived_variables.items()
            }

            lhs = list()
            rhs = list()
            for k, v in self.get_equations(format="state-equations").items():
                lhs.append(Symbol(k))
                expr = (
                    v.rhs.subs(params)
                    .subs(derived_variables)
                    .subs(derived_variables)
                    .subs(derived_variables)
                    .subs(params)
                )
                rhs.append(expr)

            module_decl = Module()
            module_decl.add(OdeFast("robertson", Symbol("t"), lhs, rhs))
            imported_module = module_decl.compile_and_load()
            return imported_module

        elif format in ["bifurcation-numcont", "bifurcation-auto7p"]:
            try:
                import importlib

                _numcont = importlib.import_module("numcont")
                cs = getattr(_numcont, "ContinuationSystem")
            except Exception as e:
                raise RuntimeError(
                    "numcont is not installed. Install it to use bifurcation formats."
                ) from e

            namespace = {"join": join, "cs": cs, "np": np}
            exec(
                self.render_code(format="bifurcation-numcont", **kwargs),
                globals(),
                namespace,
            )
            DynamicalSystem = namespace[self.name + "BifModel"]

            tempdir = tempfile.mkdtemp(prefix=self.name + "_")
            auto_files_dir = join(tempdir, "AutoFiles")
            os.makedirs(auto_files_dir, exist_ok=True)
            model_file_path = join(auto_files_dir, "model.f90")

            model_code = self.render_code("bifurcation-auto7p")
            with open(model_file_path, "w") as model_file:
                model_file.write(model_code)

            system = DynamicalSystem(
                fortran_file=join(tempdir, "AutoFiles/model"),
                data_path=join(tempdir, "AutoFiles/BifurcationData"),
            )

            return system

        else:
            rendered_code = clean_code(self.render_code(format=format, **kwargs))
            namespace = {}
            exec(clean_code(rendered_code), namespace)
            model_dfun = namespace[self.name]
            return model_dfun

    def to_lems(self, initial_conditions=1, component_id=None):
        """Build a LEMS model for this local neural mass model.

        Parameters:
        - initial_conditions: number or dict; if number, used for all SVs; if dict, keys are sv name or sv_name_0
        - component_id: optional id for the component; defaults to model label

        Returns:
        - lems.Model instance containing a ComponentType and a Component for this model
        """
        import lems.api as lems  # lazy import
        from tvbo.knowledge import ontology as _ontology  # avoid shadowing
        from tvbo.export.lemsgenerator import setup_lems_model  # lazy to avoid cycles

        model = setup_lems_model()

        def _unit_to_dimension(u: str):
            if not u:
                return "none"
            u = str(u).lower()
            if u in {
                "s",
                "sec",
                "second",
                "seconds",
                "ms",
                "millisecond",
                "millisecond(s)",
                "millisecond",
                "millisecond(s)",
            }:
                return "time"
            if u in {"v", "mv", "volt", "millivolt", "millivolts", "millivolt(s)"}:
                return "voltage"
            if u in {"a", "ma", "ampere", "amp", "milliampere", "milliamp"}:
                return "current"
            # Fallback
            return "none"

        local_ct = lems.ComponentType(
            name=self.name,
            description=(
                self.ontology.description.first()
                if self.ontology and self.ontology.description
                else None
            ),
        )
        model.add(local_ct)

        # Parameters from metadata
        for k, p in self.parameters.items():
            local_ct.add(
                lems.Parameter(
                    name=k,
                    dimension=_unit_to_dimension(getattr(p, "unit", None)),
                )
            )

        # No extra network input channel; coupling terms (e.g., c_pop0) are already in metadata

        # Coupling parameters as exposed Parameters
        cterms = _ontology.get_model_coupling_terms(self.ontology)
        p_coup_defaults = {}
        for k, cterm in cterms.items():
            p_coup_defaults[k] = 0.0
            local_ct.add(lems.Parameter(name=k, dimension="none"))
        if "local_coupling" not in cterms.keys():
            local_ct.add(lems.Parameter(name="local_coupling", dimension="none"))
            p_coup_defaults["local_coupling"] = 0.0

        # Derived variables / functions
        if self.derived_variables:
            for dp in self.derived_variables.values():
                if getattr(dp, "conditional", False):
                    cv = lems.ConditionalDerivedVariable(
                        name=dp.name,
                        dimension=_unit_to_dimension(getattr(dp, "unit", None)),
                        exposure=dp.name,
                    )
                    for case in dp.cases:
                        condition_str = (
                            None if case.condition is True else str(case.condition)
                        )
                        cv.add_case(
                            lems.Case(
                                condition=condition_str,
                                value=str(case.equation.rhs).replace("**", "^"),
                            )
                        )
                    local_ct.dynamics.add(cv)
                else:
                    local_ct.dynamics.add(
                        lems.DerivedVariable(
                            name=dp.name,
                            dimension=_unit_to_dimension(getattr(dp, "unit", None)),
                            value=str(dp.equation.rhs).replace("**", "^"),
                        )
                    )

        # Dynamics and state variables (time in milliseconds)
        local_ct.add(lems.Constant(name="ms", value="1ms", dimension="time"))
        onstart = lems.OnStart()

        if isinstance(initial_conditions, dict):
            init_conds = dict(initial_conditions)
            assign_uniform = False
        else:
            init_conds = {}
            assign_uniform = True

        for sv in _ontology.get_model_statevariables(self.ontology).values():
            sv_name = _ontology.replace_suffix(sv)
            dimension = _unit_to_dimension(
                sv.has_unit.first().label.first() if sv.has_unit.first() else None
            )
            sv_start = sv_name + "_0"

            if assign_uniform:
                init_conds[sv_start] = initial_conditions
            else:
                init_conds[sv_start] = init_conds.get(
                    sv_start, init_conds.get(sv_name, 0.0)
                )

            deriv = sv.has_derivative.first()

            local_ct.add(lems.Parameter(name=sv_start, dimension=dimension))
            local_ct.add(lems.Exposure(name=sv_name, dimension=dimension))

            local_ct.dynamics.add(
                lems.StateVariable(name=sv_name, dimension=dimension, exposure=sv_name)
            )
            # Base derivative from ontology
            base_expr = str(equations.sympify_value(deriv)).replace("**", "^")
            # Do not inject extra inputs here; global coupling is represented via coupling_terms (e.g., c_pop0)

            local_ct.dynamics.add(
                lems.TimeDerivative(
                    variable=sv_name,
                    value=f"({base_expr}) / ms",
                ),
            )

            onstart.add(lems.StateAssignment(variable=sv_name, value=sv_start))

        local_ct.dynamics.add(onstart)

        # Component instance
        parameter_values = {k: p.value for k, p in self.parameters.items()}
        parameter_values.update(init_conds)
        parameter_values.update(p_coup_defaults)
        model.add(
            lems.Component(
                id_=(component_id or (self.name or "model")),
                type_=local_ct.name,
                **parameter_values,
            )
        )

        return model

    def get_run_filename(self, format, **kwargs):
        from tvbo import tempdir

        for k in [
            "filename",
            "force",
            "periodic_orbits",
            "bif_point",
            "verbose",
            "verbosity",
        ]:
            kwargs.pop(k, None)
        kwargs = {k: kwargs[k] for k in sorted(kwargs.keys())}
        filename = join(
            tempdir,
            self.name
            + f"_format-{format}_"
            + "_".join(f"{k}-{v}" for k, v in kwargs.items()),
        )

        return filename

    def get_initial_values(self, default=0.1, random=False, N=1, **kwargs):
        if random:
            init = []
            for k, sv in self.state_variables.items():
                lo, hi = sv.domain.lo if sv.domain else -10, (
                    sv.domain.hi if sv.domain else 10
                )
                if sv.boundaries:
                    lo = max(lo, sv.boundaries.lo)
                    hi = min(hi, sv.boundaries.hi)
                if random:
                    sv_init = np.random.uniform(lo, hi, size=N)
                if random == "normal":
                    sv_init = np.random.normal(loc=(lo + hi) / 2, scale=(hi - lo) / 6)
                self.state_variables[k].initial_value = sv_init
                init.append(sv_init)
        else:
            init = [
                float(sv.initial_value) if sv.initial_value is not None else default
                for sv in self.state_variables.values()
            ]
        return np.array(init)

    def run(
        self, format="python", verbose=0, save=True, run_kwargs={}, **kwargs
    ) -> TimeSeries | BifurcationResult:
        if save:
            kwargs.update({"filename": self.get_run_filename(format=format, **kwargs)})

        if "xi" in kwargs:
            xi = kwargs.pop("xi")

        if "julia" in format:
            code = self.render_code(format=format, **kwargs)
            from tvbo.utils.julia import get_julia

            jl, Main = get_julia(compiled_modules=True)
            Main.eval(code)
            if format == "julia":
                t = Main.eval("Array(sol.t)")
                u_mat = Main.eval("Array(hcat(sol.u...))")  # states x time
                import numpy as _np

                t_arr = _np.array(t, dtype=float)
                data = _np.array(u_mat, dtype=float).T  # time x states
                data4 = data[:, :, None, None]
                labels_dimensions = {
                    "State Variable": list(self.state_variables.keys()),
                    "Region": ["Region0"],
                }
                return TimeSeries(
                    time=t_arr,
                    data=data4,
                    title=self.name,
                    sample_period=(t_arr[1] - t_arr[0]) if t_arr.size > 1 else None,
                    labels_dimensions=labels_dimensions,
                )
            elif format == "bifurcation-julia":
                import numpy as _np

                br_obj = Main.eval("bifurcation_result")
                bif_res = BifurcationResult(br=br_obj, **kwargs)
                if "periodic_orbits" in kwargs and kwargs["periodic_orbits"]:
                    po = Main.eval("po_results")

                    bif_res.periodic_orbits = [
                        BifurcationResult(br=p, **kwargs) for p in po.branches
                    ]
                return bif_res

        elif "python" == format:
            from scipy.integrate import odeint

            # Discrete-time systems: iterate map instead of integrating ODEs
            if getattr(self, "system_type", "continuous") == "discrete":
                # Initial conditions
                if "u_0" not in kwargs:
                    u_0 = self.get_initial_values(
                        random=kwargs.get("random_initial_conditions", False)
                    )
                else:
                    u_0 = kwargs.pop("u_0")

                steps = int(kwargs.pop("steps", kwargs.pop("duration", 1000)))
                dt = float(kwargs.pop("dt", 1.0))
                t = np.arange(steps) * dt

                # Build RHS expressions once
                eqs_state = self.get_equations(format="state-equations")
                state_order = list(self.state_variables.keys())
                ssyms = [Symbol(k) for k in state_order]
                rhs_exprs = [eqs_state[k].rhs for k in state_order]

                # Resolve derived parameters and variables into RHS symbolically first
                eqs_all = self.get_equations(format="dict")
                dp_eqs = eqs_all.get("derived-parameters", []) or []
                dv_eqs = eqs_all.get("derived-variables", []) or []
                dp_subs = {eq.lhs: eq.rhs for eq in dp_eqs}
                dv_subs = {eq.lhs: eq.rhs for eq in dv_eqs}
                # Apply a few rounds to cover simple dependency chains
                for _ in range(3):
                    rhs_exprs = [expr.subs(dv_subs).subs(dp_subs) for expr in rhs_exprs]

                # Parameter substitutions
                param_subs = {Symbol(p.name): p.value for p in self.parameters.values()}

                data = np.zeros((steps, len(state_order)), dtype=float)
                data[0, :] = np.asarray(u_0, dtype=float).reshape(-1)

                for i in range(1, steps):
                    # Substitute previous state and parameter numeric values
                    sub = param_subs.copy()
                    sub.update({sym: val for sym, val in zip(ssyms, data[i - 1, :])})
                    next_vals = [float(expr.subs(sub)) for expr in rhs_exprs]

                    data[i, :] = next_vals

                return TimeSeries(
                    data=data.reshape(*data.shape, 1, 1),
                    time=t,
                    labels_dimensions={"State Variable": list(self.state_variables)},
                    sample_period=dt,
                )

            if self.stimulus:
                stimulus = Stimulus(self.stimulus).execute("python")
            else:
                stimulus = None

            if stimulus and not "stimulus" in run_kwargs:
                run_kwargs.update({"stimulus": stimulus})

            model_dfun = self.execute(format=format, **kwargs)

            if "u_0" not in kwargs:
                # Initial conditions
                u_0 = self.get_initial_values(
                    random=kwargs.get("random_initial_conditions", False)
                )
            else:
                u_0 = kwargs.pop("u_0")

            if "dt" not in kwargs:
                dt = 0.1
            else:
                dt = kwargs.pop("dt")
            if "t" not in kwargs:
                duration = kwargs.pop("duration", 8000)
                t = np.arange(0, duration, dt)
            else:
                t = kwargs.pop("t")
            # Run the simulation with the updated parameters
            solution_slider = odeint(
                lambda u, t: model_dfun(u, t, **run_kwargs), u_0, t
            )

            return TimeSeries(
                data=solution_slider.reshape(*solution_slider.shape, 1, 1),
                time=t,
                labels_dimensions={"State Variable": list(self.state_variables)},
                sample_period=dt,
            )

        if format.lower() in ["c", "sympy2c"]:
            u_0 = kwargs.pop("u_0", self.get_initial_values())
            dt = kwargs.pop("dt", 0.1)
            duration = kwargs.pop("duration", 8000)
            rtol = kwargs.pop("rtol", 1e-6)
            atol = kwargs.pop("atol", 1e-6)
            T = kwargs.pop("t", np.arange(0, duration, dt, dtype=np.float64))

            compiled_module = self.execute(format=format, **kwargs)
            result, diagnostics = compiled_module.solve_fast_robertson(
                u_0, T, rtol=rtol, atol=atol
            )
            return TimeSeries(
                data=result.reshape(*result.shape, 1, 1),
                time=T,
                labels_dimensions={"State Variable": list(self.state_variables)},
                sample_period=dt,
            )
        else:
            raise ValueError(f"Format {format} not supported.")

    def add_stimulus(self, stimulus, as_derived_variable=True):

        if not any(
            [sv.stimulation_variable for sv in self.state_variables.values()]
        ) and not any(
            ["stim_t" in sv.equation.rhs for sv in self.state_variables.values()]
        ):
            print(
                "CAUTION! No state variable with attribute `stimulation_variable=True` set.\nStimulation will have no effect."
            )
        if isinstance(stimulus, Stimulus) and not as_derived_variable:
            self.stimulus = stimulus

        elif stimulus.equation is not None and as_derived_variable:
            eq, params = stimulus.get_expression()
            param_map = {k: Symbol(str(k) + "_stim") for k in params.keys()}
            params = {param_map[k]: v for k, v in params.items()}
            eq = eq.subs(param_map)
            self.derived_variables.update(
                {
                    "stim_t": tvbo_datamodel.DerivedVariable(
                        name="stim_t", equation=tvbo_datamodel.Equation(rhs=eq)
                    )
                }
            )
            self.parameters.update(
                {
                    str(k): tvbo_datamodel.Parameter(name=str(k), value=v)
                    for k, v in params.items()
                }
            )

    def find_periodic_orbits(self, f):
        # Get the directory and the basename without extension
        directory = dirname(f)
        base_name_no_ext = splitext(basename(f))[0]

        # Find all files in the directory that start with the basename
        matching_files = [
            join(directory, file)
            for file in os.listdir(directory)
            if file.startswith(base_name_no_ext + "_po") and file != basename(f)
        ]
        return matching_files

    def plot_bifurcation_timeseries(
        self,
        ICS,
        VOI,
        n_runs=2,
        t=np.arange(0, 500, 0.1),
        offset=2,
        ax1=None,
        ax2=None,
        **kwargs,
    ):
        colors = colormaps["viridis"](np.linspace(0, 1, n_runs + 1))[:-1]

        if ax1 is None:
            fig, axs = plt.subplots(ncols=2)
            ax1, ax2 = axs
            return_fig = True
        else:
            return_fig = False

        ax1.set_title("Bifurcation diagram")
        ax2.set_title("Time series")
        # Compute bifurcation result directly and plot via its method
        bif = self.run(format="bifurcation-julia", ICS=ICS, **kwargs)
        bif.plot(VOI=VOI, ICS=ICS, ax=ax1)
        y0, y1 = ax1.get_ylim()

        # Parameter sweep across continuation parameter range inferred from result
        p_min, p_max = bif.parameter.min(), bif.parameter.max()
        a_values = np.linspace(p_min, p_max, n_runs)

        for i, p in enumerate(a_values):
            self.parameters[ICS].value = p

            data = self.run(format="python", t=t)
            # exp = experiment.SimulationExperiment(
            #     id=i, model=self, connectivity={"number_of_regions": 1}
            # )
            # self.experiments.update({ICS: {p: exp}})

            # data = self.experiments[ICS][p].simulation_data
            # data = exp.run(simulation_length=simulation_length)["Raw"]

            ax2.plot(
                data.time,
                data.get_state_variable(VOI).data.squeeze(),
                alpha=0.5,
                color=colors[i],
                label=f"{p:.2f}",
            )

            ax1.vlines(
                p, y0, y1, color=colors[i], alpha=0.5, linestyle="--", linewidth=0.5
            )
            ax1.annotate(
                f"{p:.2f}",
                xy=(p, y1),
                xytext=(p, y1),
                # arrowprops=dict(facecolor="black", shrink=0.05),
                ha="left",
                fontsize=8,
            )
        ax2.set_xlabel("Time in ms")
        ax2.set_ylabel(f"${latex(Symbol(VOI))}$")

        ax2.legend(
            title=ICS,
            handlelength=0.5,
            fontsize=8,
            handletextpad=0.5,
            loc="lower right",
        )
        ax1.legend(
            handlelength=1,
            fontsize=8,
            handletextpad=0.5,
            loc="lower right",
        )
        ax1.set_ylim(y0, y1)

        if return_fig:
            for ax in fig.axes:
                ax.set_box_aspect(1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.tight_layout()

            plt.close()
            return fig

    def parameter_table(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "Parameter": p.name,
                    "Value": p.value,
                    "Description": p.description,
                }
                for p in self.parameters.values()
            ]
        )
        return df

    def save_model_metadata(self, filename):
        from linkml_runtime.dumpers import yaml_dumper

        yaml_dumper.dump(self, filename)

    def save_python_class(self, directory="."):
        fpath = join(directory, f"{self.name}.py")
        with open(fpath, "w") as f:
            f.write(
                """
import math
import numpy as np
from tvb.simulator.models.base import ModelNumbaDfun, Model
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final"""
            )
            f.write(self.render_code())

    def generate_report(
        self, format="markdown", template_name="tvbo-report-model", outputfile=None
    ):
        self.update_metadata()
        if format in ["markdown", "pdf"]:
            template = templates.lookup.get_template(f"{template_name}.md.mako")
        elif format == "html":
            template = templates.lookup.get_template(f"{template_name}.html.mako")

        render = (
            template.render(model=self)
            .replace(r"\mathcal{lo}_{coupling}", "c_{local}")
            .replace("c_{pop0}", "c_{global}")
        )

        if outputfile:
            if format == "pdf":
                report.to_pdf(render, outputfile)
            else:
                with open(outputfile, "w") as f:
                    f.write(render)

        return render

    def save_report(self, opath, format="markdown"):
        self.report_path = opath
        if format == "markdown":
            extension = "md"
        elif format == "latex":
            extension = "tex"
        else:
            extension = format

        with open(join(opath, f"{self.name}." + extension), "w") as f:
            f.write(self.generate_report(format=format))


class Model(Dynamics):
    def __init__(self, name, ontology=None, metadata=None, **kwargs):
        super().__init__(name=name, **kwargs)
