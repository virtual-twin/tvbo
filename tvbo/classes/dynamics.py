#
# Module: localdynamics.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""Python behaviour layer for `Dynamics` models.

Defines [`DynamicalSystem`](#tvbo.classes.dynamics.DynamicalSystem) — the base
class that augments the generated LinkML `Dynamics` datamodel with model
construction and ontology resolution, a symbolic (SymPy) representation,
equation normalization and dependency-ordered sorting, multi-backend code
generation, simulation and bifurcation runs, plotting, and report export —
together with the `Model` and `Dynamics` convenience subclasses.
"""

import logging
import copy as _copy
import functools
import os
import re
import tempfile
from os.path import basename, dirname, join, splitext
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import owlready2
from tvbo.utils import initial_value, yaml_loader
from matplotlib import colormaps
from sympy import Derivative, Eq, Function, Symbol, latex, symbols
from sympy.core.sympify import SympifyError

from tvbo import templates
from tvbo.analysis import BifurcationResult
from tvbo.classes import equation as _equation_mod
from tvbo.classes.perturbation import Stimulus
from tvbo.codegen import templater
from tvbo.data.types import TimeSeries
from tvbo.datamodel import pydantic as _pdm
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.datamodel.schema import Case, ConditionalBlock, DerivedVariable, Equation
from tvbo.ontology import owl as ontology
from tvbo.ontology import query
from tvbo.parse.expression import parse_eq
from tvbo.utils import report

logger = logging.getLogger(__name__)

TEMPLATES = templates.root

@functools.cache
def _available_neural_mass_models():
    """The ontology's neural-mass-model classes, resolved and memoised on first use.

    Kept lazy so importing this module (through ``import tvbo``) does not force the
    ontology to load — it is only needed to validate/build models from the ontology.
    """
    return set(ontology.get_models().values())


def __getattr__(name):  # PEP 562: keep ``available_neural_mass_models`` importable, lazily.
    if name == "available_neural_mass_models":
        return _available_neural_mass_models()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


## BifurcationResult moved to tvbo.analysis.bifurcation


def clean_code(code):
    """Replace Unicode infinity (`∞`) with the Python literal `inf`.

    Generated model code occasionally carries the ∞ glyph from upstream
    ontology labels; SymPy and most backends can't parse it.
    """
    cleaned_code = re.sub(r"∞", "inf", code)
    return cleaned_code


def _normalize_conditionals(model):
    """Ensure dv.equation.conditionals is populated for all conditional DVs.

    If dv.cases is populated but dv.equation.conditionals is empty,
    copy the cases into equation.conditionals as ConditionalBlock objects
    and build the Piecewise rhs string. This makes dv.equation.conditionals
    the single canonical location for conditional data.

    dv.cases is deprecated — new models should define conditionals
    directly on the equation.
    """
    for dv in model.derived_variables.values():
        cases = getattr(dv, "cases", None)
        if not cases:
            continue
        # Already normalized — skip
        if (
            getattr(dv.equation, "conditionals", None)
            and len(dv.equation.conditionals) > 0
        ):
            continue
        # Populate equation.conditionals from dv.cases
        dv.equation.conditionals = [
            ConditionalBlock(condition=case.condition, expression=case.equation.rhs)
            for case in cases
        ]
        # Mark the DV as conditional if not already
        if not getattr(dv, "conditional", False):
            dv.conditional = True


def _migrate_coupling_terms(model):
    """Bidirectional sync between coupling_terms and coupling_inputs.

    coupling_terms (dict[str, Parameter]) is deprecated in favor of
    coupling_inputs (dict[str, CouplingInput]).  This function:
    1. Copies coupling_terms entries into coupling_inputs (forward migration)
    2. Copies coupling_inputs entries back into coupling_terms as Parameters
       (backward compat for templates that still read coupling_terms)
    """
    ct = getattr(model, "coupling_terms", None) or {}
    getattr(model, "coupling_inputs", None) or {}

    # Forward: coupling_terms → coupling_inputs
    if ct:
        for name, param in ct.items():
            if name not in model.coupling_inputs:
                model.coupling_inputs[name] = tvbo_datamodel.CouplingInput(
                    name=str(name),
                    description=getattr(param, "description", None),
                )

    # Backward: coupling_inputs → coupling_terms (for template compat)
    if model.coupling_inputs and model.coupling_terms is not None:
        for name, ci_obj in model.coupling_inputs.items():
            if name not in model.coupling_terms:
                model.coupling_terms[name] = tvbo_datamodel.Parameter(
                    name=str(name),
                    description=getattr(ci_obj, "description", None),
                )


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


def class2metadata(ontoclass: Any, metadata: Any):
    """Populate a `Dynamics` metadata object from an owlready2 ontology class.

    Fills in description, state variables (with equations, boundaries, and
    coupling-variable flags), derived variables, and parameters by querying
    the TVB-O ontology for the corresponding semantic annotations.

    Args:
        ontoclass: The owlready2 class to read from.
        metadata: The `Dynamics` instance to populate in place.
    """
    if not metadata.description:
        metadata.description = ontology.get_def(ontoclass, mode="long")
    dependent_equations = _equation_mod.sort_equations_by_dependencies(
        _equation_mod.symbolic_model_equations(ontoclass)
    )
    state_variables = order_by_equations(
        ontology.get_model_statevariables(ontoclass), dependent_equations
    )
    state_variables = ontology.get_model_statevariables(ontoclass)

    functions = order_by_equations(
        ontology.get_model_functions(ontoclass), dependent_equations
    )

    for k, v in state_variables.items():
        range = ontology.get_range(v)

        if v.stateVariableBoundaries:
            boundary = ontology.get_range(v.stateVariableBoundaries.first())
            boundaries = tvbo_datamodel.Range(lo=boundary[0], hi=boundary[1])
        else:
            boundaries = None

        # Preserve the descriptive stateVariableRange (IC-sampling support) as the
        # sampling distribution when a clamp exists — mirrors the file/adapter paths.
        # (Previously `range` was computed but dropped, so it was lost on round-trip.)
        sv_range = (
            tvbo_datamodel.Range(lo=float(range[0]), hi=float(range[1]))
            if range and range[0] is not None and range[1] is not None
            else None
        )
        _sv_domain, _sv_distribution = _fold_range_boundaries(sv_range, boundaries)

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
                        domain=_sv_domain,
                        distribution=_sv_distribution,
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
                # An ontology-declared clamp (stateVariableBoundaries) is the
                # operative constraint, so it wins over a pre-existing
                # descriptive (unenforced) domain — consistent with how the
                # file loader folds boundaries → domain+enforce=clamp. The
                # descriptive range is kept as the sampling distribution (a
                # pre-existing distribution takes precedence).
                "domain": _sv_domain or state_var.domain,
                "distribution": state_var.distribution or _sv_distribution,
                "coupling_variable": state_var.coupling_variable
                or (v in ontoclass.has_cvar),
            }

            for attr, value in updates.items():
                setattr(state_var, attr, value)

    # Update parameters AFTER state_variables are populated so that
    # update_parameters can parse their equations and determine which
    # ontology parameters are actually used.
    update_parameters(metadata, ontoclass)

    # Collect all free symbols from state variable equations
    # Build a local_dict to avoid sympy interpreting 'I' as imaginary unit, etc.
    from sympy import Symbol, sympify

    # First pass: collect all known symbol names to build local_dict
    known_names = (
        set(metadata.parameters.keys())
        | set(metadata.state_variables.keys())
        | set(metadata.derived_variables.keys())
        | set(metadata.derived_parameters.keys())
        | set(functions.keys())
    )
    # Add coupling terms from ontology
    onto_coupling_terms = ontology.get_model_coupling_terms(
        ontoclass, only_global=False
    )
    known_names |= set(onto_coupling_terms.keys())

    local_dict = {name: Symbol(name) for name in known_names}

    required_symbols = set()
    for sv in metadata.state_variables.values():
        if sv.equation and sv.equation.rhs:
            try:
                eq = sympify(sv.equation.rhs, locals=local_dict)
                required_symbols.update(str(s) for s in eq.free_symbols)
            except Exception:
                pass

    # Already defined symbols (no need to fetch from ontology)
    defined_symbols = (
        set(metadata.parameters.keys())
        | set(metadata.state_variables.keys())
        | set(metadata.derived_variables.keys())
        | set(metadata.derived_parameters.keys())
        | set(metadata.output.keys() if isinstance(metadata.output, dict) else [])
    )

    # Only add ontology functions if they are required but not yet defined
    for k, v in functions.items():
        if k in required_symbols and k not in defined_symbols:
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
        name = ontology.replace_suffix(condpar)
        # Only add conditional if it's required but not yet defined
        if name in required_symbols and name not in defined_symbols:
            val = _equation_mod.sympify_value(condpar)
            metadata.derived_variables.update(
                {
                    name: DerivedVariable(
                        name=name,
                        symbol=condpar.symbol.first(),
                        conditional=True,
                        cases=[
                            Case(
                                condition=condition,
                                equation=Equation(lhs=name, rhs=expr),
                            )
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

    # Only add ontology coupling terms if they are required in state equations
    # (onto_coupling_terms was fetched earlier for building local_dict)
    # Store them in coupling_inputs (canonical) with fallback to coupling_terms
    # for backward compat until coupling_terms is fully removed from schema.
    ci_dict = metadata.coupling_inputs
    ct_dict = metadata.coupling_terms
    for k, v in onto_coupling_terms.items():
        if k in required_symbols and k not in ci_dict and k not in ct_dict:
            metadata.coupling_inputs[k] = tvbo_datamodel.CouplingInput(name=k)

    for r in ontoclass.has_reference:
        if r.name not in metadata.references:
            metadata.references.append(r.name)


def update_parameters(metadata, ontoclass, verbose=0, only_used=True, **kwargs):
    """
    Update parameters from ontology.

    Parameters
    ----------
    metadata : Dynamics
        Model metadata to update
    ontoclass : owlready2.ThingClass
        Ontology class
    verbose : int
        Verbosity level
    only_used : bool
        If True (default), only add parameters that are referenced in equations.
        If False, add all parameters from ontology (legacy behavior).
    **kwargs : dict
        Parameter overrides
    """
    # Collect all symbols used in equations if only_used=True
    used_symbols = set()
    if only_used:
        from sympy.parsing.sympy_parser import parse_expr

        # Build local_dict so that ALL names in the model are treated as plain
        # Symbols.  Without this, sympy interprets 'e' as Euler's number (E)
        # and 'I' as the imaginary unit, so they never appear as free_symbols
        # and the corresponding parameters are silently dropped.
        all_names: set[str] = set()
        eq_dicts = [
            getattr(metadata, "parameters", {}),
            getattr(metadata, "state_variables", {}),
            getattr(metadata, "derived_variables", {}),
            getattr(metadata, "derived_parameters", {}),
        ]
        for eq_dict in eq_dicts:
            all_names.update(str(k) for k in eq_dict.keys())
        # Also include ontology parameter labels so they aren't shadowed
        for k in ontology.get_default_values(ontoclass, class_as_key=True):
            label = ontology.replace_suffix(k.label.first())
            all_names.add(label)
            all_names.update(k.synonym + k.symbol)
        local_dict = {n: Symbol(n) for n in all_names}

        for eq_dict in eq_dicts:
            for item in eq_dict.values():
                if (
                    hasattr(item, "equation")
                    and item.equation
                    and hasattr(item.equation, "rhs")
                ):
                    expr = parse_expr(str(item.equation.rhs), local_dict=local_dict)
                    used_symbols.update(str(s) for s in expr.free_symbols)

    for k, v in ontology.get_default_values(ontoclass, class_as_key=True).items():
        label = ontology.replace_suffix(k.label.first())

        # Skip if only_used=True and this parameter isn't referenced in equations
        if only_used and label not in used_symbols:
            # Also check synonyms/symbols
            if not any(syn in used_symbols for syn in (k.synonym + k.symbol)):
                continue

        if range := ontology.get_range(k):
            lo, hi, step = range
            domain = tvbo_datamodel.Range(lo=lo, hi=hi, step=step)
        else:
            domain = None

        if label not in metadata.parameters and not any(
            synonym in metadata.parameters for synonym in k.synonym + k.symbol
        ):
            if verbose > 0:
                logger.debug("using parameter %s from the ontology", label)
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


# When False (default), authored equation term order is preserved end-to-end
# (parse unevaluated + stringify order='none'); set True to restore SymPy's
# canonical Add/Mul re-sorting. Generated dynamics then read like the source.
REORDER_EQUATIONS = False


def update_equations(model):
    """Normalize equation symbols on *model* (in place).

    Builds a substitution map that rewrites raw RHS strings into canonical
    SymPy form: `*_dot` / `dot*` names become time derivatives, derived
    variables are inlined, and Heaviside / acronym placeholders are resolved.
    """
    _evaluate = REORDER_EQUATIONS
    substitutions = {}

    t = symbols("t")
    equations = model.get_equations(evaluate=_evaluate)

    for k, eq in model.get_equations(evaluate=_evaluate).items():
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

        # Coupling inputs (and time ``t``) are defined by the model spec, not missing
        # specifications — excluding them keeps a fully-specified model from reaching into
        # the ontology (an expensive, load-triggering lookup) just to resolve a symbol that
        # is already known. The ontology is consulted only for genuinely unresolved symbols.
        missing_symbols = [
            s
            for s in eq.free_symbols
            if str(s) not in model.parameters
            and str(s) not in model.state_variables
            and str(s) not in (model.coupling_inputs or {})
            and str(s) != "t"
            and str(s) not in (model.derived_variables or {})
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
                            [
                                ontology.replace_suffix(lbl) == str(s)
                                for lbl in labelsearch
                            ]
                        ]
                    )

                if not labelsearch:
                    # The suffix filter above can drop every candidate when no
                    # label matches the symbol exactly; nothing to substitute.
                    continue

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
        """Rewrite each variable's equation with `substitutions` and store it back.

        Iterates `metadata_dict` (state or derived variables), resolves each
        entry's equation (from `equations` or the entry's own `equation`),
        applies the Symbol→Symbol `substitutions` while preserving authored
        term order, and writes the resulting `sympy.Eq` back into `equations`
        keyed by variable name.

        Args:
            metadata_dict: Mapping of variable name to its schema object.
            substitutions: Symbol→Symbol replacement map applied to each RHS.
            equations: Equation store, read for existing entries and updated
                in place.
            time_derivative: If `True`, build the LHS as a time derivative of
                the variable rather than the bare symbol.
        """
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

            # xreplace + order='none' preserve authored term order (substitutions
            # is Symbol->Symbol, so this matches subs but does not re-canonicalize)
            eq_sub = eq.xreplace(substitutions)

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


def stable_rhs(stored: Any, expression: Any, scope: Mapping, serialise: Callable) -> str:
    """The stored right-hand side if it already denotes `expression`, else a fresh one.

    Serialising is not a fixed point. SymPy's `StrPrinter` renders an unevaluated
    ``Mul(Integer(-1), Float(3.5))`` as ``- 1*3.5``, which parses back to a differently
    nested product that prints as ``- 3.5`` — so the round trip oscillates with period two.
    `Dynamics.update_metadata` re-serialises every equation and runs on every
    `render_code`, so an equation flipped between two spellings and the emitted code
    depended on how many times the model had been rendered.

    Re-serialising an expression the stored text already denotes cannot add information,
    only spelling, so the stored text wins and normalisation becomes idempotent.
    """
    if stored:
        try:
            if parse_eq(str(stored), local_dict=scope, evaluate=False) == expression:
                return str(stored)
        except (SympifyError, TypeError, ValueError, AttributeError, SyntaxError):
            pass
    return serialise(expression)


def sort_equations(model: Any, variable_type: str):
    """Reorder `model[variable_type]` by topological dependency order, in place.

    Resolves the model's equation dependency DAG and reorders the variables
    so each equation appears after the variables it references — required by
    backends that emit straight-line code (JAX, NumPy printers).

    Args:
        model: The dynamics model whose equations should be sorted.
        variable_type: Attribute name — typically `"state_variables"`,
            `"derived_variables"`, or `"functions"`.
    """
    # Skip sorting for list format (e.g., output as list of names)
    if isinstance(model[variable_type], list):
        return

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

    # Variables the dependency graph does not mention keep the order they came in and stay
    # ahead of the sorted ones. Prepending them one at a time instead reverses them, and
    # because each call re-sorts whatever order the last one left behind, that made the
    # result alternate: rendering a model twice emitted its derived variables in opposite
    # orders, so generated code depended on how often it had been generated before.
    sorted_variables_metadata = {**original_metadata, **sorted_variables_metadata}

    # Update the original dictionary in-place
    model[variable_type].clear()
    model[variable_type].update(sorted_variables_metadata)


# Slot aliases: YAML keys that map to canonical slot names.
# Keeps YAML files readable (e.g. ``components:`` instead of ``modes:``) while
# the datamodel uses a single canonical attribute.
_DYNAMICS_SLOT_ALIASES = {
    "components": "modes",
}


def _clamp_domain(rng):
    """Mark a bounds Range as a hard clamp (``enforce='clamp'``) and return it.

    Folds a legacy ``boundaries`` Range into the unified ``domain`` representation:
    ``boundaries`` always meant "clamp the trajectory to [lo, hi]", which is now
    expressed as a ``domain`` with ``enforce='clamp'``. Returns None unchanged.
    """
    if rng is None:
        return None
    try:
        rng.enforce = "clamp"
    except (AttributeError, ValueError):
        pass
    return rng


def _fold_range_boundaries(rng, boundaries):
    """Fold a descriptive range + hard-clamp boundaries into ``(domain, distribution)``.

    Mirrors ``adapters.tvb`` so ontology / programmatic imports match file ingestion:
    the clamp (``boundaries``) becomes the enforced ``domain``; the descriptive range
    (the IC-sampling support) is preserved as the sampling ``distribution`` — but only
    when it differs from the clamp, since an identical clamp already conveys it. With no
    clamp, the descriptive range is the (unenforced) ``domain`` and there is no separate
    distribution. ``rng``/``boundaries`` are ``Range`` objects or ``None``.
    """
    if boundaries is None:
        return rng, None
    domain = _clamp_domain(boundaries)
    distribution = None
    if rng is not None and (rng.lo, rng.hi) != (boundaries.lo, boundaries.hi):
        distribution = tvbo_datamodel.Distribution(domain=rng)
    return domain, distribution


def _fold_component_alias(d: dict) -> None:
    """Recursively rename the Dynamics-only ``components`` → ``modes`` slot alias.

    ``components`` is a ``modes`` alias only inside a Dynamics, so it is folded here
    (and by the class-scoped fold in the loader) rather than anywhere a ``components``
    key appears. Mutates ``d`` in place at every nesting level.
    """
    for alias, canonical in _DYNAMICS_SLOT_ALIASES.items():
        if alias in d:
            if canonical in d:
                raise ValueError(
                    f"Cannot specify both '{alias}' and '{canonical}' — "
                    f"'{alias}' is an alias for '{canonical}'."
                )
            d[canonical] = d.pop(alias)
    modes = d.get("modes")
    if isinstance(modes, dict):
        for v in modes.values():
            if isinstance(v, dict):
                _fold_component_alias(v)


def _resolve_dynamics_aliases(d: dict) -> dict:
    """Normalize a Dynamics kwargs/metadata dict through the SINGLE shared route.

    Every construction path — ``Dynamics(**dict)``, ``from_file``, ``from_string``,
    the ``iri`` backfill, and the network/experiment coercion helpers — funnels
    through here, so they apply identical conveniences and cannot drift:

    * the Dynamics-specific ``components`` → ``modes`` alias (recursively), then
    * :func:`tvbo.utils.yaml_loader._normalize_loaded` — the one implementation
      shared with the LinkML ``load``/``loads``/``load_as_dict`` path: the aliases
      ``Dynamics`` declares, the legacy ``boundaries``/``range`` → ``domain`` fold (``boundaries``
      gaining ``enforce: clamp``; a co-existing descriptive ``domain`` preserved as
      the IC-sampling ``distribution``), and the terse ``distribution: {lo, hi}``
      lift. A bare ``domain`` is left untouched (``enforce`` defaults to ``none``),
      so clamping stays opt-in.

    ``_normalize_loaded`` rebuilds mappings, so the normalized content is written
    back into ``d`` in place (``clear`` + ``update``) to honour the in-place
    contract the coercion callers rely on; ``d`` is also returned for convenience.
    """
    _fold_component_alias(d)
    normalized = yaml_loader._normalize_loaded(d)
    if normalized is not d:
        d.clear()
        d.update(normalized)
    return d


def _validate_dynamics_kwargs(kwargs: dict) -> None:
    """Validate Dynamics kwargs and provide helpful error messages for schema mistakes.

    Common mistakes:
    - Using 'output' as a dict of derived variables (should be 'derived_variables')
    - Using raw dicts instead of LinkML loader (should use Dynamics.from_string())
    """
    # Resolve slot aliases recursively (e.g. components → modes)
    _resolve_dynamics_aliases(kwargs)

    # Check if 'output' is misused as derived variables
    output = kwargs.get("output")
    if output is not None and isinstance(output, dict):
        # Check if it looks like derived variable definitions
        first_val = next(iter(output.values()), None)
        if isinstance(first_val, dict) and (
            "equation" in first_val or "rhs" in first_val
        ):
            raise ValueError(
                "'output' should be a list of variable names, not variable definitions. "
                "Did you mean 'derived_variables'?\n\n"
                "Change:\n"
                "  output:\n"
                "    x:\n"
                "      equation: ...\n\n"
                "To:\n"
                "  derived_variables:\n"
                "    x:\n"
                "      equation: ...\n"
                "  output: [x]  # optional: list of outputs to include"
            )


class DynamicalSystem(tvbo_datamodel.Dynamics):
    """Enhanced base class for `Dynamics` adding Python-side behaviour.

    Wraps the generated LinkML `Dynamics` datamodel with the methods that
    make a model usable: ontology resolution (`use_ontology=True`), symbolic
    representation via SymPy, equation reordering, backend code generation,
    YAML / JSON / Pydantic round-tripping, and matplotlib plotting hooks.

    Most users should construct via [`Dynamics`](#tvbo.classes.dynamics.Dynamics)
    or `Dynamics.from_db(name)` — this class is the implementation base.
    """

    def __init__(
        self,
        name="Dynamics",
        _skip_ontology: bool = False,
        use_ontology: bool = False,
        **kwargs,
    ):
        iri = kwargs.get("iri")
        if iri and (name is None or name == "Dynamics"):
            from tvbo.data.registry import resolve, local_name
            from tvbo.utils import deep_merge

            local = local_name(iri)
            try:
                loaded = yaml_loader.load_as_dict(str(resolve("Dynamics", local)))
                _resolve_dynamics_aliases(loaded)
                # Registry entry is the base; inline kwargs override at the leaf
                # (e.g. parameters.a.value wins, siblings kept from the entry).
                merged = deep_merge(loaded, kwargs)
                kwargs.clear()
                kwargs.update(merged)
                name = kwargs.pop("name", local)
                _skip_ontology = True
            except (FileNotFoundError, RuntimeError):
                pass

        if name is not None:
            kwargs["name"] = str(name)

        # Validate common schema mistakes before LinkML processing
        _validate_dynamics_kwargs(kwargs)

        # Initialize datamodel (base class sets up empty containers)
        super().__init__(**kwargs)

        # Skip ontology lookup when model is fully specified (e.g., from PyRates import)
        if _skip_ontology:
            return

        # Auto-populate only when a name was provided; keep default Dynamics() empty
        if name != "Dynamics":
            # Opt-in: resolve ontology class by name and backfill missing fields
            if use_ontology:
                self._populate_from_ontology_by_name()

            # Finalize metadata/equations
            self.update_metadata()
            self.calculate_derived_parameters()

    @property
    def components(self):
        """Alias for ``modes`` — sub-dynamics contained in this model."""
        return self.modes

    @components.setter
    def components(self, value):
        """Set the sub-dynamics (`modes`) contained in this model."""
        self.modes = value

    # Factory constructors
    @classmethod
    def from_datamodel(
        cls, model_meta: tvbo_datamodel.Dynamics, use_ontology: bool = False
    ):
        """Create from a datamodel Dynamics instance by copying its
        already-normalized state (avoids ``_as_dict`` re-init crash on
        ``inlined_as_dict`` fields)."""
        inst = cls.__new__(cls)
        inst.__dict__.update(model_meta.__dict__)
        if use_ontology:
            inst._populate_from_ontology_by_name()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_ontology(cls, ontoclass: owlready2.ThingClass | str, **kwargs):
        """Create a model populated from an ontology class.

        Args:
            ontoclass: An owlready2 model class, or a label string that is
                resolved against the `NeuralMassModel` ontology branch.
            **kwargs: Extra fields forwarded to the constructor and ontology
                population.

        Returns:
            A populated instance with metadata and derived parameters
            finalized.
        """
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
    def from_file(
        cls, path: str | os.PathLike, use_ontology: bool = False
    ) -> "Dynamics":
        """Load a model from a YAML/JSON specification file on disk.

        Args:
            path: Path to a TVBO model specification file.
            use_ontology: If `True`, backfill missing fields from the
                ontology after loading.

        Returns:
            The instance parsed from the file.
        """
        data = yaml_loader.load_as_dict(str(path))
        _resolve_dynamics_aliases(data)
        inst = cls(**data)
        if use_ontology:
            inst._populate_from_ontology_by_name()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_string(cls, str: str, use_ontology: bool = False) -> "Dynamics":
        """Load a model from a YAML specification string.

        Args:
            str: A YAML document describing the model.
            use_ontology: If `True`, backfill missing fields from the
                ontology after parsing.

        Returns:
            The instance parsed from the string.
        """
        data = yaml_loader.load_as_dict(str) or {}
        _resolve_dynamics_aliases(data)
        inst = cls(**data)
        if use_ontology:
            inst._populate_from_ontology_by_name()
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    # ── Platform retrieval ────────────────────────────────────────

    TVBO_PLATFORM_URL = "https://tvbo.charite.de"

    @classmethod
    def from_platform(
        cls,
        name: str,
        base_url: str = TVBO_PLATFORM_URL,
    ) -> "Dynamics":
        """Load a dynamics model from the tvbo platform API.

        Fetches the full LinkML-valid YAML definition from the platform
        and constructs a Dynamics instance.

        Parameters
        ----------
        name : str
            Model name (e.g., "JansenRit", "ReducedWongWang").
        base_url : str
            Platform base URL.

        Returns
        -------
        Dynamics
            Dynamics instance loaded from the platform.
        """
        import requests

        api = f"{base_url.rstrip('/')}/api/v1/dynamics"
        resp = requests.get(f"{api}/{name}/sidecar", params={"format": "yaml"})
        resp.raise_for_status()
        return cls.from_string(resp.text)

    @classmethod
    def list_platform_models(cls, base_url: str = TVBO_PLATFORM_URL, **filters) -> list:
        """List available dynamics models on the tvbo platform.

        Parameters
        ----------
        base_url : str
            Platform base URL.
        **filters
            Filtering parameters (e.g., system_type="continuous").

        Returns
        -------
        list[dict]
            List of model summaries.
        """
        import requests

        api = f"{base_url.rstrip('/')}/api/v1/dynamics"
        resp = requests.get(api, params=filters)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def from_pyrates(cls, path: str, operator_key: str | None = None) -> "Dynamics":
        """Load a Dynamics model from a PyRates YAML template file.

        Parameters
        ----------
        path : str
            Path to PyRates YAML file.
        operator_key : str, optional
            Name of the specific OperatorTemplate to load (without _op suffix).
            If None, loads the first OperatorTemplate found.
            Use SimulationExperiment.from_pyrates() to load all operators.

        Returns
        -------
        Dynamics
            New Dynamics instance populated from the PyRates template.

        Example
        -------
        >>> model = Dynamics.from_pyrates("jansen_rit.yaml")
        >>> # Load specific operator from multi-operator file
        >>> tsodyks = Dynamics.from_pyrates("synaptic_plasticity.yaml", operator_key="tsodyks")
        """
        from tvbo.codegen.pyrates import from_pyrates_yaml

        data = from_pyrates_yaml(path, operator_key=operator_key)
        # Skip ontology lookup - PyRates YAML provides complete model definition
        inst = cls(_skip_ontology=True, **data)
        # Only calculate derived parameters if needed
        if inst.derived_parameters:
            inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_db(cls, name: str) -> "Dynamics":
        """Load a Dynamics model by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Dynamics", name)))

    @classmethod
    def list_db(cls, model_type: str | None = None) -> list[str]:
        """List available models in the tvbo database.

        Parameters
        ----------
        model_type : str, optional
            Filter by model category.  Valid values: ``mean_field``,
            ``neural_mass``, ``phase_oscillator``, ``phenomenological``,
            ``spiking``, ``generic``, ``field``.

        Examples
        --------
        >>> Dynamics.list_db()                           # all models
        >>> Dynamics.list_db(model_type='mean_field')    # mean-field only
        >>> Dynamics.list_db(model_type='spiking')       # spiking models
        """
        from tvbo.data.registry import list_entries, list_entries_with_metadata

        if model_type is None:
            return list_entries("Dynamics")
        rows = list_entries_with_metadata("Dynamics")
        return sorted(r["name"] for r in rows if r.get("model_type") == model_type)

    @classmethod
    def db_overview(cls, model_type: str | None = None):
        """Return a pandas DataFrame summarising the Dynamics database.

        Columns: ``name``, ``model_type``, ``system_type``, ``description``.

        Parameters
        ----------
        model_type : str, optional
            If given, only show models of that category.

        Examples
        --------
        >>> Dynamics.db_overview()
        >>> Dynamics.db_overview(model_type='neural_mass')
        """
        import pandas as pd

        from tvbo.data.registry import list_entries_with_metadata

        rows = list_entries_with_metadata("Dynamics")
        if model_type is not None:
            rows = [r for r in rows if r.get("model_type") == model_type]
        df = (
            pd.DataFrame(
                [
                    {
                        "name": r.get("name", ""),
                        "model_type": r.get("model_type", ""),
                        "system_type": r.get("system_type", "continuous"),
                        "description": (r.get("description") or "")[:80],
                    }
                    for r in rows
                ]
            )
            .sort_values(["model_type", "name"])
            .reset_index(drop=True)
        )
        return df

    # -------  Ontology enrichment  -------

    def enrich_from_ontology(self):
        """Explicitly enrich this model from the ontology by name.

        Looks up the model name in the TVB ontology and backfills missing
        parameter values, descriptions, ranges, state-variable metadata, and
        derived variables.  Useful when you define a partial model spec and
        want the ontology to fill in the gaps.

        Example
        -------
        >>> d = Dynamics.from_string(partial_spec)
        >>> d.enrich_from_ontology()  # fill in defaults from the knowledge base
        """
        self._populate_from_ontology_by_name()
        self.update_metadata()
        self.calculate_derived_parameters()
        return self

    # Internal helpers
    def _populate_from_ontology_by_name(self):
        """Resolve the ontology class by name and populate fields from it."""
        oc = self.ontology  # auto-discover by name (read-only property)
        if oc is not None:
            self._populate_from_ontology(oc)

    def _populate_from_ontology(self, oc, **kwargs):
        # Fill schema fields from ontology, without persisting runtime-only state
        class2metadata(oc, self)
        update_parameters(self, oc, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name} - {len(self.parameters)} parameters and {len(self.state_variables)} state variables"

    def to_yaml(self, filepath: str | None = None, format: str = "tvbo") -> str:
        """Export the model to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to write the YAML file. If None, returns the YAML string.
        format : str
            Output format: "tvbo" (default) or "pyrates".
            PyRates format generates a complete experiment YAML (model + network).

        Returns
        -------
        str
            YAML string or filepath if written to file.

        Example
        -------
        >>> model.to_yaml("model.yaml")  # TVBO format
        >>> model.to_yaml("model.yaml", format="pyrates")  # PyRates experiment format
        """
        if format.lower() == "pyrates":
            from tvbo.codegen.pyrates import to_pyrates_yaml_string

            return to_pyrates_yaml_string(dynamics=self, filepath=filepath)
        else:
            from tvbo.utils import to_yaml as _to_yaml

            return _to_yaml(self, filepath)

    # ---- Runtime convenience properties (no extra attributes) ----
    @property
    def metadata(self):
        """The underlying datamodel instance holding the schema fields (this object)."""
        # Alias: the datamodel instance itself holds the schema fields
        return self

    @property
    def ontology(self):
        """The ontology class matching this model's name, or `None` if it is not a known neural mass model."""
        name = getattr(self, "name", None)
        if not name:
            return None
        cl = ontology.onto.search_one(label=str(name))
        return cl if cl in _available_neural_mass_models() else None

    def search_ontology(self, search_str: str, **kwargs):
        """Search this model's ontology subtree for a term.

        Args:
            search_str: Text to search for among the model's ontology
                labels and synonyms.
            **kwargs: Forwarded to the underlying ontology search.

        Returns:
            The ontology search matches for `search_str` within this model.
        """
        return ontology.search_in_model(search_str, self.ontology, **kwargs)

    @property
    def keyed_parameters(self):
        """Mapping of each parameter as a SymPy `Symbol` to its numeric value."""
        return {
            Symbol(p.name): p.value for p in getattr(self, "parameters", {}).values()
        }

    @property
    def symbolic(self):
        """Full symbolic ODE system using proper SymPy conventions.

        State variables are represented as ``Function(name)(t)`` so that
        ``Derivative(theta(t), t)`` stays unevaluated.  Derived variables
        and derived parameters are included as algebraic equations.

        Returns
        -------
        dict
            ``{'state': [...], 'derived': [...], 'parameters': {...}}``
            where each list contains ``sympy.Eq`` objects and parameters
            maps ``Symbol → value``.

        Example
        -------
        >>> model.symbolic['state']
        [Eq(Derivative(theta(t), t), I + omega)]
        >>> model.symbolic['derived']
        [Eq(signal(t), sin(theta(t)))]
        """
        import sympy as sp

        from tvbo.parse.expression import parse_eq

        t = Symbol("t")

        # Build scope: state variables as Function(name)(t),
        # everything else as Symbol
        scope = {}
        sv_funcs = {}
        for name in self.state_variables:
            f = Function(str(name))
            sv_funcs[name] = f
            scope[str(name)] = f(t)

        for p in self.parameters.values():
            scope[str(p.name)] = Symbol(str(p.name))
        for ci in getattr(self, "coupling_inputs", {}).keys():
            scope[str(ci)] = Symbol(str(ci))
        for name in getattr(self, "derived_parameters", {}):
            scope[str(name)] = Symbol(str(name))
        for name in getattr(self, "derived_variables", {}):
            # Derived variables that appear in state equations
            # should resolve to their Function(t) form
            scope[str(name)] = Function(str(name))(t)
        for fname, f in getattr(self, "functions", {}).items():
            scope[str(fname)] = Function(str(fname))
            for name in f.arguments:  # arguments is a dict keyed by name
                scope[str(name)] = Symbol(str(name))
        scope["t"] = t
        if "e" not in scope:
            scope["e"] = sp.E

        # Wrap all parsing in evaluate=False to preserve expression order
        # (prevents SymPy from rewriting e.g. sin(v0-v) → -sin(v-v0))
        with sp.evaluate(False):
            # State equations: Eq(d/dt state(t), rhs)
            state_eqs = []
            for name, sv in self.state_variables.items():
                rhs = parse_eq(sv.equation, local_dict=scope)
                order = int(getattr(sv, "equation_order", 1) or 1)
                discrete = getattr(self, "system_type", "continuous") == "discrete"
                if discrete:
                    lhs = sv_funcs[name](t)
                else:
                    lhs = sp.Derivative(sv_funcs[name](t), *([t] * order))
                state_eqs.append(sp.Eq(lhs, rhs))

            # Derived parameter equations
            dp_eqs = []
            for name, dp in getattr(self, "derived_parameters", {}).items():
                rhs = parse_eq(dp.equation, local_dict=scope)
                dp_eqs.append(sp.Eq(Symbol(str(name)), rhs))

            # Derived variable equations
            dv_eqs = []
            for name, dv in getattr(self, "derived_variables", {}).items():
                has_conds = (
                    bool(getattr(dv.equation, "conditionals", None))
                    and len(getattr(dv.equation, "conditionals", [])) > 0
                )
                if getattr(dv, "conditional", False) and has_conds:
                    rhs = _equation_mod.conditionals2piecewise(dv.equation)
                else:
                    rhs = parse_eq(dv.equation, local_dict=scope)
                dv_eqs.append(sp.Eq(Function(str(name))(t), rhs))

            # Function definitions: Eq(Sigm(v), 2*e0/(1+exp(r*(v0-v))))
            func_eqs = []
            for fname, f in getattr(self, "functions", {}).items():
                arguments = [Symbol(str(name)) for name in f.arguments]
                lhs = Function(str(fname))(*arguments)
                rhs = parse_eq(f.equation, local_dict=scope)
                func_eqs.append(sp.Eq(lhs, rhs))

        # Parameters: Symbol → numeric value
        params = {Symbol(str(p.name)): p.value for p in self.parameters.values()}

        return {
            "state": state_eqs,
            "functions": func_eqs,
            "derived_parameters": dp_eqs,
            "derived": dv_eqs,
            "parameters": params,
        }

    def get_symbolic_elements(self, include_time_symbol: bool = True):
        """Build a unified local_dict for parsing model expressions.

        Includes symbols for parameters, coupling terms, derived parameters, derived
        variables, output transforms, state variables, event names, function names, and
        (optionally) the time symbol 't'.

        Every declared name must appear here so it shadows SymPy's own global namespace:
        `Q` is SymPy's assumptions object, `S` its sympify shortcut, `O` big-O, `N`
        numeric evaluation and `I` the imaginary unit, so a model that names a quantity
        after any of them would otherwise fail to parse.

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

        # Coupling inputs (named inputs from coupling function)
        for ci in getattr(self, "coupling_inputs", {}).keys():
            scope[str(ci)] = Symbol(str(ci))

        # Derived parameters / variables / output transforms as Symbols
        for name in getattr(self, "derived_parameters", {}).keys():
            scope[str(name)] = Symbol(str(name))
        for name in getattr(self, "derived_variables", {}).keys():
            scope[str(name)] = Symbol(str(name))

        # Output is a list of string references
        for name in getattr(self, "output", []):
            scope[str(name)] = Symbol(str(name))

        # State variables as Symbols
        for name in getattr(self, "state_variables", {}).keys():
            scope[str(name)] = Symbol(str(name))

        # Functions: undefined function heads; also add their argument symbols
        for fname, f in getattr(self, "functions", {}).items():
            scope[str(fname)] = Function(str(fname))
            for name in f.arguments:  # arguments is a dict keyed by name
                scope[str(name)] = Symbol(str(name))

        for name in getattr(self, "events", {}) or {}:
            scope[str(name)] = Symbol(str(name))

        if "e" not in scope:
            from sympy import E

            scope["e"] = E

        return scope

    def symbol_map(self):
        """Display-symbol overrides for report rendering: ``{identifier Symbol: LaTeX str}``.

        For each element that declares a ``symbol`` (e.g. ``w_+`` for the identifier
        ``w_plus``, or ``S^{(E)}`` for ``S_e``), map its identifier Symbol to the LaTeX
        of that override, so ``sympy.latex(expr, symbol_names=model.symbol_map())``
        renders the source's own notation. Elements without an override are omitted (they
        render from their identifier). Fully sympy-native: the override is itself rendered
        through ``sympy.latex(Symbol(...))``, inheriting Greek/sub/superscript handling.

        Keyed by the canonical collection keys (the identifiers used in the equations),
        over the same element collections as
        [`get_symbolic_elements`](#tvbo.classes.dynamics.Dynamics.get_symbolic_elements).
        """
        collections = (
            self.parameters,
            self.state_variables,
            self.derived_variables,
            self.derived_parameters,
            self.coupling_inputs,
        )
        return {
            Symbol(str(key)): latex(Symbol(str(el.symbol)))
            for coll in collections
            for key, el in (coll or {}).items()
            if getattr(el, "symbol", None)
        }

    def update_metadata(self):
        """Normalize and finalize the model's equation metadata in place.

        Migrates deprecated fields (`cases`→`conditionals`, `coupling_terms`→
        `coupling_inputs`), canonicalizes every state and derived-variable
        equation via
        [`update_equations`](#tvbo.classes.dynamics.update_equations), and
        sorts derived parameters, derived variables, and outputs into
        dependency order.
        """
        # Normalize dv.cases → dv.equation.conditionals (dv.cases is deprecated)
        _normalize_conditionals(self)

        # Migrate coupling_terms → coupling_inputs (coupling_terms is deprecated)
        _migrate_coupling_terms(self)

        # Collect all equations (state + derived) and update stored Equation objects
        all_eqs = update_equations(self)

        from sympy.printing import StrPrinter

        _rhs_str = (
            (lambda e: str(e))
            if REORDER_EQUATIONS
            else StrPrinter(settings={"order": "none"}).doprint
        )
        scope = self.get_symbolic_elements()
        for v, eq in all_eqs.items():
            target = self.state_variables.get(v, self.derived_variables.get(v))
            stored = getattr(getattr(target, "equation", None), "rhs", None)
            equation = tvbo_datamodel.Equation(
                lhs=str(eq.lhs), rhs=stable_rhs(stored, eq.rhs, scope, _rhs_str)
            )
            if v in self.state_variables:
                self.state_variables[v].equation = equation
            elif v in self.derived_variables:
                # Preserve conditionals through the equation update
                old_conds = getattr(
                    self.derived_variables[v].equation, "conditionals", None
                )
                if old_conds:
                    equation.conditionals = old_conds
                self.derived_variables[v].equation = equation
        # Build dependency order without storing state
        _ = self.get_dependency_tree()
        sort_equations(self, "derived_parameters")
        sort_equations(self, "derived_variables")
        sort_equations(self, "output")
        # sort_equations(self, "state_variables") #TODO: Test if sorting is really not necessary

    # -----------------------
    # Fluent builder helpers and setters
    # -----------------------
    @staticmethod
    def _coerce_range(domain):
        """Accept None | Range | (lo, hi[, step]) | {lo, hi[, step, enforce]} → Range."""
        if domain is None or isinstance(domain, tvbo_datamodel.Range):
            return domain
        if isinstance(domain, dict):
            kw = {k: v for k, v in domain.items() if k in ("lo", "hi", "step", "enforce") and v is not None}
            for k in ("lo", "hi", "step"):
                if k in kw:
                    kw[k] = float(kw[k])
            return tvbo_datamodel.Range(**kw) if kw else None
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
        from tvbo.datamodel import pydantic as _pdm

        return _pdm.Dynamics.model_validate(self._as_dict)

    @classmethod
    def from_pydantic(cls, pyd_obj, use_ontology: bool = False):
        """Create a Dynamics from a tvbopydantic.Dynamics (or dict-like)."""
        data = pyd_obj.model_dump() if hasattr(pyd_obj, "model_dump") else dict(pyd_obj)
        inst = cls(use_ontology=use_ontology, **data)
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
        """Add a parameter to the model.

        Args:
            name: Parameter name (also its dict key).
            value: Numeric default value.
            unit: Physical unit.
            description: Human-readable description.
            domain: Valid range as a `Range`, `(lo, hi[, step])` tuple, or
                dict.
            definition: Formal definition or ontology reference.
            symbol: Display symbol.

        Returns:
            `self`, to allow fluent chaining.
        """
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
        nonparam_known |= set(map(str, self.output))  # output is list of strings
        nonparam_known |= set(map(str, self.derived_parameters.keys()))
        for f in self.functions.values():
            nonparam_known |= {str(name) for name in f.arguments}
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
        """Add a state variable (with its differential equation) to the model.

        Any free symbols in `equation` that are not yet known are
        auto-registered as parameters. A legacy `boundaries` clamp is folded
        into `domain` (with the descriptive range preserved as the sampling
        `distribution`).

        Args:
            name: State-variable name (also its dict key).
            equation: RHS of its time-derivative equation; accepts a string,
                `sympy.Eq`/`Expr`, or `Equation`.
            description: Human-readable description.
            domain: Valid/sampling range as a `Range`, tuple, or dict.
            boundaries: Legacy hard-clamp range, folded into `domain`.
            initial_value: Default initial condition.
            unit: Physical unit.
            coupling_variable: Mark this variable as observed for network
                coupling.
            stimulation_variable: Mark this variable as a stimulation target.
            symbol: Display symbol.

        Returns:
            `self`, to allow fluent chaining.
        """
        eq = (
            self._coerce_equation(equation, lhs=str(name))
            if equation is not None
            else None
        )
        # ``boundaries`` is the legacy name for a hard clamp; fold it into the
        # unified ``domain`` with enforce='clamp'. When both are given, the clamp
        # is the operative domain and the descriptive ``domain`` (the IC-sampling
        # range) is preserved as the sampling ``distribution`` rather than dropped.
        _domain, _distribution = _fold_range_boundaries(
            self._coerce_range(domain),
            self._coerce_range(boundaries) if boundaries is not None else None,
        )
        self.state_variables[str(name)] = tvbo_datamodel.StateVariable(
            name=str(name),
            equation=eq,
            description=description,
            domain=_domain,
            distribution=_distribution,
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
        """Add a derived (algebraic) variable to the model.

        Args:
            name: Derived-variable name (also its dict key).
            expression: RHS expression; accepts a string, `sympy.Eq`/`Expr`,
                or `Equation`.
            conditionals: Optional list of `(expression, condition)` pairs
                defining a piecewise/conditional variable.
            unit: Physical unit.
            description: Human-readable description.
            symbol: Display symbol.

        Returns:
            `self`, to allow fluent chaining.
        """
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
        """Add a reusable function definition to the model.

        Args:
            name: Function name (also its dict key).
            expression: Function body; accepts a string, `sympy.Eq`/`Expr`,
                or `Equation`.
            arguments: Argument names as a sequence, or a mapping of name to
                `Parameter`.
            description: Human-readable description.
            definition: Formal definition or ontology reference.

        Returns:
            `self`, to allow fluent chaining.
        """
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
    def add_coupling_input(
        self,
        name: str,
        description: str | None = None,
        unit: str | None = None,
        dimension: int = 1,
        keys: list[str] | None = None,
    ):
        """Add a coupling input (a network-supplied term) to the model.

        Any existing parameter with the same name is removed so the name
        resolves to the coupling input.

        Args:
            name: Coupling-input name (also its dict key).
            description: Human-readable description.
            unit: Accepted for backward compatibility; not currently stored on
                the coupling input.
            dimension: Number of components the input carries.
            keys: Optional sub-keys addressed by the coupling input.

        Returns:
            `self`, to allow fluent chaining.
        """
        key = str(name)
        self.coupling_inputs[key] = tvbo_datamodel.CouplingInput(
            name=key,
            description=description,
            dimension=dimension,
            keys=keys or [],
        )
        # Keep parameters clean: remove any parameter with same name
        if key in self.parameters:
            del self.parameters[key]

        return self

    def add_coupling_term(self, name, description=None, unit=None):
        """Deprecated. Use add_coupling_input() instead."""
        import warnings

        warnings.warn(
            "add_coupling_term() is deprecated. Use add_coupling_input() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_coupling_input(name=name, description=description)

    def add_output(
        self,
        name: str,
        expression=None,
        *,
        unit: str | None = None,
        description: str | None = None,
    ):
        """Add an output variable. Creates a derived_variable and adds its name to output list."""
        name_str = str(name)
        # Create derived variable with the equation
        eq = (
            self._coerce_equation(expression, lhs=name_str)
            if expression is not None
            else None
        )
        self.derived_variables[name_str] = tvbo_datamodel.DerivedVariable(
            name=name_str, equation=eq, unit=unit, description=description
        )
        # Add reference to output list
        if name_str not in self.output:
            self.output.append(name_str)
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
        """Add a derived parameter (computed from other parameters) to the model.

        Args:
            name: Derived-parameter name (also its dict key).
            expression: RHS expression; accepts a string, `sympy.Eq`/`Expr`,
                or `Equation`.
            unit: Physical unit.
            description: Human-readable description.
            symbol: Display symbol.

        Returns:
            `self`, to allow fluent chaining.
        """
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
        """Plot this model's ontology graph.

        Args:
            **kwargs: Forwarded to `tvbo.plot.ontology.plot_model`.

        Returns:
            The rendered ontology plot.
        """
        from tvbo.plot import ontology

        return ontology.plot_model(self.ontology, **kwargs)

    def plot(self, *dims, **kwargs):
        """Plot trajectories of this dynamics in 1D, 2D, or 3D.

        See :func:`tvbo.plot.dynamics.plot_dynamics` for parameters.
        """
        from tvbo.plot.dynamics import plot_dynamics
        from tvbo.plot.dynamics_layout import plot_dynamics_layout

        layout = kwargs.pop("layout", None)
        panels = kwargs.pop("panels", None)
        if layout is not None or panels is not None:
            return plot_dynamics_layout(
                self,
                layout=layout,
                panels=panels,
                figsize=kwargs.pop("figsize", None),
                subplot_kwargs=kwargs.pop("subplot_kwargs", None),
                fig=kwargs.pop("fig", None),
                axes=kwargs.pop("axes", None),
            )

        return plot_dynamics(self, *dims, **kwargs)

    def animate(self, parameter, values, *dims, **kwargs):
        """Animate by sweeping one parameter through ``values``.

        See :func:`tvbo.plot.dynamics.animate_dynamics` for parameters.
        Returns a :class:`matplotlib.animation.FuncAnimation`.
        """
        from tvbo.plot.dynamics import animate_dynamics

        return animate_dynamics(self, parameter, values, *dims, **kwargs)

    def render_equation(self, obj, format="latex", inline_functions=False, **kwargs):
        """Render a model element's equation to a string.

        Handles conditional derived variables (converting `conditionals` to a
        SymPy `Piecewise`) and can optionally inline the model's function
        definitions.

        Args:
            obj: A model element exposing an `equation` (state/derived
                variable, derived parameter, …).
            format: Output format, e.g. `"latex"`, `"numpy"`, `"julia"`.
            inline_functions: If `True`, substitute model function bodies
                inline instead of emitting function calls.
            **kwargs: Forwarded to `tvbo.codegen.code.render_equation`.

        Returns:
            The rendered equation in the requested format.
        """
        from tvbo.classes.equation import sympify as tvbo_sympify
        from tvbo.codegen.code import render_equation

        scope = self.get_symbolic_elements()
        # Tell the printer which names are functions so it emits f(x) cleanly
        uf = {str(name): str(name) for name in getattr(self, "functions", {}).keys()}

        # Build inline_funcs dict if requested
        inline_funcs = None
        if inline_functions and hasattr(self, "functions") and self.functions:
            inline_funcs = {}
            for fname, fdef in self.functions.items():
                arg_names = [str(name) for name in fdef.arguments]
                body = tvbo_sympify(fdef.equation.rhs)
                inline_funcs[fname] = (arg_names, body)
            # Don't emit function names as user_functions if we're inlining them
            uf = {}

        # For conditional derived variables, use conditionals2piecewise
        # which reads from dv.equation.conditionals (canonical location).
        eq_to_render = obj.equation
        if getattr(obj, "conditional", False) and getattr(
            obj.equation, "conditionals", None
        ):
            eq_rhs_str = str(obj.equation.rhs) if obj.equation.rhs else ""
            if "Piecewise" not in eq_rhs_str:
                pw = _equation_mod.conditionals2piecewise(obj.equation)
                from types import SimpleNamespace

                eq_to_render = SimpleNamespace(rhs=str(pw))

        return render_equation(
            eq_to_render,
            local_dict=scope,
            format=format,
            user_functions=uf,
            inline_funcs=inline_funcs,
            **kwargs,
        )

    def render_equation_cse(self, obj, format="numpy", inline_functions=False, **kwargs):
        """Common-subexpression-eliminated variant of :meth:`render_equation`.

        Returns ``(setup, final)`` — a list of ``(name, expr)`` assignments plus the
        return expression — so interpreted backends (TVB / numpy) evaluate each
        shared subexpression (notably repeated model-function calls) once instead of
        per occurrence. Builds the same symbolic scope / user-function set as
        :meth:`render_equation`; see :func:`tvbo.codegen.code.render_equation_cse`.
        """
        from tvbo.classes.equation import sympify as tvbo_sympify
        from tvbo.codegen.code import render_equation_cse

        scope = self.get_symbolic_elements()
        uf = {str(name): str(name) for name in getattr(self, "functions", {}).keys()}

        inline_funcs = None
        if inline_functions and getattr(self, "functions", None):
            inline_funcs = {}
            for fname, fdef in self.functions.items():
                arg_names = [str(name) for name in fdef.arguments]
                inline_funcs[fname] = (arg_names, tvbo_sympify(fdef.equation.rhs))
            uf = {}

        eq_to_render = obj.equation
        if getattr(obj, "conditional", False) and getattr(obj.equation, "conditionals", None):
            if "Piecewise" not in (str(obj.equation.rhs) if obj.equation.rhs else ""):
                from types import SimpleNamespace

                pw = _equation_mod.conditionals2piecewise(obj.equation)
                eq_to_render = SimpleNamespace(rhs=str(pw))

        return render_equation_cse(
            eq_to_render,
            local_dict=scope,
            format=format,
            user_functions=uf,
            inline_funcs=inline_funcs,
            **kwargs,
        )

    def get_equations(self, format="metadata", evaluate=True):
        """Collect the model's equations as SymPy `Eq` objects.

        Builds equations for derived parameters, functions, derived variables,
        state equations (as time derivatives, or plain maps for discrete
        systems), and output transformations.

        Args:
            format: Shape of the result. `"dict"` groups equations by
                category; `"state-equations"` returns only state equations
                keyed by variable name; any other value (e.g. `"metadata"`)
                returns a single flat mapping of variable name to `Eq`.
            evaluate: If `True`, let SymPy evaluate/simplify parsed
                right-hand sides; if `False`, preserve authored term order.

        Returns:
            A mapping of equations whose structure depends on `format`.

        Raises:
            ValueError: If an entry in `output` names neither a derived nor a
                state variable.
        """
        # if format == "sympy":
        #     return _equation_mod.symbolic_model_equations(self.ontology)
        # elif format == "latex":
        #     return equations.render_latex_equations(self.ontology)

        scope = self.get_symbolic_elements()
        equations = {}
        # Determine system type (default to continuous)
        discrete = getattr(self, "system_type", "continuous") == "discrete"

        equations["derived-parameters"] = []
        for k, dp in self.derived_parameters.items():
            equations["derived-parameters"].append(
                Eq(lhs=Symbol(k), rhs=parse_eq(dp.equation, local_dict=scope, evaluate=evaluate))
            )

        equations["functions"] = []
        for k, f in self.functions.items():
            arguments = [Symbol(name) for name in f.arguments]
            k = Function(k)(*arguments)
            equations["functions"].append(
                Eq(lhs=k, rhs=parse_eq(f.equation, local_dict=scope, evaluate=evaluate))
            )

        equations["derived-variables"] = []
        for k, dv in self.derived_variables.items():
            # Use equation.conditionals (canonical location for conditional data)
            has_conditionals = bool(getattr(dv.equation, "conditionals", None)) and (
                len(getattr(dv.equation, "conditionals", [])) > 0
            )
            if getattr(dv, "conditional", False) and has_conditionals:
                expression = _equation_mod.conditionals2piecewise(dv.equation)
            else:
                expression = parse_eq(dv.equation, local_dict=scope, evaluate=evaluate)

            equations["derived-variables"].append(Eq(lhs=Symbol(k), rhs=expression))

        equations["state-equations"] = []
        for k, sv in self.state_variables.items():
            if not getattr(sv, "equation", None):
                continue
            t = Symbol("t")
            sv_symbol = Symbol(k)
            # Prefer conditionals on the Equation if present; fallback to rhs parsing
            has_conditionals = bool(getattr(sv.equation, "conditionals", None)) and (
                len(getattr(sv.equation, "conditionals", [])) > 0
            )
            if has_conditionals:
                expression = _equation_mod.conditionals2piecewise(sv.equation)
            else:
                expression = parse_eq(sv.equation, local_dict=scope, evaluate=evaluate)

            order = int(getattr(sv, "equation_order", 1) or 1)
            if discrete:
                lhs_expr = sv_symbol
            elif order > 1:
                lhs_expr = Derivative(sv_symbol, *([t] * order))
            else:
                lhs_expr = Derivative(sv_symbol, t)
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
        # Output is a list of string references to derived_variables or state_variables
        for var_name in self.output:
            var_name_str = str(var_name)
            if var_name_str in self.derived_variables:
                dv = self.derived_variables[var_name_str]
                equations["output-transformations"].append(
                    Eq(
                        lhs=Symbol(var_name_str),
                        rhs=parse_eq(dv.equation, local_dict=scope, evaluate=evaluate),
                    )
                )
            elif var_name_str in self.state_variables:
                # State variable directly as output - no transformation needed
                # Don't add identity equation Eq(S, S) as it would overwrite the
                # real state equation in the flat dict returned by get_equations()
                pass
            else:
                raise ValueError(
                    f"Output variable '{var_name_str}' not found in derived_variables or state_variables"
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
        """Substitute parameter values (and any overrides) into every equation.

        Parameter symbols are replaced with their numeric values, then any
        `**kwargs` overrides are applied, and finally all coupling inputs are
        forced to `0` (for fixed-point / equilibrium analysis) — so a `kwargs`
        entry named after a coupling input is overridden by that `0`.

        Args:
            **kwargs: Additional symbol-name to value substitutions.

        Returns:
            The list of equations with substitutions applied.
        """
        # Substitute parameters and defaults into equations (useful for fixed-point search)
        sub = self.keyed_parameters
        sub.update(kwargs)
        # Set all coupling inputs to 0 for fixed-point analysis
        for ci in getattr(self, "coupling_inputs", {}).keys():
            sub[ci] = 0
        return [eq.subs(sub) for eq in self.get_equations().values()]

    def calculate_derived_parameters(self):
        """Evaluate and cache each derived parameter's numeric value.

        Scalar parameters are substituted into every derived-parameter
        equation and the result stored on the model. Array-valued or otherwise
        unresolved derived parameters keep a `None` value and are recomputed at
        runtime by the generated code.

        Returns:
            A mapping of derived-parameter name to its computed value (or
            `None`), or `None` if the model has no derived parameters.
        """
        if self.derived_parameters is None:
            return None
        from tvbo.utils import is_array_valued

        # Loop-invariant: build the symbol scope and the scalar-parameter
        # substitution once. Array-valued parameters (mode-coupling matrices,
        # quadrature vectors) are excluded — they have no load-time scalar and
        # would make ``subs`` raise — so the derived parameters that depend on
        # them stay symbolic and are recomputed at runtime.
        local_dict = self.get_symbolic_elements()
        scalar_subs = {
            Symbol(p.name): p.value
            for p in self.parameters.values()
            if not is_array_valued(p.value)
        }
        for k, dp in self.derived_parameters.items():
            try:
                eq = parse_eq(dp.equation, local_dict=local_dict, evaluate=False).subs(scalar_subs)
                # Convert SymPy Float to Python float for YAML serialization.
                self.derived_parameters[k].value = float(eq.evalf())
            except (TypeError, ValueError):
                # Array-valued / unresolved derived parameter: no load-time scalar;
                # recomputed at runtime by the generated update_derived_parameters.
                self.derived_parameters[k].value = None
        return {k: self.derived_parameters[k].value for k in self.derived_parameters}

    def get_dependency_tree(self, ontomapping=False, include_state_equations=False):
        """Build the equation dependency graph for this model.

        Nodes are the model's symbols; each edge points from a dependency to
        the quantity whose equation uses it (dependencies → dependents). State
        equations are excluded by default to avoid cycles in discrete systems.

        Args:
            ontomapping: If `True`, also build an ontology-class version of
                the graph and the symbol↔ontology-class mappings.
            include_state_equations: If `True`, include state equations in
                the graph.

        Returns:
            The dependency graph, or — when `ontomapping` is `True` — the
            tuple `(graph, ontology_graph, symbol_to_onto, onto_to_symbol)`.
        """
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

        G = _equation_mod.dependency_tree(eq_list)

        if not ontomapping:
            return G

        symbol_onto_mapping = {}
        onto_symbol_mapping = {}
        # Coupling inputs don't have model-specific suffixes in ontology
        coupling_term_names = set(getattr(self, "coupling_inputs", {}).keys())
        for n in G.nodes:
            suffix = (
                ontology.get_model_suffix(self.ontology or self.name)
                if str(n) not in coupling_term_names
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
        """Plot the model's equation dependency graph.

        Args:
            ax: Existing matplotlib axis to draw into; if omitted, a new
                figure is created and returned.
            edgecolor: Node edge color.
            color_nodes_by: Ontology attribute used to color nodes by
                category.
            pos: Node layout, `"graphviz"` (hierarchical) or otherwise a
                Kamada–Kawai layout.
            edgekwargs: Extra keyword arguments for edge drawing.
            **kwargs: Forwarded to the node-drawing helper.

        Returns:
            The created figure when `ax` was not supplied, otherwise `None`.
        """

        import sympy

        from tvbo.plot import ontology as ontology_plot

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
            cat_dict, categories = ontology_plot.get_node_color_mapping(
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

        ontology_plot.draw_custom_nodes(
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
        ax.set_xlim([1.01 * lim for lim in ax.get_xlim()])

        if return_fig:
            for ax in fig.axes:
                ax.set_box_aspect(1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.tight_layout()

            plt.close()
            return fig

    def render_code(self, format="tvb", alt_label=None, **kwargs):
        """Generate backend source code for this model.

        Dispatches to the template (or adapter) for the requested backend and returns the
        formatted source. Reads the model and does not modify it, so the source depends on
        the model alone and not on how often it has been rendered.

        Normalisation belongs to construction: every path that builds a `Dynamics` runs
        [`update_metadata`](#tvbo.classes.dynamics.DynamicalSystem.update_metadata)
        already. Repeating it here made rendering a command as well as a query, and because
        the normalisation was not idempotent the emitted code alternated between two
        spellings depending on the number of previous renders. A model mutated by hand
        after construction should be normalised by whoever mutated it.

        Args:
            format: Target backend, e.g. `"tvb"`, `"jax"`, `"numpy"`,
                `"tvboptim"`, `"julia"`, `"bifurcation-julia"`, `"pde-fem"`,
                or `"neuroml"`.
            alt_label: Optional alternative label for the generated model.
            **kwargs: Forwarded to the template/adapter (e.g. `continuation`).

        Returns:
            The generated code as a formatted string.

        Raises:
            ValueError: If `format` is not a supported backend.
        """
        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-model.py.mako")

        elif format in ["scipy", "python", "jax-python", "python-jax"]:
            # scipy-compatible signature: func(y, t, param=val, ...)
            template = templates.lookup.get_template("tvbo-python-model.py.mako")

        elif format == "python-network":
            template = templates.lookup.get_template("tvbo-python-model.py.mako")
            kwargs.update({"coupling_as_argument": True})

        elif format.lower() == "tvboptim":
            # Full AbstractDynamics subclass for tvboptim
            template = templates.lookup.get_template("tvbo-tvboptim-dynamics.py.mako")

        elif format.lower() in ["autodiff", "jax", "numpy"]:
            # standard signature: dfun(current_state, t, cX, _p)
            template = templates.lookup.get_template("tvbo-jax-dfuns.py.mako")

        elif format == "julia":
            template = templates.lookup.get_template(
                "tvbo-julia-DifferentialEquations.jl.mako"
            )
        elif format == "bifurcation-julia":
            from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

            continuation = kwargs.pop("continuation", None)
            ctx = BifurcationKitAdapter._prepare_context(self, continuation, **kwargs)
            template = templates.lookup.get_template(
                "tvbo-julia-BifurcationKit.jl.mako"
            )
            rendered_code = template.render(**ctx)
            return templater.format_code(rendered_code, format=format)
        elif format == "bifurcation-numcont":
            # The numcont backend now consumes the f90 source directly.
            template = templates.lookup.get_template("tvbo-auto7p.py.mako")
        elif format == "bifurcation-auto7p":
            template = templates.lookup.get_template("tvbo-auto7p.py.mako")
        elif format in ["pde-fem", "pde-python", "pde"]:
            # Generic Python FEM (scikit-fem) template
            template = templates.lookup.get_template("tvbo-pde-fem.py.mako")
        elif format.lower() in ["neuroml", "nml", "lems"]:
            from tvbo.adapters.neuroml import NeuroMLAdapter

            adapter = NeuroMLAdapter(self)
            return adapter.render_dynamics(**kwargs)
        else:
            raise ValueError(f"Format {format} not supported.")

        rendered_code = template.render(
            model=self, format=format, jax="jax" in format, **kwargs
        )
        return templater.format_code(rendered_code, format=format)

    def render(self, format="yaml", **kwargs) -> str:
        """Unified entry point for rendering the model in any output format.

        Dispatches to the appropriate renderer based on *format*:

        - ``'yaml'`` — TVBO YAML specification
        - ``'pyrates-yaml'`` — PyRates YAML
        - ``'report'`` / ``'markdown'`` / ``'md'`` — human-readable Markdown report
        - ``'pdf'`` — report rendered to PDF (requires *outputfile* kwarg)
        - ``'neuroml'`` / ``'nml'`` / ``'lems'`` — LEMS XML via NeuroMLAdapter
        - Any code format accepted by :meth:`render_code` (``'tvb'``,
          ``'jax'``, ``'julia'``, ``'bifurcation-julia'``, …)

        Parameters
        ----------
        format : str
            Target output format.
        **kwargs
            Forwarded to the underlying renderer.

        Returns
        -------
        str
        """
        fmt = format.lower()

        # ── Serialisation ────────────────────────────────────────────────
        if fmt == "yaml":
            return self.to_yaml(filepath=kwargs.get("filepath"))
        if fmt == "pyrates-yaml":
            return self.to_yaml(filepath=kwargs.get("filepath"), format="pyrates")

        # ── Report ───────────────────────────────────────────────────────
        if fmt in ("report", "markdown", "md", "pdf"):
            report_fmt = "pdf" if fmt == "pdf" else "markdown"
            return self.generate_report(format=report_fmt, **kwargs)

        # ── Code generation (all other formats) ──────────────────────────
        return self.render_code(format=format, **kwargs)

    def display_markdown(self, format="tvb", **kwargs):
        """Render generated code as an IPython Markdown code block.

        Args:
            format: Backend passed to
                [`render_code`](#tvbo.classes.dynamics.DynamicalSystem.render_code).
            **kwargs: Forwarded to `render_code`.

        Returns:
            An `IPython.display.Markdown` object wrapping the generated code.
        """
        from IPython.display import Markdown

        code = templater.format_code(
            self.render_code(format=format, **kwargs), format=format
        )
        return Markdown(
            f"```{'python' if format in ['tvb', 'python', 'jax', 'autodiff'] else format}\n{code}\n```"
        )

    def execute(self, format="tvb", **kwargs):
        """Generate and execute the model code, returning a runnable object.

        Dispatches on `format`: builds a configured TVB model instance, a
        tvboptim dynamics instance, a compiled C module (`sympy2c`), a
        bifurcation/continuation run, or a plain dfun callable.

        Args:
            format: Backend to execute, e.g. `"tvb"`, `"tvboptim"`, `"c"`,
                `"bifurcation-auto7p"`, or a code format yielding a dfun.
            **kwargs: Constructor/runtime arguments forwarded to the executed
                code.

        Returns:
            The executed object, whose type depends on `format`.
        """

        if format == "tvb":
            rendered_code = clean_code(self.render_code(format=format, **kwargs))
            namespace = {}
            exec(rendered_code, namespace)
            tvb_obj = namespace[self.name](**kwargs)
            tvb_obj.title = self.label
            tvb_obj.configure()
            return tvb_obj

        elif format.lower() in ("tvboptim", "tvb-optim"):
            namespace = {}
            exec(clean_code(self.render_code(format="tvboptim")), namespace)
            cls = namespace[self.name]
            return cls(**kwargs)

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
            params.update(
                {Symbol(str(ci)): 0.0 for ci in getattr(self, "coupling_inputs", {})}
            )
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
            # Standalone in-tree AUTO-07p backend (no external `numcont` package).
            # Builds a one-off SimulationExperiment wrapping this Dynamics and
            # delegates to NumContAdapter.
            from tvbo.adapters.numcont import NumContAdapter
            from tvbo.classes.continuation import Continuation
            from tvbo.classes.experiment import SimulationExperiment

            cont = kwargs.pop("continuation", None) or Continuation(
                name=self.name + "_eq"
            )
            exp = SimulationExperiment(
                name=self.name,
                label=getattr(self, "label", self.name),
                dynamics=self,
                continuations={cont.name: cont},
            )
            return NumContAdapter(exp).run(**kwargs)

        else:
            rendered_code = clean_code(self.render_code(format=format, **kwargs))
            namespace = {}
            exec(clean_code(rendered_code), namespace)
            model_dfun = namespace[self.name]
            return model_dfun

    def to_lems(self, initial_conditions=1, component_id=None):
        """Build a LEMS model for this local neural mass model.

        .. deprecated::
            Use ``NeuroMLAdapter(model).render_code()`` from
            ``tvbo.adapters.neuroml`` instead.  This method returns a
            ``lems.Model`` object (PyLEMS API); the adapter produces a
            validated XML string.

        Parameters:
        - initial_conditions: number or dict; if number, used for all SVs; if dict, keys are sv name or sv_name_0
        - component_id: optional id for the component; defaults to model label

        Returns:
        - lems.Model instance containing a ComponentType and a Component for this model
        """
        import warnings

        warnings.warn(
            "Dynamics.to_lems() is deprecated. Use NeuroMLAdapter(model).render_code() from tvbo.adapters.neuroml instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import lems.api as lems  # lazy import

        from tvbo.codegen.lems import setup_lems_model  # lazy to avoid cycles
        from tvbo.ontology import owl as _ontology  # avoid shadowing

        model = setup_lems_model()

        from tvbo.utils.units import unit_to_lems_dimension

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
                    dimension=unit_to_lems_dimension(getattr(p, "unit", None)),
                )
            )

        # No extra network input channel; coupling terms (e.g., c_glob) are already in metadata

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
                        dimension=unit_to_lems_dimension(getattr(dp, "unit", None)),
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
                            dimension=unit_to_lems_dimension(getattr(dp, "unit", None)),
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
            dimension = unit_to_lems_dimension(
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
            base_expr = str(_equation_mod.sympify_value(deriv)).replace("**", "^")
            # Do not inject extra inputs here; global coupling is represented via coupling_inputs (e.g., c_glob)

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
        """Build a deterministic cache filename for a run in the temp directory.

        Non-identifying keyword arguments (e.g. `filename`, `force`,
        `verbose`) are dropped and the rest are sorted so the same run maps to
        the same path.

        Args:
            format: Backend format string included in the filename.
            **kwargs: Run parameters encoded into the filename.

        Returns:
            The cache-file path (without extension) inside the temp directory.
        """
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
        """Build the initial state vector for a simulation.

        If any state variable defines a `distribution` (or `random=True`),
        initial values are sampled from it (Gaussian or uniform over the
        finite domain bounds); otherwise each variable's `initial_value` (or
        `default`) is used.

        Args:
            default: Fallback value for variables without an initial value.
            random: Deprecated flag to sample from each variable's domain.
            N: Number of samples per state variable.
            **kwargs: Ignored extra arguments.

        Returns:
            A NumPy array of initial values. When sampling from a distribution
            (or `random=True`) the shape is `(n_state_variables, N)`; otherwise
            it is 1-D with one entry per state variable.
        """
        if random:
            import warnings

            warnings.warn(
                "random=True is deprecated. Set distribution on state variables instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Auto-detect: if any SV has a distribution, sample from it
        has_distributions = any(
            getattr(sv, "distribution", None) for sv in self.state_variables.values()
        )
        if random or has_distributions:
            init = []
            for k, sv in self.state_variables.items():
                dist = getattr(sv, "distribution", None)
                if dist:
                    # Use distribution.domain, fall back to sv.domain
                    domain = getattr(dist, "domain", None) or getattr(
                        sv, "domain", None
                    )
                    # Guard against non-finite / missing bounds: a distribution
                    # without its own domain falls back to sv.domain, which may be a
                    # half-open clamp (e.g. [0, inf)); uniform(0, inf) would overflow.
                    _dlo = getattr(domain, "lo", None) if domain else None
                    _dhi = getattr(domain, "hi", None) if domain else None
                    lo = float(_dlo) if (isinstance(_dlo, (int, float)) and np.isfinite(_dlo)) else -10.0
                    hi = float(_dhi) if (isinstance(_dhi, (int, float)) and np.isfinite(_dhi)) else 10.0
                    dist_name = str(getattr(dist, "name", "Uniform")).lower()
                    if dist_name in ("gaussian", "normal"):
                        sv_init = np.random.normal(
                            loc=(lo + hi) / 2, scale=(hi - lo) / 6, size=N
                        )
                    else:
                        sv_init = np.random.uniform(lo, hi, size=N)
                elif random:
                    # Legacy fallback: sample from the domain range. The domain
                    # may carry a one-sided clamp (e.g. [0, inf) for a firing
                    # rate), so guard against non-finite / inverted bounds —
                    # uniform(0, inf) would yield inf/NaN initial states.
                    dlo = getattr(sv.domain, "lo", None) if sv.domain else None
                    dhi = getattr(sv.domain, "hi", None) if sv.domain else None
                    lo = dlo if (isinstance(dlo, (int, float)) and np.isfinite(dlo)) else -10.0
                    hi = dhi if (isinstance(dhi, (int, float)) and np.isfinite(dhi)) else 10.0
                    if hi <= lo:
                        hi = lo + 1.0
                    sv_init = np.random.uniform(lo, hi, size=N)
                else:
                    # No distribution, no random flag → use initial_value
                    sv_init = np.repeat(initial_value(sv, default), N)
                init.append(sv_init)
        else:
            init = [
                initial_value(sv, default) for sv in self.state_variables.values()
            ]
        return np.array(init)

    def run(
        self, format="python", verbose=0, save=True, run_kwargs={}, **kwargs
    ) -> TimeSeries | BifurcationResult:
        """Generate, execute, and integrate the model, returning its output.

        Supports Julia (ODE and bifurcation), Python (SciPy `odeint`, or an
        iterated map for discrete systems), and compiled C backends.

        Args:
            format: Backend to run, e.g. `"python"`, `"julia"`,
                `"bifurcation-julia"`, or `"c"`.
            verbose: Verbosity level.
            save: If `True`, cache results under a deterministic run filename.
            run_kwargs: Extra arguments forwarded to the integrated dfun
                (e.g. `stimulus`).
            **kwargs: Simulation settings such as `duration`, `dt`, `t`, and
                `u_0`.

        Returns:
            A [`TimeSeries`](#tvbo.data.types.TimeSeries) for time-domain
            runs, or a `BifurcationResult` for bifurcation formats.

        Raises:
            ValueError: If `format` is not supported.
        """
        if save:
            kwargs.update({"filename": self.get_run_filename(format=format, **kwargs)})

        if "xi" in kwargs:
            kwargs.pop("xi")

        if "julia" in format:
            code = self.render_code(format=format, **kwargs)
            from tvbo.run.julia import (
                extract_bifurcation_result,
                extract_ode_solution,
                run_julia_code,
            )

            run_julia_code(code)
            if format == "julia":
                t, u_raw, sol = extract_ode_solution()
                # Single-node ODE: u_raw is (n_sv, n_t), hcat(sol.u...) gives states×time
                data = u_raw.T  # time x states
                data4 = data[:, :, None, None]
                labels_dimensions = {
                    "State Variable": list(self.state_variables.keys()),
                    "Region": ["Region0"],
                }
                return TimeSeries(
                    time=t,
                    data=data4,
                    title=self.name,
                    sample_period=(t[1] - t[0]) if t.size > 1 else None,
                    labels_dimensions=labels_dimensions,
                )
            elif format == "bifurcation-julia":
                br_obj = extract_bifurcation_result()
                bif_res = BifurcationResult(br=br_obj, model=self, **kwargs)
                # Auto-detect PO branches from continuation object or explicit kwarg
                cont = kwargs.get("continuation", None)
                _has_branches = (
                    "periodic_orbits" in kwargs and kwargs["periodic_orbits"]
                ) or (cont and getattr(cont, "branches", None))
                if _has_branches:
                    from tvbo.adapters.julia import eval_with_auto_install

                    try:
                        po = eval_with_auto_install("po_results")
                        bif_res.periodic_orbits = [
                            BifurcationResult(br=p, model=self, **kwargs)
                            for p in po.branches
                        ]
                    except Exception as e:
                        import warnings

                        warnings.warn(f"Periodic orbit extraction failed: {e}")
                        bif_res.periodic_orbits = []
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

            if stimulus and "stimulus" not in run_kwargs:
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
        """Attach a stimulus to the model.

        Warns if no state variable is marked as a stimulation target.
        Depending on `as_derived_variable`, the stimulus is either stored on
        `self.stimulus` or lowered into a `stim_t` derived variable plus
        suffixed stimulus parameters.

        Args:
            stimulus: A
                [`Stimulus`](#tvbo.classes.perturbation.Stimulus) to apply.
            as_derived_variable: If `True`, inline the stimulus as a `stim_t`
                derived variable; if `False`, store the `Stimulus` object
                directly.
        """

        if not any(
            [sv.stimulation_variable for sv in self.state_variables.values()]
        ) and not any(
            ["stim_t" in sv.equation.rhs for sv in self.state_variables.values()]
        ):
            import warnings

            warnings.warn(
                "No state variable with attribute `stimulation_variable=True` set. "
                "Stimulation will have no effect.",
                stacklevel=2,
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
        """Find sibling periodic-orbit output files for a run.

        Args:
            f: Path to the base run file whose periodic-orbit companions
                (`<base>_po*`) are searched for in the same directory.

        Returns:
            The list of matching periodic-orbit file paths.
        """
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
        """Plot a bifurcation diagram alongside representative time series.

        Builds two linked panels — a bifurcation diagram over `ICS` and time
        series of `VOI` sampled at several parameter values — and either
        returns the combined figure or draws into the supplied axes.

        Args:
            ICS: Name of the continuation/bifurcation parameter to vary.
            VOI: Variable of interest to plot.
            n_runs: Number of parameter values sampled for the time-series
                panel.
            t: Time vector for the time-series simulations.
            offset: Vertical offset between successive time-series traces.
            ax1: Axis for the bifurcation panel; a new layout is made if
                omitted.
            ax2: Axis for the time-series panel.
            **kwargs: Forwarded to the bifurcation run.

        Returns:
            The combined figure when axes are not supplied, otherwise `None`.
        """
        panels = {
            "a": {
                "kind": "bifurcation",
                "run": {"format": "bifurcation-julia", "ICS": ICS, **kwargs},
                "plot": {"VOI": VOI, "ICS": ICS},
                "title": "Bifurcation diagram",
            },
            "b": {
                "kind": "parameter_sweep_timeseries",
                "parameter": ICS,
                "dims": (VOI,),
                "t": t,
                "from_panel": "a",
                "n_values": n_runs,
                "legend_title": ICS,
                "title": "Time series",
            },
        }

        if ax1 is None and ax2 is None:
            return self.plot(layout="ab", panels=panels)

        from tvbo.plot.dynamics_layout import (
            render_dynamics_panel as _render_dynamics_panel,
        )
        from tvbo.plot.layout_mosaic import finish_panel as _finish_panel

        cache = {}
        _render_dynamics_panel(self, panels["a"], ax1, cache)
        cache["a"] = self.run(format="bifurcation-julia", ICS=ICS, **kwargs)
        ax1.clear()
        cache["a"].plot(VOI=VOI, ICS=ICS, ax=ax1)
        _render_dynamics_panel(self, panels["b"], ax2, cache)
        _finish_panel(ax1, panels["a"])
        _finish_panel(ax2, panels["b"])

    def parameter_table(self):
        """Return a pandas DataFrame of the model's parameters.

        Returns:
            A DataFrame with `Parameter`, `Value`, and `Description` columns.
        """
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
        """Serialize the model metadata to a YAML file.

        Args:
            filename: Destination path for the dumped YAML.
        """
        from linkml_runtime.dumpers import yaml_dumper

        yaml_dumper.dump(self, filename)

    def save_python_class(self, directory="."):
        """Write the model as a standalone TVB Python class file.

        Emits `<name>.py` in `directory` with the required imports followed
        by the rendered TVB model code.

        Args:
            directory: Target directory for the generated `<name>.py` file.
        """
        fpath = join(directory, f"{self.name}.py")
        with open(fpath, "w") as f:
            f.write("""
import math
import numpy as np
from tvb.simulator.models.base import ModelNumbaDfun, Model
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final""")
            f.write(self.render_code())

    def generate_report(
        self,
        format="markdown",
        template_name="tvbo-report-model",
        outputfile=None,
        derivative_notation: str = "dot",
        baseline=None,
        citeformat=None,
    ):
        """Render a human-readable report of the model.

        Reads the model and does not modify it, for the same reason as
        [`render_code`](#tvbo.classes.dynamics.DynamicalSystem.render_code): normalisation
        belongs to construction, and repeating it here made a report a command as well as a
        query.

        Refreshes metadata and renders the Markdown report template; the
        result is optionally written to `outputfile` (as Markdown or, for
        `format="pdf"`, a PDF).

        Args:
            format: `"markdown"`/`"md"` or `"pdf"`.
            template_name: Base name of the report Mako template.
            outputfile: If given, path the report is written to.
            derivative_notation: Notation for time derivatives, e.g. `"dot"`.
            baseline: Another `Dynamics` to diff against. When given, the report
                lists only the state variables, parameters, derived variables and
                couplings that are new or changed relative to it (a "relative to"
                note replaces the shared rows) — e.g. a controlled variant shown
                against its uncontrolled base without repeating every shared term.
            citeformat: How references are emitted. Default (`None`) renders a
                formatted **References** section at the end (a standalone report).
                `"quarto"` instead emits inline `@key` citations in the fulltext and
                omits the list, so the report can be embedded in a Quarto document
                whose own `bibliography:` resolves the citations into one bibliography.

        Returns:
            The rendered Markdown report string.

        Raises:
            ValueError: If `format` is not one of `markdown`, `md`, or `pdf`.
        """
        normalized_format = format.lower() if isinstance(format, str) else "markdown"
        if normalized_format not in ["markdown", "md", "pdf"]:
            raise ValueError("format must be one of: markdown, pdf")

        md_template = templates.lookup.get_template(f"{template_name}.md.mako")
        md_render = (
            md_template.render(model=self, derivative_notation=derivative_notation, baseline=baseline, citeformat=citeformat)
            .replace(r"\mathcal{lo}_{coupling}", "c_{local}")
            .replace("c_{pop0}", "c_{global}")
        )

        render = md_render

        if outputfile:
            if normalized_format == "pdf":
                report.to_pdf(md_render, outputfile)
            else:
                with open(outputfile, "w") as f:
                    f.write(render)

        return render

    def save_report(self, opath, format="markdown"):
        """Generate the model report and write it to a directory.

        Args:
            opath: Directory the report file is written to (as
                `<name>.<ext>`).
            format: Report format passed to `generate_report` — `"markdown"`
                (written as `.md`) or `"pdf"`.
        """
        self.report_path = opath
        if format in ("markdown", "md"):
            extension = "md"
        else:
            extension = format

        with open(join(opath, f"{self.name}." + extension), "w") as f:
            f.write(self.generate_report(format=format))

    def copy(self, **overrides) -> "Dynamics":
        """Return a deep copy of this experiment.

        Use keyword overrides to set attributes on the returned copy.

        Errors are not swallowed; if a field can't be copied, an exception is raised.
        """
        new_obj = _copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    # Python copy protocol hooks
    def __copy__(self):
        # Keep Python's copy.copy semantics: shallow copy
        cls = self.__class__
        clone = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(clone, k, v)
        return clone

    def __deepcopy__(self, memo):
        import dataclasses

        cls = self.__class__
        # For dataclasses, we need to copy all fields, not just __dict__
        # __dict__ may not include fields that are still at their default values
        data = {}
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                value = getattr(self, field.name, None)
                data[field.name] = _copy.deepcopy(value, memo)
        else:
            # Fallback for non-dataclass
            for k, v in self.__dict__.items():
                data[k] = _copy.deepcopy(v, memo)

        # Create clone using proper constructor to ensure all defaults are set
        clone = cls(**data)
        memo[id(self)] = clone
        return clone


class Model(DynamicalSystem):
    """Deprecated alias for [`Dynamics`](#tvbo.classes.dynamics.Dynamics).

    Kept for backwards compatibility — new code should use `Dynamics`.
    """

    def __init__(self, name, ontology=None, metadata=None, **kwargs):
        super().__init__(name=name, **kwargs)


class Dynamics(DynamicalSystem):
    """A named local neural-mass / population model: parameters, state variables, equations.

    The smallest runnable unit in TVBO. A `Dynamics` binds a name to a set
    of parameters and an ODE system, and is round-trippable through YAML,
    SymPy, and any of the supported backends (JAX, TVB, PyRates, Julia, …).

    Construct one inline, from the curated TVB-O database, or by IRI:

    Examples:
        ```python
        from tvbo import Dynamics

        # Inline
        lorenz = Dynamics(
            parameters={"sigma": {"value": 10.0}, "rho": {"value": 28.0},
                        "beta": {"value": 8/3}},
            state_variables={
                "X": {"equation": {"rhs": "sigma * (Y - X)"}},
                "Y": {"equation": {"rhs": "X * (rho - Z) - Y"}},
                "Z": {"equation": {"rhs": "X * Y - beta * Z"}},
            },
        )

        # From the curated database
        rww = Dynamics.from_db("ReducedWongWangExcInh")

        # By IRI (resolved at construction time)
        rww = Dynamics(iri="tvbo:ReducedWongWangExcInh")
        ```

    See the [writing-models](../../../skills/writing-models/SKILL.md) skill
    for the YAML form and equation conventions.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

