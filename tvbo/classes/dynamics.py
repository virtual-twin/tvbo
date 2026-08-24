# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Dynamics models.

The public import location for :class:`Dynamics`, alongside the helpers that read a model out of the ontology.

There is no wrapper class: a model's own methods live in [`tvbo.behaviour.dynamics`](../behaviour/dynamics.qmd) and are attached to the generated class itself, so a `Dynamics` carries them however it was built — loaded through LinkML, validated through Pydantic, or resolved onto an edge. The equations live on a [`SymbolicSystem`](../parse/system.qmd), one per model.

Examples:
    ```python
    from tvbo import Dynamics

    lorenz = Dynamics(
        parameters={"sigma": {"value": 10.0}, "rho": {"value": 28.0},
                    "beta": {"value": 8/3}},
        state_variables={
            "X": {"equation": {"rhs": "sigma * (Y - X)"}},
            "Y": {"equation": {"rhs": "X * (rho - Z) - Y"}},
            "Z": {"equation": {"rhs": "X * Y - beta * Z"}},
        },
    )

    rww = Dynamics.from_db("ReducedWongWangExcInh")
    rww = Dynamics(iri="tvbo:ReducedWongWangExcInh")
    ```

See the [writing-models](../../../skills/writing-models/SKILL.md) skill for the YAML form and equation conventions.
"""

import functools
import logging
import re
from typing import Any

from sympy import Symbol

from tvbo import templates
from tvbo.classes import equation as _equation_mod
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.datamodel.schema import DerivedVariable, Dynamics
from tvbo.ontology import owl as ontology
from tvbo.utils import yaml_loader

logger = logging.getLogger(__name__)

TEMPLATES = templates.root


@functools.cache
def _available_neural_mass_models():
    """The ontology's neural-mass-model classes, resolved and memoised on first use.

    Kept lazy so importing this module (through ``import tvbo``) does not force the ontology to load — it is only needed to validate/build models from the ontology.
    """
    return set(ontology.get_models().values())


def __getattr__(name):  # PEP 562: keep ``available_neural_mass_models`` importable, lazily.
    if name == "available_neural_mass_models":
        return _available_neural_mass_models()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ontology_class(name):
    """The ontology neural mass model labelled *name*, or ``None``.

    Restricted to that branch on purpose: the ontology holds classes of every kind under one label space, and a model must not be filled from a coupling function or an integrator that happens to share its name.
    """
    if not name:
        return None
    found = ontology.onto.search_one(label=str(name))
    return found if found in _available_neural_mass_models() else None


def populate_from_ontology(model, ontoclass, **kwargs):
    """Fill *model*'s unset schema fields from *ontoclass*.

    The one implementation behind both ways in: `Dynamics.from_ontology`, which is handed the class, and `enrich(source="ontology")`, which resolves it from what the model names. Nothing runtime-only is written, so the model stays serializable.
    """
    class2metadata(ontoclass, model)
    update_parameters(model, ontoclass, **kwargs)


## BifurcationResult moved to tvbo.analysis.bifurcation


def clean_code(code):
    """Replace Unicode infinity (`∞`) with the Python literal `inf`.

    Generated model code occasionally carries the ∞ glyph from upstream ontology labels; SymPy and most backends can't parse it.
    """
    cleaned_code = re.sub(r"∞", "inf", code)
    return cleaned_code


def order_by_equations(derived_variables, dependent_equations):
    """Orders the `derived_variables` dictionary based on the key order of the `dependent_equations` dictionary.

    Parameters:
    derived_variables (dict): Dictionary to be ordered.
    dependent_equations (dict): Dictionary providing the key order for sorting.

    Returns:
    dict: A new dictionary ordered by the key order from `dependent_equations`.
    """
    dependency = {k.replace("dot", ""): v for k, v in dependent_equations.items()}
    # Order derived_variables based on the order in dependent_equations
    ordered_dict = {k: derived_variables[k] for k in dependency if k in derived_variables}

    return ordered_dict


def class2metadata(ontoclass: Any, metadata: Any):
    """Populate a `Dynamics` metadata object from an owlready2 ontology class.

    Fills in description, state variables (with equations, boundaries, and coupling-variable flags), derived variables, and parameters by querying the TVB-O ontology for the corresponding semantic annotations.

    Every name in the model is put into a `local_dict` as a plain `Symbol` before any equation is parsed. Without it, sympy reads `e` as Euler's number and `I` as the imaginary unit, so neither turns up in `free_symbols` and the parameters behind them are dropped in silence. An ontology coupling term is added only where a state equation actually requires it.

    Args:
        ontoclass: The owlready2 class to read from.
        metadata: The `Dynamics` instance to populate in place.
    """
    if not metadata.description:
        metadata.description = ontology.get_def(ontoclass, mode="long")
    dependent_equations = _equation_mod.sort_equations_by_dependencies(_equation_mod.symbolic_model_equations(ontoclass))
    state_variables = ontology.get_model_statevariables(ontoclass)

    functions = order_by_equations(ontology.get_model_functions(ontoclass), dependent_equations)

    for k, v in state_variables.items():
        range = ontology.get_range(v)

        if v.stateVariableBoundaries:
            boundary = ontology.get_range(v.stateVariableBoundaries.first())
            boundaries = tvbo_datamodel.Range(lo=boundary[0], hi=boundary[1])
        else:
            boundaries = None

        # Kept as the sampling distribution when a clamp exists, mirroring the file and adapter paths.
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
                            rhs=td.value.first().replace("numpy.", "").replace("np.", ""),
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
                # An ontology-declared clamp (stateVariableBoundaries) is the operative constraint, so it wins over a pre-existing descriptive (unenforced) domain — consistent with how the file loader folds boundaries → domain+enforce=clamp. The descriptive range is kept as the sampling distribution (a pre-existing distribution takes precedence).
                "domain": _sv_domain or state_var.domain,
                "distribution": state_var.distribution or _sv_distribution,
                "coupling_variable": state_var.coupling_variable or (v in ontoclass.has_cvar),
            }

            for attr, value in updates.items():
                setattr(state_var, attr, value)

    # Update parameters AFTER state_variables are populated so that update_parameters can parse their equations and determine which ontology parameters are actually used.
    update_parameters(metadata, ontoclass)

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
    onto_coupling_terms = ontology.get_model_coupling_terms(ontoclass, only_global=False)
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
                            rhs=v.value.first().replace("numpy.", "").replace("np.", ""),
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
                        equation=tvbo_datamodel.Equation(
                            lhs=name,
                            conditionals=[
                                tvbo_datamodel.ConditionalBlock(condition=condtion, expression=expr)
                                for expr, condtion in val.args
                            ],
                        ),
                    )
                }
            )

    # Only the ontology coupling terms the state equations actually name.
    for k in onto_coupling_terms:
        if k in required_symbols and k not in metadata.coupling_inputs:
            metadata.coupling_inputs[k] = tvbo_datamodel.CouplingInput(name=k)

    for r in ontoclass.has_reference:
        if r.name not in metadata.references:
            metadata.references.append(r.name)


def update_parameters(metadata, ontoclass, verbose=0, only_used=True, **kwargs):
    """Update a model's parameters from the ontology.

    As in `class2metadata`, every model name is bound as a plain `Symbol` first, so `e` and `I` are not read as Euler's number and the imaginary unit and their parameters silently lost.

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
                if hasattr(item, "equation") and item.equation and hasattr(item.equation, "rhs"):
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

        if label not in metadata.parameters and not any(synonym in metadata.parameters for synonym in k.synonym + k.symbol):
            if verbose > 0:
                logger.debug("using parameter %s from the ontology", label)
            metadata.parameters.update(
                {
                    label: tvbo_datamodel.Parameter(
                        name=label,
                        value=kwargs.get(k, v),
                        description=ontology.get_def(k, mode="short").replace("\n", " "),
                        domain=domain,
                        definition=k.definition.first(),
                    )
                }
            )

        if label in metadata.parameters:
            if metadata.parameters[label].description is None:
                metadata.parameters[label].description = ontology.get_def(k, mode="short").replace("\n", " ")

            if metadata.parameters[label].unit is None:
                metadata.parameters[label].unit = k.has_unit.first().name if k.has_unit else k.unit.first()

            if metadata.parameters[label].value is None:
                metadata.parameters[label].value = k.defaultValue.first()


def _clamp_domain(rng):
    """Mark a bounds Range as a hard clamp (``enforce='clamp'``) and return it.

    Folds a legacy ``boundaries`` Range into the unified ``domain`` representation:
    ``boundaries`` always meant "clamp the trajectory to [lo, hi]", which is now expressed as a ``domain`` with ``enforce='clamp'``. Returns None unchanged.
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
    the clamp (``boundaries``) becomes the enforced ``domain``; the descriptive range (the IC-sampling support) is preserved as the sampling ``distribution`` — but only when it differs from the clamp, since an identical clamp already conveys it. With no clamp, the descriptive range is the (unenforced) ``domain`` and there is no separate distribution. ``rng``/``boundaries`` are ``Range`` objects or ``None``.
    """
    if boundaries is None:
        return rng, None
    domain = _clamp_domain(boundaries)
    distribution = None
    if rng is not None and (rng.lo, rng.hi) != (boundaries.lo, boundaries.hi):
        distribution = tvbo_datamodel.Distribution(domain=rng)
    return domain, distribution


def _resolve_dynamics_aliases(d: dict) -> dict:
    """Normalize a Dynamics kwargs/metadata dict through the SINGLE shared route.

    Every construction path — ``Dynamics(**dict)``, ``from_file``, ``from_string`` and the network/experiment coercion helpers — funnels through here, so they apply identical conveniences and cannot drift.

    It applies only what a class cannot: the legacy ``boundaries``/``range`` → ``domain`` fold (``boundaries`` gaining ``enforce: clamp``; a co-existing descriptive ``domain`` preserved as the IC-sampling ``distribution``), and the terse ``distribution: {lo, hi}`` lift. A bare ``domain`` is left untouched (``enforce`` defaults to ``none``), so clamping stays opt-in. Declared slot aliases — ``components`` → ``modes`` among them — are folded by the dialect at construction, at every nesting level, from the schema's own ``aliases:``.

    ``_normalize_loaded`` rebuilds mappings, so the normalized content is written back into ``d`` in place (``clear`` + ``update``) to honour the in-place contract the coercion callers rely on; ``d`` is also returned for convenience.
    """
    normalized = yaml_loader._normalize_loaded(d)
    if normalized is not d:
        d.clear()
        d.update(normalized)
    return d


DynamicalSystem = Dynamics
Model = Dynamics
"""Former runtime subclasses of the generated `Dynamics`, now the class itself.

Both wrapped it only to add methods, which a `Dynamics` carries on its own; keeping the
names spelled differently would say there were three kinds of model when there is one.
``Model`` also declared ``ontology=`` and ``metadata=`` and discarded them — that now
raises, as an unknown keyword should.
"""
