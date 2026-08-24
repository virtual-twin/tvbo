# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Python behaviour layer for `Dynamics` models.

Defines [`DynamicalSystem`](#tvbo.classes.dynamics.DynamicalSystem) — the base class that augments the generated LinkML `Dynamics` datamodel with model construction and ontology resolution, a symbolic (SymPy) representation, equation normalization and dependency-ordered sorting, multi-backend code generation, simulation and bifurcation runs, plotting, and report export — together with the `Model` and `Dynamics` convenience subclasses.
"""

import copy as _copy
import functools
import logging
import os
import re
from os.path import basename, dirname, join, splitext
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import owlready2
from sympy import Derivative, Eq, Function, Symbol, latex

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
from tvbo.parse.expression import function_bodies, parse_eq, states_an_expression
from tvbo.parse.symbols import assumptions_of, symbol_in
from tvbo.utils import initial_value, report, yaml_loader

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


## BifurcationResult moved to tvbo.analysis.bifurcation


def clean_code(code):
    """Replace Unicode infinity (`∞`) with the Python literal `inf`.

    Generated model code occasionally carries the ∞ glyph from upstream ontology labels; SymPy and most backends can't parse it.
    """
    cleaned_code = re.sub(r"∞", "inf", code)
    return cleaned_code


def _normalize_conditionals(model):
    """Ensure dv.equation.conditionals is populated for all conditional DVs.

    If dv.cases is populated but dv.equation.conditionals is empty, copy the cases into equation.conditionals as ConditionalBlock objects. This makes dv.equation.conditionals the single canonical location for conditional data; nothing is written back to `equation.rhs`, which stays whatever the model's author wrote.

    dv.cases is deprecated — new models should define conditionals directly on the equation.
    """
    for dv in model.derived_variables.values():
        cases = getattr(dv, "cases", None)
        if not cases:
            continue
        # Already normalized — skip
        if getattr(dv.equation, "conditionals", None) and len(dv.equation.conditionals) > 0:
            continue
        # Populate equation.conditionals from dv.cases
        dv.equation.conditionals = [ConditionalBlock(condition=case.condition, expression=case.equation.rhs) for case in cases]
        # Mark the DV as conditional if not already
        if not getattr(dv, "conditional", False):
            dv.conditional = True


def _migrate_coupling_terms(model):
    """Bidirectional sync between coupling_terms and coupling_inputs.

    coupling_terms (dict[str, Parameter]) is deprecated in favor of coupling_inputs (dict[str, CouplingInput]).  This function:
    1. Copies coupling_terms entries into coupling_inputs (forward migration) 2. Copies coupling_inputs entries back into coupling_terms as Parameters (backward compat for templates that still read coupling_terms)
    """
    ct = getattr(model, "coupling_terms", None) or {}

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
    state_variables = order_by_equations(ontology.get_model_statevariables(ontoclass), dependent_equations)
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
                                tvbo_datamodel.ConditionalBlock(condition=condtion, expression=expr)
                                for expr, condtion in val.args
                            ],
                        ),
                    )
                }
            )

    # An ontology coupling term is added — to the canonical `coupling_inputs`, with the deprecated `coupling_terms` still consulted — only where a state equation reads it.
    ci_dict = metadata.coupling_inputs
    ct_dict = metadata.coupling_terms
    for k in onto_coupling_terms:
        if k in required_symbols and k not in ci_dict and k not in ct_dict:
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


def sort_equations(model: Any, variable_type: str, graph=None):
    """Reorder `model[variable_type]` by topological dependency order, in place.

    Resolves the model's equation dependency DAG and reorders the variables so each equation appears after the variables it references — required by backends that emit straight-line code (JAX, NumPy printers).

    Variables the dependency graph does not mention keep the order they came in and stay ahead of the sorted ones. Prepending them one at a time instead reverses them, and because each call re-sorts whatever order the last one left behind, that made the result alternate: rendering a model twice emitted its derived variables in opposite orders, so generated code depended on how often it had been generated before.

    Args:
        model: The dynamics model whose equations should be sorted.
        variable_type: Attribute name — typically `"state_variables"`,
            `"derived_variables"`, or `"functions"`.
        graph: An already-built dependency graph. Reordering a collection does not change
            the graph's edges, so a caller sorting several collections builds it once.
    """
    # Skip sorting for list format (e.g., output as list of names)
    if isinstance(model[variable_type], list):
        return

    G_dep = model.get_dependency_tree() if graph is None else graph
    if isinstance(G_dep, tuple):
        G_dep = G_dep[0]
    sorted_variables = []
    for tg in nx.dag.topological_generations(G_dep):
        sorted_variables.extend(sorted(tg, key=lambda x: str(x)))

    original_metadata = model[variable_type].copy()

    sorted_variables_metadata = {}
    for var_name in sorted_variables:
        if str(var_name) in model[variable_type]:
            sorted_variables_metadata[str(var_name)] = original_metadata.pop(str(var_name))

    sorted_variables_metadata = {**original_metadata, **sorted_variables_metadata}

    # Update the original dictionary in-place
    model[variable_type].clear()
    model[variable_type].update(sorted_variables_metadata)


# Readable YAML keys (`components:`) over the datamodel's single canonical attribute (`modes`).
_DYNAMICS_SLOT_ALIASES = {
    "components": "modes",
}


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


def _fold_component_alias(d: dict) -> None:
    """Recursively rename the Dynamics-only ``components`` → ``modes`` slot alias.

    ``components`` is a ``modes`` alias only inside a Dynamics, so it is folded here (and by the class-scoped fold in the loader) rather than anywhere a ``components`` key appears. Mutates ``d`` in place at every nesting level.
    """
    for alias, canonical in _DYNAMICS_SLOT_ALIASES.items():
        if alias in d:
            if canonical in d:
                raise ValueError(f"Cannot specify both '{alias}' and '{canonical}' — '{alias}' is an alias for '{canonical}'.")
            d[canonical] = d.pop(alias)
    modes = d.get("modes")
    if isinstance(modes, dict):
        for v in modes.values():
            if isinstance(v, dict):
                _fold_component_alias(v)


def _resolve_dynamics_aliases(d: dict) -> dict:
    """Normalize a Dynamics kwargs/metadata dict through the SINGLE shared route.

    Every construction path — ``Dynamics(**dict)``, ``from_file``, ``from_string``, the ``iri`` backfill, and the network/experiment coercion helpers — funnels through here, so they apply identical conveniences and cannot drift:

    * the Dynamics-specific ``components`` → ``modes`` alias (recursively), then
    * :func:`tvbo.utils.yaml_loader._normalize_loaded` — the one implementation
      shared with the LinkML ``load``/``loads``/``load_as_dict`` path: the aliases ``Dynamics`` declares, the legacy ``boundaries``/``range`` → ``domain`` fold (``boundaries`` gaining ``enforce: clamp``; a co-existing descriptive ``domain`` preserved as the IC-sampling ``distribution``), and the terse ``distribution: {lo, hi}`` lift. A bare ``domain`` is left untouched (``enforce`` defaults to ``none``), so clamping stays opt-in.

    ``_normalize_loaded`` rebuilds mappings, so the normalized content is written back into ``d`` in place (``clear`` + ``update``) to honour the in-place contract the coercion callers rely on; ``d`` is also returned for convenience.
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
        if isinstance(first_val, dict) and ("equation" in first_val or "rhs" in first_val):
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

    Wraps the generated LinkML `Dynamics` datamodel with the methods that make a model usable: ontology resolution (`use_ontology=True`), symbolic representation via SymPy, equation reordering, backend code generation, YAML / JSON / Pydantic round-tripping, and matplotlib plotting hooks.

    Most users should construct via [`Dynamics`](#tvbo.classes.dynamics.Dynamics) or `Dynamics.from_db(name)` — this class is the implementation base.

    `_skip_ontology` says the model arrives fully specified — an `iri=` registry entry, a PyRates import — so the slow ontology lookup is pointless. It does **not** mean the record needs no normalising: every path runs `update_metadata`, which migrates the deprecated `cases:` slot into conditionals and sorts derived variables into the dependency order the straight-line JAX and NumPy emitters require. Returning early on this flag left `cases:` models rendering with no branches at all.
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
            from tvbo.data.registry import local_name, resolve
            from tvbo.utils import deep_merge

            local = local_name(iri)
            try:
                # Merged into constructor kwargs, so the entry's file envelope has to go: `load_as_dict` keeps it for callers that dispatch on it.
                loaded = yaml_loader.strip_envelope(yaml_loader.load_as_dict(str(resolve("Dynamics", local))))
                _resolve_dynamics_aliases(loaded)
                # Registry entry is the base; inline kwargs override at the leaf (e.g. parameters.a.value wins, siblings kept from the entry).
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

        # Auto-populate only when a name was provided; keep default Dynamics() empty
        if name != "Dynamics":
            if use_ontology and not _skip_ontology:
                self._populate_from_ontology_by_name()

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
    def from_datamodel(cls, model_meta: tvbo_datamodel.Dynamics, use_ontology: bool = False):
        """Create from a datamodel Dynamics instance by copying its already-normalized state (avoids ``_as_dict`` re-init crash on ``inlined_as_dict`` fields)."""
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
            ontoclass = query.label_search(ontoclass, root_class="NeuralMassModel", exact_match=["label"])[0]
        inst = cls(name=ontoclass.name, **kwargs)
        inst._populate_from_ontology(ontoclass, **kwargs)
        inst.update_metadata()
        inst.calculate_derived_parameters()
        return inst

    @classmethod
    def from_file(cls, path: str | os.PathLike, use_ontology: bool = False) -> "Dynamics":
        """Load a model from a YAML/JSON specification file on disk.

        Args:
            path: Path to a TVBO model specification file.
            use_ontology: If `True`, backfill missing fields from the
                ontology after loading.

        Returns:
            The instance parsed from the file.
        """
        data = yaml_loader.strip_envelope(yaml_loader.load_as_dict(str(path)))
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
        data = yaml_loader.strip_envelope(yaml_loader.load_as_dict(str)) or {}
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

        Fetches the full LinkML-valid YAML definition from the platform and constructs a Dynamics instance.

        Parameters
        ----------
        name : str
            Model name (e.g., "JansenRit", "ReducedWongWang").
        base_url : str
            Platform base URL.

        Returns:
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

        Returns:
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

        Returns:
        -------
        Dynamics
            New Dynamics instance populated from the PyRates template.

        Example:
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

        Examples:
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

        Examples:
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

        Looks up the model name in the TVB ontology and backfills missing parameter values, descriptions, ranges, state-variable metadata, and derived variables.  Useful when you define a partial model spec and want the ontology to fill in the gaps.

        Example:
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

        Returns:
        -------
        str
            YAML string or filepath if written to file.

        Example:
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
        return {Symbol(p.name): p.value for p in self.parameters.values()}

    @property
    def symbolic(self):
        """Full symbolic ODE system using proper SymPy conventions.

        State variables are represented as ``Function(name)(t)`` so that ``Derivative(theta(t), t)`` stays unevaluated.  Derived variables and derived parameters are included as algebraic equations.

        Returns:
        -------
        dict
            ``{'state': [...], 'derived': [...], 'parameters': {...}}``
            where each list contains ``sympy.Eq`` objects and parameters
            maps ``Symbol → value``. That map is keyed by the scope's own
            symbols: rebuilt keys look identical, compare unequal, and make
            substituting it into these equations silently replace nothing.

        Example:
        -------
        >>> model.symbolic['state']
        [Eq(Derivative(theta(t), t), I + omega)]
        >>> model.symbolic['derived']
        [Eq(signal(t), sin(theta(t)))]
        >>> model.symbolic['units']
        {omega: 'per_ms', I: None}
        """
        from tvbo.analysis.units import declared_units

        form = self._symbolic_form(notation="function")
        scope = self.get_symbolic_elements(time_dependent=True)
        return {
            "state": list(form["state-equations"].values()),
            "functions": list(form["functions"].values()),
            "derived_parameters": list(form["derived-parameters"].values()),
            "derived": list(form["derived-variables"].values()),
            "parameters": {scope[str(p.name)]: p.value for p in self.parameters.values() if str(p.name) in scope},
            "units": declared_units(self),
        }

    def check_units(self, strictness: str = "dimensional", time_unit: str | None = None):
        """Per-equation dimensional verdicts for this model.

        See [`tvbo.analysis.units.check_units`](../analysis/units.qmd#check_units). Each verdict is `consistent`, `inconsistent` or `underdetermined`; the third is a distinct answer, not a soft failure, because 24 of the 39 curated models declare no units and calling those wrong would pressure fake declarations into the published record.
        """
        from tvbo.analysis.units import check_units

        return check_units(self, strictness=strictness, time_unit=time_unit)

    def get_symbolic_elements(self, include_time_symbol: bool = True, time_dependent: bool = False):
        """Build a unified local_dict for parsing model expressions.

        Includes symbols for parameters, coupling terms, derived parameters, derived variables, output transforms, state variables, event names, function names, and (optionally) the time symbol 't'.

        Every declared name must appear here so it shadows SymPy's own global namespace:
        `Q` is SymPy's assumptions object, `S` its sympify shortcut, `O` big-O, `N` numeric evaluation and `I` the imaginary unit, so a model that names a quantity after any of them would otherwise fail to parse.

        Args:
            include_time_symbol: Bind `t` to `Symbol("t")`.
            time_dependent: Bind state and derived variables to `Function(name)(t)` rather than `Symbol(name)`, so `Derivative(x(t), t)` stays unevaluated and the result reads as a system of ODEs. This is the only difference between the two symbolic views of a model — everything downstream of the scope is shared, which is why it is a parameter here and not a second builder.

        Returns:
        -------
        dict
            Mapping of names to SymPy objects suitable for parse_eq(local_dict=...).
            A copy, so a caller may keep or adapt it; the model's own is cached.
        """
        key = (bool(include_time_symbol), bool(time_dependent))
        scopes = self._symbolic_state()["scopes"]
        if key not in scopes:
            scopes[key] = self._build_symbolic_elements(include_time_symbol, time_dependent)
        return dict(scopes[key])

    def _build_symbolic_elements(self, include_time_symbol: bool, time_dependent: bool):
        """Assemble the symbol table. See `get_symbolic_elements`.

        Holds only the names the *model* declares. A function's formal arguments are bound by that function, exactly as a lambda binds its parameters, and are supplied as an overlay while its body is parsed — see `_assemble_equations`. Registering them here let a formal shadow a variable of the same name: `ReducedWongWangTvboptim` declares both `H(x)` and a derived variable `x`, and the formal won, so the analysis view held `x` constant and dropped the chain-rule term from every Jacobian through `H`.

        Assumptions ride on the time-dependent view only. They are what SymPy's analysis machinery needs — without `real=True` the fixed points of a two-variable model do not come back inside a minute — but they also enter `Symbol.sort_key`, so the same product prints as `q*alpha` instead of `alpha*q`. That is no gain for a backend that parses, inlines and prints without ever simplifying, and every emitted file is compared against a frozen reference. The codegen view therefore stays plain, and the two are never mixed: a `Symbol` from one does not compare equal to the same name from the other, so nothing can substitute across them by accident.

        Function heads are the exception, and carry `assumptions_of()` in both views: a head is notation-independent — `Sigm` is the same function whether the variables around it are Symbols or Functions of `t`. Building it per view made `Function("Sigm", real=True)` and `Function("Sigm")`, which print identically, compare unequal, and make `expr.has(Sigm)` False on an expression that visibly calls it, so every inliner matched nothing, silently.
        """

        def _assume(element=None):
            return assumptions_of(element) if time_dependent else {}

        def _symbol(name, element=None):
            return Symbol(str(name), **_assume(element))

        t = _symbol("t")
        scope: dict[str, object] = {}

        def _variable(name, element=None):
            if time_dependent:
                return Function(str(name), **_assume(element))(t)
            return _symbol(name, element)

        if include_time_symbol:
            scope["t"] = t

        for p in self.parameters.values():
            scope[str(p.name)] = _symbol(p.name, p)

        # Coupling inputs (named inputs from coupling function)
        for ci in self.coupling_inputs.keys():
            scope[str(ci)] = _symbol(ci)

        # A derived parameter is constant in time, so it stays a Symbol in both views.
        for name in self.derived_parameters.keys():
            scope[str(name)] = _symbol(name)
        for name, dv in self.derived_variables.items():
            scope[str(name)] = _variable(name, dv)

        # Output is a list of string references
        for name in self.output:
            scope[str(name)] = _variable(name, self.derived_variables.get(str(name)))

        for name, sv in self.state_variables.items():
            scope[str(name)] = _variable(name, sv)

        for fname in self.functions:
            scope[str(fname)] = Function(str(fname), **assumptions_of())

        for name in self.events:
            scope[str(name)] = _symbol(name)

        if "e" not in scope:
            from sympy import E

            scope["e"] = E

        return scope

    _GROUP_COLLECTIONS = {
        "derived-parameters": "derived_parameters",
        "functions": "functions",
        "derived-variables": "derived_variables",
        "state-equations": "state_variables",
        "output-transformations": "output",
    }
    """Which collection each equation group is built from, and takes its order from."""

    def _equation_inputs(self):
        """What `_build_symbolic_form` reads, split into content and order.

        The cache is sound only if *content* changes whenever a built equation would, so it walks the same collections the builder walks rather than a hand-listed subset: a slot the builder starts reading without being added here would serve a stale equation forever. Content is compared as dicts, which ignore key order.

        *order* is tracked separately because `sort_equations` reorders collections into
        dependency order without changing a single equation — five times over one load.
        Treating that as a content change would re-parse everything to produce the same expressions in a different sequence.
        """

        def _equation(element):
            equation = getattr(element, "equation", None)
            if equation is None:
                return None
            return (
                equation.rhs,
                tuple((c.condition, c.expression) for c in equation.conditionals),
                bool(equation.latex),
            )

        def _assumed(element):
            """Keyed on `assumptions_of` itself, so the key cannot drift from what it reads.

            A `domain` is not an equation, but it decides whether a symbol is `positive` or merely `real`, and `Symbol('a', positive=True) != Symbol('a', real=True)`. Naming the fields here instead would leave the key stale the day `assumptions_of` starts reading one more of them.
            """
            return tuple(sorted(assumptions_of(element).items()))

        content = (
            self.system_type,
            {str(name): _assumed(p) for name, p in self.parameters.items()},
            frozenset(str(name) for name in self.coupling_inputs),
            frozenset(str(name) for name in self.events),
            frozenset(str(name) for name in self.output),
            {str(k): _equation(v) for k, v in self.derived_parameters.items()},
            {str(k): (_equation(v), _assumed(v)) for k, v in self.derived_variables.items()},
            {
                str(k): (
                    _equation(v),
                    int(v.equation_order or 1) if v.equation_order else 1,
                    _assumed(v),
                )
                for k, v in self.state_variables.items()
            },
            {str(k): (tuple(str(a) for a in v.arguments), _equation(v)) for k, v in self.functions.items()},
        )
        order = tuple(
            tuple(str(name) for name in getattr(self, collection)) for collection in self._GROUP_COLLECTIONS.values()
        )
        return content, order

    def _reordered(self, form):
        """The same equations, re-keyed into their collections' current order."""
        return {
            group: {
                name: equations[name]
                for name in (str(n) for n in getattr(self, self._GROUP_COLLECTIONS[group]))
                if name in equations
            }
            for group, equations in form.items()
        }

    def _symbolic_form(self, notation: str = "symbol", evaluate: bool = True):
        """The model's equations, parsed once per (notation, evaluate) and remembered.

        The single symbolic layer between a model's metadata and everything rendered from it. Both public views — [`get_equations`](#tvbo.classes.dynamics.Dynamics.get_equations) and [`symbolic`](#tvbo.classes.dynamics.Dynamics.symbolic) — are projections of this, as is the function-body table the inliner consumes, so an equation is parsed once no matter how many consumers ask for it.

        Before this existed each caller re-derived from metadata: loading `ZerlautAdaptationSecondOrder` parsed its 27 equations 264 times, and every `render_code` and `generate_report` parsed all 27 again because nothing was kept.

        The cache is discarded whole whenever `_equation_inputs` changes, which is what makes it safe on a mutable model. Rendering is a query — `render_code` no longer runs `update_metadata` — so no consumer can invalidate it mid-use.

        Args:
            notation: `"symbol"` binds variables to `Symbol(name)`; `"function"` binds
                them to `Function(name)(t)`.
            evaluate: Let SymPy evaluate right-hand sides, or preserve authored term order.

        Returns:
            `{group: {name: Eq}}` over the five groups, each keyed by the variable it
            defines so no consumer has to recover a name from an `Eq`'s left-hand side.
        """
        forms = self._symbolic_state()["forms"]
        key = (notation, bool(evaluate))
        if key not in forms:
            forms[key] = self._build_symbolic_form(notation, evaluate)
        return forms[key]

    def _symbolic_state(self):
        """The per-content cache the symbol table and the equations share.

        One invalidation point for both, because they have to agree: a scope built from one set of names and equations parsed against another is precisely the drift this layer exists to remove.

        A reorder keeps the parsed equations and re-keys them; the scopes are dropped instead, since rebuilding a symbol table is a few hundred `Symbol` constructions while reparsing is the expensive half.
        """
        content, order = self._equation_inputs()
        cache = self.__dict__.get("_symbolic_cache")
        if cache is None or cache[0] != content:
            cache = (content, order, {"scopes": {}, "forms": {}})
        elif cache[1] != order:
            reordered = {key: self._reordered(form) for key, form in cache[2]["forms"].items()}
            cache = (content, order, {"scopes": {}, "forms": reordered})
        else:
            return cache[2]
        object.__setattr__(self, "_symbolic_cache", cache)
        return cache[2]

    def _build_symbolic_form(self, notation: str, evaluate: bool):
        """Parse every equation the model states, once. See `_symbolic_form`.

        The two views differ in what they are for, and the evaluation policy follows from that rather than the other way round.

        `"symbol"` feeds codegen, which parses, inlines and prints. It honours the caller's
        *evaluate* so a backend can keep the term order its author wrote.

        `"function"` is the analysis view — the one `Matrix.jacobian`, `solve` and `dsolve` act on — so it is canonical. It used to suppress evaluation globally, which kept `Derivative(theta(t), t)` from collapsing but also left the right-hand sides in a nested unevaluated form that SymPy's solvers cannot make progress on: asked for the fixed points of `Generic2dOscillator` in that form, `solve` returns nothing in 45 s;
        canonical and with real symbols it answers in under one. `Derivative` is built explicitly here, so nothing needs the global suppression to survive.
        """
        time_dependent = notation == "function"
        return self._assemble_equations(
            time_dependent=time_dependent,
            evaluate=True if time_dependent else evaluate,
        )

    def _assemble_equations(self, time_dependent: bool, evaluate: bool):
        """Build the five equation groups against one scope. See `_build_symbolic_form`.

        Every symbol an equation's left-hand side names is resolved through `scope` — the same table the right-hand sides were parsed against. Minting one here instead produces a name that prints identically and compares unequal once the analysis view attaches assumptions, and `subs` across that mismatch replaces nothing rather than raising: a derivative taken w.r.t. a freshly built `t` leaves `doit()` returning 0, and a derived parameter's definition substitutes into none of its own equations.
        """
        scope = self.get_symbolic_elements(time_dependent=time_dependent)
        t = symbol_in(scope, "t")
        discrete = self.system_type == "discrete"

        def _lhs(name):
            return symbol_in(scope, name)

        def _states(element):
            """Whether the element has anything to parse — see `states_an_expression`.

            An element declared with no `rhs` and no conditionals is skipped rather than parsed. Every rendering path now funnels through here, so one such element used to break all of them at once instead of only `get_equations`.
            """
            return states_an_expression(getattr(element, "equation", None))

        def _parse(element, namespace=None):
            return parse_eq(element.equation, local_dict=namespace or scope, evaluate=evaluate)

        def _formal(name):
            """A function's bound argument — a quantity, never a state, so never `name(t)`."""
            return Symbol(str(name), **(assumptions_of() if time_dependent else {}))

        def _function_scope(function):
            """The model's names with this function's formals bound over them."""
            return {**scope, **{str(a): _formal(a) for a in function.arguments}}

        form = {
            "derived-parameters": {
                str(k): Eq(lhs=_lhs(k), rhs=_parse(dp)) for k, dp in self.derived_parameters.items() if _states(dp)
            },
            "functions": {
                str(k): Eq(
                    lhs=_lhs(k)(*[_formal(a) for a in f.arguments]),
                    rhs=_parse(f, _function_scope(f)),
                )
                for k, f in self.functions.items()
                if _states(f) and f.arguments
            },
            "derived-variables": {
                str(k): Eq(lhs=_lhs(k), rhs=_parse(dv)) for k, dv in self.derived_variables.items() if _states(dv)
            },
            "state-equations": {},
            "output-transformations": {},
        }

        for k, sv in self.state_variables.items():
            if not _states(sv):
                continue
            order = int(sv.equation_order or 1)
            lhs = _lhs(k) if discrete else Derivative(_lhs(k), *([t] * order))
            form["state-equations"][str(k)] = Eq(lhs=lhs, rhs=_parse(sv))

        # An identity equation for an output that IS a state variable overwrites its real one.
        for name in self.output:
            name = str(name)
            if name in self.derived_variables:
                if _states(self.derived_variables[name]):
                    form["output-transformations"][name] = Eq(lhs=_lhs(name), rhs=_parse(self.derived_variables[name]))
            elif name not in self.state_variables:
                raise ValueError(f"Output variable '{name}' not found in derived_variables or state_variables")

        return form

    def _items(self):
        """What the LinkML dumpers see: schema slots only.

        Mirrors the guard on `Network`. `_symbolic_cache` holds SymPy objects that `yaml.SafeDumper` cannot represent, and LinkML slot names are never underscore-prefixed, so excluding every leading-underscore key keeps this correct as further caches are added rather than depending on a maintained denylist.
        """
        for k, v in super()._items():
            if not str(k).startswith("_"):
                yield k, v

    def symbol_map(self):
        """Display-symbol overrides for report rendering: ``{identifier Symbol: LaTeX str}``.

        For each element that declares a ``symbol`` (e.g. ``w_+`` for the identifier ``w_plus``, or ``S^{(E)}`` for ``S_e``), map its identifier Symbol to the LaTeX of that override, so ``sympy.latex(expr, symbol_names=model.symbol_map())`` renders the source's own notation. Elements without an override are omitted (they render from their identifier). Fully sympy-native: the override is itself rendered through ``sympy.latex(Symbol(...))``, inheriting Greek/sub/superscript handling.

        Keyed by the canonical collection keys (the identifiers used in the equations), over the same element collections as [`get_symbolic_elements`](#tvbo.classes.dynamics.Dynamics.get_symbolic_elements).
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
        """Normalize the model's equation metadata in place.

        Migrates the deprecated `cases` and `coupling_terms` slots onto `conditionals` and `coupling_inputs`, then sorts derived parameters, derived variables and outputs into dependency order — which the backends emitting straight-line code (JAX, NumPy) require. Every construction path runs this, including the ones that skip the ontology lookup because the model already arrived fully specified.

        It used to also call `update_equations`, whose result it discarded. Once `get_equations` became a projection of the symbolic layer, returning `Eq` objects already keyed by the bare variable name, that call's loop re-filed each entry under the key it already had: verified a no-op on all 106 curated models.
        """
        _normalize_conditionals(self)
        _migrate_coupling_terms(self)

        # Reordering a collection cannot change the graph's edges, so all three sorts share one build.
        graph = self.get_dependency_tree()
        for collection in ("derived_parameters", "derived_variables", "output"):
            sort_equations(self, collection, graph=graph)

    # Fluent builder helpers and setters
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
                return tvbo_datamodel.Range(lo=float(lo), hi=float(hi), step=float(step))
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

    def update_parameters_from_equations(self, default_value: float = 1.0, overwrite: bool = False):
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
        to_remove = [pname for pname in list(self.parameters.keys()) if str(pname) in nonparam_known]
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
                self.parameters[s] = tvbo_datamodel.Parameter(name=s, value=float(default_value))
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

        Any free symbols in `equation` that are not yet known are auto-registered as parameters. A legacy `boundaries` clamp is folded into `domain` (with the descriptive range preserved as the sampling `distribution`).

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
        eq = self._coerce_equation(equation, lhs=str(name)) if equation is not None else None
        # ``boundaries`` is the legacy name for a hard clamp; fold it into the unified ``domain`` with enforce='clamp'. When both are given, the clamp is the operative domain and the descriptive ``domain`` (the IC-sampling range) is preserved as the sampling ``distribution`` rather than dropped.
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
                cond_blocks.append(tvbo_datamodel.ConditionalBlock(condition=str(cond), expression=str(expr)))
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
                str(k): (v if isinstance(v, tvbo_datamodel.Parameter) else tvbo_datamodel.Parameter(name=str(k)))
                for k, v in arguments.items()
            }
        else:
            args = list(arguments) if isinstance(arguments, (list, tuple)) else []
            args_dict = {str(a): tvbo_datamodel.Parameter(name=str(a)) for a in args}
        eq = self._coerce_equation(expression, lhs=str(name)) if expression is not None else None
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

        Any existing parameter with the same name is removed so the name resolves to the coupling input.

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
        eq = self._coerce_equation(expression, lhs=name_str) if expression is not None else None
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
        eq = self._coerce_equation(expression, lhs=str(name)) if expression is not None else None
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

    def symbolic_rhs(self, obj, evaluate: bool = True):
        """The parsed right-hand side of one of this model's elements.

        Resolved through the symbolic layer, so rendering an element reuses the expression already parsed for it rather than parsing its metadata again — the reason a second `render_code` on the same model costs nothing. Falls back to the element's own `Equation` for anything the model does not declare (a stimulus, a caller's ad-hoc element); `parse_eq` accepts either, so the caller does not need to know which.

        *evaluate* must match what the caller would have parsed with. A backend that
        preserves authored term order needs the unevaluated form: SymPy canonicalises `a + V*b + c*V**2` out of the order its author wrote it in, and the emitted source is compared against a frozen reference.
        """
        name = str(obj.name) if getattr(obj, "name", None) else ""
        if name:
            for group in self._symbolic_form(evaluate=evaluate).values():
                if name in group:
                    return group[name].rhs
        return obj.equation

    def _printer_arguments(self, obj, inline_functions: bool, kwargs: dict) -> dict:
        """What both render paths hand the printer, stated once.

        Inlining and naming are exclusive: a body substituted into the expression leaves no function head for the printer to name, so the user-function table empties exactly when a body table is supplied. `evaluate` follows the caller's `preserve_order`, so the expression is parsed the way the caller is about to print it.
        """
        inline_funcs = function_bodies(self) if inline_functions else None
        return {
            "equation": self.symbolic_rhs(obj, evaluate=not kwargs.get("preserve_order", False)),
            "local_dict": self.get_symbolic_elements(),
            "user_functions": {} if inline_funcs else {str(name): str(name) for name in self.functions},
            "inline_funcs": inline_funcs,
        }

    def render_equation(self, obj, format="latex", inline_functions=False, **kwargs):
        """Render a model element's equation to a string.

        Handles conditional derived variables (converting `conditionals` to a SymPy `Piecewise`) and can optionally inline the model's function definitions.

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
        from tvbo.codegen.code import render_equation

        return render_equation(format=format, **self._printer_arguments(obj, inline_functions, kwargs), **kwargs)

    def render_equation_cse(self, obj, format="numpy", inline_functions=False, **kwargs):
        """Common-subexpression-eliminated variant of :meth:`render_equation`.

        Returns ``(setup, final)`` — a list of ``(name, expr)`` assignments plus the return expression — so interpreted backends (TVB / numpy) evaluate each shared subexpression (notably repeated model-function calls) once instead of per occurrence. Builds the same symbolic scope / user-function set as :meth:`render_equation`; see :func:`tvbo.codegen.code.render_equation_cse`.
        """
        from tvbo.codegen.code import render_equation_cse

        return render_equation_cse(format=format, **self._printer_arguments(obj, inline_functions, kwargs), **kwargs)

    def get_equations(self, format="metadata", evaluate=True):
        """Collect the model's equations as SymPy `Eq` objects.

        Builds equations for derived parameters, functions, derived variables, state equations (as time derivatives, or plain maps for discrete systems), and output transformations.

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
        form = self._symbolic_form(notation="symbol", evaluate=evaluate)

        if format == "state-equations":
            return dict(form["state-equations"])
        if format == "dict":
            return {group: list(equations.values()) for group, equations in form.items()}
        return {name: equation for group in form.values() for name, equation in group.items()}

    def fill_in_equations(self, **kwargs):
        """Substitute parameter values (and any overrides) into every equation.

        Parameter symbols are replaced with their numeric values, then any `**kwargs` overrides are applied, and finally all coupling inputs are forced to `0` (for fixed-point / equilibrium analysis) — so a `kwargs` entry named after a coupling input is overridden by that `0`.

        Args:
            **kwargs: Additional symbol-name to value substitutions.

        Returns:
            The list of equations with substitutions applied.
        """
        # Substitute parameters and defaults into equations (useful for fixed-point search)
        sub = self.keyed_parameters
        sub.update(kwargs)
        # Set all coupling inputs to 0 for fixed-point analysis
        for ci in self.coupling_inputs.keys():
            sub[ci] = 0
        return [eq.subs(sub) for eq in self.get_equations().values()]

    def calculate_derived_parameters(self):
        """Evaluate and cache each derived parameter's numeric value.

        Scalar parameters are substituted into every derived-parameter equation and the result stored on the model. Array-valued or otherwise unresolved derived parameters keep a `None` value and are recomputed at runtime by the generated code.

        Returns:
            A mapping of derived-parameter name to its computed value (or
            `None`), or `None` if the model has no derived parameters.
        """
        if self.derived_parameters is None:
            return None
        from tvbo.utils import is_array_valued

        # Loop-invariant: build the symbol scope and the scalar-parameter substitution once. Array-valued parameters (mode-coupling matrices, quadrature vectors) are excluded — they have no load-time scalar and would make ``subs`` raise — so the derived parameters that depend on them stay symbolic and are recomputed at runtime.
        local_dict = self.get_symbolic_elements()
        scalar_subs = {Symbol(p.name): p.value for p in self.parameters.values() if not is_array_valued(p.value)}
        for k, dp in self.derived_parameters.items():
            try:
                eq = parse_eq(dp.equation, local_dict=local_dict, evaluate=False).subs(scalar_subs)
                # Convert SymPy Float to Python float for YAML serialization.
                self.derived_parameters[k].value = float(eq.evalf())
            except (TypeError, ValueError):
                # Array-valued or unresolved: recomputed at runtime by update_derived_parameters.
                self.derived_parameters[k].value = None
        return {k: self.derived_parameters[k].value for k in self.derived_parameters}

    def get_dependency_tree(self, ontomapping=False, include_state_equations=False):
        """Build the equation dependency graph for this model.

        Nodes are the model's symbols; each edge points from a dependency to the quantity whose equation uses it (dependencies → dependents). The graph exists mainly to sort derived quantities, and state equations are excluded by default because a discrete system makes them cyclic: an algebraic derived variable depends on the states, and the states depend on that derived variable at the same step.

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
        coupling_term_names = set(self.coupling_inputs.keys())
        for n in G.nodes:
            suffix = ontology.get_model_suffix(self.ontology or self.name) if str(n) not in coupling_term_names else ""
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
        edgekwargs=None,
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

        if edgekwargs is None:
            edgekwargs = {"connectionstyle": "arc3,rad=0", "edge_color": "grey"}
        if not ax:
            fig, ax = plt.subplots(figsize=(12, 2))
            return_fig = True
        else:
            return_fig = False

        G = self.get_dependency_tree(include_state_equations=True)
        if isinstance(G, tuple):
            G = G[0]

        if color_nodes_by is not None:
            G, G_onto, symbol_onto_mapping, onto_symbol_mapping = self.get_dependency_tree(
                ontomapping=True, include_state_equations=True
            )
            edgecolor = None
            cat_dict, categories = ontology_plot.get_node_color_mapping(G_onto, color_nodes_by, return_categories=True)
            kwargs.update({"node_colors": [cat_dict[categories[symbol_onto_mapping[n]]] for n in G.nodes]})

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
                pos = {key: (x, min_y) if isinstance(key, sympy.Derivative) else (x, y) for key, (x, y) in pos.items()}
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

        Dispatches to the template (or adapter) for the requested backend and returns the formatted source. Reads the model and does not modify it, so the source depends on the model alone and not on how often it has been rendered.

        Normalisation belongs to construction: every path that builds a `Dynamics` runs [`update_metadata`](#tvbo.classes.dynamics.DynamicalSystem.update_metadata) already. Repeating it here made rendering a command as well as a query, and because the normalisation was not idempotent the emitted code alternated between two spellings depending on the number of previous renders. A model mutated by hand after construction should be normalised by whoever mutated it.

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
            template = templates.lookup.get_template("tvbo-julia-DifferentialEquations.jl.mako")
        elif format == "bifurcation-julia":
            from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

            continuation = kwargs.pop("continuation", None)
            ctx = BifurcationKitAdapter._prepare_context(self, continuation, **kwargs)
            template = templates.lookup.get_template("tvbo-julia-BifurcationKit.jl.mako")
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
        elif format.lower() in ["pyrates", "pyrates-yaml", "pyrates_yaml"]:
            from tvbo.codegen.pyrates import to_pyrates_model_yaml

            rendered_code = to_pyrates_model_yaml(self, **kwargs)
            return templater.format_code(rendered_code, format="pyrates")
        else:
            raise ValueError(f"Format {format} not supported.")

        rendered_code = template.render(model=self, format=format, jax="jax" in format, **kwargs)
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

        Returns:
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

        code = templater.format_code(self.render_code(format=format, **kwargs), format=format)
        return Markdown(f"```{'python' if format in ['tvb', 'python', 'jax', 'autodiff'] else format}\n{code}\n```")

    def execute(self, format="tvb", **kwargs):
        """Generate and execute the model code, returning a runnable object.

        Dispatches on `format`: builds a configured TVB model instance, a tvboptim dynamics instance, a compiled C module (`sympy2c`), a bifurcation/continuation run, or a plain dfun callable.

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
                Module = _sympy2c.Module
                OdeFast = _sympy2c.OdeFast
            except Exception as e:
                raise RuntimeError("sympy2c is not installed. Install it to use format='c' or 'sympy2c'.") from e

            params = self.keyed_parameters
            params.update({Symbol(str(ci)): 0.0 for ci in self.coupling_inputs})
            params.update({Symbol("local_coupling"): 0.0})

            scope = self.get_symbolic_elements()
            derived_variables = {Symbol(k): parse_eq(v.equation, local_dict=scope) for k, v in self.derived_variables.items()}

            lhs = list()
            rhs = list()
            for k, v in self.get_equations(format="state-equations").items():
                lhs.append(Symbol(k))
                expr = v.rhs.subs(params).subs(derived_variables).subs(derived_variables).subs(derived_variables).subs(params)
                rhs.append(expr)

            module_decl = Module()
            module_decl.add(OdeFast("robertson", Symbol("t"), lhs, rhs))
            imported_module = module_decl.compile_and_load()
            return imported_module

        elif format in ["bifurcation-numcont", "bifurcation-auto7p"]:
            # In-tree AUTO-07p: a one-off SimulationExperiment around this Dynamics, run through NumContAdapter.
            from tvbo.adapters.numcont import NumContAdapter
            from tvbo.classes.continuation import Continuation
            from tvbo.classes.experiment import SimulationExperiment

            cont = kwargs.pop("continuation", None) or Continuation(name=self.name + "_eq")
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

    def get_run_filename(self, format, **kwargs):
        """Build a deterministic cache filename for a run in the temp directory.

        Non-identifying keyword arguments (e.g. `filename`, `force`, `verbose`) are dropped and the rest are sorted so the same run maps to the same path.

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
            self.name + f"_format-{format}_" + "_".join(f"{k}-{v}" for k, v in kwargs.items()),
        )

        return filename

    def get_initial_values(self, default=0.1, N=1, **kwargs):
        """Build the initial state vector for a simulation.

        A state variable that declares a `distribution` is sampled from it — Gaussian, or uniform over the finite domain bounds; every other variable uses its `initial_value`, or *default*. Declaring a distribution is the only way to ask for sampling: the old `random=True` flag sampled every variable's raw domain regardless of what the model said, and is gone.

        Args:
            default: Fallback value for variables without an initial value.
            N: Number of samples per state variable.
            **kwargs: Ignored extra arguments.

        Returns:
            A NumPy array of initial values, shaped `(n_state_variables, N)` when any variable declares a distribution, and 1-D with one entry per state variable otherwise.
        """
        has_distributions = any(getattr(sv, "distribution", None) for sv in self.state_variables.values())
        if has_distributions:
            init = []
            for sv in self.state_variables.values():
                dist = getattr(sv, "distribution", None)
                if dist:
                    # Use distribution.domain, fall back to sv.domain
                    domain = getattr(dist, "domain", None) or getattr(sv, "domain", None)
                    # Guard against non-finite / missing bounds: a distribution without its own domain falls back to sv.domain, which may be a half-open clamp (e.g. [0, inf)); uniform(0, inf) would overflow.
                    _dlo = getattr(domain, "lo", None) if domain else None
                    _dhi = getattr(domain, "hi", None) if domain else None
                    lo = float(_dlo) if (isinstance(_dlo, (int, float)) and np.isfinite(_dlo)) else -10.0
                    hi = float(_dhi) if (isinstance(_dhi, (int, float)) and np.isfinite(_dhi)) else 10.0
                    dist_name = str(getattr(dist, "name", "Uniform")).lower()
                    if dist_name in ("gaussian", "normal"):
                        sv_init = np.random.normal(loc=(lo + hi) / 2, scale=(hi - lo) / 6, size=N)
                    else:
                        sv_init = np.random.uniform(lo, hi, size=N)
                else:
                    sv_init = np.repeat(initial_value(sv, default), N)
                init.append(sv_init)
        else:
            init = [initial_value(sv, default) for sv in self.state_variables.values()]
        return np.array(init)

    def run(self, format="python", verbose=0, save=True, run_kwargs=None, **kwargs) -> TimeSeries | BifurcationResult:
        """Generate, execute, and integrate the model, returning its output.

        Supports Julia (ODE and bifurcation), Python (SciPy `odeint`, or an iterated map for discrete systems), and compiled C backends.

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
        if run_kwargs is None:
            run_kwargs = {}
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
                _has_branches = ("periodic_orbits" in kwargs and kwargs["periodic_orbits"]) or (
                    cont and getattr(cont, "branches", None)
                )
                if _has_branches:
                    from tvbo.adapters.julia import eval_with_auto_install

                    try:
                        po = eval_with_auto_install("po_results")
                        bif_res.periodic_orbits = [BifurcationResult(br=p, model=self, **kwargs) for p in po.branches]
                    except Exception as e:
                        import warnings

                        warnings.warn(f"Periodic orbit extraction failed: {e}", stacklevel=2)
                        bif_res.periodic_orbits = []
                return bif_res

        elif "python" == format:
            from scipy.integrate import odeint

            # Discrete-time systems: iterate map instead of integrating ODEs
            if getattr(self, "system_type", "continuous") == "discrete":
                # Initial conditions
                if "u_0" not in kwargs:
                    u_0 = self.get_initial_values()
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
                    sub.update({sym: val for sym, val in zip(ssyms, data[i - 1, :], strict=True)})
                    next_vals = [float(expr.subs(sub)) for expr in rhs_exprs]

                    data[i, :] = next_vals

                return TimeSeries(
                    data=data.reshape(*data.shape, 1, 1),
                    time=t,
                    labels_dimensions={"State Variable": list(self.state_variables)},
                    sample_period=dt,
                )

            stimulus = self.stimulus.execute("python") if self.stimulus else None

            if stimulus and "stimulus" not in run_kwargs:
                run_kwargs.update({"stimulus": stimulus})

            model_dfun = self.execute(format=format, **kwargs)

            if "u_0" not in kwargs:
                # Initial conditions
                u_0 = self.get_initial_values()
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
            solution_slider = odeint(lambda u, t: model_dfun(u, t, **run_kwargs), u_0, t)

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
            result, diagnostics = compiled_module.solve_fast_robertson(u_0, T, rtol=rtol, atol=atol)
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
        Depending on `as_derived_variable`, the stimulus is either stored on `self.stimulus` or lowered into a `stim_t` derived variable plus suffixed stimulus parameters.

        Args:
            stimulus: A
                [`Stimulus`](#tvbo.classes.perturbation.Stimulus) to apply.
            as_derived_variable: If `True`, inline the stimulus as a `stim_t`
                derived variable; if `False`, store the `Stimulus` object
                directly.
        """
        if not any([sv.stimulation_variable for sv in self.state_variables.values()]) and not any(
            ["stim_t" in sv.equation.rhs for sv in self.state_variables.values()]
        ):
            import warnings

            warnings.warn(
                "No state variable with attribute `stimulation_variable=True` set. Stimulation will have no effect.",
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
                {"stim_t": tvbo_datamodel.DerivedVariable(name="stim_t", equation=tvbo_datamodel.Equation(rhs=eq))}
            )
            self.parameters.update({str(k): tvbo_datamodel.Parameter(name=str(k), value=v) for k, v in params.items()})

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

        Builds two linked panels — a bifurcation diagram over `ICS` and time series of `VOI` sampled at several parameter values — and either returns the combined figure or draws into the supplied axes.

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

        Emits `<name>.py` in `directory` with the required imports followed by the rendered TVB model code.

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

        Reads the model and does not modify it, for the same reason as [`render_code`](#tvbo.classes.dynamics.DynamicalSystem.render_code): normalisation belongs to construction, and repeating it here made a report a command as well as a query.

        Renders the Markdown report template; the result is optionally written to `outputfile` (as Markdown or, for `format="pdf"`, a PDF).

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
        """Deep-copy every declared field, not just those present in `__dict__`.

        A dataclass field still holding its default may be absent from `__dict__`, so copying that alone would silently drop it.
        """
        import dataclasses

        cls = self.__class__
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

    The smallest runnable unit in TVBO. A `Dynamics` binds a name to a set of parameters and an ODE system, and is round-trippable through YAML, SymPy, and any of the supported backends (JAX, TVB, PyRates, Julia, …).

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

    See the [writing-models](../../../skills/writing-models/SKILL.md) skill for the YAML form and equation conventions.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
