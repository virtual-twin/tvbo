"""What a `Dynamics` does, on every `Dynamics` however it was built.

Attached through [`DynamicsBehaviour`](dynamics.qmd), so a model loaded through LinkML, validated through Pydantic or resolved onto an edge answers the same questions as one constructed through :mod:`tvbo.classes.dynamics`. There is no runtime subclass to promote a record into: the methods are on the generated class itself.

Split out of [`DynamicsBehaviour`](dynamics.qmd) only to keep one file readable; the mixin discovery in ``hatch_build`` attaches classes whose name ends in ``Behaviour``, so this one adds no second attachment point. :mod:`tvbo.classes.dynamics` remains the public import location for the class itself.
"""

from __future__ import annotations

import copy as _copy
import os
from os.path import basename, dirname, join, splitext
from typing import TYPE_CHECKING

import numpy as np
from sympy import Eq, Symbol

if TYPE_CHECKING:
    from tvbo.datamodel.schema import Dynamics


class _Lazy:
    """A module imported on first attribute access.

    This mixin is a base of the generated `Dynamics`, so whatever it imports at module scope is paid for by every ``import tvbo.datamodel.schema`` — including the JAX and codegen processes that never plot or query the ontology. The tvbo half would also close an import cycle, since the schema module is what imports this one.
    """

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attribute):
        import importlib

        return getattr(importlib.import_module(self._name), attribute)


analysis = _Lazy("tvbo.analysis")
data_types = _Lazy("tvbo.data.types")
expression = _Lazy("tvbo.parse.expression")
model_helpers = _Lazy("tvbo.classes.dynamics")
nx = _Lazy("networkx")
ontology = _Lazy("tvbo.ontology.owl")
owlready2 = _Lazy("owlready2")
perturbation = _Lazy("tvbo.classes.perturbation")
plt = _Lazy("matplotlib.pyplot")
query = _Lazy("tvbo.ontology.query")
report = _Lazy("tvbo.utils.report")
templater = _Lazy("tvbo.codegen.templater")
templates = _Lazy("tvbo.templates")
tvbo_datamodel = _Lazy("tvbo.datamodel.schema")
utils = _Lazy("tvbo.utils")
yaml_loader = _Lazy("tvbo.utils.yaml_loader")
_equation_mod = _Lazy("tvbo.classes.equation")
_pdm = _Lazy("tvbo.datamodel.pydantic")


class DynamicsRuntime:
    """Construction, code generation, simulation, plotting and reporting for a model."""

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
    def from_datamodel(cls, model_meta: tvbo_datamodel.Dynamics):
        """Create from a datamodel Dynamics instance by copying its already-normalized state (avoids ``_as_dict`` re-init crash on ``inlined_as_dict`` fields)."""
        inst = cls.__new__(cls)
        inst.__dict__.update(model_meta.__dict__)
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
            A populated instance.
        """
        if isinstance(ontoclass, str):
            ontoclass = query.label_search(ontoclass, root_class="NeuralMassModel", exact_match=["label"])[0]
        inst = cls(name=ontoclass.name, **kwargs)
        model_helpers.populate_from_ontology(inst, ontoclass, **kwargs)
        return inst

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Dynamics":
        """Load a model from a YAML/JSON specification file on disk.

        Args:
            path: Path to a TVBO model specification file.

        Returns:
            The instance parsed from the file.
        """
        data = yaml_loader.strip_envelope(yaml_loader.load_as_dict(str(path)))
        model_helpers._resolve_dynamics_aliases(data)
        return cls(**data)

    @classmethod
    def from_string(cls, str: str) -> "Dynamics":
        """Load a model from a YAML specification string.

        Args:
            str: A YAML document describing the model.

        Returns:
            The instance parsed from the string.
        """
        data = yaml_loader.strip_envelope(yaml_loader.load_as_dict(str)) or {}
        model_helpers._resolve_dynamics_aliases(data)
        return cls(**data)

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
        return cls(**data)

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
        return model_helpers.ontology_class(getattr(self, "name", None))

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
    def _peer(self):
        """The generated module a member of this record has to be built in.

        See [`peer_module`](../datamodel/dialect.qmd#peer_module): the builders serve both generated forms, and the strict Pydantic models reject a LinkML dataclass where they want their own peer.
        """
        from tvbo.datamodel.dialect import peer_module

        return peer_module(self)

    def _collection(self, slot: str) -> dict:
        """*slot*'s mapping, created if this form left it unset.

        The LinkML dataclass defaults a keyed collection to an empty mapping and the Pydantic model defaults it to ``None``, so a builder that assumed the first raised ``'NoneType' object does not support item assignment`` on the second. Members are added by mutating the mapping, never by reassigning the slot.
        """
        if getattr(self, slot, None) is None:
            setattr(self, slot, {})
        return getattr(self, slot)

    def _sequence(self, slot: str) -> list:
        """*slot*'s list, created if this form left it unset.

        The list-valued counterpart of [`_collection`](#_collection): the Pydantic model defaults ``output`` to ``None``, so a builder that assumed a list raised ``argument of type 'NoneType' is not iterable`` on that form.
        """
        if getattr(self, slot, None) is None:
            setattr(self, slot, [])
        return getattr(self, slot)

    def _coerce_range(self, domain):
        """Accept None | Range | (lo, hi[, step]) | {lo, hi[, step, enforce]} → Range."""
        if domain is None or isinstance(domain, self._peer.Range):
            return domain
        if isinstance(domain, dict):
            kw = {k: v for k, v in domain.items() if k in ("lo", "hi", "step", "enforce") and v is not None}
            for k in ("lo", "hi", "step"):
                if k in kw:
                    kw[k] = float(kw[k])
            return self._peer.Range(**kw) if kw else None
        if isinstance(domain, (list, tuple)):
            if len(domain) == 2:
                lo, hi = domain
                return self._peer.Range(lo=float(lo), hi=float(hi))
            if len(domain) == 3:
                lo, hi, step = domain
                return self._peer.Range(lo=float(lo), hi=float(hi), step=float(step))
        return domain

    def _coerce_equation(self, expr, lhs: str | None = None):
        """Coerce str | sympy.Eq | sympy.Expr | Equation into this record's own Equation.

        Also normalizes common implicit multiplication patterns in string expressions, e.g.:
        - a(Y - X) -> a*(Y - X) when 'a' is not a known function
        - XY -> X*Y, cZ -> c*Z, 2X -> 2*X, X2 -> X*2

        An Equation of the *other* generated form is rebuilt rather than passed through:
        the strict Pydantic models reject a LinkML dataclass on assignment.
        """
        if isinstance(expr, self._peer.Equation):
            return expr
        if hasattr(expr, "model_dump"):
            return self._peer.Equation(**expr.model_dump())
        if hasattr(expr, "rhs") and not isinstance(expr, Eq):
            return self._peer.Equation(lhs=getattr(expr, "lhs", lhs), rhs=expr.rhs)
        if isinstance(expr, Eq):
            return self._peer.Equation(lhs=str(expr.lhs), rhs=str(expr.rhs))
        return self._peer.Equation(lhs=lhs, rhs=expr)

    # Interop with Pydantic models
    def to_pydantic(self):
        """Return a tvbopydantic.Dynamics validated instance for this model."""
        from tvbo.datamodel import pydantic as _pdm

        return _pdm.Dynamics.model_validate(self._as_dict)

    @classmethod
    def from_pydantic(cls, pyd_obj):
        """Create a Dynamics from a tvbopydantic.Dynamics (or dict-like)."""
        data = pyd_obj.model_dump() if hasattr(pyd_obj, "model_dump") else dict(pyd_obj)
        return cls(**data)

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
        self._collection("parameters")[str(name)] = self._peer.Parameter(
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
        nonparam_known = set(map(str, self._collection("state_variables")))
        nonparam_known |= set(map(str, self._collection("derived_variables")))
        nonparam_known |= set(map(str, self.output or []))  # output is list of strings
        nonparam_known |= set(map(str, self._collection("derived_parameters")))
        for f in self._collection("functions").values():
            nonparam_known |= {str(name) for name in f.arguments}
        nonparam_known.add("t")

        parameters = self._collection("parameters")
        # If any existing parameters clash with known entities, remove them (they were falsely inferred earlier)
        to_remove = [pname for pname in list(parameters) if str(pname) in nonparam_known]
        for pname in to_remove:
            del parameters[pname]

        # Known symbols also include existing parameters (after clean-up above)
        known = set(map(str, parameters)) | nonparam_known

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
                self._collection("parameters")[s] = self._peer.Parameter(name=s, value=float(default_value))
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
        _domain, _distribution = model_helpers._fold_range_boundaries(
            self._coerce_range(domain),
            self._coerce_range(boundaries) if boundaries is not None else None,
        )
        self._collection("state_variables")[str(name)] = self._peer.StateVariable(
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
        eq = self._coerce_equation(expression, lhs=str(name)) if expression is not None else self._peer.Equation(lhs=str(name))
        if conditionals:
            eq.conditionals = [
                self._peer.ConditionalBlock(condition=str(cond), expression=str(expr)) for expr, cond in conditionals
            ]
        self._collection("derived_variables")[str(name)] = self._peer.DerivedVariable(
            name=str(name),
            equation=eq,
            unit=unit,
            description=description,
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
                str(k): (v if isinstance(v, self._peer.Parameter) else self._peer.Parameter(name=str(k)))
                for k, v in arguments.items()
            }
        else:
            args = list(arguments) if isinstance(arguments, (list, tuple)) else []
            args_dict = {str(a): self._peer.Parameter(name=str(a)) for a in args}
        eq = self._coerce_equation(expression, lhs=str(name)) if expression is not None else None
        self._collection("functions")[str(name)] = self._peer.Function(
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
        self._collection("coupling_inputs")[key] = self._peer.CouplingInput(
            name=key,
            description=description,
            dimension=dimension,
            keys=keys or [],
        )
        # Keep parameters clean: remove any parameter with same name
        parameters = self._collection("parameters")
        if key in parameters:
            del parameters[key]

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
        self._collection("derived_variables")[name_str] = self._peer.DerivedVariable(
            name=name_str, equation=eq, unit=unit, description=description
        )
        # Add reference to output list
        output = self._sequence("output")
        if name_str not in output:
            output.append(name_str)
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
        self._collection("derived_parameters")[str(name)] = self._peer.DerivedParameter(
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

        scope = self.get_symbolic_elements()
        # Tell the printer which names are functions so it emits f(x) cleanly
        uf = {str(name): str(name) for name in self.functions}

        inline_funcs = expression.function_bodies(self) if inline_functions else None
        if inline_funcs:
            # Inlined calls leave no function heads for the printer to name
            uf = {}

        return render_equation(
            self.symbolic_rhs(obj, evaluate=not kwargs.get("preserve_order", False)),
            local_dict=scope,
            format=format,
            user_functions=uf,
            inline_funcs=inline_funcs,
            **kwargs,
        )

    def render_equation_cse(self, obj, format="numpy", inline_functions=False, **kwargs):
        """Common-subexpression-eliminated variant of :meth:`render_equation`.

        Returns ``(setup, final)`` — a list of ``(name, expr)`` assignments plus the return expression — so interpreted backends (TVB / numpy) evaluate each shared subexpression (notably repeated model-function calls) once instead of per occurrence. Builds the same symbolic scope / user-function set as :meth:`render_equation`; see :func:`tvbo.codegen.code.render_equation_cse`.
        """
        from tvbo.codegen.code import render_equation_cse

        scope = self.get_symbolic_elements()
        uf = {str(name): str(name) for name in self.functions}

        inline_funcs = expression.function_bodies(self) if inline_functions else None
        if inline_funcs:
            uf = {}

        return render_equation_cse(
            self.symbolic_rhs(obj, evaluate=not kwargs.get("preserve_order", False)),
            local_dict=scope,
            format=format,
            user_functions=uf,
            inline_funcs=inline_funcs,
            **kwargs,
        )

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

    def get_dependency_tree(self, ontomapping=False, include_state_equations=False):
        """Build the equation dependency graph for this model.

        Nodes are the model's symbols; each edge points from a dependency to the quantity whose equation uses it (dependencies → dependents). State equations are excluded by default to avoid cycles in discrete systems.

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

        # State equations are excluded by default: in a discrete system an algebraic derived variable and the states read each other within one step, which is a cycle.
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

        Dispatches to the template (or adapter) for the requested backend and returns the formatted source. Reads the model and does not modify it, so the source depends on the model alone and not on how often it has been rendered — nor on whether anything reordered it first. The dependency order the straight-line emitters need is a view the symbolic layer computes, not a state the model is put into: see [`in_dependency_order`](../behaviour/dynamics.qmd#in_dependency_order).

        Args:
            format: Target backend, e.g. `"tvb"`, `"jax"`, `"numpy"`,
                `"tvboptim"`, `"julia"`, `"bifurcation-julia"`, `"pde-fem"`,
                or `"neuroml"`. The template-rendered ones are declared in
                [`CODE_FORMATS`](#tvbo.codegen.templater.CODE_FORMATS); the rest
                are built by an adapter.
            alt_label: Optional alternative label for the generated model.
            **kwargs: Forwarded to the template/adapter (e.g. `continuation`).

        Returns:
            The generated code as a formatted string.

        Raises:
            ValueError: If `format` is not a supported backend.
        """
        spec = templater.code_format(format)
        if spec is not None and spec.template:
            template = templates.lookup.get_template(spec.template)
            # The format's own kwargs are defaults, so an explicit argument still wins.
            for key, value in spec.render_kwargs:
                kwargs.setdefault(key, value)

        elif format == "bifurcation-julia":
            from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

            continuation = kwargs.pop("continuation", None)
            ctx = BifurcationKitAdapter._prepare_context(self, continuation, **kwargs)
            template = templates.lookup.get_template("tvbo-julia-BifurcationKit.jl.mako")
            rendered_code = template.render(**ctx)
            return templater.format_code(rendered_code, format=format)
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
                [`render_code`](#tvbo.behaviour.dynamics_runtime.DynamicsRuntime.render_code).
            **kwargs: Forwarded to `render_code`.

        Returns:
            An `IPython.display.Markdown` object wrapping the generated code.
        """
        from IPython.display import Markdown

        code = templater.format_code(self.render_code(format=format, **kwargs), format=format)
        return Markdown(f"```{templater.source_language(format) or format}\n{code}\n```")

    def execute(self, format="tvb", **kwargs):
        """Generate and execute the model code, returning a runnable object.

        Dispatches on `format`: builds a configured TVB model instance, a tvboptim dynamics instance, a compiled C module (`sympy2c`), a bifurcation/continuation run, or a plain dfun callable.

        Every code format resolves its binding through [`entry_point_name`](#tvbo.codegen.templater.entry_point_name), the same declaration the templates emit against, so any backend that renders Python hands back a usable object for a custom JAX, NumPy or SciPy workflow rather than failing on a name the caller had to guess.

        Args:
            format: Backend to execute, e.g. `"tvb"`, `"tvboptim"`, `"c"`,
                `"bifurcation-auto7p"`, or a code format yielding a dfun.
            **kwargs: Constructor/runtime arguments forwarded to the executed
                code.

        Returns:
            The executed object, whose type depends on `format`.

        Raises:
            ValueError: If `format` declares no entry point, so nothing can be
                re-entered from the rendered source.
        """

        if format == "tvb":
            rendered_code = model_helpers.clean_code(self.render_code(format=format, **kwargs))
            namespace = {}
            exec(rendered_code, namespace)
            tvb_obj = namespace[templater.entry_point_name(self, format)](**kwargs)
            tvb_obj.title = self.label
            tvb_obj.configure()
            return tvb_obj

        elif format.lower() in ("tvboptim", "tvb-optim"):
            namespace = {}
            exec(model_helpers.clean_code(self.render_code(format="tvboptim")), namespace)
            cls = namespace[templater.entry_point_name(self, "tvboptim")]
            return cls(**kwargs)

        elif format.lower() in ["c", "sympy2c"]:
            try:
                import importlib

                _sympy2c = importlib.import_module("sympy2c")
                Module = getattr(_sympy2c, "Module")
                OdeFast = getattr(_sympy2c, "OdeFast")
            except Exception as e:
                raise RuntimeError("sympy2c is not installed. Install it to use format='c' or 'sympy2c'.") from e

            params = self.keyed_parameters
            params.update({Symbol(str(ci)): 0.0 for ci in self.coupling_inputs})
            params.update({Symbol("local_coupling"): 0.0})

            scope = self.get_symbolic_elements()
            derived_variables = {
                Symbol(k): expression.parse_eq(v.equation, local_dict=scope) for k, v in self.derived_variables.items()
            }

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
            # In-tree AUTO-07p backend: a one-off SimulationExperiment wraps this Dynamics for NumContAdapter.
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
            entry = templater.entry_point_name(self, format)
            rendered_code = model_helpers.clean_code(self.render_code(format=format, **kwargs))
            namespace = {}
            exec(rendered_code, namespace)
            return namespace[entry]

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
            description=(self.ontology.description.first() if self.ontology and self.ontology.description else None),
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
                branches = getattr(dp.equation, "conditionals", None)
                if branches:
                    cv = lems.ConditionalDerivedVariable(
                        name=dp.name,
                        dimension=unit_to_lems_dimension(getattr(dp, "unit", None)),
                        exposure=dp.name,
                    )
                    for branch in branches:
                        condition_str = None if branch.condition is True else str(branch.condition)
                        cv.add_case(
                            lems.Case(
                                condition=condition_str,
                                value=str(branch.expression).replace("**", "^"),
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
            dimension = unit_to_lems_dimension(sv.has_unit.first().label.first() if sv.has_unit.first() else None)
            sv_start = sv_name + "_0"

            if assign_uniform:
                init_conds[sv_start] = initial_conditions
            else:
                init_conds[sv_start] = init_conds.get(sv_start, init_conds.get(sv_name, 0.0))

            deriv = sv.has_derivative.first()

            local_ct.add(lems.Parameter(name=sv_start, dimension=dimension))
            local_ct.add(lems.Exposure(name=sv_name, dimension=dimension))

            local_ct.dynamics.add(lems.StateVariable(name=sv_name, dimension=dimension, exposure=sv_name))
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

    def get_initial_values(self, default=0.1, random=False, N=1, **kwargs):
        """Build the initial state vector for a simulation.

        If any state variable defines a `distribution` (or `random=True`), initial values are sampled from it (Gaussian or uniform over the finite domain bounds); otherwise each variable's `initial_value` (or `default`) is used.

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
        has_distributions = any(getattr(sv, "distribution", None) for sv in self.state_variables.values())
        if random or has_distributions:
            init = []
            for k, sv in self.state_variables.items():
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
                elif random:
                    # Legacy fallback: sample from the domain range. The domain may carry a one-sided clamp (e.g. [0, inf) for a firing rate), so guard against non-finite / inverted bounds — uniform(0, inf) would yield inf/NaN initial states.
                    dlo = getattr(sv.domain, "lo", None) if sv.domain else None
                    dhi = getattr(sv.domain, "hi", None) if sv.domain else None
                    lo = dlo if (isinstance(dlo, (int, float)) and np.isfinite(dlo)) else -10.0
                    hi = dhi if (isinstance(dhi, (int, float)) and np.isfinite(dhi)) else 10.0
                    if hi <= lo:
                        hi = lo + 1.0
                    sv_init = np.random.uniform(lo, hi, size=N)
                else:
                    # No distribution, no random flag → use initial_value
                    sv_init = np.repeat(utils.initial_value(sv, default), N)
                init.append(sv_init)
        else:
            init = [utils.initial_value(sv, default) for sv in self.state_variables.values()]
        return np.array(init)

    def run(
        self, format="python", verbose=0, save=True, run_kwargs={}, **kwargs
    ) -> data_types.TimeSeries | analysis.BifurcationResult:
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
                return data_types.TimeSeries(
                    time=t,
                    data=data4,
                    title=self.name,
                    sample_period=(t[1] - t[0]) if t.size > 1 else None,
                    labels_dimensions=labels_dimensions,
                )
            elif format == "bifurcation-julia":
                br_obj = extract_bifurcation_result()
                bif_res = analysis.BifurcationResult(br=br_obj, model=self, **kwargs)
                # Auto-detect PO branches from continuation object or explicit kwarg
                cont = kwargs.get("continuation", None)
                _has_branches = ("periodic_orbits" in kwargs and kwargs["periodic_orbits"]) or (
                    cont and getattr(cont, "branches", None)
                )
                if _has_branches:
                    from tvbo.adapters.julia import eval_with_auto_install

                    try:
                        po = eval_with_auto_install("po_results")
                        bif_res.periodic_orbits = [analysis.BifurcationResult(br=p, model=self, **kwargs) for p in po.branches]
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
                    u_0 = self.get_initial_values(random=kwargs.get("random_initial_conditions", False))
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

                return data_types.TimeSeries(
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
                u_0 = self.get_initial_values(random=kwargs.get("random_initial_conditions", False))
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

            return data_types.TimeSeries(
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
            return data_types.TimeSeries(
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
        if isinstance(stimulus, perturbation.Stimulus) and not as_derived_variable:
            self.stimulus = stimulus

        elif stimulus.equation is not None and as_derived_variable:
            eq, params = stimulus.get_expression()
            param_map = {k: Symbol(str(k) + "_stim") for k in params.keys()}
            params = {param_map[k]: v for k, v in params.items()}
            eq = eq.subs(param_map)
            self.derived_variables.update(
                {"stim_t": self._peer.DerivedVariable(name="stim_t", equation=self._peer.Equation(rhs=eq))}
            )
            self.parameters.update({str(k): self._peer.Parameter(name=str(k), value=v) for k, v in params.items()})

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

        Reads the model and does not modify it, for the same reason as [`render_code`](#tvbo.behaviour.dynamics_runtime.DynamicsRuntime.render_code): normalisation belongs to construction, and repeating it here made a report a command as well as a query.

        Refreshes metadata and renders the Markdown report template; the result is optionally written to `outputfile` (as Markdown or, for `format="pdf"`, a PDF).

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

    def __copy__(self):
        """A shallow copy, through the generated form's own protocol where it has one.

        Pydantic implements the copy hooks itself and ``model_copy()`` routes through them, so building a clone with ``cls.__new__`` leaves ``__pydantic_extra__`` unset and the first assignment raises. The LinkML dataclasses have no hook, hence the manual path.
        """
        inherited = getattr(super(), "__copy__", None)
        if inherited is not None:
            return inherited()
        cls = self.__class__
        clone = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(clone, k, v)
        return clone

    def __deepcopy__(self, memo):
        """A deep copy, through the generated form's own protocol where it has one.

        See [`__copy__`](#__copy__): reconstructing a Pydantic model by hand is what its own hook exists to avoid.
        """
        import dataclasses

        inherited = getattr(super(), "__deepcopy__", None)
        if inherited is not None:
            return inherited(memo)

        cls = self.__class__
        # For dataclasses, we need to copy all fields, not just __dict__ __dict__ may not include fields that are still at their default values
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
