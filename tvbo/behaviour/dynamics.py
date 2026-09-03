"""The symbolic layer, on every :class:`Dynamics` however it was built.

Attached to the generated classes by name (``DynamicsBehaviour`` -> ``Dynamics``), so a model loaded through LinkML, validated through Pydantic or resolved onto an edge answers the same symbolic questions as one constructed through :mod:`tvbo.classes.dynamics`.

That reach is the point. Consumers used to ask whether the model in hand had a symbolic layer and fall back to parsing its metadata directly when it did not — three copies of one dispatch, and two parses of the same equation against namespaces built differently.

The equations themselves live on a [`SymbolicSystem`](../parse/system.qmd), one per model, held under a private attribute: both generated forms accept one without a slot and neither serializes it (``tests/test_spec_model_contract.py``).
"""

from __future__ import annotations

from tvbo.behaviour.dynamics_runtime import DynamicsRuntime


class DynamicsBehaviour(DynamicsRuntime):
    """Everything a model does, gathered where the generated class can inherit it.

    The symbolic view is here; construction, code generation, simulation, plotting and reporting come from [`DynamicsRuntime`](dynamics_runtime.qmd). One class because ``hatch_build`` attaches behaviour by name — ``DynamicsBehaviour`` to ``Dynamics`` — and splitting the source by concern is what keeps that one name readable.
    """

    def _from_ontology(self, key: str) -> bool:
        """Gap-fill from the ontology neural mass model *key* labels. True when found."""
        from tvbo.classes.dynamics import ontology_class, populate_from_ontology

        ontoclass = ontology_class(key)
        if ontoclass is None:
            return False
        populate_from_ontology(self, ontoclass)
        return True

    @property
    def symbolic_system(self):
        """This model's [`SymbolicSystem`](../parse/system.qmd), built once and kept.

        The symbolic layer is a service object rather than a further set of methods here: a scope and the equations parsed against it must agree, and holding both in one place makes that agreement — and the single point the cache is invalidated at — visible.

        A layer holds the model it was built for, so one that arrived on a *different* model is discarded and rebuilt. Every copy protocol carries the attribute across — `copy.copy`, Pydantic's `model_copy`, unpickling — and a carried layer would answer the clone's questions about the original: the clone's edits would neither invalidate the cache nor reach the scope. Checking ownership here fixes all of them at once, rather than one exclusion rule per copy hook.
        """
        system = self.__dict__.get("_symbolic_system")
        if system is None or system.model is not self:
            from tvbo.parse.system import SymbolicSystem

            system = SymbolicSystem(self)
            object.__setattr__(self, "_symbolic_system", system)
        return system

    def get_symbolic_elements(self, include_time_symbol: bool = True, time_dependent: bool = False):
        """The symbol table this model's expressions are parsed against.

        See [`SymbolicSystem.scope`](../parse/system.qmd#scope). Returns a copy, so a caller may keep or adapt it.
        """
        return self.symbolic_system.scope(include_time_symbol=include_time_symbol, time_dependent=time_dependent)

    def _symbolic_form(self, notation: str = "symbol", evaluate: bool = True):
        """This model's parsed equations. See [`SymbolicSystem.form`](../parse/system.qmd#form)."""
        return self.symbolic_system.form(notation=notation, evaluate=evaluate)

    def in_dependency_order(self, collection: str):
        """*collection* ordered so each member follows what it reads.

        What every backend emitting straight-line code iterates, in place of the collection itself. See [`SymbolicSystem.in_dependency_order`](../parse/system.qmd#in_dependency_order):
        the order is settled on the parsed equations, so rendering stays a query and a model does not have to be rewritten before it can be emitted.
        """
        return self.symbolic_system.in_dependency_order(collection)

    def get_equations(self, format: str = "metadata", evaluate: bool = True):
        """Collect the model's equations as SymPy `Eq` objects.

        The flat projection of [`SymbolicSystem.form`](../parse/system.qmd#form): derived parameters, functions, derived variables, state equations (as time derivatives, or plain maps for discrete systems), and output transformations.

        It rides on the layer rather than on the runtime `Dynamics` because the consumers that ask for it hold either flavour — a report's per-node model on a heterogeneous network is the generated one, and promoting it to a runtime model just to ask re-parsed every equation the layer already held.

        Args:
            format: Shape of the result. `"dict"` groups equations by category;
                `"state-equations"` returns only state equations keyed by variable name;
                any other value (e.g. `"metadata"`) returns a single flat mapping of
                variable name to `Eq`.
            evaluate: If `True`, let SymPy evaluate/simplify parsed right-hand sides; if
                `False`, preserve authored term order.

        Returns:
            A mapping of equations whose structure depends on `format`.

        Raises:
            ValueError: If an entry in `output` names neither a derived nor a state
                variable.
        """
        form = self._symbolic_form(notation="symbol", evaluate=evaluate)

        if format == "state-equations":
            return dict(form["state-equations"])
        if format == "dict":
            return {group: list(equations.values()) for group, equations in form.items()}
        return {name: equation for group in form.values() for name, equation in group.items()}

    @property
    def symbolic(self):
        """Full symbolic ODE system using proper SymPy conventions.

        See [`SymbolicSystem.view`](../parse/system.qmd#view).

        Example:
            ```python
            model.symbolic['state']    # [Eq(Derivative(theta(t), t), I + omega)]
            model.symbolic['derived']  # [Eq(signal(t), sin(theta(t)))]
            model.symbolic['units']    # {omega: 'per_ms', I: None}
            ```
        """
        return self.symbolic_system.view()

    @property
    def keyed_parameters(self):
        """Each parameter's symbol mapped to its value, keyed for the codegen view.

        See [`SymbolicSystem.keyed_parameters`](../parse/system.qmd#keyed_parameters); this is the view `get_equations` returns, which is what such a map is substituted into.
        """
        return self.symbolic_system.keyed_parameters()

    def symbol_map(self):
        """Display-symbol overrides for report rendering: ``{identifier Symbol: LaTeX str}``.

        See [`SymbolicSystem.symbol_map`](../parse/system.qmd#symbol_map).
        """
        return self.symbolic_system.symbol_map()

    def check_units(self, strictness: str = "dimensional", time_unit: str | None = None):
        """Per-equation dimensional verdicts for this model.

        See [`tvbo.analysis.units.check_units`](../analysis/units.qmd#check_units). Each verdict is `consistent`, `inconsistent` or `underdetermined`; the third is a distinct answer, not a soft failure, because 24 of the 39 curated models declare no units and calling those wrong would pressure fake declarations into the published record.
        """
        from tvbo.analysis.units import check_units

        return check_units(self, strictness=strictness, time_unit=time_unit)

    def _items(self):
        """What the LinkML dumpers see: schema slots only.

        Mirrors the guard on `Network`. `_symbolic_system` holds SymPy objects that `yaml.SafeDumper` cannot represent, and LinkML slot names are never underscore-prefixed, so excluding every leading-underscore key keeps this correct as further caches are added rather than depending on a maintained denylist.

        Only the LinkML form has slots to filter — Pydantic serializes through its own machinery and offers nothing to delegate to — so the mixin yields nothing there rather than raising on a `super()` that does not exist.
        """
        inherited = getattr(super(), "_items", None)
        if inherited is None:
            return
        for k, v in inherited():
            if not str(k).startswith("_"):
                yield k, v
