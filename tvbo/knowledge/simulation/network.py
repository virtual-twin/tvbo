#  coupling.py
#
# Created on Mon Jan 22 2024
# Author: Leon K. Martin
#
# Copyright (c) 2024 Charité Universitätsmedizin Berlin
#
"""
TVB-O wrapper for Coupling functions
====================================

```{seealso}
- [Coupling](![wiki]/Coupling/index.html)
```

"""
import copy
from os.path import join

import networkx as nx
import numpy as np
import owlready2
from mako.template import Template
from sympy import pycode

from tvbo import templates
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import templater
from tvbo.export.code import parse_eq
from tvbo.knowledge import constants, ontology, query
from tvbo.knowledge.simulation import equations, localdynamics
from tvbo.parse import metadata as metadata_mod
from tvbo.run import compgraph

TEMPLATES = templates.root


def get_parameters(CF):
    if isinstance(CF, str):
        CF = ontology.get_coupling_function(CF)

    parameters = {}
    for p in CF.has_parameter:
        param_props = {"domain": {}}
        (
            param_props["domain"]["lo"],
            param_props["domain"]["hi"],
            param_props["domain"]["step"],
        ) = (
            ontology.get_range(p) if ontology.get_range(p) else ("-inf", "inf", "0.001")
        )
        param_props["value"] = (
            float(p.defaultValue.first())
            if len(p.defaultValue) > 0 and p.defaultValue.first() != "None"
            else 0
        )
        param_props["definition"] = p.definition.first()
        param_props["label"] = ontology.replace_suffix(p.label.first())
        param_props["name"] = p.name
        parameters[p] = param_props
    return parameters


def coupling_class2metadata(ontoclass, metadata, overwrite: bool = False):
    """Populate coupling metadata from an ontology class.

    If overwrite is False (default), only fill missing fields.
    If overwrite is True, always set name and pre/post expressions.
    Parameters are added if missing; existing parameter value/description are
    only filled if missing regardless of overwrite.
    """
    # Name
    try:
        if overwrite or not getattr(metadata, "name", None):
            metadata.name = ontoclass.label.first()
    except Exception:
        pass

    # Equations
    try:
        eqs = equations.get_symbolic_coupling(ontoclass)
    except Exception:
        eqs = None
    if eqs:
        if overwrite or getattr(metadata, "pre_expression", None) is None:
            metadata.pre_expression = tvbo_datamodel.Equation(rhs=str(eqs["pre"]))
        if overwrite or getattr(metadata, "post_expression", None) is None:
            metadata.post_expression = tvbo_datamodel.Equation(rhs=str(eqs["post"]))

    # Parameters
    for key, param in get_parameters(ontoclass).items():
        label = param["label"]
        if getattr(metadata, "parameters", None) is None:
            metadata.parameters = {}
        if label not in metadata.parameters:
            metadata.parameters[label] = tvbo_datamodel.Parameter(
                name=param["label"],
                value=param["value"],
                description=param["definition"],
            )
        else:
            if getattr(metadata.parameters[label], "value", None) is None:
                metadata.parameters[label].value = param["value"]
            if getattr(metadata.parameters[label], "description", None) is None:
                metadata.parameters[label].description = param["definition"]


class Coupling(tvbo_datamodel.Coupling):
    """Runtime Coupling that is also a direct instance of tvbo_datamodel.Coupling.

    - If a name matches an ontology Coupling, missing fields are populated from ontology.
    - Backward compatibility:  returns self so existing code keeps working.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-populate from ontology without overwriting existing fields
        self._populate_from_ontology()

    def _populate_from_ontology(self):
        """Fill missing metadata fields from ontology based on `name`.

        Only empties are set: name (if missing), pre_expression/post_expression
        (if missing), and parameters (add new or fill missing value/description).
        """
        try:
            oc = self.ontoclass
        except Exception:
            oc = None
        if not oc:
            return

        # Reuse shared helper; non-destructive fill
        coupling_class2metadata(oc, self, overwrite=False)

    @classmethod
    def from_ontology(cls, ontoclass):
        """Create a Coupling instance from an ontology Coupling class."""
        if isinstance(ontoclass, str):
            ontoclass = query.label_search(
                ontoclass, root_class="Coupling", exact_match=["label"]
            )[0]
        if not isinstance(ontoclass, owlready2.entity.ThingClass):
            raise ValueError(
                "ontoclass must be a string or an ontology Coupling class."
            )
        metadata = tvbo_datamodel.Coupling(name=ontoclass.label.first())
        coupling_class2metadata(ontoclass, metadata, overwrite=True)
        return cls(**metadata._as_dict)

    @classmethod
    def from_datamodel(cls, datamodel_instance):
        """Create a Coupling instance from an existing tvbo_datamodel.Coupling instance."""
        if not isinstance(datamodel_instance, tvbo_datamodel.Coupling):
            raise ValueError(
                "datamodel_instance must be a tvbo_datamodel.Coupling instance."
            )
        return cls(metadata=datamodel_instance)

    # Back-compat: expose  pointing to self
    @property
    def metadata(self):
        return self

    # def __str__(self):
    #     return (
    #         self.name if self.name else f"Coupling{self.id}"
    #     )

    # def __repr__(self):
    #     # You can reuse __str__ or return a more detailed representation
    #     return self.__str__()

    def to_yaml(self, filepath: str | None = None):
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    def render_code(self, format="tvb", model=None, alt_label=None, **kwargs):
        if format == "tvb":
            rendered_code = templates.lookup.get_template(
                "tvbo-tvb-coupling.py.mako"
            ).render(coupling=self)

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("tvbo-jax-coupling.py.mako")
            rendered_code = template.render(coupling=self, model=model, **kwargs)

        elif format.lower() == "python":
            from tvbo.export.code import NumPyPrinter, render_expression

            render_expression(self.equation, format="python")

        return templater.format_code(rendered_code)

    def execute(self, format="tvb", alt_label=None, **kwargs):
        if format == "tvb":
            local_vars = {}
            exec(
                self.render_code(alt_label=alt_label),
                templater.exec_globals,
                local_vars,
            )
            tvb_obj = local_vars[self.name if not alt_label else alt_label](**kwargs)
            return tvb_obj

        elif format.lower() == "python":
            from sympy import Symbol, lambdify

            return lambdify(
                [Symbol("x"), Symbol("g"), Symbol("N"), Symbol("i")]
                + [Symbol(p) for p in self.parameters],
                self.equation,
            )

    # ---- Runtime properties (no extra attributes) ----
    @property
    def ontoclass(self):
        try:
            hits = (
                query.label_search(self.name, root_class="Coupling")
                if getattr(self, "name", None)
                else []
            )
            return hits[0] if hits else None
        except Exception:
            return None

    @property
    def pre(self):
        return parse_eq(self.pre_expression)

    @property
    def post(self):
        return parse_eq(self.post_expression)

    @property
    def equation(self):
        try:
            pre = self.pre
            post = self.post
            if pre is None or post is None:
                return None
            return equations.generate_global_coupling_function(pre, post)
        except Exception:
            return None

    def plot(self, weights=None, node_idx=0, xs=None, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        import sympy as sp

        if weights is None:
            weights = np.random.normal(loc=0.0, scale=1.0, size=(3, 3))
            np.fill_diagonal(weights, 0)

        i, N = sp.symbols("i N", integer=True)
        x = sp.IndexedBase("x")
        g = sp.IndexedBase("g")

        all_param_names = list(self.parameters.keys())
        used_param_names = sorted(
            [
                name
                for name in all_param_names
                if sp.Symbol(name) in self.equation.free_symbols
            ]
        )
        param_syms = tuple(sp.symbols(used_param_names))
        f = sp.lambdify((x, g, i, N) + param_syms, self.equation, modules="numpy")

        if xs is None:
            xs = np.linspace(-2.0, 2.0, 100)

        varnames = f.__code__.co_varnames

        if node_idx is not None:
            k = 0
            i_plot = 1
            x0 = xs.copy()
            ys = []
            for xv in xs:
                x_tmp = x0.copy()
                x_tmp[k] = xv
                ys.append(
                    f(
                        x_tmp,
                        weights,
                        i_plot,
                        weights.shape[0],
                        **{
                            p: self.parameters[p].value
                            for p in used_param_names
                            if p in varnames
                        },
                    )
                )

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(xs, ys)
            ax.set_xlabel(f"x[{k}]")
            ax.set_ylabel("y(i)")
            ax.set_title("Coupling vs single input component")
            plt.close()
            return fig


def get_global_coupling_functions():
    onto = ontology.get_onto()
    CouplingFunctions = onto.Coupling.subclasses()

    # for CF in CouplingFunctions:
    #     CF.pre = MethodType(get_pre_summation_coupling_function, CF)
    return list(CouplingFunctions)


available_coupling_functions = set(get_global_coupling_functions())


class Network:
    def __init__(self, connectome, normalize_weights=True):
        if normalize_weights:
            # Normalize using Connectome's schema-safe method
            try:
                connectome.normalize_weights()
            except Exception:
                pass
        # Build a graph snapshot from current connectome
        self.graph = connectome.create_graph()

    def add_local_model(self, model):
        if isinstance(model, localdynamics.Model) or isinstance(
            model, localdynamics.Dynamics
        ):
            for node in self.graph.nodes:
                self.graph.nodes[node]["model"] = model

        elif isinstance(model, dict):
            for node in model:
                self.graph.nodes[node]["model"] = model[node]

    def add_coupling(self, coupling):
        is_multi = isinstance(self.graph, (nx.MultiDiGraph, nx.MultiGraph))

        if isinstance(coupling, Coupling):
            # Copy same coupling instance to all edges
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    self.graph[src][tgt][key]["coupling"] = copy.deepcopy(coupling)
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = copy.deepcopy(coupling)

        elif isinstance(coupling, tvbo_datamodel.Coupling):
            # Wrap a pure datamodel instance
            wrapped = Coupling(metadata=coupling)
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    self.graph[src][tgt][key]["coupling"] = copy.deepcopy(wrapped)
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = copy.deepcopy(wrapped)

        elif isinstance(coupling, dict):
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    # Support mapping by (src,tgt,key) with fallback to (src,tgt)
                    val = coupling.get((src, tgt, key), coupling.get((src, tgt)))
                    self.graph[src][tgt][key]["coupling"] = val
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = coupling[src, tgt]

    def add_stimulus(self, node, stimulus, stvar=None, as_derived_variable=False):
        if as_derived_variable:
            self.graph.nodes[node]["model"].add_stimulus(
                stimulus, as_derived_variable=True
            )
        else:
            self.graph.nodes[node]["stimulus"] = stimulus
            if stvar is not None:
                if not isinstance(stvar, list):
                    stvar = [stvar]
                for var in stvar:
                    self.graph.nodes[node]["model"].state_variables[
                        var
                    ].stimulation_variable = True

    def setup_dfuns(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["dfun"] = self.graph.nodes[node]["model"].execute(
                "python-network"
            )

    def setup_cfuns(self):
        from sympy import Symbol, lambdify

        is_multi = isinstance(self.graph, (nx.MultiDiGraph, nx.MultiGraph))

        if is_multi:
            for src, tgt, key in self.graph.edges(keys=True):
                coup = self.graph[src][tgt][key]["coupling"]
                self.graph[src][tgt][key]["cfun"] = coup.execute("python")
                self.graph[src][tgt][key]["prefun"] = lambdify(
                    [Symbol("x_j")],
                    coup.pre.subs({k: p.value for k, p in coup.parameters.items()}),
                )
                self.graph[src][tgt][key]["postfun"] = lambdify(
                    [Symbol("gx")],
                    coup.post.subs({k: p.value for k, p in coup.parameters.items()}),
                )
        else:
            for src, tgt in self.graph.edges:
                coup = self.graph[src][tgt]["coupling"]
                self.graph[src][tgt]["cfun"] = coup.execute("python")
                self.graph[src][tgt]["prefun"] = lambdify(
                    [Symbol("x_j")],
                    coup.pre.subs({k: p.value for k, p in coup.parameters.items()}),
                )
                self.graph[src][tgt]["postfun"] = lambdify(
                    [Symbol("gx")],
                    coup.post.subs({k: p.value for k, p in coup.parameters.items()}),
                )

    def setup_initial_conditions(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["state"] = np.array(
                [
                    sv.initial_value
                    for sv in self.graph.nodes[node]["model"].state_variables.values()
                ]
            )
            # self.graph.nodes[node]["state"] = np.random.uniform(-1, 1, size=2)

    def setup_stimulation(self, sampling_rate=500, duration=2000):
        for node in self.graph.nodes:
            if (
                "stimulus" in self.graph.nodes[node].keys()
                and self.graph.nodes[node]["stimulus"] is not None
            ):
                stimulus = self.graph.nodes[node]["stimulus"]
                self.graph.nodes[node]["stimfun"] = stimulus.execute(
                    format="python",
                    duration=stimulus.duration,
                    sampling_rate=sampling_rate,
                )

    def run(self, duration=1000, dt=1, format="graph"):
        self.setup_initial_conditions()
        self.setup_stimulation()
        self.setup_dfuns()
        self.setup_cfuns()

        compgraph.initialize_graph_states_with_history(self.graph, delay_buffer=1000)
        time_points = compgraph.simulate_graph_dynamics_with_delay(
            self.graph, T=duration, dt=dt
        )

        ts = compgraph.collect_time_series(self.graph, time_points)

        return ts
