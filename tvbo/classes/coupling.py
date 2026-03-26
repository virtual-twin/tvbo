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
import os
from os.path import join

import networkx as nx
import numpy as np
import owlready2
from mako.template import Template
from sympy import pycode

from tvbo import templates
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.codegen import templater
from tvbo.codegen.code import parse_eq
from tvbo.ontology import constants
from tvbo.ontology import owl as ontology
from tvbo.ontology import query
from tvbo.classes import equation as equations
from tvbo.classes import dynamics as localdynamics


TEMPLATES = templates.root

# Path to database coupling function YAML files
_COUPLING_DB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'database', 'coupling_functions'
)


def _load_coupling_from_database(name, coupling):
    """Fill coupling metadata from a database YAML file.

    Looks for ``tvbo/database/coupling_functions/<name>.yaml`` and fills
    missing ``pre_expression``, ``post_expression``, ``parameters``,
    and ``delayed`` on the coupling instance.

    Parameters
    ----------
    name : str
        Coupling function name (e.g. ``"KuramotoCoupling"``).
    coupling : tvbo_datamodel.Coupling
        Coupling instance to fill (modified in-place).

    Returns
    -------
    bool
        True if a database file was found and applied.
    """
    import yaml as _yaml

    db_file = os.path.join(_COUPLING_DB_DIR, f'{name}.yaml')
    if not os.path.exists(db_file):
        return False

    with open(db_file) as f:
        data = _yaml.safe_load(f)

    if 'pre_expression' in data and not getattr(coupling, 'pre_expression', None):
        pe = data['pre_expression']
        coupling.pre_expression = tvbo_datamodel.Equation(
            **(pe if isinstance(pe, dict) else {'rhs': pe})
        )
    if 'post_expression' in data and not getattr(coupling, 'post_expression', None):
        pe = data['post_expression']
        coupling.post_expression = tvbo_datamodel.Equation(
            **(pe if isinstance(pe, dict) else {'rhs': pe})
        )
    if 'parameters' in data:
        for pname, pval in data['parameters'].items():
            if pname not in coupling.parameters:
                if isinstance(pval, dict):
                    if 'name' not in pval:
                        pval['name'] = pname
                    coupling.parameters[pname] = tvbo_datamodel.Parameter(**pval)
                else:
                    coupling.parameters[pname] = tvbo_datamodel.Parameter(
                        name=pname, value=pval
                    )
    if 'delayed' in data and data['delayed'] is not None:
        if getattr(coupling, 'delayed', None) is None:
            coupling.delayed = data['delayed']

    return True


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

    If ``iri`` is set (e.g. ``tvbo:SigmoidalJansenRit``), missing fields
    are automatically populated from the ontology/database.
    Use ``Coupling.from_ontology(name)`` for explicit ontology lookup.
    """

    def __init__(self, **kwargs):
        # Legacy: accept and ignore use_ontology kwarg
        kwargs.pop('use_ontology', None)
        super().__init__(**kwargs)
        # Auto-populate from ontology if iri is set and expressions are missing
        if getattr(self, 'iri', None) and not getattr(self, 'pre_expression', None):
            self._populate_from_ontology()

    def _populate_from_ontology(self, lookup_name=None):
        """Fill missing metadata fields from ontology/database.

        Parameters
        ----------
        lookup_name : str, optional
            Name or CURIE to look up. If None, uses ``self.name``.
            Supports plain names (``KuramotoCoupling``) and CURIEs
            (``tvbo:KuramotoCoupling``).
        """
        if lookup_name:
            # Strip CURIE prefix if present
            lookup = lookup_name.split(':', 1)[-1] if ':' in lookup_name else lookup_name
        else:
            lookup = getattr(self, 'name', None)

        # Try database YAML first (fast, no ontology deps needed)
        if lookup and _load_coupling_from_database(lookup, self):
            return

        # Fallback: ontology lookup
        try:
            if lookup:
                hits = query.label_search(lookup, root_class="Coupling")
                oc = hits[0] if hits else None
            else:
                oc = self.ontoclass
        except Exception:
            oc = None
        if not oc:
            return

        # Reuse shared helper; non-destructive fill
        coupling_class2metadata(oc, self, overwrite=False)

    def populate_from_type(self, type_ref):
        """Fill missing pre/post expressions and parameters from a type reference.

        This is used when ``network.coupling`` entries specify a ``type`` field
        to reference a known coupling function (e.g. ``KuramotoCoupling`` or
        ``tvbo:KuramotoCoupling``).

        Parameters
        ----------
        type_ref : str
            Coupling function name or CURIE (e.g. ``"KuramotoCoupling"``
            or ``"tvbo:KuramotoCoupling"``).
        """
        self._coupling_type = type_ref
        self._populate_from_ontology(lookup_name=type_ref)
        self._resolve_xi_xj()

    def _resolve_xi_xj(self):
        """Auto-populate local_states/incoming_states from x_i/x_j in expression.

        Coupling database equations use generic placeholders ``x_i`` (local
        node state) and ``x_j`` (source node state).  When the user has
        declared ``incoming_states`` (the actual state variable names to
        pull from connected nodes) but not ``local_states``, and the
        expression references ``x_i``, we mirror ``incoming_states`` into
        ``local_states`` so the template can generate correct assignments.
        """
        pre_rhs = str(self.pre_expression.rhs) if getattr(self, 'pre_expression', None) else ''
        incoming = getattr(self, 'incoming_states', None) or []
        local = getattr(self, 'local_states', None) or []

        if 'x_i' in pre_rhs and not local and incoming:
            # x_i refers to local copy of the same states as incoming
            self.local_states = list(incoming)

        if 'x_j' in pre_rhs and not incoming and local:
            # x_j refers to remote copy of the same states as local
            self.incoming_states = list(local)

    @classmethod
    def from_ontology(cls, ontoclass):
        """Create a Coupling instance from an ontology Coupling class or name.

        Accepts an owlready2 class, a plain name (``"SigmoidalJansenRit"``),
        or a CURIE (``"tvbo:SigmoidalJansenRit"``).
        Tries the database YAML first, then falls back to ontology lookup.
        """
        if isinstance(ontoclass, str):
            # Strip CURIE prefix if present
            lookup = ontoclass.split(':', 1)[-1] if ':' in ontoclass else ontoclass
            # Try database YAML first (fast, no ontology deps needed)
            coup = cls(name=lookup)
            if _load_coupling_from_database(lookup, coup):
                return coup
            # Fall back to ontology lookup
            hits = query.label_search(
                lookup, root_class="Coupling", exact_match=["label"]
            )
            if not hits:
                raise ValueError(f"Coupling '{lookup}' not found in database or ontology.")
            ontoclass = hits[0]
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

    @classmethod
    def from_file(cls, filepath: str) -> "Coupling":
        """Load a Coupling from a YAML file."""
        from linkml_runtime.loaders import yaml_loader
        return yaml_loader.load(str(filepath), target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "Coupling":
        """Load a Coupling by name from the tvbo database."""
        from tvbo.data.registry import resolve
        return cls.from_file(str(resolve("Coupling", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available coupling functions in the tvbo database."""
        from tvbo.data.registry import list_entries
        return list_entries("Coupling")

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

    def render(self, format="yaml", **kwargs) -> str:
        fmt = format.lower()
        if fmt == "yaml":
            return self.to_yaml(filepath=kwargs.get("filepath"))
        return self.render_code(format=format, **kwargs)

    def render_code(self, format="tvb", model=None, alt_label=None, **kwargs):
        if format == "tvb":
            rendered_code = templates.lookup.get_template(
                "tvbo-tvb-coupling.py.mako"
            ).render(coupling=self)

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("tvbo-jax-coupling.py.mako")
            rendered_code = template.render(coupling=self, model=model, **kwargs)

        elif format.lower() in ("tvboptim", "tvb-optim"):
            template = templates.lookup.get_template(
                "tvbo-tvboptim-coupling.py.mako"
            )
            rendered_code = template.render(coupling=self, **kwargs)

        elif format.lower() == "python":
            from tvbo.codegen.code import NumPyPrinter, render_expression

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

        elif format.lower() in ("tvboptim", "tvb-optim"):
            namespace = {}
            exec(self.render_code(format="tvboptim"), namespace)
            cls = namespace[self.name]
            return cls(**kwargs)

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

    def symbolic(self, delays=False):
        """Full symbolic coupling equation with proper indexed state variables.

        Resolves all expression styles (``theta_j``/``theta_i``, ``x_j``/``x_i``,
        ``incoming_states``/``local_states``) into proper ``IndexedBase`` notation
        and wraps the pre-expression in a weighted summation over connected nodes.

        Parameters
        ----------
        delays : bool
            If True, incoming states carry an explicit time-delay index:
            ``y1[j, t - tau[i, j]]`` instead of plain ``y1[j]``.

        Returns
        -------
        sympy.Expr
            E.g. ``Sum(w[i, j]*sin(theta[j] - theta[i]), (j, 0, N - 1))/N``
        """
        import sympy as sp
        from sympy import IndexedBase, Symbol, symbols, Sum
        from tvbo.parse.expression import parse_eq

        i, j, N, gx = symbols("i j N gx")
        w = IndexedBase("w")

        # State variable names from coupling metadata
        incoming = [str(s) for s in (self.incoming_states or [])]
        local = [str(s) for s in (self.local_states or [])]

        # Build IndexedBase per state and substitution map
        state_bases = {}
        subs_map = {}
        for state_name in set(incoming + local):
            state_bases[state_name] = IndexedBase(state_name)

        # Index expressions: delayed adds a time argument to incoming states
        if delays:
            t = Symbol("t")
            tau = IndexedBase("tau")
            def _incoming(sn):
                return state_bases[sn][j, t - tau[i, j]]
        else:
            def _incoming(sn):
                return state_bases[sn][j]

        def _local(sn):
            return state_bases[sn][i]

        # Bare state names: y1 → y1[j] (incoming), y1 → y1[i] (local)
        # This handles expressions like "2*e0 / (1 + exp(r*(v0 - (y1 - y2))))"
        for sn in incoming:
            subs_map[Symbol(sn)] = _incoming(sn)
        for sn in local:
            subs_map[Symbol(sn)] = _local(sn)

        # State-subscript aliases: theta_j → theta[j], theta_i → theta[i]
        for sn in incoming:
            subs_map[Symbol(f"{sn}_j")] = _incoming(sn)
        for sn in local:
            subs_map[Symbol(f"{sn}_i")] = _local(sn)

        # x_j / x_i fallback → first state
        if incoming:
            subs_map[Symbol("x_j")] = _incoming(incoming[0])
        if local:
            subs_map[Symbol("x_i")] = _local(local[0])

        # Literal incoming_states / local_states → first state
        if incoming:
            subs_map[Symbol("incoming_states")] = _incoming(incoming[0])
        if local:
            subs_map[Symbol("local_states")] = _local(local[0])

        # local_dict for parse_eq: ensure alias tokens parse as Symbols
        local_dict = {str(k): k for k in subs_map}
        local_dict["gx"] = gx
        local_dict["N"] = N
        for pname in (self.parameters or {}):
            local_dict[str(pname)] = Symbol(str(pname))

        # Parse and substitute inside evaluate=False to prevent:
        #  - sin.eval() sign canonicalization (Function.__new__, L301)
        #  - Add.flatten() alphabetical reordering (AssocOp.__new__, L95)
        # Parsing must also be inside the block so that e.g.
        # v0 - (y1 - y2) isn't flattened to v0 - y1 + y2 before subs.
        with sp.evaluate(False):
            pre_expr = parse_eq(self.pre_expression, local_dict=local_dict)
            post_expr = parse_eq(self.post_expression, local_dict=local_dict)
            pre_indexed = pre_expr.subs(subs_map)

        # Full coupling: Sum(w[i,j] * pre, (j, 0, N-1)), substituted into post
        gx_sum = Sum(w[i, j] * pre_indexed, (j, 0, N - 1))
        return post_expr.subs({gx: gx_sum})

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

