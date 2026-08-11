"""Ontology population, symbolic reading, code generation and plotting for coupling.

Attached to the generated classes by name (``CouplingBehaviour`` -> ``Coupling``), so a
coupling carries these however it was built. The experiment loader used to reassign
``__class__`` to reach the one nested in a loaded experiment, which left couplings reached
any other way — ``network.coupling`` entries, an inner coupling, a hand-built one — as
plain records.

Unlike the other mixins this one hooks construction. It can: the generated
``__post_init__`` ends in ``super().__post_init__(**kwargs)``, and the mixin sits directly
before ``YAMLRoot`` in the MRO, so the hook runs once the generated normalization is done,
whichever path built the object. That is where a coupling given only an ``iri`` adopts the
name and the expressions that ``iri`` points at, as the wrapper's ``__init__`` used to.
"""

from __future__ import annotations

import numpy as np


def _schema_default(cls, slot: str):
    """The default the schema gives *slot*, whichever generated form *cls* is.

    A post-init hook cannot see the keywords as they were passed — the defaults are
    already applied — so "the recipe did not name this" is read as "the value still equals
    the schema default", and the two forms keep that default in different places.
    """
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields is not None:
        return fields[slot].default
    return cls.model_fields[slot].default


class CouplingBehaviour:
    """Population, rendering and symbolic reading for a coupling function."""

    def __post_init__(self, *args, **kwargs):
        """The LinkML dataclasses' construction hook."""
        super().__post_init__(*args, **kwargs)
        self._adopt_iri_entry()

    def model_post_init(self, context, /):
        """The Pydantic models' construction hook.

        The two generated forms do not share one. Without this the same YAML produced two
        different couplings — the dataclass populated from its ``iri``, the Pydantic model
        left as a bare default ``Linear`` — which is the divergence
        ``tests/test_schema_aliases.py`` exists to prevent.
        """
        super().model_post_init(context)
        self._adopt_iri_entry()

    def _adopt_iri_entry(self):
        """Adopt the ontology entry an ``iri``-only coupling points at.

        A recipe that names the coupling solely by ``iri`` gets that entry's local name and
        its expressions; an explicitly named one keeps its own name and is only filled in.
        `name` carries a schema default, so "explicitly named" is read as differing from
        that default — the one thing ``__init__`` could see that a post-init hook cannot.
        """
        if not getattr(self, "iri", None) or getattr(self, "pre_expression", None):
            return

        from tvbo.data.registry import local_name

        local = local_name(self.iri)
        if self.name == _schema_default(type(self), "name"):
            self.name = local
        self._populate_from_ontology(lookup_name=local)

    def _populate_from_ontology(self, lookup_name=None):
        """Fill missing metadata fields from ontology/database.

        Parameters
        ----------
        lookup_name : str, optional
            Name or CURIE to look up. If None, uses ``self.name``.
            Supports plain names (``KuramotoCoupling``) and CURIEs
            (``tvbo:KuramotoCoupling``).
        """
        from tvbo.classes.coupling import _load_coupling_from_database, coupling_class2metadata
        from tvbo.ontology import query

        if lookup_name:
            lookup = lookup_name.split(":", 1)[-1] if ":" in lookup_name else lookup_name
        else:
            lookup = getattr(self, "name", None)

        if lookup and _load_coupling_from_database(lookup, self):
            return

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
        pre_rhs = str(self.pre_expression.rhs) if getattr(self, "pre_expression", None) else ""
        incoming = getattr(self, "incoming_states", None) or []
        local = getattr(self, "local_states", None) or []

        if "x_i" in pre_rhs and not local and incoming:
            self.local_states = list(incoming)

        if "x_j" in pre_rhs and not incoming and local:
            self.incoming_states = list(local)

    @classmethod
    def from_ontology(cls, ontoclass):
        """Create a Coupling instance from an ontology Coupling class or name.

        Accepts an owlready2 class, a plain name (``"SigmoidalJansenRit"``),
        or a CURIE (``"tvbo:SigmoidalJansenRit"``).
        Tries the database YAML first, then falls back to ontology lookup.
        """
        import owlready2

        from tvbo.classes.coupling import _load_coupling_from_database, coupling_class2metadata
        from tvbo.datamodel import schema as tvbo_datamodel
        from tvbo.ontology import query

        if isinstance(ontoclass, str):
            lookup = ontoclass.split(":", 1)[-1] if ":" in ontoclass else ontoclass
            coup = cls(name=lookup)
            if _load_coupling_from_database(lookup, coup):
                return coup
            hits = query.label_search(lookup, root_class="Coupling", exact_match=["label"])
            if not hits:
                raise ValueError(f"Coupling '{lookup}' not found in database or ontology.")
            ontoclass = hits[0]
        if not isinstance(ontoclass, owlready2.entity.ThingClass):
            raise ValueError("ontoclass must be a string or an ontology Coupling class.")
        metadata = tvbo_datamodel.Coupling(name=ontoclass.label.first())
        coupling_class2metadata(ontoclass, metadata, overwrite=True)
        return cls(**metadata._as_dict)

    @classmethod
    def from_datamodel(cls, datamodel_instance):
        """Copy an existing datamodel `Coupling`'s fields into a new one."""
        from tvbo.datamodel import schema as tvbo_datamodel

        if not isinstance(datamodel_instance, tvbo_datamodel.Coupling):
            raise ValueError("datamodel_instance must be a tvbo_datamodel.Coupling instance.")
        return cls(**datamodel_instance._as_dict)

    @classmethod
    def from_file(cls, filepath: str):
        """Load a Coupling from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(filepath), target_class=cls)

    @classmethod
    def from_db(cls, name: str):
        """Load a Coupling by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Coupling", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available coupling functions in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Coupling")

    @property
    def metadata(self):
        """The coupling's own metadata, i.e. this object itself (back-compat accessor)."""
        return self

    def to_yaml(self, filepath: str | None = None):
        """Serialize this coupling to YAML.

        Args:
            filepath: Optional path to write the YAML to. If omitted, the
                YAML is only returned.

        Returns:
            The YAML representation of the coupling as a string.
        """
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    def render(self, format="yaml", **kwargs) -> str:
        """Render this coupling in the requested output format.

        Dispatches to `to_yaml`, `report`, or `render_code` depending on
        `format`.

        Args:
            format: Output format. `"yaml"` serializes to YAML; `"report"`,
                `"markdown"`, `"md"`, or `"pdf"` produce a human-readable
                report; any other value is forwarded to `render_code` to
                generate backend code.
            **kwargs: Forwarded to the underlying renderer (e.g. `filepath`).

        Returns:
            The rendered output as a string.
        """
        fmt = format.lower()
        if fmt == "yaml":
            return self.to_yaml(filepath=kwargs.get("filepath"))
        if fmt in ("report", "markdown", "md", "pdf"):
            report_fmt = "pdf" if fmt == "pdf" else "markdown"
            return self.report(format=report_fmt, **kwargs)
        return self.render_code(format=format, **kwargs)

    def render_code(self, format="tvb", model=None, alt_label=None, **kwargs):
        """Generate backend-specific code for this coupling.

        Args:
            format: Target backend (case-insensitive). One of `"tvb"`,
                `"autodiff"`/`"jax"`, `"tvboptim"`/`"tvb-optim"`, or
                `"python"`.
            model: Model context passed to the JAX template when relevant.
            alt_label: Alternative label accepted for signature
                compatibility; not used directly by this method.
            **kwargs: Additional arguments forwarded to the selected template.

        Returns:
            The formatted, backend-specific source code as a string.

        Raises:
            ValueError: If `format` is not a supported backend.
        """
        from tvbo import templates
        from tvbo.codegen import templater

        fmt = format.lower()
        if fmt == "tvb":
            rendered_code = templates.lookup.get_template("tvbo-tvb-coupling.py.mako").render(coupling=self)

        elif fmt in ("autodiff", "jax"):
            template = templates.lookup.get_template("tvbo-jax-coupling.py.mako")
            rendered_code = template.render(coupling=self, model=model, **kwargs)

        elif fmt in ("tvboptim", "tvb-optim"):
            template = templates.lookup.get_template("tvbo-tvboptim-coupling.py.mako")
            rendered_code = template.render(coupling=self, **kwargs)

        elif fmt == "python":
            from tvbo.codegen.code import render_expression

            rendered_code = render_expression(self.equation, format="python")

        else:
            raise ValueError(f"Unsupported render_code format: {format!r}")

        return templater.format_code(rendered_code)

    def report(self, format: str = "markdown", outputfile: str | None = None,
               parameters: bool = True, equations=None) -> str:
        """Render a human-readable markdown (or pdf) report for this coupling.

        Includes pre/post expressions, the full assembled coupling equation
        (``Coupling.equation``), and the parameter table.

        Args:
            format: ``markdown``/``md`` or ``pdf``.
            outputfile: Where to write the rendering, if anywhere.
            parameters: Emit the parameter table. A host report that already
                glossaries these symbols passes ``False`` — a study's Methods lists
                them beside the model's, so the table here would repeat rows the
                reader has just read, uncaptioned and unnumbered.
            equations: A ``tvbo.utils.report.Equations`` to number and anchor the
                coupling equation with. Without one it renders bare, and in a study
                report that is the only unnumbered equation on the page — the reader
                can cite every state equation and not the coupling that joins them.
        """
        from tvbo import templates

        fmt = format.lower()
        if fmt not in ("markdown", "md", "pdf"):
            raise ValueError("format must be one of: markdown, md, pdf")

        template = templates.lookup.get_template("report/tvbo-report-coupling.md.mako")
        md = template.render(coupling=self, parameters=parameters, equations=equations)

        if outputfile:
            if fmt == "pdf":
                from tvbo.utils import report as _report

                _report.to_pdf(md, outputfile)
            else:
                with open(outputfile, "w", encoding="utf-8") as f:
                    f.write(md)
        return md

    def generate_report(self, format: str = "markdown", outputfile: str | None = None) -> str:
        """Backward-compatible alias for :meth:`report`."""
        return self.report(format=format, outputfile=outputfile)

    def execute(self, format="tvb", alt_label=None, **kwargs):
        """Render, execute, and instantiate this coupling for a backend.

        Renders the coupling code via `render_code`, executes it, and returns
        the resulting runtime object.

        Args:
            format: Target backend. `"tvb"` returns an instantiated TVB
                coupling object; `"tvboptim"`/`"tvb-optim"` returns an
                instantiated tvboptim coupling class; `"python"` returns a
                `sympy.lambdify`-based callable for the coupling equation.
            alt_label: Alternative name to instantiate the coupling under
                (TVB backend only).
            **kwargs: Constructor arguments forwarded to the instantiated
                coupling object.

        Returns:
            The instantiated backend coupling object, or a callable for the
            `"python"` format.
        """
        from tvbo.codegen import templater

        fmt = format.lower()
        if fmt == "tvb":
            local_vars = {}
            exec(self.render_code(alt_label=alt_label), templater.exec_globals, local_vars)
            return local_vars[self.name if not alt_label else alt_label](**kwargs)

        if fmt in ("tvboptim", "tvb-optim"):
            namespace = {}
            exec(self.render_code(format="tvboptim"), namespace)
            return namespace[self.name](**kwargs)

        if fmt == "python":
            from sympy import Symbol, lambdify

            return lambdify(
                [Symbol("x"), Symbol("g"), Symbol("N"), Symbol("i")] + [Symbol(p) for p in self.parameters],
                self.equation,
            )

    @property
    def ontoclass(self):
        """The ontology `Coupling` class matching this coupling's name, or `None` if not found."""
        from tvbo.ontology import query

        try:
            hits = query.label_search(self.name, root_class="Coupling") if getattr(self, "name", None) else []
            return hits[0] if hits else None
        except Exception:
            return None

    @property
    def pre(self):
        """The parsed pre-summation expression of the coupling function."""
        from tvbo.codegen.code import parse_eq

        return parse_eq(self.pre_expression)

    @property
    def post(self):
        """The parsed post-summation expression of the coupling function."""
        from tvbo.codegen.code import parse_eq

        return parse_eq(self.post_expression)

    @property
    def equation(self):
        """The full assembled global coupling equation (pre and post combined), or `None` if it cannot be built."""
        from tvbo.classes import equation as equations

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

        Notes
        -----
        Parsing and substitution both stay inside ``evaluate(False)`` so that sympy
        neither canonicalizes signs nor reorders an ``Add`` before the states are
        indexed. In the factored case a bare state name refers to the summed (``j``)
        node even where it is declared ``local``; only an explicit ``_i`` stays local.
        """
        import sympy as sp
        from sympy import IndexedBase, Sum, Symbol, symbols

        from tvbo.parse.expression import parse_eq

        i, j, N, gx = symbols("i j N gx")
        w = IndexedBase("w")

        incoming = [str(s) for s in (self.incoming_states or [])]
        local = [str(s) for s in (self.local_states or [])]

        state_bases = {}
        subs_map = {}
        for state_name in set(incoming + local):
            state_bases[state_name] = IndexedBase(state_name)

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

        for sn in incoming:
            subs_map[Symbol(sn)] = _incoming(sn)
        for sn in local:
            subs_map[Symbol(sn)] = _local(sn)

        for sn in incoming:
            subs_map[Symbol(f"{sn}_j")] = _incoming(sn)
        for sn in local:
            subs_map[Symbol(f"{sn}_i")] = _local(sn)

        if incoming:
            subs_map[Symbol("x_j")] = _incoming(incoming[0])
        if local:
            subs_map[Symbol("x_i")] = _local(local[0])

        if incoming:
            subs_map[Symbol("incoming_states")] = _incoming(incoming[0])
        if local:
            subs_map[Symbol("local_states")] = _local(local[0])

        local_dict = {str(k): k for k in subs_map}
        local_dict["gx"] = gx
        local_dict["N"] = N
        for pname in self.parameters or {}:
            local_dict[str(pname)] = Symbol(str(pname))

        with sp.evaluate(False):
            pre_expr = parse_eq(self.pre_expression, local_dict=local_dict)
            post_expr = parse_eq(self.post_expression, local_dict=local_dict)
            if not isinstance(pre_expr, (list, tuple)):
                pre_indexed = pre_expr.subs(subs_map)

        if isinstance(pre_expr, (list, tuple)):
            pre_subs = dict(subs_map)
            for sn in set(incoming) | set(local):
                pre_subs[Symbol(sn)] = _incoming(sn)
            return self._factored_symbolic(pre_expr, post_expr, subs_map, pre_subs, w, i, j, N)

        gx_sum = Sum(w[i, j] * pre_indexed, (j, 0, N - 1))
        return post_expr.subs({gx: gx_sum})

    def _factored_symbolic(self, pre_expr, post_expr, subs_map, pre_subs, w, i, j, N):
        """Assemble the symbolic form of a factored (vectorized) coupling.

        The k-th pre component is summed into ``gx_k`` and the post recombines them. When
        the post is linear in the ``gx_k`` — the usual case — this collapses to the
        canonical single sum ``c = Sum_j w[i,j] * sum_k a_k(x_i) * pre_k(x_j)`` and lets
        ``trigsimp`` fold the per-edge term; otherwise the explicit
        ``gx_k = Sum(w * pre_k)`` form is kept. Built outside the caller's
        ``evaluate(False)`` block so the sum, the coefficients and ``trigsimp`` evaluate.

        A folded odd-trig term is shown in the physics convention ``f(incoming - local)``:
        sympy canonicalises ``f(x_j - x_i)`` to ``-f(x_i - x_j)``, so it is rebuilt
        positive-first.
        """
        import sympy as sp
        from sympy import Sum, Symbol

        pre_k = [comp.subs(pre_subs) for comp in pre_expr]
        gxk = [Symbol(f"gx_{k}") for k in range(len(pre_k))]
        post_indexed = post_expr.subs(subs_map)
        coeffs = [post_indexed.coeff(g) for g in gxk]
        if sp.expand(post_indexed - sum(a * g for a, g in zip(coeffs, gxk))) == 0:
            edge = sp.trigsimp(sum(a * p for a, p in zip(coeffs, pre_k)))
            c0, rest = edge.as_coeff_Mul()
            odd = (sp.sin, sp.tan, sp.sinh, sp.tanh)
            if c0 == -1 and getattr(rest, "func", None) in odd and rest.args[0].is_Add:
                terms = sorted(
                    (-t for t in rest.args[0].as_ordered_terms()),
                    key=lambda t: t.could_extract_minus_sign(),
                )
                with sp.evaluate(False):
                    edge = rest.func(sp.Add(*terms, evaluate=False))
            return Sum(w[i, j] * edge, (j, 0, N - 1))
        return post_indexed.subs({g: Sum(w[i, j] * p, (j, 0, N - 1)) for g, p in zip(gxk, pre_k)})

    def summed_inputs(self, delays=False):
        """Summed inputs ``gx_k`` of a factored / vectorized coupling.

        A factored coupling emits a *list* pre-expression whose k-th component is summed
        over the graph into ``gx_k = Sum_j w[i,j] * (c_pre)_k(x_j)``, which the
        post-expression then recombines. Returns ``[(gx_k, sum_expr), ...]`` so a report
        can state precisely what ``gx_0``, ``gx_1``, … mean; empty for a scalar coupling.
        """
        import sympy as sp
        from sympy import IndexedBase, Sum, Symbol, symbols

        from tvbo.parse.expression import parse_eq

        with sp.evaluate(False):
            pre = parse_eq(self.pre_expression)
        if not isinstance(pre, (list, tuple)):
            return []

        i, j, N = symbols("i j N")
        w = IndexedBase("w")
        states = {str(s) for s in (self.incoming_states or [])} | {str(s) for s in (self.local_states or [])}
        t, tau = Symbol("t"), IndexedBase("tau")

        def _incoming(sn):
            base = IndexedBase(sn)
            return base[j, t - tau[i, j]] if delays else base[j]

        # In a summed pre a bare state (or its `_j` alias) is the incoming (j) node.
        subs = {}
        for sn in states:
            subs[Symbol(sn)] = _incoming(sn)
            subs[Symbol(f"{sn}_j")] = _incoming(sn)
        return [
            (Symbol(f"gx_{k}"), Sum(w[i, j] * comp.subs(subs), (j, 0, N - 1)))
            for k, comp in enumerate(pre)
        ]

    def plot(self, weights=None, node_idx=0, xs=None, ax=None, **kwargs):
        """Plot the coupling output against a single input state component.

        Lambdifies the assembled coupling `equation` and evaluates it while
        sweeping one node's state over `xs`, holding the other components
        fixed.

        Args:
            weights: Connectivity weight matrix. If omitted, a random 3x3
                matrix with a zeroed diagonal is used.
            node_idx: Plotting gate only. When not `None` (the default is `0`)
                the plot is drawn; the value is not otherwise used, as the swept
                component is always index 0. Pass `None` to skip plotting.
            xs: Values to sweep the selected state component over. Defaults
                to 100 points on the interval `[-2, 2]`.
            ax: Matplotlib axes to draw on. If omitted, a new figure is
                created and returned.
            **kwargs: Accepted for signature flexibility; currently unused.

        Returns:
            The created Matplotlib figure when `ax` is omitted and `node_idx`
            is not `None`; otherwise `None`.
        """
        import matplotlib.pyplot as plt
        import sympy as sp

        if node_idx is None:
            return None

        if weights is None:
            weights = np.random.normal(loc=0.0, scale=1.0, size=(3, 3))
            np.fill_diagonal(weights, 0)

        i, N = sp.symbols("i N", integer=True)
        x = sp.IndexedBase("x")
        g = sp.IndexedBase("g")

        used_param_names = sorted(
            name for name in self.parameters if sp.Symbol(name) in self.equation.free_symbols
        )
        param_syms = tuple(sp.symbols(used_param_names))
        f = sp.lambdify((x, g, i, N) + param_syms, self.equation, modules="numpy")

        if xs is None:
            xs = np.linspace(-2.0, 2.0, 100)

        varnames = f.__code__.co_varnames
        params = {p: self.parameters[p].value for p in used_param_names if p in varnames}

        ys = []
        for xv in xs:
            x_tmp = xs.copy()
            x_tmp[0] = xv
            ys.append(f(x_tmp, weights, 1, weights.shape[0], **params))

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(xs, ys)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("y(i)")
        ax.set_title("Coupling vs single input component")
        if return_fig:
            plt.close()
            return fig
