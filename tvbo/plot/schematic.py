"""Population-schematic diagrams for neural-mass / mean-field / spiking models.

Renders the excitatory/inhibitory **population circuit** of a model: population nodes plus
**typed, signed** connections — synapse type (NMDA / AMPA / GABA / …) sets the line colour,
biophysical sign sets the terminal (excitatory ``▶``, inhibitory ``●``) — within a population, between populations, as recurrent self-loops, and across ``n_areas`` coupled areas. It is the
standing "what is this model" diagram every recipe wants (Deco 2014 Fig. 1a/2a, Jansen-Rit, …).

**Rendering backend = graphviz layout + matplotlib draw.** graphviz (the ``dot`` engine) solves
node placement and edge-spline routing — its strength — and matplotlib draws the result with full styling control: crisp custom arrowheads, endpoints snapped to node edges, small compact
self-loops, neuron-dot population glyphs, and publication typography. This beats either alone (graphviz styling is rigid; hand-placed matplotlib arrows don't route cleanly).

Two entry points:

* **Declarative** — build ``Population`` / ``Connection`` objects → ``PopulationSchematic``.
* **Model-derived** — ``PopulationSchematic.from_dynamics(dynamics)`` reads a tvbo ``Dynamics``:
  populations are its state variables (``S_E`` → ``E``); connections are recovered by expanding the derived variables into each state variable's dfun and classifying every cross-population
  term by the **sign** of its coefficient and the **synapse** named in the multiplying parameter.

Flexible layout/style, all via ``plot(...)`` keywords:

* ``layout``: ``"stacked"`` (E-row over I-row, the DMF look) · ``"mirror"`` (``[I E]…[E I]``, the
  paired-area spiking look) · ``"row"`` (single rank).
* ``glyph``: ``"circle"`` · ``"neurons"`` (dotted population = spiking) · a callable.
* ``self_loop``, ``coupling_blob``, ``ellipsis``, ``legend``/``legend_labels``, ``synapse_colors``,
  ``node_size``, ``ranksep``/``nodesep`` — all tunable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Synapse-type → line colour (Deco 2014 convention: NMDA/GABA black, AMPA grey).
SYNAPSE_COLORS = {"NMDA": "#141414", "GABA": "#141414", "AMPA": "#9a9a9a", "GLUT": "#141414", None: "#141414"}


@dataclass
class Population:
    """One neural population (a node). ``sign`` = ``"excitatory"`` (open node) /
    ``"inhibitory"`` (filled). ``label`` is the glyph inside the node (defaults to ``name``)."""

    name: str
    sign: str = "excitatory"
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name


@dataclass
class Connection:
    """Directed connection ``source → target``. ``synapse`` (NMDA/AMPA/GABA) sets colour;
    ``sign`` (``excitatory`` ▶ / ``inhibitory`` ●) sets the terminal; ``scope`` is ``"local"`` (within an area) or ``"long_range"`` (between adjacent areas). ``recurrent`` (self-loop) is
    inferred from ``source == target`` when omitted."""

    source: str
    target: str
    synapse: Optional[str] = None
    sign: str = "excitatory"
    scope: str = "local"
    label: Optional[str] = None
    recurrent: Optional[bool] = None

    def __post_init__(self):
        if self.recurrent is None:
            self.recurrent = self.source == self.target


# Model-derived connection recovery (from a tvbo Dynamics)
def _synapse_from_symbol(name: str) -> Optional[str]:
    up = name.upper()
    for tag in ("NMDA", "AMPA", "GABA", "GLUT"):
        if tag in up:
            return tag
    return None


def _population_of(state_var: str) -> str:
    return state_var.rsplit("_", 1)[-1] if "_" in state_var else state_var


def _pop_sign(pop: str) -> str:
    return "inhibitory" if pop.upper().startswith("I") else "excitatory"


def _coeff_sign(coeff) -> int:
    val = coeff.subs({s: 1 for s in coeff.free_symbols})
    try:
        return 1 if float(val) >= 0 else -1
    except TypeError:
        import sympy as sp

        return 1 if sp.sign(coeff) != -1 else -1


def _edge_synapse(coeff, synapse_map, default_sign):
    names = sorted(str(s) for s in coeff.free_symbols)
    if synapse_map:
        for n in names:
            if n in synapse_map:
                return synapse_map[n]
    for n in names:
        syn = _synapse_from_symbol(n)
        if syn:
            return syn
    return "GABA" if default_sign == "inhibitory" else None


def _derive_from_dynamics(dynamics, coupling_symbols, long_range_synapse, synapse_map):
    """Recover (populations, connections) from a ``Dynamics`` by reading the **input-current** derived variables (affine in state vars + coupling), which capture the connections that the
    nonlinear transfer function ``r = H(I)`` hides from a plain state-variable coefficient."""
    import sympy as sp

    svs = list(dynamics.state_variables.keys())
    derived = dict(getattr(dynamics, "derived_variables", {}) or {})
    coupling_symbols = list(coupling_symbols)
    syms = {n: sp.Symbol(n) for n in set(svs) | set(derived) | set(dynamics.parameters) | set(coupling_symbols)}

    def _rhs(obj):
        eq = getattr(obj, "equation", obj)
        return sp.sympify(str(getattr(eq, "rhs", getattr(obj, "rhs", obj))), locals=syms)

    dvals = {syms[n]: _rhs(d) for n, d in derived.items()}
    for _ in range(len(derived) + 1):
        dvals = {k: v.subs(dvals) for k, v in dvals.items()}

    drivers = [syms[s] for s in svs] + [syms[c] for c in coupling_symbols]

    def _affine_score(expr):
        deg = 0
        for v in drivers:
            first = sp.diff(expr, v)
            if first == 0:
                continue
            deg += 1
            if first.free_symbols & set(drivers):
                return (False, 0)
        return (deg > 0, deg)

    inputs: dict[str, tuple] = {}
    for n in derived:
        ok, deg = _affine_score(dvals[syms[n]])
        if ok:
            pop = _population_of(n)
            if pop not in inputs or deg > inputs[pop][1]:
                inputs[pop] = (dvals[syms[n]], deg)

    populations: dict[str, Population] = {}
    connections: list[Connection] = []

    def _add_pop(pop):
        populations.setdefault(pop, Population(pop, sign=_pop_sign(pop), label=pop))

    for tgt, (expr, _deg) in inputs.items():
        _add_pop(tgt)
        for src_sv in svs:
            coeff = sp.simplify(sp.diff(expr, syms[src_sv]))
            if coeff == 0:
                continue
            src = _population_of(src_sv)
            _add_pop(src)
            sign = "excitatory" if _coeff_sign(coeff) >= 0 else "inhibitory"
            connections.append(
                Connection(
                    src, tgt, sign=sign, scope="local", synapse=_edge_synapse(coeff, synapse_map, sign), recurrent=src == tgt
                )
            )
        for cs in coupling_symbols:
            coeff = sp.simplify(sp.diff(expr, syms[cs]))
            if coeff == 0:
                continue
            sign = "excitatory" if _coeff_sign(coeff) >= 0 else "inhibitory"
            syn = long_range_synapse or _edge_synapse(coeff, synapse_map, sign)
            connections.append(
                Connection(_population_of(svs[0]), tgt, sign=sign, scope="long_range", synapse=syn, recurrent=False)
            )
    return list(populations.values()), connections


# matplotlib drawing primitives (styling)
def _snap(p, center, r):
    v = np.asarray(p, float) - np.asarray(center[:2], float)
    L = np.hypot(*v) or 1.0
    return tuple(np.asarray(center[:2], float) + v / L * r)


def _head(ax, tip, direction, color, kind, hlen=10.0, hw=6.5):
    d = np.asarray(direction, float)
    d /= np.hypot(*d) or 1.0
    n = np.array([-d[1], d[0]])
    from matplotlib.patches import Circle, Polygon

    if kind == "dot":
        ax.add_patch(Circle(np.asarray(tip) - d * hw * 0.62, hw * 0.72, fc=color, ec=color, zorder=7))
    else:
        b = np.asarray(tip) - d * hlen
        ax.add_patch(Polygon([tip, b + n * hw * 0.5, b - n * hw * 0.5], closed=True, fc=color, ec=color, zorder=7))


def _bspline(ax, pts, color, lw, dashed):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    verts = [pts[0]]
    codes = [Path.MOVETO]
    for i in range(1, len(pts) - 2, 3):
        verts += [pts[i], pts[i + 1], pts[i + 2]]
        codes += [Path.CURVE4] * 3
    ls = (0, (6, 4)) if dashed else "solid"
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, edgecolor=color, lw=lw, linestyle=ls, zorder=3, capstyle="round"))


def _self_loop(ax, x, y, r, up, color, kind, height, width):
    """Small compact self-loop tangent to the node edge (little rope). ``height``/``width`` are the loop's extent beyond the edge in node-radius units."""
    s = 1 if up else -1
    a = np.deg2rad(26)
    p0 = (x - r * np.sin(a), y + s * r * np.cos(a))
    p3 = (x + r * np.sin(a), y + s * r * np.cos(a))
    apex = y + s * r * (1 + height)
    spread = width * r
    _bspline(ax, [p0, (x - spread, apex), (x + spread, apex), p3], color, 1.6, False)
    _head(ax, p3, (p3[0] - (x + spread), p3[1] - apex), color, kind)


def _neuron_glyph(ax, x, y, r, sign, label):
    """A population circle filled with small open 'neuron' dots (spiking-network glyph)."""
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), r, fc="white" if sign == "excitatory" else "#c9c9c9", ec="#141414", lw=2.0, zorder=5))
    pts = [(0, 0)]
    for rr, k, ph in ((0.5, 6, 0.0), (0.82, 9, 0.35)):
        for j in range(k):
            aa = ph + 2 * np.pi * j / k
            pts.append((rr * np.cos(aa), rr * np.sin(aa)))
    for px, py in pts:
        ax.add_patch(Circle((x + px * r * 0.72, y + py * r * 0.72), r * 0.09, fc="none", ec="#141414", lw=1.0, zorder=6))


def _plain_node(ax, x, y, r, sign, label, fontsize):
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), r, fc="white" if sign == "excitatory" else "#c9c9c9", ec="#141414", lw=1.9, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=6)


def _stretch_x(pos, splines, target):
    """Scale x about the centre so the circuit's width:height ≥ ``target`` (nodes keep radius
    ``r`` and stay round; only inter-node spacing + edge splines stretch). This makes a narrow stacked circuit span the same width as a wide mirror circuit, so panels align edge to edge."""
    if not target or not pos:
        return pos, splines
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    r0 = next(iter(pos.values()))[2]
    Sx, Sy = max(xs) - min(xs), max(ys) - min(ys)
    Wc, Hc = Sx + 5 * r0, Sy + 5 * r0
    if Sx <= 0 or Wc / Hc >= target:
        return pos, splines
    sx = 1 + (Hc * target - Wc) / Sx
    cx = (max(xs) + min(xs)) / 2
    pos = {n: (cx + (x - cx) * sx, y, r) for n, (x, y, r) in pos.items()}
    splines = [(t, h, [(cx + (px - cx) * sx, py) for px, py in pts]) for t, h, pts in splines]
    return pos, splines


# Schematic
class PopulationSchematic:
    """A renderable population circuit: populations + typed/signed connections over N areas."""

    def __init__(self, populations, connections, n_areas: int = 1):
        self.populations = list(populations)
        self.connections = list(connections)
        self.n_areas = int(n_areas)
        self._pop = {p.name: p for p in self.populations}

    @classmethod
    def from_dynamics(
        cls,
        dynamics,
        n_areas: int = 1,
        long_range_synapse: Optional[str] = "AMPA",
        synapse_map: Optional[dict] = None,
        coupling_symbols=("c_glob", "coupling", "gx", "local_coupling"),
    ):
        pops, conns = _derive_from_dynamics(dynamics, coupling_symbols, long_range_synapse, synapse_map)
        return cls(pops, conns, n_areas=n_areas)

    # -- graph structure --------------------------------------------------
    def _ranks(self, layout):
        N = self.n_areas
        exc = [p.name for p in self.populations if p.sign == "excitatory"]
        inh = [p.name for p in self.populations if p.sign == "inhibitory"]
        if layout == "mirror":
            row = [f"{p}0" for p in inh] + [f"{p}0" for p in exc]
            for a in range(1, N):
                row += [f"{p}{a}" for p in exc] + [f"{p}{a}" for p in inh]
            return [row]
        if layout == "row":
            return [[f"{p.name}{a}" for a in range(N) for p in self.populations]]
        return [[f"{p}{a}" for a in range(N) for p in grp] for grp in (exc, inh) if grp]

    def _nodes_edges(self, layout, hide=None):
        hide = hide or (lambda c: False)
        N = self.n_areas
        nodes = [f"{p.name}{a}" for a in range(N) for p in self.populations]
        edges, meta, loops = [], {}, {}
        for a in range(N):
            for c in self.connections:
                if c.scope != "local" or hide(c):
                    continue
                t, h = f"{c.source}{a}", f"{c.target}{a}"
                if c.recurrent:
                    loops[t] = (c.synapse, c.sign)
                else:
                    edges.append((t, h))
                    meta[(t, h)] = (c.synapse, c.sign, False)
        for a in range(N - 1):
            for c in self.connections:
                if c.scope != "long_range" or hide(c):
                    continue
                dashed = c.target != c.source  # E→I long-range (FFI) drawn dashed
                for lo, hi in ((a, a + 1), (a + 1, a)):
                    t, h = f"{c.source}{lo}", f"{c.target}{hi}"
                    edges.append((t, h))
                    meta[(t, h)] = (c.synapse, c.sign, dashed)
        return nodes, edges, meta, loops, self._ranks(layout)

    def _layout(self, layout, node_size, ranksep, nodesep, hide=None, ellipsis=False):
        """Run graphviz (dot); return {node:(x,y,r)}, [(t,h,spline_pts)], meta, loops.

        When ``ellipsis``, adds a left/right ``…`` placeholder to every rank as its OWN column (aligned across ranks), so the ellipsis reserves layout space instead of being painted
        over the edges, and the real areas pack toward the centre.
        """
        import json
        from graphviz import Digraph

        nodes, edges, meta, loops, ranks = self._nodes_edges(layout, hide)
        dots = []
        if ellipsis:
            ranks = [[f"dotL{ri}"] + list(row) + [f"dotR{ri}"] for ri, row in enumerate(ranks)]
            dots = [f"dotL{ri}" for ri in range(len(ranks))] + [f"dotR{ri}" for ri in range(len(ranks))]
        rank_of = {n: i for i, row in enumerate(ranks) for n in row}
        g = Digraph(engine="dot")
        g.attr(rankdir="TB", ranksep=str(ranksep), nodesep=str(nodesep), splines="spline")
        g.attr("node", shape="circle", fixedsize="true", width=str(node_size))
        for n in nodes:
            g.node(n)
        for n in dots:  # narrow ellipsis columns so the real circuit zooms in
            g.node(n, shape="none", label="", width=str(node_size * 0.45), height="0.1")
        for row in ranks:
            with g.subgraph() as s:
                s.attr(rank="same")
                for n in row:
                    s.node(n)
                # heavy invisible chain fixes the left→right order inside the rank
                for a, b in zip(row, row[1:]):
                    s.edge(a, b, style="invis", weight="50")
        for ri in range(len(ranks) - 1):  # align the … columns vertically
            g.edge(f"dotL{ri}", f"dotL{ri + 1}", style="invis")
            g.edge(f"dotR{ri}", f"dotR{ri + 1}", style="invis")
        # Only CROSS-rank edges may set ranks; an edge whose endpoints share a rank (all edges in a single-row `mirror` layout, or E→E long-range in `stacked`) must NOT constrain ranking, or it fights the rank=same and scrambles the node order.
        for t, h in edges:
            same_rank = rank_of.get(t) is not None and rank_of.get(t) == rank_of.get(h)
            g.edge(t, h, constraint="false" if (same_rank or t == h) else "true")
        data = json.loads(g.pipe(format="json").decode())
        objs = data["objects"]
        pos = {
            o["name"]: (
                float(o["pos"].split(",")[0]),
                float(o["pos"].split(",")[1]),
                float(o.get("width", node_size)) * 72 / 2,
            )
            for o in objs
            if "pos" in o
        }
        splines = []
        for e in data.get("edges", []):
            if e.get("style") == "invis":
                continue
            t, h = objs[e["tail"]]["name"], objs[e["head"]]["name"]
            pts = next(
                (
                    [tuple(p) for p in op["points"]]
                    for op in e.get("_draw_", [])
                    if op.get("op") in ("b", "B") and op.get("points")
                ),
                None,
            )
            if pts:
                splines.append((t, h, pts))
        return pos, splines, meta, loops

    # -- render -----------------------------------------------------------
    def plot(
        self,
        ax=None,
        layout="stacked",
        glyph="circle",
        node_size=0.7,
        self_loop=(0.62, 0.55),
        coupling_blob=False,
        ellipsis=None,
        legend=True,
        legend_labels=None,
        legend_loc="below",
        synapse_colors=None,
        fontsize=None,
        ranksep=1.15,
        nodesep=1.0,
        hide=None,
        long_range_targets=None,
        fill_width=2.6,
        fit="tight",
        figsize=(7.8, 4.0),
    ):
        """Render the schematic onto ``ax`` (created if ``None``). Returns the axis.

        ``layout`` ∈ {stacked, mirror, row}; ``glyph`` ∈ {circle, neurons} or a
        ``callable(ax, x, y, r, sign, label)``; ``self_loop`` = ``(height, width)`` or ``None``;
        ``coupling_blob`` shades a blob between the inner excitatory nodes (mirror layout);
        ``ellipsis`` draws ``…`` on both ends (defaults to True for stacked with N>1);
        ``legend_labels`` overrides the synapse legend text (e.g. ``{"AMPA": "AMPA, NMDA"}``).

        Hiding connections (declare in the figure caption when you do):
        ``hide`` — a predicate ``callable(Connection) -> bool`` to drop any connection; and
        ``long_range_targets`` — a convenience restricting long-range projections to those target populations (e.g. ``["E"]`` = E→E only, no FFI, for the spiking Fig 1a view).

        Width control (two independent knobs):
        ``fill_width`` (float, default 2.6) stretches the circuit horizontally to at least this width:height aspect (how wide the circuit spreads; ``None``/0 disables); and
        ``fit`` controls how the axes box maps to that circuit — ``"tight"`` shrinks the box to the circuit (minimal whitespace, panels may differ in width) · ``"width"`` keeps the
        box's given width and pads the data limits (panels in a mosaic column stay the SAME width) · a ``float`` forces that exact width:height box aspect. Circles stay round in all.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        if fontsize is None:  # inherit the figure's metadata font_size
            fontsize = plt.rcParams["font.size"]
        colors = {**SYNAPSE_COLORS, **(synapse_colors or {})}
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        user_hide = hide or (lambda c: False)

        def _hide(c):
            return bool(user_hide(c)) or (
                long_range_targets is not None and c.scope == "long_range" and c.target not in long_range_targets
            )

        if ellipsis is None:
            ellipsis = layout == "stacked" and self.n_areas > 1
        pos, splines, meta, loops = self._layout(layout, node_size, ranksep, nodesep, _hide, ellipsis)
        pos, splines = _stretch_x(pos, splines, fill_width)
        real = {n: v for n, v in pos.items() if not n.startswith("dot")}
        r0 = next(iter(real.values()))[2]

        # edges (snapped to node edges) — behind nodes
        for t, h, pts in splines:
            syn, sign, dashed = meta[(t, h)]
            pts = list(pts)
            pts[0] = _snap(pts[0], pos[t], pos[t][2])
            pts[-1] = _snap(pts[-1], pos[h], pos[h][2])
            _bspline(ax, pts, colors.get(syn, "#141414"), 1.6, dashed)
            _head(
                ax,
                pts[-1],
                (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]),
                colors.get(syn, "#141414"),
                "dot" if sign == "inhibitory" else "tri",
            )
        # self-loops (excitatory up, inhibitory down in stacked; all up otherwise)
        if self_loop:
            hgt, wid = self_loop
            for n, (syn, sign) in loops.items():
                x, y, r = pos[n]
                up = True if layout != "stacked" else (self._pop[n[:-1]].sign == "excitatory")
                _self_loop(ax, x, y, r, up, colors.get(syn, "#141414"), "dot" if sign == "inhibitory" else "tri", hgt, wid)
        # coupling blob between inner excitatory nodes (paired/mirror areas)
        if coupling_blob and self.n_areas >= 2:
            exc = [p.name for p in self.populations if p.sign == "excitatory"]
            if exc:
                a0, a1 = pos[f"{exc[0]}0"], pos[f"{exc[0]}1"]
                ax.add_patch(
                    Ellipse(
                        ((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2),
                        abs(a1[0] - a0[0]) * 1.15,
                        a0[2] * 2.3,
                        fc="#cfe0f2",
                        ec="none",
                        alpha=0.7,
                        zorder=0,
                    )
                )
        # population nodes (skip the … placeholder columns)
        for n, (x, y, r) in real.items():
            p = self._pop[n[:-1]]
            if callable(glyph):
                glyph(ax, x, y, r, p.sign, p.label)
            elif glyph == "neurons":
                _neuron_glyph(ax, x, y, r, p.sign, p.label)
            else:
                _plain_node(ax, x, y, r, p.sign, p.label, fontsize)
        # ellipsis: draw … at the centre of its reserved left/right columns
        cy = np.mean([y for _, y, _ in real.values()])
        for side in ("dotL", "dotR"):
            col = [v[0] for n, v in pos.items() if n.startswith(side)]
            if col:  # compact "…", scales with the base font
                ax.text(np.mean(col), cy, "…", fontsize=fontsize * 1.3, ha="center", va="center")

        allx = [x for x, _, _ in pos.values()]
        ally = [y for _, y, _ in pos.values()]
        x0, x1 = min(allx) - 1.0 * r0, max(allx) + 1.0 * r0
        y0, y1 = min(ally) - 2.0 * r0, max(ally) + 2.0 * r0  # just enough for the self-loops
        if isinstance(fit, (int, float)) and not isinstance(fit, bool):
            # force an exact box width:height aspect — pad the shorter data dimension to match
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = x1 - x0, y1 - y0
            if w / h < fit:
                w = h * fit
            else:
                h = w / fit
            x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        # "tight": box shrinks to the content (no whitespace, panels may differ in width).
        # "width": box keeps its given width, data limits pad (mosaic panels stay equal width).
        adjustable = "datalim" if fit == "width" else "box"
        ax.set_aspect("equal", adjustable=adjustable)
        ax.axis("off")
        if legend:
            self._legend(ax, colors, legend_labels, fontsize, legend_loc)
        return ax

    def _legend(self, ax, colors, legend_labels, fontsize, loc="below"):
        from matplotlib.lines import Line2D

        seen, items = set(), []
        for c in self.connections:
            if c.synapse and c.synapse not in seen:
                seen.add(c.synapse)
                lbl = (legend_labels or {}).get(c.synapse, c.synapse)
                kind = "o" if c.sign == "inhibitory" else ">"
                items.append((lbl, colors.get(c.synapse, "#141414"), kind, c.synapse))
        # one legend entry per synapse; terminal reflects the dominant sign of that synapse
        handles = [
            Line2D([0], [0], color=col, lw=2.4, marker=mk, markersize=8, markerfacecolor=col, markeredgecolor=col, label=lbl)
            for (lbl, col, mk, _s) in items
        ]
        if handles:
            # "below": legend TOP just under the axis (grows down); "top": legend BOTTOM just above the axis (grows up — use when a panel sits directly below the schematic, so the legend does not overlap it). Layout engine reserves the space either way.
            anchor, mloc = ((0.5, 1.02), "lower center") if loc == "top" else ((0.5, -0.02), "upper center")
            leg = ax.legend(
                handles=handles,
                loc=mloc,
                ncol=len(handles),
                frameon=False,
                fontsize=fontsize - 2,
                handlelength=1.8,
                columnspacing=1.6,
                bbox_to_anchor=anchor,
            )
            leg.set_clip_on(False)


def plot_population_schematic(dynamics=None, ax=None, n_areas=1, long_range_synapse="AMPA", synapse_map=None, **plot_kwargs):
    """Convenience: derive a schematic from ``dynamics`` and render it (see ``PopulationSchematic``)."""
    schem = PopulationSchematic.from_dynamics(
        dynamics, n_areas=n_areas, long_range_synapse=long_range_synapse, synapse_map=synapse_map
    )
    return schem.plot(ax=ax, **plot_kwargs)
