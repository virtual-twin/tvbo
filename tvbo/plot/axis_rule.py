# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""The axis rule a theme declares, applied to a drawn figure.

A theme states who places an axis's ticks (``ticks``), where the axis ends relative to them (``axis_ends``), how far the two scale-bearing spines stand off the plot (``spine_offset``) and whether they stop at their outer ticks (``spine_trim``). :func:`place` and :func:`frame` carry those statements out over every axes of a figure, and both do nothing under an empty rule, which is what a figure whose study declares no rule gets: its axes stay exactly as its panels drew them.

An axis cannot always have round ticks, a tick at each end and no empty space at once. Which of the three gives way is the rule's to say and never this module's: ``axis_ends: data`` keeps the limits and lets the outer ticks sit inside them, ``ticks`` widens the axis to a tick at both ends, and ``near`` widens an end only where the next tick lies within ``end_tick_reach`` of the data.

Whatever a panel or a drawer chose stands. An axis with fixed ticks keeps them and its limits, one with declared limits or an image keeps its limits, and a log, categorical or deliberately bare axis is not touched at all.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9
DEFAULT_STEPS = (1.0, 2.0, 5.0)
DEFAULT_MAX_TICKS = 4
DEFAULT_REACH = 0.2

TICKS = ("native", "round", "declared")
ENDS = ("data", "ticks", "near")

SLOTS = (
    "ticks",
    "max_ticks",
    "tick_steps",
    "axis_ends",
    "end_tick_reach",
    "spine_offset",
    "spine_trim",
    "align_labels",
    "colorbar_ends",
)
"""The Theme slots that make up the axis rule: the ones a figure's ``auto_format: false`` switches off together."""


def _round_steps(span: float, max_ticks: int, steps) -> list[float]:
    """Round step sizes from fine to coarse, starting a decade below the finest that could fit *max_ticks* ticks on *span*."""
    multiples = sorted({float(s) for s in steps})
    if not multiples or multiples[0] < 1.0 or multiples[-1] >= 10.0:
        raise ValueError(
            f"tick_steps {list(steps)!r}: each step multiple must lie in [1, 10); it is read times any power of ten"
        )
    unit = 10.0 ** (np.floor(np.log10(span / (max_ticks - 1))) - 1)
    return [unit * m * 10.0**e for e in range(0, 14) for m in multiples]


def round_ticks(a, b, lo=None, hi=None, *, max_ticks=DEFAULT_MAX_TICKS, steps=DEFAULT_STEPS, ends="near", reach=DEFAULT_REACH):
    """Round ticks for data on [*a*, *b*] drawn in the view [*lo*, *hi*], and the limits to draw them in.

    The ticks are the multiples of the finest round step that puts at most *max_ticks* of them on the axis. *ends* decides the limits. ``data`` keeps the view and places the ticks inside it. ``ticks`` takes the multiple at or below *a* and the one at or above *b*, so both ends carry a tick however far that widens the axis. ``near`` lets an end take that multiple only where it widens the axis by at most *reach* of the data's span and keeps the view's end otherwise, preferring the placement with the most ticks and then the most ends on one, and widening past *reach* only where the axis would otherwise carry fewer than three ticks.

    Where no step fits the budget with two ticks or more, the step that exceeds it by least is taken: one tick too many states the scale, two non-round ticks at the data's own ends do not.
    """
    if ends not in ENDS:
        raise ValueError(f"axis_ends {ends!r}: expected one of {', '.join(ENDS)}")
    a, b = float(a), float(b)
    lo, hi = (a if lo is None else min(float(lo), a)), (b if hi is None else max(float(hi), b))
    max_ticks, span = max(2, int(max_ticks)), b - a
    if not span > 0 or not np.isfinite(span):
        raise ValueError(f"round ticks need an extent, got [{a}, {b}]")

    def run(step, low, high):
        first, last = np.ceil(low / step - EPS), np.floor(high / step + EPS)
        return int(last - first) + 1, first, last

    def ticks_of(step, first, last):
        decimals = max(0, int(-np.floor(np.log10(step))) + 2)
        return np.round(np.arange(first, last + 1) * step, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0

    def placed(step, limit):
        """Every way to end the axis at *step*, as (ticks, ends on a tick, width, first, last, low, high), an end moving outward by at most *limit* of the span."""
        below, above = np.floor(a / step + EPS) * step, np.ceil(b / step - EPS) * step
        lows = [(lo, 0)] if ends != "ticks" else []
        highs = [(hi, 0)] if ends != "ticks" else []
        if ends != "data" and a - below <= limit * span:
            lows.append((below, 1))
        if ends != "data" and above - b <= limit * span:
            highs.append((above, 1))
        return [
            (*run(step, low, high), on_low + on_high, high - low, low, high) for low, on_low in lows for high, on_high in highs
        ]

    spill = None
    for step in _round_steps(span, max_ticks, steps):
        within = placed(step, np.inf if ends == "ticks" else reach)
        fitting = [o for o in within if 2 <= o[0] <= max_ticks]
        best = max(fitting, key=lambda o: (o[0], o[3])) if fitting else None
        if ends == "near" and (best is None or best[0] < 3):
            wider = [
                o for o in placed(step, np.inf) if 3 <= o[0] <= max_ticks
            ]  # two ticks are too few to read a value between
            best = min(wider, key=lambda o: (o[4], -o[0])) if wider else best
        if best is not None:
            _, first, last, _, _, low, high = best
            return ticks_of(step, first, last), (float(low), float(high))
        over = [o for o in within if o[0] > max_ticks]
        if over:
            spill = (step, min(over, key=lambda o: (o[0], -o[3])))
    if spill is None:
        return np.array([a, b], dtype=float), (lo, hi)
    step, (_, first, last, _, _, low, high) = spill
    return ticks_of(step, first, last), (float(low), float(high))


def figure_axes(fig) -> list:
    """Every axes of *fig*: its panels, their twins, and the insets a composite panel opened inside them."""
    pending, seen = list(fig.axes), []
    while pending:
        ax = pending.pop()
        seen.append(ax)
        pending.extend(getattr(ax, "child_axes", []))
    return seen


def ruled(ax) -> bool:
    """Whether *ax* is a drawn two-dimensional axes carrying scales, which is all an axis rule speaks about."""
    return ax.name == "rectilinear" and ax.axison and ax.get_label() != "<colorbar>"


def boxed(ax) -> bool:
    """Whether *ax* draws a closed frame, which has no free spine to stand off or to trim."""
    return all(ax.spines[side].get_visible() for side in ("top", "right", "left", "bottom"))


def mend_gridspecs(fig) -> None:
    """Give every zero row or column ratio of *fig*'s gridspecs a width when *fig* lays out with the constrained or compressed engine, the two that divide by those ratios.

    A colour bar on an equal-aspect image inserts zero-height pad rows into its gridspec, and matplotlib's constrained and compressed engines then raise ``ZeroDivisionError`` at draw time. Under any other engine those pad rows are what keeps the bar beside its image at full height, so they stay as drawn.
    """
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    if not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        return
    for ax in fig.axes:
        spec = ax.get_gridspec()
        if spec is None:
            continue
        for get, put in ((spec.get_height_ratios, spec.set_height_ratios), (spec.get_width_ratios, spec.set_width_ratios)):
            ratios = get()
            if ratios is not None and 0 in ratios:
                put([r or 1.0 for r in ratios])


def _scaled(ax, name: str) -> bool:
    """Whether the *name* axis of *ax* is a linear scale that draws tick marks or tick numbers."""
    axis = _axis(ax, name)
    if axis.get_scale() != "linear" or not axis.get_visible():
        return False
    drawn = axis.get_tick_params(which="major")
    sides = ("bottom", "top") if name == "x" else ("left", "right")
    return any(drawn.get(key, False) for side in sides for key in (side, f"label{side}"))


def _autoscaled(ax, name: str) -> bool:
    """Whether nothing has fixed the limits of the *name* axis of *ax*: no declared limits, and no image or mesh whose extent is its frame."""
    from matplotlib.collections import QuadMesh

    framed = bool(ax.images) or any(isinstance(c, QuadMesh) for c in ax.collections)
    return (ax.get_autoscalex_on() if name == "x" else ax.get_autoscaley_on()) and not framed


def _extent(ax, name: str):
    """The extent of what *ax* draws along *name*, without the margin autoscaling adds, or None when it has drawn nothing there."""
    lo, hi = ax.dataLim.intervalx if name == "x" else ax.dataLim.intervaly
    return (float(lo), float(hi)) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def _view(ax, name: str) -> tuple[float, float]:
    """The limits of the *name* axis of *ax*, low end first whichever way it runs."""
    return tuple(sorted(ax.get_xlim() if name == "x" else ax.get_ylim()))


def _axis(ax, name: str):
    """The x or the y axis of *ax*, by name."""
    return ax.xaxis if name == "x" else ax.yaxis


def _placeable(ax, name: str) -> bool:
    """Whether the *name* axis of *ax* is a drawn linear scale whose ticks the backend's own locator still places, so nobody has chosen them."""
    from matplotlib.ticker import MaxNLocator

    return _scaled(ax, name) and isinstance(_axis(ax, name).get_major_locator(), MaxNLocator)


def _members(ax, name: str, groups: dict) -> list:
    """The axes that must end up on one scale with *ax* along *name*: the ones the backend shares it with and the ones a declared shared scale groups it with."""
    shared = (ax.get_shared_x_axes() if name == "x" else ax.get_shared_y_axes()).get_siblings(ax)
    out = list(shared)
    for group in groups.get(name, ()):
        if any(member in out for member in group):
            out.extend(member for member in group if member not in out)
    return out


def _native_ends(locator, extent, view, ends: str, reach: float) -> tuple[float, float]:
    """The limits that put an axis's ends on the ticks its own locator places, each end moving outward by the rule *ends* allows."""
    (a, b), (lo, hi) = extent, view
    span = b - a
    placed = np.asarray(locator.tick_values(a, b), dtype=float)
    below, above = placed[placed <= a + EPS * span], placed[placed >= b - EPS * span]
    low = float(below.max()) if below.size and (ends == "ticks" or a - below.max() <= reach * span) else min(lo, a)
    high = float(above.min()) if above.size and (ends == "ticks" or above.min() - b <= reach * span) else max(hi, b)
    return low, high


def place(fig, rule: dict, *, skip=(), shared=None, names=None) -> None:
    """Place the ticks and the ends of every axis of *fig* that nobody placed, by the theme's *rule*.

    ``ticks: round`` re-places an automatically located axis on round values (:func:`round_ticks`); ``native`` leaves the backend's locator in charge; ``declared`` refuses a figure in which any scale still carries ticks no panel gave, naming every one of them at once. ``axis_ends`` then moves the limits of an axis that is still autoscaled. Axes that share a scale, by the backend or by the figure's ``share_x``/``share_y`` (*shared*, ``{"x": [[axes, ...]], "y": [...]}``), are placed from their common extent so they stay on one scale. *skip* holds the ``id`` of every axes the rule leaves alone, and *names* maps an axes' ``id`` to the panel key an error should call it by.
    """
    from matplotlib.ticker import AutoLocator

    ticks, ends = rule.get("ticks") or "native", rule.get("axis_ends") or "data"
    if ticks not in TICKS:
        raise ValueError(f"theme ticks {ticks!r}: expected one of {', '.join(TICKS)}")
    if ends not in ENDS:
        raise ValueError(f"theme axis_ends {ends!r}: expected one of {', '.join(ENDS)}")
    if ticks == "native" and ends == "data":
        return
    budget = {
        "max_ticks": rule.get("max_ticks") or DEFAULT_MAX_TICKS,
        "steps": rule.get("tick_steps") or DEFAULT_STEPS,
        "reach": DEFAULT_REACH if rule.get("end_tick_reach") is None else float(rule["end_tick_reach"]),
    }
    groups, names, unchosen, done = shared or {}, names or {}, {}, set()
    owner = _owners(fig)
    for ax in figure_axes(fig):
        if id(ax) in skip or not ruled(ax):
            continue
        for name in ("x", "y"):
            if (id(ax), name) in done or not _placeable(ax, name):
                continue  # fixed, bare, categorical or logarithmic: someone chose, and it stands
            if ticks == "declared":
                unchosen.setdefault(_name(ax, owner, names), []).append(name)
                continue
            members = [m for m in _members(ax, name, groups) if id(m) not in skip and ruled(m)] or [ax]
            done.update((id(m), name) for m in members)
            movable = ends != "data" and all(_autoscaled(m, name) for m in members)
            extents = [e for e in (_extent(m, name) for m in members) if e]
            views = [_view(m, name) for m in members]
            view = (min(v[0] for v in views), max(v[1] for v in views))
            extent = (min(e[0] for e in extents), max(e[1] for e in extents)) if extents and movable else view
            if not extent[1] > extent[0]:
                continue
            automatic = [
                m for m in members if _placeable(m, name) and isinstance(_axis(m, name).get_major_locator(), AutoLocator)
            ]
            placed, limits = None, view
            if ticks == "round" and automatic:
                placed, limits = round_ticks(*extent, *view, **budget, ends=ends if movable else "data")
            elif movable:
                limits = _native_ends(_axis(ax, name).get_major_locator(), extent, view, ends, budget["reach"])
            for member in members if movable else ():
                (member.set_xlim if name == "x" else member.set_ylim)(
                    *(limits[::-1] if _axis(member, name).get_inverted() else limits)
                )
            for member in automatic if placed is not None else ():
                _axis(member, name).set_ticks(placed.tolist())
    if unchosen:
        listed = "; ".join(f"{panel}: {', '.join(axes)}" for panel, axes in sorted(unchosen.items()))
        raise ValueError(
            f"theme `ticks: declared`: these axes carry ticks no panel declared — {listed}. "
            "Give each its `xticks`/`yticks`, or state `ticks: round` or `ticks: native` in the theme."
        )


def frame(fig, rule: dict, *, skip=()) -> None:
    """Stand the two scale-bearing spines of every open-framed axes of *fig* off the plot and end them on their outer ticks, as the theme's *rule* states, and align the figure's axis labels where it says so.

    Run once the limits and ticks are final, since a trimmed spine describes the ticks standing at that moment. A boxed frame is skipped: it has no free spine, and moving two sides of a closed box opens it.
    """
    offset, trim = rule.get("spine_offset"), rule.get("spine_trim")
    if offset is not None or trim:
        for ax in figure_axes(fig):
            if id(ax) in skip or not ruled(ax) or boxed(ax):
                continue
            if offset is not None:
                ax.spines["left"].set_position(("axes", -float(offset)))
                ax.spines["bottom"].set_position(("axes", -float(offset)))
            if trim:
                trim_spine(ax, "x")
                trim_spine(ax, "y")
    if rule.get("align_labels"):
        fig.align_labels()


def trim_spine(ax, name: str) -> None:
    """End one scale-bearing spine of *ax* on the outermost ticks now inside its limits. An axis with fewer than two ticks has nothing to end on and keeps its whole spine."""
    spine = ax.spines["bottom" if name == "x" else "left"]
    lo, hi = _view(ax, name)
    tol = EPS * max(hi - lo, 1.0)
    ticks = [float(t) for t in (ax.get_xticks() if name == "x" else ax.get_yticks()) if lo - tol <= float(t) <= hi + tol]
    if len(ticks) > 1:
        spine.set_bounds(min(ticks), max(ticks))


def _owners(fig) -> dict:
    """``{id(inset): the figure-level axes it was opened in}``, so an inset is named by the panel it belongs to."""
    out: dict = {}
    for top in fig.axes:
        pending = list(getattr(top, "child_axes", []))
        while pending:
            child = pending.pop()
            out[id(child)] = top
            pending.extend(getattr(child, "child_axes", []))
    return out


def _name(ax, owner: dict, names: dict) -> str:
    """What an error calls *ax*: its panel's key, the key of the panel an inset or twin belongs to, or its own label."""
    top = owner.get(id(ax), ax)
    if id(top) in names:
        return f"panel {names[id(top)]}" + ("" if top is ax else " (inset)")
    spec = top.get_subplotspec()
    for other in top.figure.axes:
        if id(other) in names and other.get_subplotspec() is spec:
            return f"panel {names[id(other)]} (twin)"
    return f"axes {top.get_label()!r}"
