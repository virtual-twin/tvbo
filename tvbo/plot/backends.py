# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""What each simulation backend can do, as a table or a figure, read from the table the workflow planner itself consults.

Rows are the backends of :data:`tvbo.run.backends.BACKENDS`, in registration order. Columns are grouped: the integration problems and study tasks a backend performs (``tvbo:SimulationTask`` individuals), the sweep axes it batches inside one job rather than fanning them out as separate jobs, and capability classes, each marked when the backend declares any of its members, so the comparison reads "compiled execution" rather than one toolbox's mechanism. A capability outside every class (how a backend is built rather than what it can do) is not shown. No mark is written by hand: the output is what the installed tvbo can do.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

TASK_GROUPS = {
    "Integration": {
        "ODEIntegration": "ODE",
        "SDEIntegration": "SDE",
        "DDEIntegration": "DDE",
        "SDDEIntegration": "SDDE",
        "EventDrivenIntegration": "event driven",
    },
    "Study task": {
        "ParameterExploration": "exploration",
        "GradientBasedOptimization": "optimization",
        "NumericalContinuation": "continuation",
        "BifurcationAnalysis": "bifurcations",
    },
}
CAPABILITIES = {
    "Execution": {"compiled": ("JITCompilation", "JuliaJIT"), "GPU": ("GPUSupport",), "gradients": ("Autodiff",)},
    "Solvers": {
        "delays": ("DelayHistoryBuffer",),
        "stochastic": ("StochasticSolver", "VectorizedRNG"),
        "stiff": ("StiffSolver", "DiffEqIntegrators"),
        "continuation": ("ContinuationSolver",),
        "spiking neurons": ("SpikingSimulation",),
    },
}
SWEEP_GROUP = "Batched axes"


def columns() -> tuple[list[str], list[tuple[str, list[tuple[str, list[bool]]]]]]:
    """The backend labels, and the matrix as ``(group, [(column, one flag per backend)])``."""
    from tvbo.run.backends import AXIS_KINDS, list_backends

    backends = list_backends()
    groups = [
        (title, [(label, [task in b.tasks for b in backends]) for task, label in names.items()])
        for title, names in TASK_GROUPS.items()
    ]
    axes = [a for a in AXIS_KINDS if any(a in b.vectorize_axes for b in backends)]
    groups.append((SWEEP_GROUP, [(a.replace("_", " "), [a in b.vectorize_axes for b in backends]) for a in axes]))
    groups += [
        (title, [(name, [any(m in b.capabilities for m in members) for b in backends]) for name, members in classes.items()])
        for title, classes in CAPABILITIES.items()
    ]
    return [b.label for b in backends], groups


def markdown() -> str:
    """The matrix as a GitHub-flavoured Markdown table, one row per backend, ``●`` where the backend can."""
    labels, groups = columns()
    heads = [f"{name} <sub>{title}</sub>" if i == 0 else name for title, cols in groups for i, (name, _) in enumerate(cols)]
    rows = ["| Backend | " + " | ".join(heads) + " |", "|---|" + "|".join(":-:" for _ in heads) + "|"]
    for k, label in enumerate(labels):
        rows.append(f"| {label} | " + " | ".join("●" if flags[k] else "" for _, cols in groups for _, flags in cols) + " |")
    return "\n".join(rows) + "\n"


def svg(theme: str | Path | Mapping = "tvbo:themes/default", *, hue: str = "palette.2") -> str:
    """The matrix as a standalone SVG: backends down the side, grouped columns along the top, a filled dot where a backend can and a hairline ring where it cannot.

    Args:
        theme: The colours, as :func:`tvbo.plot.lifecycle.wheel` takes them.
        hue: The palette slot the dots are drawn in; ``palette.2`` is the Generate stage's.
    """
    from tvbo.plot.lifecycle import FONT, _document, _mix, _theme
    from tvbo.plot.palette import _hue

    colours = _theme(theme)
    page, ink = colours["background"], colours["ink"]
    dot = _hue(hue, colours["palette"])
    ring = _mix(ink, page, 0.75)
    labels, groups = columns()
    step, gap, row_h, label_w, head_h, group_h = 30.0, 14.0, 30.0, 170.0, 130.0, 26.0
    xs, x = [], label_w
    spans = []
    for _, cols in groups:
        start = x
        for _ in cols:
            xs.append(x + step / 2)
            x += step
        spans.append((start, x))
        x += gap
    names = [name for _, cols in groups for name, _ in cols]
    reach = max(cx + 4 + 0.6 * 13 * len(name) * math.cos(math.radians(55)) for cx, name in zip(xs, names, strict=True))
    width, top = max(x - gap, reach) + 16, group_h + head_h
    height = top + row_h * len(labels) + 12
    out = [
        f"<style>text{{font-family:{FONT};fill:{ink}}}.g{{font-size:14px;font-weight:700}}.c{{font-size:13px}}.r{{font-size:15px;font-weight:600}}</style>",
    ]
    for (title, _), (a, b) in zip(groups, spans, strict=True):
        out.append(f'<text class="g" x="{(a + b) / 2:.1f}" y="18" text-anchor="middle">{title}</text>')
        out.append(f'<path d="M {a + 3:.1f} 26 H {b - 3:.1f}" stroke="{ring}" stroke-width="1.5"/>')
    for cx, name in zip(xs, names, strict=True):
        out.append(f'<text class="c" transform="translate({cx + 4:.1f} {top - 8:.1f}) rotate(-55)">{name}</text>')
    flags = [f for _, cols in groups for _, f in cols]
    for k, label in enumerate(labels):
        cy = top + row_h * k + row_h / 2
        out.append(f'<text class="r" x="{label_w - 14:.1f}" y="{cy + 5:.1f}" text-anchor="end">{label}</text>')
        for cx, f in zip(xs, flags, strict=True):
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.5" fill="{dot}"/>'
                if f[k]
                else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="none" stroke="{ring}" stroke-width="1.2"/>'
            )
    return _document(out, (0.0, 0.0, width, height), "Backend capability matrix", page)
