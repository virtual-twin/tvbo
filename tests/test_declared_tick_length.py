"""A panel that declares `tick_length: 0` draws no tick marks.

The emitted `_apply_axopts` kept a declared tick length only when it was truthy, so zero, the one value that removes the marks, fell back to the theme's length. Koller2024's Fig. 4a colour bar declares it because its published bar carries no ticks, and the theme's inward ticks crossed the bar and ran into the matrix beside it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot


def _apply_axopts(base):
    """The axis-directive helper a one-panel figure's emitted script defines, bound to that script's namespace."""
    layer = P.Layer(used=P.DataRef(iri="tvbo:exp/none/exp-1", output="x"))
    figure = P.Figure(name="fig", layout="a", panels={"a": P.Panel(panel_key="a", kind="cartesian", layers=[layer])})
    ns: dict = {}
    exec(compile(bsplot.render_code(figure, base, "out.png"), "<figure>", "exec"), ns)
    return ns["_apply_axopts"]


def test_a_declared_zero_tick_length_removes_the_marks(tmp_path):
    fig, ax = plt.subplots()
    ax.tick_params(length=4.0)
    _apply_axopts(tmp_path)(ax, {"tick_length": 0.0})
    assert [t.tick1line.get_markersize() for t in ax.yaxis.get_major_ticks()] == [0.0] * len(ax.get_yticks())
    plt.close(fig)


def test_an_undeclared_tick_length_leaves_the_theme_s(tmp_path):
    fig, ax = plt.subplots()
    ax.tick_params(length=4.0)
    _apply_axopts(tmp_path)(ax, {})
    assert {t.tick1line.get_markersize() for t in ax.yaxis.get_major_ticks()} == {4.0}
    plt.close(fig)
