"""A figure refuses `reconcile: by_label` on its bindings, naming the reference.

A figure draws what a binding reads in its container's order and never read `DataRef.reconcile`, so `by_label` on a heatmap's, a line's, an annotation's or a grid cell's `used:`, or on a surface's mesh binding, was accepted and ignored, and an array stored on another node order was drawn as if aligned. The figure is refused when its script is generated, before anything is opened or drawn.

Self-contained: no container is needed, since the refusal comes before any is opened.
"""

from __future__ import annotations

import pytest

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

BY_LABEL = P.DataRef(analysis="sim", output="fc", reconcile="by_label")


def _render_code(tmp_path, **panel):
    """The script of a one-panel figure."""
    figure = P.Figure(name="fc", layout="a", panel_numbers=False, panels={"a": P.Panel(panel_key="a", **panel)})
    return bsplot.render_code(figure, str(tmp_path))


@pytest.mark.parametrize(
    "panel",
    [
        {"kind": "heatmap", "layers": [P.Layer(used=BY_LABEL, encoding=P.Encoding(x="node_j", y="node_i"))]},
        {"kind": "cartesian", "layers": [P.Layer(used=BY_LABEL, encoding=P.Encoding(x="node", y="fc"))]},
        {"kind": "cartesian", "annotations": [P.Annotation(text="r = {:.2f}", used=BY_LABEL)]},
        {"kind": "grid", "cell": {"kind": "heatmap"}, "layers": [P.Layer(used=BY_LABEL)]},
    ],
    ids=["heatmap-layer", "line-layer", "annotation", "grid-cell"],
)
def test_by_label_on_a_figure_binding_is_refused_by_name(tmp_path, panel):
    """Drawn in its container's node order as if aligned, where it is now refused before anything is drawn (G209)."""
    with pytest.raises(ValueError, match=r"`reconcile: by_label` on the reference to analysis 'sim', output 'fc'"):
        _render_code(tmp_path, **panel)


def test_by_label_on_a_surfaces_mesh_binding_is_refused_by_name(tmp_path):
    """The mesh binding is read apart from the layers, and refused a stated `output`, `sel` or `transform` but not `reconcile` (G209)."""
    with pytest.raises(ValueError, match=r"`reconcile: by_label` on the reference to analysis 'mesh'"):
        _render_code(tmp_path, kind="surface", surface={"used": {"analysis": "mesh", "reconcile": "by_label"}})
