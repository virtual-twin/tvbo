"""A figure's colour scale, and the axis directives that reach it.

A ``kind: colorbar`` slot is a panel whose drawing is an inset bar. Two things live in that slot and they are not the same axes: the bar, which is what a declared frame (ticks, formats, label padding) is about, and the slot behind it, which is where the panel's letter belongs and what a shared scale groups. Keying is the other half: a centred heatmap truncates its map to the half-range the field reaches, so a bar built from the untruncated map puts the neutral colour somewhere the mesh never has it and every value the reader looks up is wrong.

Self-contained: a synthetic container in ``tmp_path``, rendered under Agg.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

FIELD = np.linspace(-0.5, 0.3, 36).reshape(6, 6)
"""A field whose range straddles zero asymmetrically, so a centred map is visibly truncated."""


@pytest.fixture
def study(tmp_path):
    """A study directory holding one container with a single 2-D output.

    The container is written where the layout record says an analysis result lives, so the layer's ``iri`` resolves through the same path the run itself writes.
    """
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    ds = xr.Dataset({"corr": (("node", "node2"), FIELD)}, coords={"node": np.arange(6), "node2": np.arange(6)})
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    ds.to_netcdf(analysis_container_path(results, "synth"), engine="h5netcdf")
    return tmp_path


def _figure(*, center=None, bar=None, panel_slots=None):
    """A heatmap beside a standalone scale, both bound to the same layer and centred alike."""
    ref = P.DataRef(iri="tvbo:synth", output="corr")
    style_opts = {} if center is None else {"center": P.Argument(name="center", value=center)}
    bar = {"colormap": "RdBu_r", **({} if center is None else {"center": center}), **(bar or {})}
    return P.Figure(
        name="scale",
        layout="ab",
        panel_numbers=True,
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="heatmap",
                layers=[
                    P.Layer(
                        used=ref,
                        encoding=P.Encoding(x="node", y="node2"),
                        style=P.Style(colormap="RdBu_r", opts=style_opts),
                    )
                ],
                colorbar={"show": False},
                **(panel_slots or {}),
            ),
            "b": P.Panel(
                panel_key="b",
                kind="colorbar",
                layers=[P.Layer(used=ref, encoding=P.Encoding(x="node", y="node2"))],
                colorbar=bar,
            ),
        },
    )


def _slot_and_bar(fig):
    """The scale panel's slot axes and the bar drawn inside it."""
    slots = [ax for ax in fig.axes if ax.child_axes]
    assert len(slots) == 1, f"expected one slot holding an inset, got {len(slots)}"
    return slots[0], slots[0].child_axes[0]


def _colormap(ax):
    """The colormap the mesh drawn on *ax* was painted with (a bar's own mesh included)."""
    return next(c for c in ax.get_children() if type(c).__name__ == "QuadMesh").get_cmap()


def test_the_panel_letter_lands_on_the_slot_not_on_the_bar(study):
    """The bar occupies a fraction of its slot, so a letter placed on it sits inside the figure rather than at the panel's corner."""
    fig = bsplot.render(_figure(), str(study), str(study / "out.png"))
    slot, bar = _slot_and_bar(fig)
    assert [t.get_text() for t in slot.texts] == ["b"]
    assert [t.get_text() for t in bar.texts] == []


def test_a_declared_frame_lands_on_the_bar(study):
    """Ticks are a property of the scale; the slot behind it carries no axis at all."""
    fig = bsplot.render(_figure(bar={"ticks": [-0.4, 0.0, 0.2]}), str(study), str(study / "out.png"))
    _slot, bar = _slot_and_bar(fig)
    np.testing.assert_allclose(bar.get_yticks(), [-0.4, 0.0, 0.2])


def test_the_standalone_bar_is_the_scale_the_mesh_was_drawn_on(study):
    """``center`` truncates the map to the half-range the data reaches, and the bar has to be truncated the same way or it keys a mesh it does not describe."""
    fig = bsplot.render(_figure(center=0.0), str(study), str(study / "out.png"))
    _slot, bar = _slot_and_bar(fig)
    sample = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(_colormap(bar)(sample), _colormap(fig.axes[0])(sample))


def test_an_uncentred_scale_spans_the_whole_map(study):
    """Without ``center`` the map is untouched, so the neutral colour sits at the middle of the bar."""
    fig = bsplot.render(_figure(), str(study), str(study / "out.png"))
    _slot, bar = _slot_and_bar(fig)
    centred = bsplot.scale_colormap("RdBu_r", FIELD.min(), FIELD.max(), 0.0)
    assert not np.allclose(_colormap(bar)(0.5), centred(0.5))


def test_tick_prune_stands_on_its_own(study):
    """``tick_prune`` is a declared panel slot, not a modifier of ``nbins``: stated alone it drops the end tick and raises nothing."""
    plain = bsplot.render(_figure(), str(study), str(study / "plain.png")).axes[0]
    pruned = bsplot.render(_figure(panel_slots={"tick_prune": "upper"}), str(study), str(study / "pruned.png")).axes[0]
    assert max(pruned.get_xticks()) < max(plain.get_xticks())
