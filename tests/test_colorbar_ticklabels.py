"""A standalone colour bar's `ticklabels` land where the spec means them, never on whatever ticks the backend picked.

Labels declared without `ticks` (`[low, high]`, the way a paper keys a scale whose numbers mean nothing to the reader) are spread evenly from the bar's lower end to its upper end, so two labels mark its two ends. Handed to the backend's automatic ticks instead, the labels fall on the first of them in order, which can lie beyond the bar's end or partway along it, and a label is lost when there are fewer ticks than labels. A label list that does not match the declared ticks one to one is refused, since any pairing of the two would be a guess.

Self-contained: a synthetic container in ``tmp_path``, rendered under Agg.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

FIELD = np.linspace(-0.5, 0.3, 36).reshape(6, 6)
"""The field the bar keys; its limits, read from the data, are the bar's two ends."""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


@pytest.fixture
def study(tmp_path):
    """A study directory holding one container with a single 2-D output, where the layout record says an analysis result lives."""
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    ds = xr.Dataset({"corr": (("node", "node2"), FIELD)}, coords={"node": np.arange(6), "node2": np.arange(6)})
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    ds.to_netcdf(analysis_container_path(results, "synth"), engine="h5netcdf")
    return tmp_path


def _bar(study, **colorbar):
    """Render a figure holding one standalone bar keyed to the field, and return the bar's axis that carries its ticks."""
    figure = P.Figure(
        name="bar",
        layout="a",
        panel_numbers=False,
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="colorbar",
                layers=[P.Layer(used=P.DataRef(iri="tvbo:synth", output="corr"), encoding=P.Encoding(x="node", y="node2"))],
                colorbar={"colormap": "viridis", **colorbar},
            )
        },
    )
    fig = bsplot.render(figure, str(study), str(study / "out.png"))
    (slot,) = [ax for ax in fig.axes if ax.child_axes]
    bar = slot.child_axes[0]
    return bar.xaxis if colorbar.get("orientation") == "horizontal" else bar.yaxis


def _labels(axis):
    return [t.get_text() for t in axis.get_ticklabels()]


def test_two_labels_without_ticks_mark_the_two_ends_of_the_bar(study):
    axis = _bar(study, ticklabels=["low", "high"])
    np.testing.assert_allclose(axis.get_ticklocs(), [FIELD.min(), FIELD.max()])
    assert _labels(axis) == ["low", "high"]


def test_labels_without_ticks_are_spread_evenly_along_a_horizontal_bar(study):
    axis = _bar(study, ticklabels=["weak", "mid", "strong"], orientation="horizontal")
    np.testing.assert_allclose(axis.get_ticklocs(), np.linspace(FIELD.min(), FIELD.max(), 3))
    assert _labels(axis) == ["weak", "mid", "strong"]


def test_labels_with_ticks_name_those_ticks_one_to_one(study):
    axis = _bar(study, ticks=[-0.4, 0.0, 0.2], ticklabels=["a", "b", "c"])
    np.testing.assert_allclose(axis.get_ticklocs(), [-0.4, 0.0, 0.2])
    assert _labels(axis) == ["a", "b", "c"]


def test_labels_that_do_not_match_the_ticks_one_to_one_are_refused(study):
    with pytest.raises(ValueError, match=r"2 `ticklabels` for 3 `ticks`"):
        _bar(study, ticks=[-0.4, 0.0, 0.2], ticklabels=["low", "high"])
