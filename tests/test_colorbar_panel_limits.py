"""A standalone colour bar states its scale with the words the maps beside it use: `symmetric`, `percentile` and `decimals`.

`kind: colorbar` is a `ColorScale`, as a surface, a volume and a network are, so a bar keying maps drawn on a scale centred on zero declares `symmetric: true` and gets that scale. Its limits come from the rule those kinds use, an end it declares still wins, and an undeclared end without a rule is the data's own. `decimals` writes every tick number to that many places; beside `ticklabels`, which replace the numbers with text, it would print nothing, so the pair is refused.

Self-contained: a synthetic container in ``tmp_path``, rendered under Agg.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

FIELD = np.linspace(-0.5, 0.3, 36).reshape(6, 6)
"""The field the bar keys: off-centre, so a centred scale and the data's own range differ at the upper end."""


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
    """Render a figure holding one standalone bar keyed to the field, and return the bar's long axis."""
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
    return slot.child_axes[0].yaxis


def test_a_symmetric_bar_is_centred_on_zero(study):
    np.testing.assert_allclose(sorted(_bar(study, symmetric=True).get_view_interval()), [-0.5, 0.5])


def test_a_percentile_clips_the_bar_to_that_share_of_the_values(study):
    axis = _bar(study, percentile=90.0)
    np.testing.assert_allclose(sorted(axis.get_view_interval()), np.percentile(FIELD, [10.0, 90.0]))


def test_an_undeclared_scale_still_spans_the_data(study):
    np.testing.assert_allclose(sorted(_bar(study).get_view_interval()), [FIELD.min(), FIELD.max()])


def test_decimals_write_every_tick_to_that_many_places(study):
    labels = [t.get_text() for t in _bar(study, decimals=2).get_ticklabels()]
    assert labels and all(re.fullmatch(r"[-\N{MINUS SIGN}]?\d+\.\d\d", t) for t in labels), labels
    assert not any(re.fullmatch(r"[-\N{MINUS SIGN}]0\.00", t) for t in labels), labels


def test_decimals_beside_ticklabels_are_refused(study):
    with pytest.raises(ValueError, match="`decimals`"):
        _bar(study, decimals=2, ticklabels=["low", "high"])
