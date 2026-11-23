"""A figure layer's `sel` takes, key by key, the exact label for a name and the nearest coordinate for a number, the rule `DataRef.sel` states.

A selection holding a label beside a number, one hemisphere at one swept coupling, matched every key exactly as soon as one key was a label, so a number that is not a stored coordinate to the last digit raised `KeyError`; only an all-numeric selection took the nearest. The emitted script and a custom or built-in panel's `load_layer` select the way every other `DataRef` reader does.

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

DATA = np.arange(24.0).reshape(2, 3, 4)
SEL = {"group": "rh", "KuramotoCoupling.a": 3.36e-06}
"""One hemisphere by its label and one coupling by a value 0.06 % off the stored 3.3598e-06."""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


@pytest.fixture
def container(tmp_path):
    """A container over hemisphere, coupling and time, written where the layout record says an analysis result lives."""
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    ds = xr.Dataset(
        {"rate": (("group", "KuramotoCoupling.a", "t"), DATA)},
        coords={"group": ["lh", "rh"], "KuramotoCoupling.a": [1.0e-06, 3.3598e-06, 1.0e-05], "t": [0.0, 1.0, 2.0, 3.0]},
    )
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    path = analysis_container_path(results, "synth")
    ds.to_netcdf(path, engine="h5netcdf")
    return str(path)


def test_the_emitted_script_takes_the_label_exactly_and_the_number_by_nearest(tmp_path, container):
    used = P.DataRef(iri="tvbo:synth", output="rate", sel={k: P.Argument(name=k, value=v) for k, v in SEL.items()})
    layer = P.Layer(used=used, encoding=P.Encoding(x="t", y="rate"))
    figure = P.Figure(
        name="sel", layout="a", panel_numbers=False, panels={"a": P.Panel(panel_key="a", kind="cartesian", layers=[layer])}
    )
    fig = bsplot.render(figure, str(tmp_path), str(tmp_path / "out.png"))
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), DATA[1, 1])


def test_a_panels_load_layer_selects_the_same_way(container):
    np.testing.assert_allclose(bsplot.load_layer({"container": container, "output": "rate", "sel": SEL}).values, DATA[1, 1])
