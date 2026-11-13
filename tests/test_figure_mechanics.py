"""Tests for the mechanics a phase-plane figure is drawn with: the vector field read from the experiment's own dynamics (`tvbo.plot.field`, `mark: stream`) and masses on springs (`kind: spring`).

The result container is written here, with the provenance sidecar an experiment run writes beside it, so the tests need no study on disk.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
import yaml

import tvbo.datamodel.pydantic as P
from tvbo import Dynamics
from tvbo.adapters import bsplot
from tvbo.plot import field

SPRING = {
    "name": "DampedSpring",
    "parameters": {"k": {"value": 1.0}, "m": {"value": 1.0}, "c": {"value": 0.5}, "F": {"value": 0.3}},
    "state_variables": {
        "x": {"equation": {"rhs": "v"}, "initial_value": 1.0, "variable_of_interest": True},
        "v": {"equation": {"rhs": "-(k/m)*x - (c/m)*v + F/m"}, "initial_value": 0.0},
    },
}


@pytest.fixture
def container(tmp_path):
    """A damped spring's result container over two nodes, with the sidecar that names its dynamics."""
    import xarray as xr

    t = np.linspace(0.0, 10.0, 201)
    x = 0.3 + np.exp(-0.25 * t)[:, None] * np.cos(t)[:, None] * np.array([0.7, 0.35])
    v = np.gradient(x, t, axis=0)
    data = np.stack([x, v], axis=1)
    ds = xr.Dataset(
        {"integration": (("time", "variable", "node"), data)},
        coords={"time": t, "variable": ["x", "v"], "node": ["far", "near"]},
        attrs={"sidecar_file": "exp-1_result.yaml"},
    )
    path = tmp_path / "exp-1_result.h5"
    ds.to_netcdf(path, engine="h5netcdf")
    (tmp_path / "exp-1_result.yaml").write_text(yaml.safe_dump({"id": 1, "dynamics": SPRING}))
    return path


def test_rest_state_is_where_the_right_hand_side_vanishes():
    assert field.rest_state(Dynamics(**SPRING)) == pytest.approx({"x": 0.3, "v": 0.0})


def test_vector_field_is_the_right_hand_side_on_the_grid():
    X, Y, U, V = field.vector_field(Dynamics(**SPRING), "x", "v", (-1, 1), (-1, 1), n=5)
    assert X.shape == U.shape == (5, 5)
    assert U[0, 0] == pytest.approx(-1.0) and V[0, 0] == pytest.approx(1 + 0.5 + 0.3), "at (x, v) = (-1, -1): dx = v, dv = -x - 0.5 v + 0.3"


def test_equilibria_are_found_once_inside_the_window():
    points = field.equilibria(Dynamics(**SPRING), "x", "v", (-1, 1), (-1, 1))
    assert points.shape == (1, 2) and points[0] == pytest.approx([0.3, 0.0])
    assert field.equilibria(Dynamics(**SPRING), "x", "v", (1, 2), (-1, 1)).shape == (0, 2), "an equilibrium outside the window is not drawn"


def test_a_swept_parameter_reaches_the_model():
    model = field.with_parameters(Dynamics(**SPRING), {"DampedSpring.F": 0.6, "node": "far"})
    assert field.rest_state(model)["x"] == pytest.approx(0.6)


def test_the_model_is_read_from_the_result_sidecar(container):
    assert field.dynamics_of(container).name == "DampedSpring"
    import xarray as xr

    x = xr.open_dataset(container)["integration"].sel(variable="x")
    (lo, hi), _ = field.window(container, "integration", "x", "v")
    assert lo < float(x.min()) and hi > float(x.max()), "the window spans the run's orbits with a margin"
    assert field.experiment_rest(str(container), ())["x"] == pytest.approx(0.3)


def test_a_state_variable_off_the_model_is_refused():
    with pytest.raises(ValueError, match="not a state variable"):
        field.vector_field(Dynamics(**SPRING), "x", "w", (-1, 1), (-1, 1))


def _figure(container, animation=None):
    """A spring panel beside a phase plane: the masses, and their orbits over the model's own field."""
    used = dict(iri=str(container), output="integration")
    return P.Figure(
        name="mechanics",
        layout="ab",
        animation=animation,
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="spring",
                spring=P.Springs(labels=True),
                layers=[P.Layer(used=P.DataRef(**used, sel={"variable": "x"}), encoding=P.Encoding(color="node"))],
            ),
            "b": P.Panel(
                panel_key="b",
                kind="cartesian",
                layers=[
                    P.Layer(used=P.DataRef(**used), mark="stream", encoding=P.Encoding(x="x", y="v")),
                    P.Layer(used=P.DataRef(**used, sel={"variable": "v"}), mark="line", frame="trail", encoding=P.Encoding(x="x", y="integration", color="node")),
                    P.Layer(used=P.DataRef(**used, sel={"variable": "v"}), mark="scatter", encoding=P.Encoding(x="x", y="integration", color="node")),
                ],
            ),
        },
    )


def test_a_spring_panel_is_a_built_in_drawer(container):
    ctx = bsplot.build_context(_figure(container), container.parent, "out.png")
    spring = next(p for p in ctx["panels"] if p["key"] == "a")
    assert spring["render"] == "spring" and spring["ctx"]["opts"]["labels"] is True
    assert "spring" in bsplot.CUSTOM_PANELS, "kind: spring needs no code_modules"


def test_a_stream_layer_is_the_static_backdrop_of_a_movie(container):
    animation = P.Animation(over="time", frames=3, fps=5)
    ctx = bsplot.build_context(_figure(container, animation), container.parent, "out.gif")
    stream = next(p for p in ctx["panels"] if p["key"] == "b")["layers"][0]
    assert stream["frame"] == "static"
    code = bsplot.render_code(_figure(container, animation), container.parent, "out.gif")
    ast.parse(code)
    assert "_field.stream(ax, " in code


def test_the_mechanics_render_still_and_animated(container, tmp_path):
    bsplot.render(_figure(container), base_dir=container.parent, outfile=str(tmp_path / "still.png"))
    assert (tmp_path / "still.png").stat().st_size > 0
    movie = tmp_path / "movie.gif"
    bsplot.render(_figure(container, P.Animation(over="time", frames=3, fps=5)), base_dir=container.parent, outfile=str(movie))
    assert movie.stat().st_size > 0
