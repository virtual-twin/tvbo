# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""The flow a model defines on the plane of two of its state variables, evaluated from the experiment's own right-hand side.

A phase-plane figure draws the vector field behind the orbits a run traced. The field is not a second recording: it is the `Dynamics` the experiment ran, read back from the result container's provenance sidecar and evaluated on a grid, with every state variable off the plane held at its declared initial value and every parameter a sweep set taken from the swept cell. So the arrows and the orbits come from one model and cannot disagree.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import numpy as np
import yaml


@functools.lru_cache(maxsize=64)
def _dynamics_record(container: str) -> dict:
    """The ``dynamics`` entry of the provenance sidecar beside *container*, parsed once per container."""
    from tvbo.data.dataref import sidecar_path

    sidecar = sidecar_path(container)
    if not sidecar.is_file():
        raise ValueError(f"{Path(container).name} has no provenance sidecar {sidecar.name}, so the model it was run with is unknown.")
    record = yaml.safe_load(sidecar.read_text())
    if not isinstance(record, dict) or not record.get("dynamics"):
        raise ValueError(f"{sidecar.name} records no dynamics: only an experiment's result carries the model a vector field is drawn from.")
    return record["dynamics"]


def dynamics_of(container) -> object:
    """The `Dynamics` an experiment result was integrated with, read from the provenance sidecar beside its container (`tvbo.data.dataref.sidecar_path`).

    The sidecar is parsed once per container, and every call builds a fresh `Dynamics` from it, so a caller that sets parameters on the model (`with_parameters`) changes no other caller's.

    Args:
        container: path of an experiment result container (``exp-<id>_..._result.h5``).

    Raises:
        ValueError: If the container has no sidecar or the sidecar records no dynamics, which is the case for an analysis result.
    """
    from tvbo import Dynamics

    return Dynamics(**copy.deepcopy(_dynamics_record(str(container))))


def cell_key(cell: dict | None) -> tuple:
    """*cell* as a hashable cache key: its single-valued entries as ``(name, value)`` pairs sorted by name, a NumPy scalar as the Python one it holds."""
    return tuple(sorted((str(k), np.asarray(v).item()) for k, v in (cell or {}).items() if np.ndim(v) == 0))


def with_parameters(dynamics, cell: dict | None):
    """*dynamics* with each parameter a swept cell names set to that cell's value.

    A sweep coordinate is named ``<Model>.<parameter>`` or ``<parameter>``; a name that is no parameter of the model (``node``, ``execution.random_seed``) leaves the model as it is.
    """
    for name, value in (cell or {}).items():
        key = str(name).rsplit(".", 1)[-1]
        if key in dynamics.parameters and np.ndim(value) == 0:
            dynamics.parameters[key].value = float(value)
    return dynamics


def _rhs(dynamics, x: str, y: str):
    """The right-hand side restricted to the (*x*, *y*) plane, as ``f(X, Y) -> (U, V)`` over arrays of any shape, every other state variable at its declared initial value."""
    from tvbo.adapters.base import ContinuationAdapter

    names = list(dynamics.state_variables)
    missing = [v for v in (x, y) if v not in names]
    if missing:
        raise ValueError(f"vector field: {', '.join(missing)} is not a state variable of {dynamics.name}; the plane is spanned by two of {names}.")
    ix, iy = names.index(x), names.index(y)
    rest = ContinuationAdapter.declared_state(dynamics)
    dfun = ContinuationAdapter.vector_field(dynamics)

    def f(X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        state = np.broadcast_to(rest.reshape(-1, *([1] * X.ndim)), (rest.size, *X.shape)).copy()
        state[ix], state[iy] = X, Y
        try:
            du = dfun(state.reshape(rest.size, -1))
        except Exception:  # a right-hand side written for one state vector at a time
            du = np.column_stack([dfun(s) for s in state.reshape(rest.size, -1).T])
        du = np.broadcast_to(du, (rest.size, X.size))
        return du[ix].reshape(X.shape), du[iy].reshape(X.shape)

    return f


def _grid(f, xlim, ylim, n: int = 30):
    """The planar field *f* on an *n* by *n* grid over *xlim* by *ylim*: ``(X, Y, U, V)``, each of shape ``(n, n)``."""
    X, Y = np.meshgrid(np.linspace(*map(float, xlim), n), np.linspace(*map(float, ylim), n))
    U, V = f(X, Y)
    return X, Y, U, V


def _roots(f, xlim, ylim, seeds: int = 6, tol: float = 1e-9) -> np.ndarray:
    """The points of the window where the planar field *f* vanishes, each once, as an array of ``(x, y)`` rows (see `equilibria`)."""
    from scipy.optimize import root

    span = np.array([np.ptp(xlim), np.ptp(ylim)], dtype=float)
    found: list = []
    for sx in np.linspace(*map(float, xlim), seeds):
        for sy in np.linspace(*map(float, ylim), seeds):
            sol = root(lambda p: np.array(f(p[0], p[1])).ravel(), [sx, sy], method="hybr", tol=tol)
            if not sol.success:
                continue
            p = sol.x
            inside = xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]
            if inside and np.max(np.abs(f(p[0], p[1]))) < 1e-6 and not any(np.all(np.abs(p - q) < 1e-6 * span + 1e-12) for q in found):
                found.append(p)
    return np.array(found).reshape(-1, 2)


def vector_field(dynamics, x: str, y: str, xlim, ylim, n: int = 30):
    """The field of *dynamics* on an *n* by *n* grid over *xlim* by *ylim*: ``(X, Y, U, V)``, each of shape ``(n, n)``.

    State variables off the plane are held at their declared initial values, which is exact for a two-variable model and a section through the flow for a larger one.
    """
    return _grid(_rhs(dynamics, x, y), xlim, ylim, n)


def equilibria(dynamics, x: str, y: str, xlim, ylim, seeds: int = 6, tol: float = 1e-9) -> np.ndarray:
    """The points of the window where the planar field vanishes, as an array of ``(x, y)`` rows.

    Found by a Newton-type root search (MINPACK's ``hybr``) from a *seeds* by *seeds* grid of starts across the window; roots that converge outside the window or onto one already found are dropped, so each equilibrium appears once.
    """
    return _roots(_rhs(dynamics, x, y), xlim, ylim, seeds, tol)


def rest_state(dynamics) -> dict:
    """The state where the whole right-hand side vanishes, as ``{variable: value}``: `ContinuationAdapter.newton_equilibrium` from the declared initial state, the search a continuation starting by ``newton`` makes.

    Raises:
        RuntimeError: If the search does not converge, naming the model.
    """
    from tvbo.adapters.base import ContinuationAdapter

    try:
        state = ContinuationAdapter.newton_equilibrium(dynamics, None)
    except RuntimeError as err:
        raise RuntimeError(f"{dynamics.name} has no rest state to draw: no state where its right-hand side vanishes was found from its initial state ({err}).") from err
    return dict(zip(dynamics.state_variables, state, strict=True))


@functools.lru_cache(maxsize=256)
def experiment_rest(container: str, cell: tuple = ()) -> dict:
    """:func:`rest_state` of the model a result container was run with, at the swept *cell* given as its `cell_key`. Cached, because a movie asks once per frame."""
    return rest_state(with_parameters(dynamics_of(container), dict(cell)))


@functools.lru_cache(maxsize=64)
def _experiment_field(container: str, x: str, y: str, xlim: tuple, ylim: tuple, cell: tuple):
    """The grid field and the equilibria of the model a container was run with, from one compiled right-hand side; cached, because a movie redraws its static backdrop once per frame."""
    f = _rhs(with_parameters(dynamics_of(container), dict(cell)), x, y)
    return _grid(f, xlim, ylim), _roots(f, xlim, ylim)


@functools.lru_cache(maxsize=64)
def window(container, output: str, x: str, y: str, pad: float = 0.25):
    """The plane a run's orbits span: the range of *x* and *y* over every sample of *output*, widened by *pad* of each span on both sides. Cached, because a movie asks once per frame."""
    import xarray as xr

    from tvbo.data.dataref import match_output

    with xr.open_dataset(container) as ds:
        da = ds[match_output(ds.data_vars, output)]
        dim = next((d for d in da.dims if d in da.coords and {x, y} <= {str(v) for v in np.atleast_1d(da.coords[d].values)}), None)
        if dim is None:
            raise ValueError(f"vector field: {output} carries no axis holding both {x} and {y}, so the window cannot be read off the run; declare xlim and ylim.")
        out = []
        for name in (x, y):
            vals = np.asarray(da.sel({dim: name}).values, dtype=float)
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            width = (hi - lo) or max(abs(hi), 1.0)
            out.append((lo - pad * width, hi + pad * width))
    return tuple(out)


def stream(ax, container, output: str, x: str, y: str, *, xlim=None, ylim=None, cell: dict | None = None, density: float = 1.2, cmap=None, fixed_points=True, **kwargs):
    """Draw the experiment's own vector field on *ax* as streamlines coloured by speed, with its equilibria marked.

    Args:
        ax: the axes to draw on; its limits are set to the window.
        container: the experiment result container; its sidecar supplies the model.
        output: the container variable whose *x* and *y* components span the default window.
        x: the state variable on the horizontal axis.
        y: the state variable on the vertical axis.
        xlim: the horizontal extent of the field; read off the run when unset.
        ylim: the vertical extent of the field; read off the run when unset.
        cell: the swept coordinates of the run the layer selected, applied to the model's parameters; a list-valued entry selects no single run and is ignored.
        density: streamline density, as matplotlib's ``streamplot`` reads it.
        cmap: colormap for the speed.
        fixed_points: a dict of ``scatter`` keywords to mark the equilibria with, ``True`` for the default mark, or ``False`` for none.
        **kwargs: forwarded to ``streamplot`` (``linewidth``, ``alpha``, ``arrowsize``, ``zorder``).

    Returns:
        The ``StreamplotSet``.
    """
    if xlim is None or ylim is None:
        wx, wy = window(str(container), output, x, y)
        xlim, ylim = xlim or wx, ylim or wy
    xlim, ylim = tuple(map(float, xlim)), tuple(map(float, ylim))
    (X, Y, U, V), points = _experiment_field(str(container), x, y, xlim, ylim, cell_key(cell))
    alpha = kwargs.pop("alpha", None)
    kwargs.pop("color", None)  # the colour is the speed
    kwargs.pop("label", None)
    kwargs.setdefault("zorder", 0)
    drawn = ax.streamplot(X, Y, U, V, color=np.hypot(U, V), cmap=cmap, density=density, **kwargs)
    if alpha is not None:
        drawn.lines.set_alpha(alpha)
        drawn.arrows.set_alpha(alpha)
    if fixed_points:
        import matplotlib as mpl

        mark = {"s": 40, "color": mpl.rcParams["text.color"], "edgecolors": mpl.rcParams["axes.facecolor"], "linewidths": 0.8, "zorder": 6}
        mark.update(fixed_points if isinstance(fixed_points, dict) else {})
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], **mark)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return drawn
