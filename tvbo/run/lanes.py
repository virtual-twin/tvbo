"""Run one experiment once per lane of the per-node values its event parameters source.

An event parameter whose `used:` DataRef resolves to an array on the network's nodes and one axis more varies per lane: every entry of that extra axis is one run of the same settled experiment with the parameter set to that entry's per-node values. A DBS cohort is the case this serves: a curated container holds one row per stimulation programme (`programme × node`, with `subject` and `session` as coordinates on `programme`), and each row drives the stimulus of one lane.

The experiment is rendered and settled once. The lanes then run as one zip space over the event parameters' leaves in `state.external` through tvboptim's `ParallelExecution`, and every declared observation is recorded in that one sweep. With `baseline=True` one more run, every lane parameter at zero, is recorded beside them as `<observation>_baseline`, the undriven reference an evoked response is read against. The result is one `xarray.Dataset`: each observation on the lane axis (its coordinates carried over from the source) and the observation's own dims, node axes labelled with the network's node labels, and the lane values themselves as `axis_points__<event>.<parameter>`, the name a saved exploration gives its array-valued axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from tvbo.utils import keyed_items

NODE_DIMS = ("node", "nodes", "region", "regions")
"""Observation dims that name the network's node axis."""

AXIS_POINTS = "axis_points__"
"""Prefix of the variables that carry each lane's parameter values, as in a saved exploration."""

BASELINE = "_baseline"
"""Suffix of the undriven reference recorded beside each observation."""


@dataclass(frozen=True)
class LaneParameter:
    """One event parameter that varies per lane: the key of its event in `state.external`, its name, and its `(lane, node)` values."""

    event: str
    name: str
    values: xr.DataArray

    @property
    def key(self) -> str:
        """`<event>.<parameter>`, the path an exploration axis spells it with."""
        return f"{self.event}.{self.name}"


def _node_labels(experiment) -> list[str]:
    """The network's node labels in model order."""
    return [str(label) for label in experiment.network.node_labels]


def _event_parameters(experiment):
    """Every `(event key in state.external, parameter name, parameter)` of the experiment's events, the key being the event's `target_variable` where it declares one and its name otherwise, the rule `SimulationExperiment._resolve_events` applies before the emitted module spells it."""
    for key, event in keyed_items(getattr(experiment, "events", None), "events"):
        event_key = str(getattr(event, "target_variable", None) or getattr(event, "name", None) or key)
        for name, param in keyed_items(getattr(event, "parameters", None), "parameters"):
            yield event_key, str(name), param


def _as_lane_parameter(event: str, name: str, values: xr.DataArray, labels: list[str]) -> LaneParameter:
    """`values` checked to be `(lane, node)` over the network's nodes, in model order, the node axis named `node`."""
    node_dims = [d for d in values.dims if d in values.coords and [str(v) for v in values.coords[d].values] == labels]
    if len(node_dims) != 1 or values.ndim != 2:
        raise ValueError(
            f"event parameter {event}.{name} varies per lane only as a (lane, node) array whose node axis is labelled with the network's {len(labels)} node labels in model order; "
            f"it resolved to dims {tuple(values.dims)} with shape {tuple(values.shape)}. Declare `reconcile: by_label` on its `used:` so the node axis is aligned by label."
        )
    lane_dim = next(d for d in values.dims if d != node_dims[0])
    return LaneParameter(event, name, values.rename({node_dims[0]: "node"}).transpose(lane_dim, "node"))


def lane_parameters(experiment, lanes: xr.Dataset | None = None, *, results_root=None) -> list[LaneParameter]:
    """The experiment's lane-varying event parameters, every one on the same lane axis.

    Without `lanes`, each event parameter whose `used:` DataRef names a container (an `iri`, an `experiment` or an `analysis`) is resolved through `tvbo.data.dataref.resolve_dataref`, reconciled by node label when the reference asks for it; the `data` parameter of a data-driven stimulus is a recording, not a lane parameter, and is left to the stimulus. With `lanes`, its `<event>.<parameter>` variables are the values instead, which is how a caller that holds the arrays in memory runs the same sweep without writing them first. Raises when nothing varies per lane or the parameters disagree on the lane axis.
    """
    from tvbo.data import dataref

    labels = _node_labels(experiment)
    found: list[LaneParameter] = []
    if lanes is not None:
        declared = {f"{event}.{name}" for event, name, _ in _event_parameters(experiment)}
        unknown = sorted(set(map(str, lanes.data_vars)) - declared)
        if unknown:
            raise ValueError(
                f"lanes carries {unknown}, which are not event parameters of the experiment (declared: {sorted(declared)})"
            )
        for key in lanes.data_vars:
            event, name = str(key).split(".", 1)
            found.append(_as_lane_parameter(event, name, lanes[key], labels))
    else:
        alias_map = experiment.network.region_alias_map()
        for event, name, param in _event_parameters(experiment):
            ref = getattr(param, "used", None)
            if ref is None or name == "data" or dataref.is_local_ref(ref):
                continue
            values = dataref.resolve_dataref(ref, results_root=results_root, alias_map=alias_map, model_labels=labels)
            found.append(_as_lane_parameter(event, name, values, labels))
    if not found:
        raise ValueError(
            "no event parameter varies per lane: none has a `used:` reference to a (lane, node) array, and no lanes were passed"
        )
    first = found[0].values
    for param in found[1:]:
        other = param.values
        if other.dims[0] != first.dims[0] or other.sizes[first.dims[0]] != first.sizes[first.dims[0]]:
            raise ValueError(
                f"{param.key} runs over {other.dims[0]}={other.sizes[other.dims[0]]} lanes but {found[0].key} over {first.dims[0]}={first.sizes[first.dims[0]]}; every lane parameter shares one lane axis"
            )
        for coord in set(first.coords) & set(other.coords):
            if first.coords[coord].dims == (first.dims[0],) and not np.array_equal(
                np.asarray(first.coords[coord].values), np.asarray(other.coords[coord].values)
            ):
                raise ValueError(
                    f"{param.key} and {found[0].key} disagree on the lane coordinate {coord!r}; their rows are not the same lanes"
                )
    return found


def build_network(experiment, namespace):
    """The emitted module's network over the spec's own weights, and its tract lengths where `create_network` takes them."""
    import inspect

    import jax.numpy as jnp

    weights = jnp.asarray(np.asarray(experiment.network.matrix("weight", format="dense", apply_transforms=False), dtype=float))
    lengths = experiment.network.matrix("length", format="dense", apply_transforms=False)
    kwargs = {}
    if lengths is not None and "distances" in inspect.signature(namespace.create_network).parameters:
        kwargs["distances"] = jnp.asarray(np.asarray(lengths, dtype=float))
    return namespace.create_network(weights, **kwargs)


def settle(experiment, namespace, network):
    """`(model_fn, state)` after the experiment's transient, the noise key reset to the declared `execution.random_seed` so every lane integrates the same realisation."""
    import jax
    import jax.numpy as jnp

    sim = namespace.run_simulation(network, run_main=False)
    state = sim.state
    if getattr(state, "noise", None) is not None:
        seed = int(getattr(getattr(experiment, "execution", None), "random_seed", None) or 0)
        state.noise.key = jax.random.key(jnp.asarray(seed, dtype=jnp.uint32))
    return sim.model_fn, state


def _observation_dims(experiment, name: str, shape: tuple[int, ...], n_nodes: int) -> tuple[str, ...]:
    """The declared `dims` of observation `name` for one lane's value of `shape`, `node` where the observation names the node axis, positional names where it declares none."""
    obs = dict(keyed_items(getattr(experiment, "observations", None), "observations"))[name]
    dims = tuple(str(d) for d in (getattr(obs, "dims", None) or []))
    if dims and len(dims) != len(shape):
        raise ValueError(f"observation {name!r} declares dims {dims} but one lane's value has shape {shape}")
    if not dims:
        dims = ("node",) if shape == (n_nodes,) else tuple(f"{name}_dim{k}" for k in range(len(shape)))
    return tuple("node" if d.lower() in NODE_DIMS and shape[k] == n_nodes else d for k, d in enumerate(dims))


def run_lanes(
    experiment,
    lanes: xr.Dataset | None = None,
    *,
    results_root=None,
    observations=None,
    batch_size: int = 8,
    n_devices: int = 1,
    baseline: bool = True,
    namespace=None,
) -> xr.Dataset:
    """Every lane of the experiment's lane parameters run once from the settled state, every observation recorded.

    `lanes` and `results_root` are as in `lane_parameters`. `observations` restricts the recorded observations (default: every declared one). `batch_size` lanes are vectorised together on each of `n_devices` devices; neither changes a lane's value. `namespace` is the experiment's rendered tvboptim module when the caller already holds it. Observations keep the precision the experiment computed in and come back as float64.
    """
    import jax
    import jax.numpy as jnp
    from tvboptim.execution import ParallelExecution
    from tvboptim.types import DataAxis, Space

    params = lane_parameters(experiment, lanes, results_root=results_root)
    labels = _node_labels(experiment)
    lane_dim = params[0].values.dims[0]
    n_lanes = params[0].values.sizes[lane_dim]
    names = (
        [str(n) for n, _ in keyed_items(getattr(experiment, "observations", None), "observations")]
        if observations is None
        else [str(n) for n in observations]
    )
    if not names:
        raise ValueError("the experiment declares no observation to record per lane")

    ns = experiment.execute("tvboptim") if namespace is None else namespace
    model_fn, state = settle(experiment, ns, build_network(experiment, ns))
    obs_fn = jax.jit(
        lambda st: tuple(ns._obs_data(getattr(ns.compute_all_observations(model_fn(st), st), name)) for name in names)
    )

    def sweep(values: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Each observation stacked over the rows of `values`, one run per row."""
        template = jax.tree_util.tree_map(lambda x: x, state)
        for param in params:
            getattr(template.external, param.event)[param.name] = DataAxis(jnp.asarray(values[param.key]))
        n = next(iter(values.values())).shape[0]
        result = ParallelExecution(
            obs_fn, Space(template, mode="zip"), n_vmap=max(1, min(int(batch_size), n)), n_pmap=max(1, int(n_devices))
        ).run()
        return [np.stack([np.asarray(result[k][i], dtype=np.float64) for k in range(n)]) for i in range(len(names))]

    driven = sweep({p.key: np.asarray(p.values.values, dtype=np.float64) for p in params})
    lane_coords = {c: v for c, v in params[0].values.coords.items() if v.dims == (lane_dim,)}
    variables: dict[str, Any] = {}
    for name, values in zip(names, driven, strict=True):
        dims = _observation_dims(experiment, name, values.shape[1:], len(labels))
        variables[name] = xr.DataArray(values, dims=(lane_dim, *dims))
    if baseline:
        for name, values in zip(names, sweep({p.key: np.zeros((1, len(labels))) for p in params}), strict=True):
            variables[name + BASELINE] = xr.DataArray(values[0], dims=variables[name].dims[1:])
    for param in params:
        variables[AXIS_POINTS + param.key] = xr.DataArray(
            np.asarray(param.values.values, dtype=np.float64), dims=(lane_dim, "node")
        )
    ds = xr.Dataset(variables, coords={**lane_coords, "node": labels})
    ds.attrs.update(
        lane_dim=lane_dim,
        n_lanes=int(n_lanes),
        lane_parameters=[p.key for p in params],
        observations=names,
        baseline=int(bool(baseline)),
    )
    return ds
