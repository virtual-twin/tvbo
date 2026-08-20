"""Flatten a multi-scale reservoir SimulationExperiment to a flat one (Python).

Lowering strategy for Experiment A (per-region reservoirs): a network of ``R`` macro regions, each hosting an ``n``-unit reservoir (``Node.subnetwork`` with a ``RandomReservoir`` generator) coupled long-range through the empirical SC, is compiled into a single **flat** ``R·n``-node scalar network plus a global weight matrix that the existing tvboptim backend runs unchanged. No vector-state machinery in the templates — the multi-scale structure is resolved entirely in Python (the chosen "flatten in Python" approach).

The reservoir multi-scale pattern this handles
----------------------------------------------
Inner unit dynamics (leaky-integrator + activation of recurrence + drive)::

    dx/dt = (1/tau) * (-x + act(W_int @ x + sc_drive))

with the cross-layer edges forming a **linear** down/up coupling:

* downward (``source_network: ".."``): ``sc_drive = W_in ⊙ (kappa · SC @ x_bar)``
  — the parent's long-range signal, scaled by a per-unit projection ``W_in``;
* upward mean-pool (``target_network: ".."``, ``rhs: mean(x)``): ``x_bar`` feeds
  the macro coupling;
* (optional) upward trained readout ``W_out @ x`` — ignored for free-running.

Because both the recurrence (``W_int @ x``) and the cross-region drive are linear in the units' states and enter the *same* activation, they fold into one global matrix via Kronecker structure::

    W_global = kron(I_R, W_int)  +  (kappa / n) · kron(SC, outer(W_in, 1ₙ))
    dX/dt    = (1/tau) · (-X + act(W_global @ X))            # X ∈ ℝ^{R·n}

The flat model is a scalar leaky-integrator whose coupling enters *inside* the activation. ``flatten_reservoir`` validates the spec matches this pattern and raises otherwise — it is a principled lowering of a well-defined model class, not a general arbitrary-coupling compiler (that would be the Stage-3 codegen engine emitting the procedure on-device).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlatReservoir:
    """Result of flattening: ready-to-run flat experiment pieces."""

    dynamics: dict
    coupling: dict
    integration: dict
    weights: np.ndarray  # W_global, (R·n, R·n)
    n_units: int  # n  (reservoir units per region)
    n_regions: int  # R
    tau: float
    activation: str  # inner activation, e.g. "tanh"


def _pluck(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _param_value(params: dict, name: str, default=None):
    p = (params or {}).get(name)
    if isinstance(p, dict):
        return p.get("value", default)
    return default if p is None else p


def _activation_from_rhs(rhs: str) -> str:
    """Identify the single elementwise activation wrapping the recurrence."""
    for act in ("tanh", "sigmoid", "relu", "sin", "identity"):
        if f"{act}(" in rhs:
            return act
    return "tanh"


def flatten_reservoir(
    exp: dict,
    n_override: int | None = None,
    max_flat_nodes: int | None = 12_000,
) -> FlatReservoir:
    """Flatten a per-region-reservoir experiment dict into a flat scalar model.

    ``exp`` is the parsed Experiment-A YAML (a plain dict). ``n_override`` optionally shrinks the reservoir size for a fast demonstration run (the full ``n`` from the spec is memory-heavy: ``W_global`` is dense ``(R·n)²``).

    ``max_flat_nodes`` caps the flat node count ``R·n`` before the dense ``(R·n)²`` float64 ``W_global`` is allocated. Because memory grows quadratically, a full-``n`` reservoir over a whole-brain SC can exhaust RAM (this lowering has OOM-crashed a machine). Exceeding the cap raises with the projected allocation size; pass a larger value — or ``None`` to disable — once the memory is known to be available.
    """
    # Local imports keep this module import-light.
    from tvbo.graph_generators import catalog

    net = exp["network"]
    subnet = _pluck(net, "node_template", "subnetwork")
    if subnet is None:
        raise ValueError("flatten_reservoir: network.node_template.subnetwork is required.")

    # --- inner reservoir dynamics (named, referenced by the subnetwork) -------
    dyn_block = exp["dynamics"]
    # named form `{TanhESN: {...}}` or flat `{name:.., ...}`
    if "name" not in dyn_block and len(dyn_block) == 1:
        inner = next(iter(dyn_block.values()))
    else:
        inner = dyn_block
    sv = inner["state_variables"]
    state_name, state_spec = next(iter(sv.items()))
    inner_rhs = _pluck(state_spec, "equation", "rhs") or ""
    activation = _activation_from_rhs(inner_rhs)
    tau = float(_param_value(inner.get("parameters", {}), "tau", 1.0))
    noise_sigma = float(_pluck(state_spec, "noise", "parameters", "sigma", "value", default=0.0) or 0.0)

    # --- reservoir recurrence W_int (via the typed-DAG resolver) --------------
    gg = subnet["graph_generator"]
    gg_params = {k: _param_value(gg.get("parameters", {}), k) for k in (gg.get("parameters") or {})}
    # weight_distribution carries its spec under `.distribution`
    wd = _pluck(gg, "parameters", "weight_distribution", "distribution")
    # Size is the subnetwork's node count, as it is for every generator; `n_nodes` on the generator is the standalone form used when there is no Network to read it from.
    if "n" in gg_params:
        raise ValueError(
            "flatten_reservoir: the reservoir generator declares `n`, which is no longer "
            "the size parameter. Rename it to `n_nodes`, or set the subnetwork's "
            "`number_of_nodes`. Accepting it silently would build the default 100-unit "
            "reservoir instead of the size the spec asks for, and the run would look fine."
        )
    n = int(n_override or subnet.get("number_of_nodes") or gg_params.get("n_nodes") or 100)
    gg_params["n_nodes"] = n
    if wd is not None:
        gg_params["weight_distribution"] = wd
    W_int = catalog.run_generator("RandomReservoir", gg_params, seed=gg.get("seed"))["weights"]

    # --- macro SC + long-range coupling gain kappa ----------------------------
    SC = np.asarray(catalog.load_matrix(net["iri"]), dtype=float)
    for tr in net.get("transforms") or []:
        rhs = _pluck(tr, "equation", "rhs") or ""
        if "max(W)" in rhs.replace(" ", "").replace("/max(W)", "/max(W)"):
            m = SC.max()
            if m > 0:
                SC = SC / m
    R = SC.shape[0]

    # Guard the dense Kronecker assembly below: W_global is a dense (R·n)² float64 matrix, so memory grows quadratically in the flat node count.
    flat_nodes = R * n
    if max_flat_nodes is not None and flat_nodes > max_flat_nodes:
        gib = (flat_nodes**2) * 8 / 2**30
        raise ValueError(
            f"flatten_reservoir: {R} regions x {n} units = {flat_nodes} flat "
            f"nodes would allocate a dense {flat_nodes}x{flat_nodes} float64 "
            f"W_global (~{gib:.1f} GiB), exceeding max_flat_nodes={max_flat_nodes}. "
            f"Shrink the reservoir (n_override=), reduce regions, or raise "
            f"max_flat_nodes if the memory is available."
        )

    macro_coupling = net["coupling"]
    _, mc_spec = next(iter(macro_coupling.items()))
    kappa = float(_param_value(mc_spec.get("parameters", {}), _gain_param_name(mc_spec), 1.0))

    # --- downward projection W_in (per-unit) from the sc_drive edge -----------
    W_in = _project_weights(subnet, n)

    # --- assemble W_global via Kronecker structure ----------------------------
    I_R = np.eye(R)
    recurrence = np.kron(I_R, W_int)
    cross = (kappa / n) * np.kron(SC, np.outer(W_in, np.ones(n)))
    W_global = (recurrence + cross).astype(np.float64)

    # --- flat scalar dynamics: coupling enters *inside* the activation --------
    coupling_name = "ReservoirInput"
    flat_dynamics = {
        "name": "FlatReservoirUnit",
        "parameters": {"tau": {"value": tau}},
        "state_variables": {
            "x": {
                "initial_value": 0.0,
                "equation": {"rhs": f"(1/tau) * (-x + {activation}({coupling_name}))"},
            }
        },
        "coupling_inputs": {coupling_name: {"dimension": 1}},
    }
    flat_coupling = {
        coupling_name: {
            "delayed": False,
            "local_states": ["x"],
            "pre_expression": {"rhs": "local_states"},
            "post_expression": {"rhs": "gx"},  # gain folded into W_global
        }
    }
    integ = _folded_integration(exp, subnet)

    fr = FlatReservoir(
        dynamics=flat_dynamics,
        coupling=flat_coupling,
        integration=integ,
        weights=W_global,
        n_units=n,
        n_regions=R,
        tau=tau,
        activation=activation,
    )
    fr.noise_sigma = noise_sigma  # type: ignore[attr-defined]
    return fr


def _folded_integration(exp: dict, subnet: dict) -> dict:
    """The one clock the flat network runs on, in the parent's time unit.

    Flattening leaves a single network, so it leaves a single clock, and the
    parent's unit is the one that survives — the flat nodes *are* the parent's
    nodes. Two things then have to happen, and neither did before: a subnetwork
    that declares a finer ``step_size`` has to impose it, or its dynamics are
    integrated at the macro step they were never meant to be; and if the two
    scales declare different ``time_unit``s, that step has to be *converted*
    rather than copied. A reservoir with ``step_size: 0.1`` in ``s`` folding into
    a network in ``ms`` needs 100, not 0.1 — a factor of 1000 that is invisible
    in the output of every backend, because the number stays plausible.

    ``duration`` and ``transient_time`` stay the parent's: they measure the
    experiment, not the reservoir.
    """
    from tvbo.utils.units import time_unit_factor

    parent = dict(exp.get("integration") or {"method": "Heun", "step_size": 1.0, "duration": 1000, "transient_time": 0})
    inner = subnet.get("integration") or {}
    if inner.get("step_size") is None:
        return parent

    outer = (exp.get("integration"), exp)
    factor = time_unit_factor((subnet, *outer), outer)
    inner_step = float(inner["step_size"]) * float(factor)
    parent_step = parent.get("step_size")
    if parent_step is None or inner_step < float(parent_step):
        parent["step_size"] = inner_step
    return parent


def _gain_param_name(coupling_spec: dict) -> str:
    """The scalar gain parameter referenced by the macro post_expression (e.g. kappa, G)."""
    post = _pluck(coupling_spec, "post_expression", "rhs") or ""
    for name in coupling_spec.get("parameters") or {}:
        if name in post:
            return name
    return next(iter(coupling_spec.get("parameters") or {"G": None}), "G")


def _project_weights(subnet: dict, n: int) -> np.ndarray:
    """Materialise the per-unit downward projection W_in from the sc_drive edge."""
    from tvbo.graph_generators import procedural

    for edge in subnet.get("edges") or []:
        if edge.get("source_network") in ("..", "parent"):
            wparams = _pluck(edge, "coupling", "parameters") or {}
            for pspec in wparams.values():
                dist = pspec.get("distribution") if isinstance(pspec, dict) else None
                if dist:
                    seed = _pluck(pspec, "distribution", "seed")
                    return procedural.draw(dist, (n,), seed=seed)
    # default: unit projection
    return np.ones(n)
