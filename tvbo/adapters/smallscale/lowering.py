"""Backend-neutral network lowering for small-scale simulators.

The functions here turn a TVB-O ``Network`` (nodes with ``size``, edges with a ``connectivity`` rule, ``Dynamics``/``Coupling``/``Event`` biology) into the two structures every point-neuron backend needs:

* **populations** — nodes grouped by their ``Dynamics``, each a block of ``Node.size`` cells, with a stable base index per node so edges can address individual cells; and
* **connections** — the explicit cell-to-cell :class:`ConnectionRecord` set that a ``connectivity`` rule (``all_to_all``/``one_to_one``) lowers to, with self-connections filtered and per-connection ``weight``/``delay`` extracted.

Everything here is independent of *how* a backend emits a synapse — that (LEMS XML, Brian2 ``Synapses``, …) stays in the backend adapter. The backend injects its own *role vocabulary* (which ``Dynamics`` are cells vs current sources vs event sources), drawn from the NeuroML type sets defined here, so the same lowering serves NeuroML, Brian2 and the rest unchanged.
"""

from __future__ import annotations

import re
import warnings
from fractions import Fraction
from typing import TypedDict

from tvbo.utils.units import unit_multiplier

# ── Identifiers ───────────────────────────────────────────────────────


def safe_id(s):
    """Make a string safe for XML id attribute."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(s or "id0"))
    return ("_" + s) if s[0].isdigit() else s


def unique_component_id(name, taken, kind="component"):
    """A component id derived from *name* that no other component already holds.

    Components are named after their Dynamics, so two differently parameterised uses of one Dynamics would collide and the second would be dropped.

    Args:
        name: the Dynamics name to derive the id from.
        taken: ids already assigned; the returned id is added to it.
        kind: what is being named, for the disambiguation warning.

    Returns:
        ``safe_id(name)``, or that with a numeric suffix when it is taken.
    """
    base = safe_id(name)
    ident = base
    n = 2
    while ident in taken:
        ident = f"{base}_{n}"
        n += 1
    if ident != base:
        warnings.warn(
            f"{kind} {base!r} is used with more than one set of parameters; "
            f"emitting the additional one as {ident!r}. Give each parameterisation "
            f"its own entry in the dynamics library to choose the names yourself.",
            stacklevel=2,
        )
    taken.add(ident)
    return ident


# ── Parameter helpers ─────────────────────────────────────────────────


def merge_params(*param_dicts):
    """Merge parameter dicts with later dicts overriding earlier ones.

    The canonical order is dynamics-library → node/edge → per-connection, i.e. the same precedence as the ``{**dyn, **node, **edge}`` spreads the backends build by hand. Keys are taken verbatim; values are not copied.
    """
    merged = {}
    for d in param_dicts:
        if d:
            merged.update(d)
    return merged


# ── Connectivity-rule lowering ────────────────────────────────────────


def connectivity_pairs(rule, src_size, tgt_size):
    """Expand a population-level connectivity rule into ``(src_idx, tgt_idx)`` pairs.

    Given the ``ConnectivityRule`` (or its string value) and the source/target population sizes, yields the local cell-index pairs a projection (or per-cell input list) enumerates.  This is the "allToAll lowering": the user declares one population-to-population Edge and the adapter generates the i x j connection set, so no O(N**2) explicit edges ever appear in the input.

    Self-connection filtering (the diagonal of a self-projection) is applied by the caller on the resolved global cell indices, so this helper simply yields the raw pattern.

    Args:
        rule: Connectivity pattern (``all_to_all`` or ``one_to_one``).
        src_size: Number of cells in the source population.
        tgt_size: Number of cells in the target population.

    Yields:
        ``(src_idx, tgt_idx)`` local cell indices.
    """
    rule_name = str(rule).lower().replace("-", "_")
    src_size = int(src_size)
    tgt_size = int(tgt_size)
    if rule_name == "one_to_one":
        for i in range(min(src_size, tgt_size)):
            yield i, i
        return
    # Default / all_to_all: fully connected projection.
    for i in range(src_size):
        for j in range(tgt_size):
            yield i, j


# ── The backend-neutral connection contract ───────────────────────────


class ConnectionRecord(TypedDict, total=False):
    """One lowered cell-to-cell connection — the contract every backend consumes.

    Produced by connectivity-rule expansion; a plain ``dict`` at runtime so templates and adapters can index it directly. The neutral core is ``from_pop``/``from_idx`` → ``to_pop``/``to_idx`` through ``synapse`` with an optional per-connection ``weight``/``delay``. ``from_rule`` records whether the connection came from a lowered ``connectivity`` rule (vs a single explicit edge). Backends may attach their own keys (e.g. ``conn_class`` for LEMS projection classification) without changing this core.
    """

    from_pop: str
    from_idx: int
    to_pop: str
    to_idx: int
    synapse: str
    weight: float | None
    delay: object
    from_rule: bool


# ── NeuroML type vocabulary ───────────────────────────────────────────

POISSON_INPUT_TYPES = frozenset({"poissonFiringSynapse", "transientPoissonFiringSynapse"})
"""Poisson inputs that are a spike source and a synapse in one component, applied onto a cell's synapses rather than wired by a projection."""

CURRENT_INPUT_TYPES = POISSON_INPUT_TYPES | frozenset(
    {
        "pulseGenerator",
        "pulseGeneratorDL",
        "compoundPulseGenerator",
        "compoundInput",
        "sineGenerator",
        "sineGeneratorDL",
        "rampGenerator",
        "rampGeneratorDL",
        "voltageClamp",
        "voltageClampTriple",
        "timedSynapticInput",
    }
)
"""NeuroML inputs that act on a cell from outside it: a standalone component injected through an ``explicitInput``, never a population."""

EVENT_SOURCE_TYPES = frozenset(
    {
        "spikeGenerator",
        "spikeGeneratorRandom",
        "spikeGeneratorRefPoisson",
        "spikeGeneratorPoisson",
        "spikeArray",
        "SpikeSourcePoisson",
    }
)
"""NeuroML spike sources: populations of their own that carry no membrane, connected to their targets by a projection."""

ONSET_PARAMETERS = {
    "pulseGenerator": "delay",
    "pulseGeneratorDL": "delay",
    "sineGenerator": "delay",
    "sineGeneratorDL": "delay",
    "rampGenerator": "delay",
    "rampGeneratorDL": "delay",
    "voltageClamp": "delay",
    "voltageClampTriple": "delay",
    "transientPoissonFiringSynapse": "delay",
    "SpikeSourcePoisson": "start",
}
"""The parameter that places each timed NeuroML input on the clock. Moving it later by a settle moves the whole input with it: every other time the input reads (its duration, a sine's phase, a ramp's slope) is relative to it."""

TIMED_BY_CHILDREN = frozenset({"compoundInput", "compoundPulseGenerator", "spikeArray", "timedSynapticInput"})
"""NeuroML inputs whose timing lives on their children: the generators a compound input sums, and the spike times a ``spikeArray`` or ``timedSynapticInput`` lists."""

STATIONARY_SOURCES = frozenset({"poissonFiringSynapse", "spikeGeneratorPoisson"})
"""Memoryless spike sources that run from the start of the run with no onset: a measured window sees the same process whether a settle precedes it or not."""


def settle_refusal(nml_type):
    """Why a NeuroML input of *nml_type* cannot be run behind a settle, or ``None`` when it can.

    An input can when a settle has nothing to move (a stationary source), or when the time it declares can be moved later by the settle: an onset parameter (`ONSET_PARAMETERS`) or children that carry their own times (`TIMED_BY_CHILDREN`). The rest are timed from the start of the run with no onset to move, so a settle would change what the measured window sees: a ``spikeGenerator`` fires on a period counted from the run start, and a ``spikeGeneratorRandom`` or ``spikeGeneratorRefPoisson`` starts its renewal process there. A type outside the input vocabulary is not an input, and is not refused here.
    """
    if nml_type not in CURRENT_INPUT_TYPES | EVENT_SOURCE_TYPES:
        return None
    if nml_type in ONSET_PARAMETERS or nml_type in TIMED_BY_CHILDREN or nml_type in STATIONARY_SOURCES:
        return None
    return f"a {nml_type} is timed from the start of the run and declares no onset to move past the settle"


def shift_onset(quantity, settle):
    """A declared time, moved from the measurement clock onto the run's.

    A run's clock opens at the start of integration, which is the start of the settle; a recipe declares its onsets against the measured window, which opens a settle later. The shift is exact and in the quantity's OWN unit, so a delay declared in seconds and a settle counted in milliseconds compose rather than meeting in whichever of the two the caller happened to write.

    Args:
        quantity: The declared time, as ``(value, unit)``.
        settle: The settle, as ``(value, unit)``.

    Returns:
        ``(value, unit)`` with the value moved later by the settle; the quantity unchanged when the settle is zero.

    Raises:
        ValueError: If either unit is not curated, so the two cannot be converted into one another.
    """
    value, unit = quantity
    if not settle[0]:
        return quantity
    scales = [unit_multiplier(str(u)) if u else None for u in (unit, settle[1])]
    if None in scales:
        raise ValueError(
            f"a declared onset in {unit!r} cannot be placed on the run clock: {unit!r} or the settle's {settle[1]!r} is "
            "not a curated unit, so the settle prepended to the measured window cannot be converted into it."
        )
    return (float(Fraction(value) + Fraction(settle[0]) * scales[1] / scales[0]), unit)


def nml_type(dynamics, default=None):
    """The NeuroML type a ``Dynamics`` names by its ``neuroml:<type>`` iri, or *default* when its iri names none."""
    iri = getattr(dynamics, "iri", None) or ""
    return iri.split(":", 1)[1] if iri.startswith("neuroml:") else default


# ── Node grouping and role classification ─────────────────────────────


def node_dynamics_name(node, default_dyn_name):
    """The ``Dynamics`` name a node runs.

    ``Node.dynamics`` is a name-reference slot, so it may arrive as a bare name or as a resolved ``Dynamics``; a node that declares none falls back to *default_dyn_name* — the experiment's top-level dynamics. One rule, shared by every backend, so they cannot disagree about which model a node runs.
    """
    node_dyn = getattr(node, "dynamics", None)
    if not node_dyn:
        return default_dyn_name
    return getattr(node_dyn, "name", None) or str(node_dyn)


def group_nodes_by_dynamics(nodes, default_dyn_name):
    """Group nodes by their ``Dynamics`` name, preserving first-encounter order."""
    from collections import OrderedDict

    groups = OrderedDict()
    for node in nodes:
        groups.setdefault(node_dynamics_name(node, default_dyn_name), []).append(node)
    return groups


def classify_node_role(dyn_name, dyn_lib_obj, vocab):
    """Classify a node group as a cell, current-input, or event-source.

    The biological type is read from ``Dynamics.iri`` (``neuroml:<type>``); a Dynamics without such an iri is a plain cell named by itself. *vocab* is the backend's role vocabulary — a mapping with ``current_input`` and ``event_source`` keys to sets of type names — so the same lowering serves any backend by swapping the sets.

    Returns ``(role, nml_type)`` with role one of ``"cell"``, ``"current_input"``, ``"event_source"``.
    """
    type_name = nml_type(dyn_lib_obj, dyn_name)
    if type_name in vocab.get("current_input", ()):
        return "current_input", type_name
    if type_name in vocab.get("event_source", ()):
        return "event_source", type_name
    return "cell", type_name


# ── Connectivity-rule expansion (the allToAll lowering) ───────────────


def expand_input_targets(tgt_base, tgt_size, rule):
    """Local target cell indices an input edge fans out to.

    A ``connectivity`` rule attaches an independent copy of the input component to every target cell (rule expansion over a size-1 "source"); without a rule the input hits the node's base cell only.
    """
    if rule:
        return [tgt_base + j for _i, j in connectivity_pairs(rule, 1, tgt_size)]
    return [tgt_base]


def expand_edge_connections(edge, *, src_pop, src_base, tgt_pop, tgt_base, src_size, tgt_size):
    """Yield ``(from_idx, to_idx, from_rule)`` for one synapse edge.

    An Edge with a ``connectivity`` rule is a population-to-population projection: expand it into the individual cell-to-cell connections here, skipping the diagonal of a self-projection when ``allow_self_connections`` is False. Without a rule the Edge is a single explicit cell-to-cell connection. ``from_rule`` marks whether the connection came from a lowered rule.
    """
    rule = getattr(edge, "connectivity", None)
    if rule:
        allow_self = getattr(edge, "allow_self_connections", True)
        same_pop = src_pop == tgt_pop
        index_pairs = connectivity_pairs(rule, src_size, tgt_size)
        from_rule = True
    else:
        index_pairs = [(0, 0)]
        allow_self = True
        same_pop = False
        from_rule = False
    for _si, _tj in index_pairs:
        from_idx = src_base + _si
        to_idx = tgt_base + _tj
        if same_pop and from_idx == to_idx and not allow_self:
            continue
        yield from_idx, to_idx, from_rule


# ── Population index assignment ───────────────────────────────────────


def assign_cell_population(dyn_name, group_nodes, node_pop_map, node_size_map):
    """Assign a cell population id and per-node base indices, filling the maps.

    Each node contributes ``Node.size`` cells laid out contiguously; the running base index lets an edge address an individual cell within the population.
    Mutates *node_pop_map* (``node_id -> (pop_id, base)``) and *node_size_map* (``node_id -> size``) in place, and returns ``(pop_id, node_ids, size)``.
    """
    pop_id = safe_id(dyn_name) + "_pop"
    node_ids = []
    base = 0
    for idx, node in enumerate(group_nodes):
        nid = getattr(node, "id", idx)
        nsize = int(getattr(node, "size", 1) or 1)
        node_pop_map[nid] = (pop_id, base)
        node_size_map[nid] = nsize
        node_ids.append(nid)
        base += nsize
    return pop_id, node_ids, base
