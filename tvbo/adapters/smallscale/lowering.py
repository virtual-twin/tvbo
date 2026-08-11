"""Backend-neutral network lowering for small-scale simulators.

The functions here turn a TVB-O ``Network`` (nodes with ``size``, edges with a ``connectivity`` rule, ``Dynamics``/``Coupling``/``Event`` biology) into the two structures every point-neuron backend needs:

* **populations** — nodes grouped by their ``Dynamics``, each a block of
  ``Node.size`` cells, with a stable base index per node so edges can address individual cells; and
* **connections** — the explicit cell-to-cell :class:`ConnectionRecord` set that a
  ``connectivity`` rule (``all_to_all``/``one_to_one``) lowers to, with self-connections filtered and per-connection ``weight``/``delay`` extracted.

Everything here is independent of *how* a backend emits a synapse — that (LEMS XML, Brian2 ``Synapses``, …) stays in the backend adapter. The backend injects its own
*role vocabulary* (which ``Dynamics`` are cells vs current sources vs event
sources) so the same lowering serves NeuroML, Brian2 and the rest unchanged.
"""

from __future__ import annotations

import re
import warnings
from typing import TypedDict

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

    The canonical order is dynamics-library → node/edge → per-connection, i.e.
    the same precedence as the ``{**dyn, **node, **edge}`` spreads the backends build by hand. Keys are taken verbatim; values are not copied.
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


# ── Node grouping and role classification ─────────────────────────────


def node_dynamics_name(node, default_dyn_name):
    """The ``Dynamics`` name a node runs.

    ``Node.dynamics`` is a name-reference slot, so it may arrive as a bare name or as a resolved ``Dynamics``; a node that declares none falls back to
    *default_dyn_name* — the experiment's top-level dynamics. One rule, shared by
    every backend, so they cannot disagree about which model a node runs.
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
    dyn_iri = getattr(dyn_lib_obj, "iri", "") or ""
    nml_type = dyn_iri.split(":", 1)[1] if dyn_iri.startswith("neuroml:") else dyn_name
    if nml_type in vocab.get("current_input", ()):
        return "current_input", nml_type
    if nml_type in vocab.get("event_source", ()):
        return "event_source", nml_type
    return "cell", nml_type


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

    An Edge with a ``connectivity`` rule is a population-to-population projection:
    expand it into the individual cell-to-cell connections here, skipping the diagonal of a self-projection when ``allow_self_connections`` is False.
    Without a rule the Edge is a single explicit cell-to-cell connection.
    ``from_rule`` marks whether the connection came from a lowered rule.
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
