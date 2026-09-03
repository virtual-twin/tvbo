"""Brain-network connectivity classes for TVBO.

Defines [`Network`](#tvbo.classes.network.Network) and its matrix-style subclass [`Connectome`](#tvbo.classes.network.Connectome), which carry the structural connectivity (weights and tract lengths), parcellation, nodes/edges and declarative transforms of a virtual brain. Includes constructors that load networks from HDF5/YAML database files and from BIDS derivatives, JAX pytree registration so a network can flow through differentiable simulations, and graph-Laplacian coupling primitives.
"""

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from jax import Array as JaxArray
from jsonasobj2 import as_dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Apply the JAX Metal-fallback guard before any Network-level JAX compute. Idempotent and cheap (jax already imported); kept out of ``import tvbo``. See tvbo.__init__._configure_jax_backend.
from tvbo import _configure_jax_backend as _cfg_jax
from tvbo.utils.pytree import register as register_pytree
from tvbo.utils.pytree import static_spec

_cfg_jax()


from tvbo.data.registry import database_dir
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.utils import edge_param, keyed_items, transform_target
from tvbo.utils.yaml_loader import resolve_edge_var_aliases

# HDF5+YAML network files — resolved via registry (works for pip & editable installs)
NETWORK_DIR = database_dir("Network")


@contextmanager
def _source_dir_on_path(source_dir):
    """Temporarily prepend ``source_dir`` to ``sys.path`` so modules living beside a study YAML (graph builders, transform callables) import by bare name.

    Skips insertion when ``source_dir`` is falsy or already on the path, and only removes what it added.
    """
    src = str(Path(source_dir).resolve()) if source_dir else None
    added = bool(src) and src not in os.sys.path
    if added:
        os.sys.path.insert(0, src)
    try:
        yield
    finally:
        if added:
            try:
                os.sys.path.remove(src)
            except ValueError:
                pass


_WEIGHT_TARGETS = ("weight", "weights", "sc")
_LENGTH_TARGETS = ("length", "lengths")


def _is_weight_name(name) -> bool:
    """Whether *name* spells the connection-weight property."""
    low = str(name).lower()
    return low in _WEIGHT_TARGETS or low in _WEIGHT_MEASURES


class LazyMaterializationWarning(UserWarning):
    """A `Network` entered a JAX transformation with arrays its companion offers still unread.

    The solver then traces nothing and reads the connectome as a constant inside the trace. Set ``TVBO_JAX_STRICT`` to make it an error.
    """


def _array_key(name: str) -> str:
    """The companion dataset path an array is kept under: ``weight`` is ``edges/weight``, a path is itself."""
    return name if "/" in name else f"edges/{name}"


def _edge_name(key: str) -> str | None:
    """The edge a companion path names, or ``None`` where it names something else — the inverse of :func:`_array_key`.

    ``edges/weight`` is the ``weight`` edge; ``mesh/vertices`` and ``edges/weight/edge_parameters/length`` are not edge matrices, and a caller that reads the resident keys as edge names offers them as matrices nothing can serve.
    """
    return key[len("edges/") :] if key.startswith("edges/") and key.count("/") == 1 else None


def _is_length_name(name) -> bool:
    """Whether *name* spells the tract-length property."""
    low = str(name).lower()
    return low in _LENGTH_TARGETS or low in _LENGTH_MEASURES


def _alias_group(name) -> tuple:
    """Every spelling of the edge property *name* names, canonical spellings first.

    The ONE place spellings are grouped. `Network.matrix` resolves a matrix through it and `Network.transforms_for` selects transforms through it, so a lookup and its transform cannot disagree about whether two names mean the same property — the failure that let `matrix("sc")` return the weight array with its declared transforms silently skipped.
    Names are lowercase; callers match source keys case-insensitively.
    """
    if _is_weight_name(name):
        return _WEIGHT_TARGETS + tuple(sorted(_WEIGHT_MEASURES.difference(_WEIGHT_TARGETS)))
    if _is_length_name(name):
        return _LENGTH_TARGETS + tuple(sorted(_LENGTH_MEASURES.difference(_LENGTH_TARGETS)))
    return (str(name).lower(),)


def _warn_superseded_accessor(old: str, new: str) -> None:
    """Emit the deprecation for a connectivity accessor superseded by `Network.matrix`."""
    warnings.warn(
        f"Network.{old} is deprecated; use Network.{new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def graph_laplacian(M):
    """Combinatorial graph Laplacian ``L = W - diag(rowsum(W))`` of a weight matrix.

    A standard network primitive (diffusive coupling operator): every row of ``L`` sums to zero. Referenced declaratively from a `Network.transforms` entry via ``callable: {module: tvbo.classes.network, name: graph_laplacian}`` — e.g. for a delay/diffusion-coupled Hopf network whose coupling matrix is the Laplacian of the (normalised) connectome. Cannot be expressed in the elementwise symbolic transform path because it needs a diagonal built from the row sums.
    """
    return M - jnp.diag(M.sum(axis=1))


def normalized_graph_laplacian(M):
    """Graph Laplacian of the max-normalised weight matrix: ``L(W / max(W))``.

    The coupling operator for a diffusion-coupled network whose global coupling strength ``G`` is expressed in the max-normalised connectome scale (so ``G`` stays O(0.01–0.1) regardless of the raw streamline-count magnitude). Referenced via ``callable: {module: tvbo.classes.network, name: normalized_graph_laplacian}``.
    Equivalent to the reference two-liner ``W = W / W.max(); L = W - diag(W.sum(1))``.
    """
    return graph_laplacian(M / jnp.max(M))


try:
    from bids.layout import BIDSLayout  # noqa: F401  # optional dep probe

    connectome_data = None
    available_connectomes = []
except ImportError:
    connectome_data = None
    available_connectomes = []


def _find_network_sidecar(
    atlas: str,
    tractogram: str,
    segmentation: str | None = None,
    scale: str | None = None,
) -> Path | None:
    """Find the YAML sidecar for a given atlas + tractogram combination.

    Searches tvbo/database/networks/ for files matching the atlas and rec- entities.
    When ``segmentation`` / ``scale`` are given, also requires ``seg-<segmentation>`` and ``scale-<scale>``. Falls back to partial matching if exact match fails.
    """
    for f in NETWORK_DIR.glob("*.yaml"):
        stem = f.stem
        if f"atlas-{atlas}" not in stem or f"rec-{tractogram}" not in stem:
            continue
        if segmentation is not None and f"seg-{segmentation}" not in stem:
            continue
        if scale is not None and f"scale-{scale}" not in stem:
            continue
        # Only skip seg- variants when the caller did NOT request one.
        if segmentation is None and "seg-" in stem:
            continue
        return f
    # Fallback: try with normalized atlas names
    atlas_map = {
        "Schaefer1000": "Schaefer2018",
        "Schaefer100017Networks": "Schaefer2018",
        "hcpmmp1": "HCPMMP1",
        "hcpmmp1ordered": "HCPMMP1",
        "DesikanKillianyranked": "DesikanKilliany",
        "Destrieuxranked": "Destrieux",
    }
    mapped = atlas_map.get(atlas, atlas)
    if mapped != atlas:
        return _find_network_sidecar(mapped, tractogram, segmentation, scale)
    return None


def _parse_bids_entities(stem: str) -> dict[str, str]:
    """Extract BIDS key-value entities from a filename stem.

    E.g. ``"tpl-MNI_atlas-DK_rec-dTOR_scale-100_desc-SC_relmat"`` → ``{"tpl": "MNI", "atlas": "DK", "rec": "dTOR", "scale": "100", "desc": "SC"}``
    """
    import re

    return dict(re.findall(r"(?:^|_)([a-zA-Z]+)-([^_]+)", stem))


def _filter_networks_by_entities(entities: dict[str, str]) -> list:
    """Return network YAML paths whose BIDS entities match all given filters."""
    matches = []
    for f in NETWORK_DIR.glob("*.yaml"):
        file_ents = _parse_bids_entities(f.stem)
        if all(file_ents.get(k) == str(v) for k, v in entities.items()):
            matches.append(f)
    return matches


# Known BEP017 weight-like and length-like measure names for auto-classification
_WEIGHT_MEASURES = {"streamlinecount", "streamlinedensity", "weight", "weights"}
_LENGTH_MEASURES = {
    "tractlength",
    "tractlengths",
    "length",
    "lengths",
    "tract_length",
    "tract_lengths",
}


def _discover_bids_measures(bids_dir) -> list[str]:
    """Auto-discover structural measures from BEP017 relmat files.

    Parses ``meas-<name>`` from filenames and classifies into [weight_measure, length_measure] order by matching against known naming conventions.  Falls back to file order if names are unknown.
    """
    import re

    bids_dir = Path(bids_dir)
    relmat_files = list(bids_dir.glob("*_relmat.dense.tsv")) + list(bids_dir.glob("*_relmat.tsv"))
    # Extract unique measure names
    measures = []
    seen = set()
    for f in relmat_files:
        m = re.search(r"meas-([^_]+)", f.name)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            measures.append(m.group(1))

    if not measures:
        return ["weight", "tract_length"]  # default canonical names

    # Classify: put weight-like first, length-like second
    weight_names = [m for m in measures if m.lower() in _WEIGHT_MEASURES]
    length_names = [m for m in measures if m.lower() in _LENGTH_MEASURES]
    other = [m for m in measures if m.lower() not in _WEIGHT_MEASURES and m.lower() not in _LENGTH_MEASURES]

    result = []
    if weight_names:
        result.append(weight_names[0])
    if length_names:
        result.append(length_names[0])
    # Append any remaining (unknown measures in file order)
    result.extend(other)
    # If no known names matched, return all in file order
    return result or measures


def _bids_measure_units(bids_dir, measures) -> dict:
    """`MeasureUnits` from each measure's BEP017 sidecar, normalised onto `UnitEnum`.

    `to_bids` has always *written* this field from an edge's `unit`; nothing read it back, so a connectome that round-tripped through BIDS came home unitless and fell to the `mm` default no matter what its sidecar said.

    Entries whose spelling does not normalise are kept verbatim rather than dropped: the value is what the dataset claims, and reporting an unrecognised unit is a better answer than reporting none.
    """
    import json

    from tvbo.utils.units import normalize_unit

    bids_dir = Path(bids_dir)
    units = {}
    for measure in measures:
        for sidecar in sorted(bids_dir.glob(f"*meas-{measure}_relmat*.json")):
            try:
                declared = json.loads(sidecar.read_text()).get("MeasureUnits")
            except (OSError, ValueError):
                continue
            if declared:
                units[measure] = normalize_unit(declared) or declared
            break
    return units


def _declared_length_unit(measure_units: dict, length_measure) -> str | None:
    """The length measure's declared unit, if it really is a length.

    A network's `distance_unit` divides its `conduction_speed` to give delays, so accepting a non-length here would produce delays in a unit that means nothing — worse than the `mm` default, which is at least wrong in a known direction.
    `streamlineCount` is declared `arbitrary` in the shipped sidecars and is the second structural measure in datasets that ship no tract lengths, so this is reached in practice rather than defensively.
    """
    from tvbo.utils.units import unit_dimensions

    declared = measure_units.get(length_measure) if length_measure else None
    if declared is None:
        return None
    return declared if unit_dimensions(declared) == {"meter": 1} else None


def get_normative_connectome_data(
    atlas: str,
    tractogram: str = "dTOR",
    segmentation: str | None = None,
    scale: str | None = None,
    with_nodes: bool = False,
):
    """Load normative connectivity matrices from tvbo/database/networks/ HDF5 files.

    Parameters
    ----------
    atlas : str
        Name of the brain parcellation atlas (e.g., "DesikanKilliany", "Destrieux")
    tractogram : str
        Tractogram/reconstruction pipeline (e.g., "dTOR", "MghUscHcp32", "PPMI85")
    segmentation, scale : str, optional
        BIDS ``seg-`` and ``scale-`` entity values used to disambiguate
        sub-resolutions of the same atlas (e.g. Schaefer2018 7Networks/1000).

    Returns:
    -------
    weights : np.ndarray
        Connection strength matrix (N x N)
    lengths : np.ndarray or None
        Tract length matrix (N x N), or None if not available
    nodes : list of Node, optional
        Only when ``with_nodes=True``: the sidecar's labelled + positioned
        region nodes, so the network is keyed by region (alignment by label,
        never by position). ``None`` if the sidecar declares no nodes.

    Examples:
    --------
    ```python
    weights, lengths = get_normative_connectome_data("DesikanKilliany", "dTOR")
    weights, lengths = get_normative_connectome_data(
        "Schaefer2018", "dTOR", segmentation="7Networks", scale="1000"
    )
    ```
    """
    sidecar = _find_network_sidecar(atlas, tractogram, segmentation, scale)
    if sidecar is None:
        raise FileNotFoundError(
            f"No network found for atlas={atlas}, tractogram={tractogram}, seg={segmentation}, scale={scale} in {NETWORK_DIR}"
        )

    from tvbo.data.network_io import load_network

    net = load_network(sidecar)
    store = getattr(net, "_store", None)
    if store is None:
        raise FileNotFoundError(f"No companion data file for {sidecar.name}")
    arrays = store.arrays
    weights = arrays.get("weight")
    if weights is None:
        weights = arrays.get("weights")
    lengths = arrays.get("length")
    if lengths is None:
        lengths = arrays.get("lengths")
    if weights is None:
        raise ValueError(f"No 'weight' edge found in {sidecar.name}")
    if with_nodes:
        return weights, lengths, (getattr(net, "nodes", None) or None)
    return weights, lengths


def _network_ref_string(net: "Network") -> str:
    """Extract a serializable reference string from a Network object.

    Resolution order:
    1. ``_save_path`` — filename from the most recent ``save()`` call
    2. ``data_file`` — companion filename (strip extension, add ``.yaml``)
    3. ``label`` — human-readable label
    4. Raise ``ValueError`` — can't derive a reference
    """
    # 1. Explicit save path (set by save())
    try:
        sp = object.__getattribute__(net, "_save_path")
        if sp:
            return str(sp)
    except AttributeError:
        pass
    # 2. data_file companion (dk_sc.h5 → dk_sc.yaml)
    df = getattr(net, "data_file", None)
    if df:
        return str(Path(df).with_suffix(".yaml").name)
    # 3. Label
    lbl = getattr(net, "label", None)
    if lbl:
        return str(lbl)
    raise ValueError(
        "Cannot derive a reference string for the parent Network. "
        "Either save it first (net.save(...)), set its label, or pass "
        "a string path instead."
    )


def _iri_local(iri: str) -> str:
    """Local part of a CURIE/IRI: the text after the first ``prefix:``."""
    return iri.split(":", 1)[-1] if ":" in iri else iri


def _backfill_name_from_iri(obj: Any, nested_key: str | None = None) -> None:
    """If ``obj`` (or ``obj[nested_key]``) is a dict with ``iri`` but no ``name``, derive ``name`` from the IRI's local part. Mutates in place."""
    if obj is None:
        return
    target = obj
    if nested_key is not None:
        target = obj.get(nested_key) if isinstance(obj, dict) else getattr(obj, nested_key, None)
    if not isinstance(target, dict):
        return
    if target.get("name"):
        return
    iri = target.get("iri")
    if not iri:
        return
    target["name"] = _iri_local(iri)


@register_pytree
class Network(tvbo_datamodel.Network):
    """A brain network: parcellation, connectome, per-node dynamics, and coupling.

    The spatial substrate of a `SimulationExperiment`. A `Network` ties an atlas/parcellation to a tractogram (structural connectivity matrix + optional path lengths) and, optionally, per-node `Dynamics` overrides and node-level coupling parameters.

    Construct inline, by IRI (resolved against the curated database), or from a NumPy / pandas matrix via [`Network.from_array`](#tvbo.classes.network.Network.from_array).

    Examples:
        ```python
        net = Network(
            parcellation={"atlas": {"iri": "tvbo:DesikanKilliany"}},
            tractogram={"iri": "tvbo:dTOR"},
        )
        ```

    See the [Network specification](/2-specify/Networks/Network.qmd) for the slot-by-slot reference and the [`Connectome`](#tvbo.classes.network.Connectome) subclass for matrix-style networks without an explicit parcellation.
    """

    @property
    def number_of_regions(self) -> int:
        """Deprecated alias for number_of_nodes."""
        return self.number_of_nodes

    @number_of_regions.setter
    def number_of_regions(self, value: int) -> None:
        """Set `number_of_nodes` through the deprecated `number_of_regions` alias."""
        self.number_of_nodes = value

    def __init__(self, **kwargs: Any) -> None:
        # Strip internal-only flags that may leak in from serialised forms.
        for _internal in ("_resolved",):
            kwargs.pop(_internal, None)

        # A top-level `iri` is a semantic pointer to a curated network in the database (e.g. a `*_relmat` structural-connectivity file). The parent LinkML Network has no `iri` slot, so resolve it here into a `data_file` reference and let `_resolve_from_data_file` load the connectivity. Inline-authored slots (transforms, parameters, coupling, node_template, …) are preserved on self. Skip when the caller already supplied explicit connectivity.
        _iri = kwargs.pop("iri", None)
        if _iri and not any(kwargs.get(k) for k in ("data_file", "nodes", "edges", "edge_matrix_files", "bids_dir")):
            _resolved_path = self._resolve_network_iri(_iri)
            if _resolved_path is not None:
                kwargs["data_file"] = _resolved_path

        # Resolve deprecated number_of_regions -> number_of_nodes
        if "number_of_regions" in kwargs:
            kwargs.setdefault("number_of_nodes", kwargs.pop("number_of_regions"))
            kwargs.pop("number_of_regions", None)

        # Derive name from iri where missing (so the dataclass post-init doesn't raise MissingRequiredField on iri-only construction).
        _backfill_name_from_iri(kwargs.get("parcellation"), nested_key="atlas")
        _backfill_name_from_iri(kwargs.get("tractogram"))

        # Check if nodes/edges or edge_matrix_files are already provided
        has_nodes = "nodes" in kwargs and kwargs["nodes"]
        has_edges = "edges" in kwargs and kwargs["edges"]
        has_edge_files = "edge_matrix_files" in kwargs and kwargs["edge_matrix_files"]

        # Load edge_matrix_files into edges
        if has_edge_files and not has_edges:
            edge_files = kwargs["edge_matrix_files"]
            # Resolve file path relative to YAML source
            from tvbo.classes.experiment import SimulationExperiment

            source_dir = None
            pending = getattr(SimulationExperiment, "_pending_source_file", None)
            if pending:
                source_dir = os.path.dirname(pending)

            emf = edge_files[0]
            fpath = str(emf)
            if source_dir and not os.path.isabs(fpath):
                fpath = os.path.join(source_dir, fpath)

            w_arr = np.loadtxt(fpath, delimiter=",")
            n_nodes = w_arr.shape[0]
            edges = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if w_arr[i, j] != 0:
                        edges.append(
                            tvbo_datamodel.Edge(
                                source=i,
                                target=j,
                                parameters=[tvbo_datamodel.Parameter(name="weight", value=float(w_arr[i, j]))],
                            )
                        )
            kwargs["edges"] = edges
            kwargs["number_of_nodes"] = n_nodes
            has_edges = True

        # Normalise an inline string parcellation -> Parcellation dict so the parent constructor accepts it. Materialisation of normative data happens later in self._resolve().
        if not has_nodes and not has_edges and not has_edge_files:
            if isinstance(kwargs.get("parcellation"), str):
                kwargs["parcellation"] = tvbo_datamodel.Parcellation(
                    label=kwargs["parcellation"],
                    atlas=tvbo_datamodel.BrainAtlas(name=kwargs["parcellation"]),
                )._as_dict

        # Infer n_nodes from nodes if present (authoritative source)
        if "nodes" in kwargs and kwargs["nodes"]:
            n_nodes = len(kwargs["nodes"])
            declared = kwargs.get("number_of_nodes") or kwargs.get("number_of_regions")
            if declared and declared != n_nodes:
                import warnings

                warnings.warn(
                    f"number_of_nodes={declared} doesn't match len(nodes)={n_nodes}. Using {n_nodes} from nodes list.",
                    stacklevel=2,
                )
            kwargs["number_of_nodes"] = n_nodes
        # Create default nodes if number_of_nodes is set but nodes list is empty
        elif kwargs.get("number_of_nodes") and not kwargs.get("nodes"):
            n_nodes = kwargs["number_of_nodes"]
            kwargs["nodes"] = [tvbo_datamodel.Node(id=i, label=f"node_{i}") for i in range(n_nodes)]

        # Fold before the base constructor and the edge_template snapshot see them.
        resolve_edge_var_aliases(kwargs.get("edges"))
        resolve_edge_var_aliases(kwargs.get("edge_template"))

        # A loader that will attach connectivity itself sets this, then calls `_resolve` once the arrays are in place. Without it the constructor-time `_resolve` would run against a half-built object: `data_file` is an INDIRECT reference (see `_resolve_from_data_file`, which reads the companion's own sidecar and would recurse into the very load in progress), so loaders must strip it — and a sidecar that also declares `parcellation:` would then fall through to the normative-database branch and cache an atlas connectome that shadows the companion's real matrices. Deferring resolves that ordering rather than special-casing the branch.
        _defer_connectivity = bool(kwargs.pop("_defer_connectivity", False))

        # `node_template` is a partial Node applied to every materialized node (see _expand_node_template). The parent constructor builds it as a real `Node`, which requires `id`; inject a sentinel so construction succeeds. We also stash the raw spec dict (sans sentinel) so template expansion works from plain dicts rather than re-serialising objects.
        import copy as _copy

        _nt = kwargs.get("node_template")
        _nt_spec = None
        if isinstance(_nt, dict):
            _nt_spec = _copy.deepcopy(_nt)
            _nt_spec.pop("id", None)
            if _nt.get("id") is None:
                _nt["id"] = -1
        _et = kwargs.get("edge_template")
        _et_spec = _copy.deepcopy(_et) if isinstance(_et, dict) else None

        # Resolve Dynamics slot aliases (components → modes) in network dynamics so the LinkML loader can construct Dynamics objects correctly.
        _net_dynamics = kwargs.get("dynamics")
        if isinstance(_net_dynamics, dict):
            from tvbo.classes.dynamics import _resolve_dynamics_aliases

            for _dk, _dv in _net_dynamics.items():
                if isinstance(_dv, dict):
                    _resolve_dynamics_aliases(_dv)

        # `Edge.coupling` is a name-reference slot at the datamodel level (its range is the identified `Coupling` class, `inlined: false`), so the base constructor would stringify an *inline* coupling definition into a bare `CouplingName`, discarding its coupling_function / parameters. Detach inline (dict) definitions here so the base constructor doesn't see them, then reattach as real `Coupling` objects below. Bare-string references are left untouched and resolve by name as before.
        _detached_couplings = self._detach_inline_edge_couplings(kwargs.get("edges"))

        super().__init__(**kwargs)

        self._reattach_inline_edge_couplings(_detached_couplings)

        # Stash raw template specs (plain dicts) for expansion in _resolve.
        object.__setattr__(self, "_node_template_spec", _nt_spec)
        object.__setattr__(self, "_edge_template_spec", _et_spec)

        # Sync number_of_nodes from nodes list (authoritative after init)
        if self.nodes:
            n_nodes = len(self.nodes)
            if self.number_of_nodes != n_nodes:
                self.number_of_nodes = n_nodes
        # Create default nodes if number_of_nodes is set but nodes list is empty
        elif self.number_of_nodes and not self.nodes:
            self.nodes = [tvbo_datamodel.Node(id=i, label=f"node_{i}") for i in range(self.number_of_nodes)]

        # Ensure conduction_speed exists in parameters
        if "conduction_speed" not in (self.parameters or {}):
            self.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
                name="conduction_speed", label="v", value=3.0, unit="mm_per_ms"
            )

        # Materialise connectivity from the declarative spec (parcellation, data_file, bids_dir, graph_generator). Idempotent; safe to call multiple times. See Network._resolve. Pick up the YAML source directory from the SimulationExperiment context (set by from_file) so relative paths resolve correctly even when Network is built as a kwarg inside SimulationExperiment.__init__.
        from tvbo.classes.experiment import SimulationExperiment as _SE

        _source_dir = None
        _pending = getattr(_SE, "_pending_source_file", None)
        if _pending:
            _source_dir = os.path.dirname(_pending)
        # Persist the source dir so lazily-applied callable transforms (resolved in _apply_transform long after load) can import modules beside the YAML.
        if _source_dir:
            self._source_dir = _source_dir
        if not _defer_connectivity:
            self._resolve(source_dir=_source_dir)

        # Runtime default: a Network with no nodes, no declared count, and no connectome to resolve is still usable as a single-node network. The serialized schema default is None (number_of_nodes is "derived from nodes if not set"), so this 1-node fallback lives only on the Python side — applied AFTER resolution so connectome-backed networks keep their resolved size rather than being pinned to 1.
        if not self.nodes and not self.number_of_nodes:
            self.number_of_nodes = 1
            self.nodes = [tvbo_datamodel.Node(id=0, label="node_0")]

    # Canonical resolver                                                   #
    def _is_materialized(self) -> bool:
        """Return True when this Network already carries connectivity data.

        A Network is "materialized" if it has cached weight matrices, a lazy array store (h5 companion), or an explicit edges list. Used by ``_resolve`` to short-circuit when no further loading is required.
        """
        if any(k.startswith("edges/") for k in self._resident()):
            return True
        if getattr(self, "_store", None) is not None:
            return True
        # A pending graph_generator owns this network's internal connectivity (e.g. a reservoir's recurrent matrix). Any `edges` present alongside it are cross-layer / coupling edges, not internal weights — so don't let them short-circuit generator resolution.
        if self._has_graph_generator():
            return False
        edges = getattr(self, "edges", None)
        if edges:
            return True
        return False

    def _resolve(self, source_dir: str | Path | None = None) -> None:
        """Materialise this Network's connectivity from its declarative spec.

        Single source of truth for "given the YAML, populate the matrices".
        Idempotent: a successful resolution sets ``self._resolved = True`` and subsequent calls are no-ops. Safe to call from ``Network.__init__``, ``Network.from_file``, and from ``SimulationExperiment.from_datamodel`` via a one-line hook.

        Resolution order (first match wins):

        1. Already materialised (cached weights / store / explicit edges):
           mark resolved and return.
        2. ``data_file`` companion (.h5 / .zarr + .yaml sidecar): load lazily via ``tvbo.data.network_io.attach_lazy_store``.
        3. ``bids_dir`` BEP017 directory: route through ``from_bids`` and copy matrices onto self.
        4. ``graph_generator.builder`` Callable: invoke (added in A2).
        5. ``parcellation`` (+ optional ``tractogram``, ``bids.segmentation``,
           ``bids.scale``): normative DB load via
           ``get_normative_connectome_data``.
        6. None of the above: no-op (Network must have been constructed via explicit ``nodes``/``edges`` or ``Network.from_matrix``).

        Parameters
        ----------
        source_dir
            Directory used to resolve relative paths in ``data_file`` or
            ``bids_dir``. When ``None``, paths are taken as absolute or
            resolved against ``cwd``. Callers loading from a YAML file
            should pass the YAML's parent directory.
        """
        if getattr(self, "_resolved", False):
            return
        # 1. Macro connectivity: skip when already materialised.
        if not self._is_materialized():
            if getattr(self, "data_file", None):
                self._resolve_from_data_file(source_dir)
            elif getattr(self, "bids_dir", None):
                self._resolve_from_bids_dir(source_dir)
            elif self._has_graph_generator():
                self._resolve_from_graph_generator(source_dir)
            elif getattr(self, "parcellation", None):
                self._resolve_from_parcellation()
        # 2. Multi-scale resolution (idempotent no-ops when unused). Runs whether or not macro connectivity was already materialised so a DB-loaded network still gets its node_template / subnetworks / sourced parameters expanded.
        self._expand_node_template()
        self._resolve_subnetworks(source_dir)
        self._resolve_parameter_sources(source_dir)
        self._resolved = True

    def invalidate_resolution(self) -> None:
        """Drop everything ``_resolve`` materialised, so the next access rebuilds it.

        Resolution is idempotent and latches, which is what stops a network being rebuilt on every access — and what makes a spec edited AFTER load silently inert. A caller that changes the declaration (a ``--set`` on a graph_generator parameter, a swapped parcellation) has to say so, or the run reports the new value and integrates the old matrix.
        """
        for attr in ("_resolved", "_producers_resolved"):
            object.__setattr__(self, attr, False)
        object.__setattr__(self, "_store", None)
        object.__setattr__(self, "_arrays", {})

    def _has_graph_generator(self) -> bool:
        """True if this Network has a resolvable GraphGenerator.

        A GraphGenerator is resolvable when *any* of:
        * `graph_generator.builder` is an explicit Callable (inline Python builder), or
        * `graph_generator.type` matches a curated entry whose symbolic
          `procedure:` block the generic engine can evaluate (the standard path for built-in generators like RandomReservoir, WeightShuffle, …), or
        * that curated entry declares a `bindings.python.callable` (the legacy /
          library-wrapper escape hatch).
        """
        gg = getattr(self, "graph_generator", None)
        if gg is None:
            return False
        if getattr(gg, "builder", None) is not None:
            return True
        if self._db_procedure_for(gg) is not None:
            return True
        return self._db_python_binding_for(gg) is not None

    @staticmethod
    def _detach_inline_edge_couplings(edges: Any) -> dict[int, dict]:
        """Pop inline (dict) coupling definitions off edge specs before construction.

        Returns ``{edge_index: coupling_dict}`` for every edge spec whose ``coupling`` is a mapping (an inline definition such as a per-edge readout or input projection). Mutates the edge dicts in place, removing the ``coupling`` key so the base ``Edge`` constructor doesn't coerce it into a bare ``CouplingName`` (which would drop coupling_function / parameters). Edge specs whose ``coupling`` is a string (a reference by name) are left untouched.
        """
        detached: dict[int, dict] = {}
        if not edges:
            return detached
        for i, e in enumerate(edges):
            if isinstance(e, dict) and isinstance(e.get("coupling"), dict):
                detached[i] = e.pop("coupling")
        return detached

    def _reattach_inline_edge_couplings(self, detached: dict[int, dict]) -> None:
        """Reattach detached inline coupling dicts as real ``Coupling`` objects."""
        if not detached or not self.edges:
            return
        for i, coupling_dict in detached.items():
            if 0 <= i < len(self.edges):
                self.edges[i].coupling = tvbo_datamodel.Coupling(**coupling_dict)

    @staticmethod
    def _resolve_network_iri(iri: str) -> str | None:
        """Resolve a curated-network IRI to its YAML sidecar path.

        Strips an optional ``tvbo:`` (or any ``prefix:``) and resolves the local name against the network registry. Returns the absolute sidecar path, or ``None`` when the IRI does not match a curated network (so the caller can fall back to treating it as a plain reference string).
        """
        local = _iri_local(iri)
        try:
            from tvbo.data.registry import resolve as registry_resolve

            return str(registry_resolve("Network", local))
        except (FileNotFoundError, ValueError, RuntimeError):
            # FileNotFoundError/ValueError: unknown name. RuntimeError: the database root could not be located (e.g. a packaging issue). In all cases fall back to treating the IRI as a plain reference.
            return None

    # Cache the curated GraphGenerator YAML entry by type so `_is_materialized` / `_resolve` don't re-read+parse the same generator from disk on every network (a reservoir net resolves one per subnetwork). A missing entry is cached as None to avoid repeat lookups.
    _GG_ENTRY_CACHE: dict[str, dict[str, Any] | None] = {}

    @classmethod
    def _db_generator_entry(cls, gg) -> dict[str, Any] | None:
        """Load and memoise the curated GraphGenerator YAML entry for `gg`.

        Returns the raw entry dict (carrying ``procedure``, ``parameters``, ``bindings``, …), or None when the type matches no curated entry — so codegen-only / unknown generators don't trigger resolver errors at load.
        """
        gtype = getattr(gg, "type", None)
        if not gtype:
            return None
        key = str(gtype)
        if key in cls._GG_ENTRY_CACHE:
            return cls._GG_ENTRY_CACHE[key]
        try:
            import yaml

            from tvbo.data.registry import resolve as registry_resolve

            entry_path = registry_resolve("GraphGenerator", key)
        except (FileNotFoundError, ValueError, RuntimeError):
            cls._GG_ENTRY_CACHE[key] = None
            return None
        with open(entry_path) as f:
            entry = yaml.safe_load(f) or {}
        cls._GG_ENTRY_CACHE[key] = entry
        return entry

    @classmethod
    def _db_python_binding_for(cls, gg) -> dict[str, Any] | None:
        """The `bindings.python` block of `gg`'s curated entry, if any.

        The legacy / escape-hatch path: most generators are now interpreted from their symbolic ``procedure:`` block (see ``_db_procedure_for`` and the generic engine), and only genuine library wrappers keep a python binding.
        """
        entry = cls._db_generator_entry(gg) or {}
        return ((entry.get("bindings") or {}).get("python")) or None

    @staticmethod
    def _callable_kwargs(fn, kwargs, defaults):
        """Drop declared defaults a builder's signature cannot accept.

        A curated entry's declared defaults describe the generator's interface, so they apply to whichever route builds it. A Callable builder, though, may implement only part of that interface — and passing it a default it never declared is a TypeError at the call, blaming the recipe for something the database supplied.

        Only DEFAULT-sourced keys are filtered. A parameter the recipe states explicitly is passed through even when the signature has no place for it, so a typo or a parameter the builder genuinely does not support still fails loudly instead of being dropped on the floor.
        """
        import inspect

        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return kwargs  # builtins / C callables expose no signature
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {name: value for name, value in kwargs.items() if name in params or name not in defaults}

    @classmethod
    def _db_procedure_for(cls, gg):
        """The typed ``procedure:`` DAG of `gg`'s curated entry, if any.

        Read from the raw curated YAML and resolved by ``tvbo/graph_generators/procedural.py``, which builds the SymPy tree and renders it through the printer tables — the same expressions every other backend emits from.
        """
        entry = cls._db_generator_entry(gg) or {}
        return entry.get("procedure") or None

    def _resolve_from_graph_generator(self, source_dir: str | Path | None) -> None:
        """Materialise the GraphGenerator and copy its result onto self.

        Three routes, in priority order:
        * **Typed `procedure:` DAG** (preferred) — the curated entry's
          backend-independent steps, resolved to SymPy and rendered through the printer tables. No per-generator Python.
        * **Explicit `graph_generator.builder`** — a user-supplied inline Callable.
        * **`bindings.python.callable`** — the library-wrapper / documented escape
          hatch for constructions the primitive set cannot express.

        Each route yields a `Network`, a dict with at least a `weights` key, or a tuple `(weights, lengths)` / `(weights, lengths, node_params)`.
        `source_dir` is forwarded so Python builders can load companion artefacts.

        A generated node keeps a positional label (`node_<i>`) unless the builder names it through a `node_labels` key. A motif whose nodes ARE particular regions (a PPC-PFC pair) has to be able to say so: every keyed selection downstream — an observation's node coord, a figure's `sel: {node: ...}` — resolves against these labels, and `node_0` forces the reader back to binding by index.
        """
        from tvbo.graph_generators.catalog import declared_defaults

        gg = self.graph_generator
        entry = self._db_generator_entry(gg) or {}

        # Flatten the GraphGenerator's Parameter list to a plain kwargs dict, shared by the DAG and the Python-callable routes. The curated entry's declared defaults sit underneath, so an optional parameter a recipe omits resolves to the value the generator documents rather than to a NameError deep inside a step.
        defaults = declared_defaults(entry)
        kwargs: dict[str, Any] = dict(defaults)
        for p in (gg.parameters or {}).values():
            pname = getattr(p, "name", None)
            if pname is None:
                continue
            val = getattr(p, "value", None)
            if val is None:
                # Distribution-valued parameters (e.g. weight_distribution) carry their spec in the `distribution` slot.
                val = getattr(p, "distribution", None)
            # A Parameter entry carrying neither a value nor a distribution states no value, so the declared default stands. The datamodel cannot distinguish that from an explicit null, and "unset means default" is the reading that matches how a curated entry documents its parameters.
            if val is not None:
                kwargs[pname] = val
        seed = getattr(gg, "seed", None)
        if seed is not None:
            kwargs.setdefault("seed", seed)

        cb = getattr(gg, "builder", None)
        procedure = None if cb is not None else self._db_procedure_for(gg)

        if procedure is not None:
            # Preferred path: resolve the typed DAG (no per-generator Python).
            from tvbo.graph_generators.procedural import materialize

            # Size is the network's, never the generator's: a generator parameter for it would be a second source of truth that can disagree with the network it builds. `_resolve` sets number_of_nodes before reaching here. `not` rather than `is None`: a declared 0 would otherwise reach the DAG and fail deep inside a step on an empty matrix, naming the step rather than the empty network that caused it.
            if not self.number_of_nodes:
                raise ValueError(
                    f"GraphGenerator {gg.type!r} builds an n_nodes x n_nodes network, so "
                    f"`network.number_of_nodes` must be set to a positive count "
                    f"(got {self.number_of_nodes!r})."
                )
            kwargs["n_nodes"] = int(self.number_of_nodes)
            result = materialize(procedure, kwargs, seed=seed)
        else:
            # Escape hatch: inline Callable or curated python binding.
            if cb is not None:
                module_name = getattr(cb, "module", None)
                func_name = getattr(cb, "name", None)
            else:
                binding = self._db_python_binding_for(gg) or {}
                lib = binding.get("library") or ""
                callable_name = binding.get("callable")
                if not callable_name:
                    raise ValueError(
                        f"GraphGenerator type {gg.type!r} has neither a `procedure:` "
                        f"block nor a `bindings.python.callable` in its database entry, "
                        f"and no inline `builder:` was provided."
                    )
                if not lib:
                    raise ValueError(
                        f"GraphGenerator type {gg.type!r} declares a `bindings.python."
                        f"callable` but no `library` — a library-wrapper binding must "
                        f"name its fully-qualified module."
                    )
                module_name = lib
                func_name = callable_name
            if not module_name or not func_name:
                raise ValueError(
                    "graph_generator builder must resolve to both `module` and `name` "
                    f"(got module={module_name!r}, name={func_name!r})"
                )

            import importlib

            # Make the YAML source directory importable so builders can live next to the study YAML.
            with _source_dir_on_path(source_dir):
                mod = importlib.import_module(module_name)
                fn = getattr(mod, func_name)
                result = fn(**self._callable_kwargs(fn, kwargs, defaults))

        # Accept a Network, a dict, or a (weights, lengths[, node_params]) tuple.
        if isinstance(result, Network):
            for attr in ("nodes", "edges", "number_of_nodes", "descriptor", "mesh"):
                val = getattr(result, attr, None)
                if val is not None:
                    setattr(self, attr, val)
            store = getattr(result, "_store", None)
            if store is not None:
                self._store = store
            self._get_arrays().update(result._get_arrays())
        else:
            # Two shapes accepted: dict with `weights` (and optional `lengths`, `node_parameters`, `node_labels`), or tuple `(weights, lengths[, node_params])`.
            node_labels = None
            if isinstance(result, dict):
                if "weights" not in result:
                    raise TypeError("graph_generator materialiser dict must include a `weights` key.")
                weights = np.asarray(result["weights"])
                lengths = np.asarray(result["lengths"]) if result.get("lengths") is not None else None
                node_params = result.get("node_parameters") or result.get("node_params") or None
                node_labels = result.get("node_labels")
                if node_labels is not None and len(node_labels) == 0:  # emptiness by length: a label array has no truth value
                    node_labels = None
            elif isinstance(result, tuple) and len(result) in (2, 3):
                weights = np.asarray(result[0])
                lengths = np.asarray(result[1]) if result[1] is not None else None
                node_params = result[2] if len(result) == 3 else None
            else:
                raise TypeError(
                    "graph_generator builder must return a Network, a dict with "
                    "a `weights` key, or a (weights, lengths) / (weights, lengths, "
                    "node_params) tuple."
                )
            n_nodes = weights.shape[0]
            self.number_of_nodes = n_nodes
            if node_labels is not None and len(node_labels) != n_nodes:
                raise ValueError(
                    f"graph_generator returned {len(node_labels)} node_labels for a "
                    f"{n_nodes}-node weight matrix; one label per node is required."
                )
            labels = [str(lbl) for lbl in node_labels] if node_labels is not None else [f"node_{i}" for i in range(n_nodes)]
            self.nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]
            # The generator supplies the internal weight matrix only. Preserve any declared edges (e.g. cross-layer routing edges on a subnetwork); only default to empty when none were authored.
            if not getattr(self, "edges", None):
                self.edges = []
            self.set_array("edges/weight", weights)
            if lengths is not None:
                self.set_array("edges/length", lengths)
            if node_params:
                # Builder may attach per-node parameters as a dict of {param_name: array of len n_nodes}. Materialise these onto each Node so downstream codegen can consume them.
                for pname, arr in node_params.items():
                    arr = np.asarray(arr)
                    for i in range(n_nodes):
                        if self.nodes[i].parameters is None:
                            self.nodes[i].parameters = {}
                        self.nodes[i].parameters[pname] = tvbo_datamodel.Parameter(name=pname, value=float(arr[i]))

    def _resolve_from_data_file(self, source_dir: str | Path | None) -> None:
        """Populate self from a companion .h5/.zarr sidecar referenced by ``self.data_file``."""
        data_file = Path(self.data_file)
        if not data_file.is_absolute():
            base = Path(source_dir) if source_dir else Path.cwd()
            data_file = (base / data_file).resolve()
        from tvbo.data.network_io import load_network, read_embedded_metadata

        # A self-describing companion carries its own metadata, so it needs no sidecar.
        if data_file.suffix in (".h5", ".zarr") and read_embedded_metadata(data_file) is None:
            sidecar = data_file.with_suffix(".yaml")
            if not sidecar.exists():
                raise FileNotFoundError(f"No YAML sidecar found for {data_file}")
        else:
            sidecar = data_file

        loaded = load_network(sidecar)
        # The sidecar is the authoritative source of connectivity. Replace connectivity-bearing fields unconditionally. Inline-authored coupling / transforms / parameters live in slots NOT listed here and are preserved on self.
        for attr in ("nodes", "edges", "number_of_nodes", "descriptor"):
            val = getattr(loaded, attr, None)
            if val is not None:
                setattr(self, attr, val)
        store = getattr(loaded, "_store", None)
        if store is not None:
            self._store = store
        loaded_mesh = getattr(loaded, "mesh", None)
        if loaded_mesh is not None and getattr(self, "mesh", None) is None:
            self.mesh = loaded_mesh
        self._resident().update(loaded._resident())

    def _resolve_from_bids_dir(self, source_dir: str | Path | None) -> None:
        """Populate self from a BEP017 BIDS directory at ``self.bids_dir``."""
        bids_dir = Path(self.bids_dir)
        if not bids_dir.is_absolute():
            base = Path(source_dir) if source_dir else Path.cwd()
            bids_dir = (base / bids_dir).resolve()
        # Reuse the from_bids classmethod, then copy its state onto self.
        atlas = None
        if self.parcellation:
            parc = self.parcellation
            atlas_obj = parc.get("atlas") if isinstance(parc, dict) else getattr(parc, "atlas", None)
            atlas = atlas_obj.get("name") if isinstance(atlas_obj, dict) else getattr(atlas_obj, "name", None)
        loaded = type(self).from_bids(
            bids_dir,
            atlas=atlas,
            structural_measures=list(self.structural_measures) if self.structural_measures else None,
            observational_measures=list(self.observational_measures) if self.observational_measures else None,
        )
        for attr in ("nodes", "edges", "number_of_nodes"):
            val = getattr(loaded, attr, None)
            if val is not None:
                setattr(self, attr, val)
        for cache in ("_bids_dir", "_store"):
            v = getattr(loaded, cache, None)
            if v is not None:
                object.__setattr__(self, cache, v)
        self._resident().update(loaded._resident())
        # Carry over the parcellation that from_bids inferred from the BIDS filenames, so the atlas resolves (and centres can be looked up by label) even when the source YAML named no parcellation.
        if getattr(self, "parcellation", None) is None:
            loaded_parc = getattr(loaded, "parcellation", None)
            if loaded_parc is not None:
                try:
                    self.parcellation = loaded_parc
                except Exception:  # noqa: BLE001
                    object.__setattr__(self, "parcellation", loaded_parc)

        self._attach_node_attributes(bids_dir)

    def _attach_node_attributes(self, bids_dir) -> None:
        """Attach per-node attributes from ``*_desc-regionSize.tsv`` sidecars as Node parameters, keyed by label.

        Every column beyond ``label`` becomes a node parameter of that name — so a symbolic weight transform can reference it (e.g. ``W / roi_size`` to normalise each target region by its size). Silent when no such sidecar or no matching labels.
        """
        import csv

        nodes = self.nodes or []
        by_label = {str(getattr(n, "label", "")): n for n in nodes}
        for f in sorted(Path(bids_dir).glob("*_desc-regionSize.tsv")):
            with open(f) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                cols = [c for c in (reader.fieldnames or []) if c != "label"]
                for row in reader:
                    node = by_label.get(row.get("label", ""))
                    if node is None:
                        continue
                    if node.parameters is None:
                        node.parameters = {}
                    for c in cols:
                        try:
                            node.parameters[c] = tvbo_datamodel.Parameter(name=c, value=float(row[c]))
                        except (TypeError, ValueError):
                            pass

    def _resolve_from_parcellation(self) -> None:
        """Populate self from a parcellation + tractogram normative DB lookup."""
        parc = self.parcellation
        atlas = parc.get("atlas") if isinstance(parc, dict) else getattr(parc, "atlas", None)
        atlas_name = None
        if atlas is not None:
            atlas_name = atlas.get("name") if isinstance(atlas, dict) else getattr(atlas, "name", None)
            if not atlas_name:
                iri = atlas.get("iri") if isinstance(atlas, dict) else getattr(atlas, "iri", None)
                if iri:
                    atlas_name = iri.split(":", 1)[-1] if ":" in iri else iri
        if not atlas_name:
            return  # No atlas → nothing to resolve.

        tractogram = self.tractogram
        trk_name = None
        if tractogram is not None:
            trk_name = tractogram.get("name") if isinstance(tractogram, dict) else getattr(tractogram, "name", None)
            if not trk_name:
                iri = tractogram.get("iri") if isinstance(tractogram, dict) else getattr(tractogram, "iri", None)
                if iri:
                    trk_name = iri.split(":", 1)[-1] if ":" in iri else iri
        trk_name = trk_name or "dTOR"

        # Optional BIDS disambiguation (seg-, scale-) when the same atlas is published at multiple resolutions or segmentations.
        bids_meta = getattr(self, "bids", None)
        seg = scale = None
        if isinstance(bids_meta, dict):
            seg = bids_meta.get("segmentation")
            scale = bids_meta.get("scale")
        elif bids_meta is not None:
            seg = getattr(bids_meta, "segmentation", None)
            scale = getattr(bids_meta, "scale", None)
        if scale is not None:
            scale = str(scale)

        try:
            w_arr, l_arr, sc_nodes = get_normative_connectome_data(
                atlas_name, trk_name, segmentation=seg, scale=scale, with_nodes=True
            )
        except FileNotFoundError:
            return

        n_nodes = w_arr.shape[0]
        if l_arr is not None and l_arr.shape[0] != n_nodes:
            import warnings

            warnings.warn(
                f"Weight matrix ({n_nodes}x{n_nodes}) and length matrix "
                f"({l_arr.shape[0]}x{l_arr.shape[1]}) have different sizes. "
                f"Using minimum size.",
                stacklevel=2,
            )
            n_nodes = min(n_nodes, l_arr.shape[0])
            w_arr = w_arr[:n_nodes, :n_nodes]
            l_arr = l_arr[:n_nodes, :n_nodes]

        if not self.nodes or len(self.nodes) != n_nodes:
            if sc_nodes and len(sc_nodes) >= n_nodes:
                # Preserve the sidecar's region labels + MNI positions so the network is keyed by region (alignment by label, never by position) for downstream by_label / crosswalk consumers. Slice in case a weight/length size mismatch truncated n_nodes above.
                self.nodes = sc_nodes[:n_nodes]
            else:
                self.nodes = [tvbo_datamodel.Node(id=i, label=f"region_{i}") for i in range(n_nodes)]
        if not self.edges:
            self.edges = []
        self.number_of_nodes = n_nodes
        self.set_array("edges/weight", np.asarray(w_arr))
        if l_arr is not None:
            self.set_array("edges/length", np.asarray(l_arr))

    # Multi-scale resolution                                               #
    def _expand_node_template(self) -> None:
        """Apply ``node_template`` to every materialized node.

        The template is a partial ``Node`` whose fields (``subnetwork``, ``dynamics``, ``edges``, ``parameters`` …) are copied onto each node that does not already set them — explicit per-node fields always win, so heterogeneous variants can override individual regions. A no-op when no template was authored. Each node receives its own deep copy of the spec so per-node resolution (seeds, overrides) stays independent.
        """
        import copy

        spec = getattr(self, "_node_template_spec", None)
        if not spec or not self.nodes:
            return

        def _is_unset(v: Any) -> bool:
            # Treat only None or an empty container as "not set on this node". Use type/len checks (not `v in (None, [], {}, ())`), which would raise on array-valued fields and mis-handle scalars like 0/False.
            if v is None:
                return True
            if isinstance(v, (list, tuple, dict)) and len(v) == 0:
                return True
            return False

        for node in self.nodes:
            for field, value in spec.items():
                if field in ("id", "label"):
                    continue
                if _is_unset(getattr(node, field, None)):
                    setattr(node, field, copy.deepcopy(value))

    def _resolve_subnetworks(self, source_dir: str | Path | None) -> None:
        """Materialize each node's ``subnetwork`` into a resolved ``Network``.

        For every node carrying a subnetwork spec (dict or partially built object), construct a `Network` and resolve it — which runs the subnetwork's own ``graph_generator`` (e.g. RandomReservoir → the recurrent matrix W_int). Idempotent: already-resolved Network instances are left untouched.
        """
        for node in self.nodes or []:
            sub = getattr(node, "subnetwork", None)
            if sub is None:
                continue
            if isinstance(sub, Network) and getattr(sub, "_resolved", False):
                continue
            if isinstance(sub, Network):
                sub._resolve(source_dir=source_dir)
                continue
            # dict or datamodel Network → build the enhanced subclass, which resolves its graph_generator during construction.
            spec = sub if isinstance(sub, dict) else as_dict(sub)
            node.subnetwork = Network(**spec)

    def _resolve_parameter_sources(self, source_dir: str | Path | None) -> None:
        """Populate network ``parameters`` declared via ``source`` + ``measure``.

        A parameter such as ``cortical_gradient`` may point at a curated Network (``source``: an IRI / path) and name a per-node ``measure`` carried by that network's nodes (``measure``: e.g. ``megalpha``). This loads the referenced network and gathers the per-node measure values into a 1-D array stored on ``parameter.value``. A no-op for ordinary scalar parameters.
        """
        params = getattr(self, "parameters", None)
        if not params:
            return
        for param in params.values():
            source = getattr(param, "source", None)
            measure = getattr(param, "measure", None)
            if not source or not measure:
                continue
            if isinstance(getattr(param, "value", None), (list, np.ndarray)):
                continue  # already materialised
            values = self._load_measure_from_source(source, measure, source_dir)
            if values is not None:
                param.value = values

    @classmethod
    def _load_measure_from_source(
        cls,
        source: str,
        measure: str,
        source_dir: str | Path | None = None,
    ) -> np.ndarray | None:
        """Load a per-node ``measure`` array from a referenced network ``source``.

        ``source`` is resolved as (1) a curated-network IRI via the registry, (2) a path relative to ``source_dir``, or (3) an absolute path. The named measure is read from each node's ``parameters[measure]`` value.
        Returns a 1-D ``ndarray`` (node order preserved), or ``None`` when the source cannot be resolved.
        """
        path = cls._resolve_network_iri(source)
        if path is None:
            cand = Path(source)
            if not cand.is_absolute() and source_dir is not None:
                cand = (Path(source_dir) / cand).resolve()
            path = str(cand) if cand.exists() else None
        if path is None:
            return None

        # from_file → load_network already resolves the network on construction; the explicit _resolve() is a harmless idempotent guard.
        net = cls.from_file(path)
        net._resolve()
        vals = []
        for node in net.nodes or []:
            node_params = getattr(node, "parameters", None) or {}
            p = node_params.get(measure) if hasattr(node_params, "get") else None
            # Per the docstring, return None (don't crash) when any node lacks a usable scalar value for the measure: missing parameter, None value, or a non-scalar (e.g. vector) value that float() can't accept.
            val = getattr(p, "value", None) if p is not None else None
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                return None
            vals.append(float(val))
        return np.asarray(vals) if vals else None

    # -- Backward-compat properties: conduction_speed, global_coupling_strength --
    @property
    def conduction_speed(self):
        """Access conduction_speed from parameters dict."""
        if self.parameters and "conduction_speed" in self.parameters:
            return self.parameters["conduction_speed"]
        return None

    @conduction_speed.setter
    def conduction_speed(self, val):
        """Store `val` as the `conduction_speed` entry in the parameters dict."""
        self.parameters["conduction_speed"] = val

    @property
    def global_coupling_strength(self):
        """Access global_coupling_strength from parameters dict."""
        if self.parameters and "global_coupling_strength" in self.parameters:
            return self.parameters["global_coupling_strength"]
        return None

    @global_coupling_strength.setter
    def global_coupling_strength(self, val):
        """Store `val` as the `global_coupling_strength` entry in the parameters dict."""
        self.parameters["global_coupling_strength"] = val

    _INTERNAL_ATTRS = frozenset(
        {
            "_store",
            "_arrays",
            "_parent_network_obj",
            "_node_template_spec",
            "_edge_template_spec",
            "_save_path",
            "_orientations",
            "_resolved",
        }
    )
    """Runtime attributes the LinkML dumpers never see; ``_items`` hides every leading-underscore key regardless, so this names them rather than gates them."""

    @property
    def parent_network_obj(self) -> Optional["Network"]:
        """The parent Network object, if assigned via object reference.

        Returns ``None`` when ``parent_network`` was set as a plain string path or was never set.
        """
        try:
            return object.__getattribute__(self, "_parent_network_obj")
        except AttributeError:
            return None

    def _items(self):
        # What the LinkML yaml_dumper / json_dumper / as_dict see. The resident arrays and the lazy store (``_arrays``, ``_store``) are runtime bookkeeping, never schema slots — and LinkML slot names are never underscore-prefixed — so every leading-underscore key is hidden by rule rather than by a denylist, and a new runtime attribute cannot reach yaml.SafeDumper as an ndarray it cannot represent. Bulk arrays belong in the binary companion via ``save_network``, referenced from the spec by ``data_file``.
        for k, v in super()._items():
            if k.startswith("_") or k in self._INTERNAL_ATTRS:
                continue
            yield k, v

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.Network) -> "Network":
        """Create a Network from a datamodel instance.

        Parameters
        ----------
        datamodel : tvbo_datamodel.Network
            Source datamodel Network instance

        Returns:
        -------
        Network
            New Network with fields copied from datamodel

        Examples:
        --------
        ```python
        from tvbo.datamodel import schema as tvbo_datamodel
        dm = tvbo_datamodel.Network(number_of_nodes=10)
        sc = Network.from_datamodel(dm)
        ```
        """
        data = as_dict(datamodel)
        # as_dict returns a dict-like object that works with **kwargs
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_matrix(
        cls,
        weights: np.ndarray | None = None,
        lengths: np.ndarray | None = None,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> "Network":
        """Create a Network from named edge-property matrices.

        This is a convenience constructor for creating networks from matrix representations. For performance, matrices are stored directly and edges are generated lazily only when needed.

        Any keyword argument whose value is array-like (ndarray, sparse matrix, or nested sequence) is treated as a named edge-property matrix and stored via ``set_matrix``. All other keyword arguments are forwarded to the ``Network`` constructor.

        Parameters
        ----------
        weights : np.ndarray, optional
            Connection weight matrix (N x N). Stored as ``"weight"``.
        lengths : np.ndarray, optional
            Tract length matrix (N x N). Stored as ``"length"``.
        labels : list of str, optional
            Node labels. If not provided, uses "node_0", "node_1", etc.
        **kwargs : Any
            Keyword arguments that are array-like are stored as named
            edge matrices (e.g. ``sc=mat`` → ``set_matrix("sc", mat)``).
            Everything else is passed to the Network constructor.

        Returns:
        -------
        Network
            New Network with nodes derived from labels and matrices stored
            for efficient access.

        Examples:
        --------
        ```python
        import numpy as np
        from tvbo import Network

        # Simple 3-node network
        W = np.array([[0, 0.5, 0.3],
                      [0.2, 0, 0.4],
                      [0.1, 0.6, 0]])
        network = Network.from_matrix(W, labels=["A", "B", "C"])
        network.plot_graph()

        # With tract lengths
        L = np.array([[0, 10, 15],
                      [10, 0, 8],
                      [15, 8, 0]])
        network = Network.from_matrix(W, lengths=L)

        # Arbitrary named edge properties
        sc = np.array([[0, 1], [1, 0]])
        fc = np.array([[1, 0.8], [0.8, 1]])
        network = Network.from_matrix(sc=sc, fc=fc, labels=["L", "R"])
        network.plot_overview()
        ```
        """
        from scipy import sparse

        def _is_matrix(v):
            """Check if value is array-like (ndarray, sparse, or nested list)."""
            if isinstance(v, np.ndarray) or sparse.issparse(v):
                return True
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return isinstance(v[0], (list, tuple, np.ndarray))
            return False

        # Separate matrix kwargs from constructor kwargs
        matrix_kwargs = {}
        ctor_kwargs = {}
        for k, v in kwargs.items():
            if _is_matrix(v):
                matrix_kwargs[k] = v
            else:
                ctor_kwargs[k] = v

        # Collect all matrices to infer n_nodes
        all_matrices = {}
        if weights is not None:
            all_matrices["weight"] = weights
        if lengths is not None:
            all_matrices["length"] = lengths
        all_matrices.update(matrix_kwargs)

        if not all_matrices:
            raise ValueError("At least one matrix must be provided.")

        # Infer n_nodes from the first matrix
        first = next(iter(all_matrices.values()))
        if sparse.issparse(first):
            n_nodes = first.shape[0]
        else:
            first = np.asarray(first)
            n_nodes = first.shape[0]

        if labels is None:
            labels = [f"node_{i}" for i in range(n_nodes)]

        # Create explicit nodes (cheap - only N objects)
        nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]

        # Build the network with nodes only - matrices stored separately for performance
        instance = cls(
            nodes=nodes,
            edges=[],  # Don't create Edge objects - too slow for large networks
            number_of_nodes=n_nodes,
            number_of_regions=n_nodes,
            **ctor_kwargs,
        )

        # Store all matrices via set_matrix for unified storage
        for name, data in all_matrices.items():
            instance.set_matrix(name, data)

        return instance

    @classmethod
    def from_bids(
        cls,
        bids_dir: str | Path,
        atlas: str | None = None,
        structural_measures: list[str] | None = None,
        observational_measures: list[str] | None = None,
        **kwargs: Any,
    ) -> "Network":
        """Create a Network from BEP017-compliant BIDS connectivity data.

        Loads structural connectivity (weights, lengths) and optionally observational targets (FC) from a BIDS derivatives directory using the BEP017 relationship matrix format.

        Parameters
        ----------
        bids_dir : str or Path
            Path to BEP017-compliant BIDS directory containing _relmat files.
        atlas : str, optional
            Atlas name to filter files (e.g., "DesikanKilliany").
            If None, uses the first atlas found.
        structural_measures : list of str, optional
            Measures to use for structural network.
            First is used as weights, second (if present) as lengths.
            If None, auto-discovered from available ``meas-*`` relmat files.
        observational_measures : list of str, optional
            Measures to load as observational targets for optimization
            (e.g., ["correlation"] for FC). Stored in network._observations.
        **kwargs : Any
            Additional keyword arguments passed to Network constructor.

        Returns:
        -------
        Network
            Network with matrices loaded from BEP017 files.
            Observational data accessible via network.observations dict.

        Examples:
        --------
        ```python
        from tvbo import Network

        # Auto-discover measures from directory
        network = Network.from_bids("tvbo/database/networks/bids/dk_average")

        # Or specify measures explicitly
        network = Network.from_bids(
            "tvbo/database/networks/bids/dk_average",
            structural_measures=["streamlineCount", "tractLength"],
            observational_measures=["BoldCorrelation"],
        )

        # Access structural connectivity
        print(network.matrix("weight").shape)  # (84, 84)
        ```
        """
        from pathlib import Path

        bids_dir = Path(bids_dir)

        if structural_measures is None:
            structural_measures = _discover_bids_measures(bids_dir)
        if observational_measures is None:
            observational_measures = []

        # Find relmat files
        relmat_files = list(bids_dir.glob("*_relmat.dense.tsv")) + list(bids_dir.glob("*_relmat.tsv"))

        if not relmat_files:
            raise ValueError(f"No BEP017 relmat files found in {bids_dir}")

        # Parse atlas from filenames if not specified
        if atlas is None:
            for f in relmat_files:
                if "atlas-" in f.name:
                    atlas = f.name.split("atlas-")[1].split("_")[0]
                    break

        # Load nodeindices file for labels
        nodeindices_files = list(bids_dir.glob(f"*atlas-{atlas}*_nodeindices.tsv"))
        labels = None
        if nodeindices_files:
            df = pd.read_csv(nodeindices_files[0], sep="\t")
            if "label" in df.columns:
                labels = df["label"].tolist()

        # Helper to load a measure
        def load_measure(measure: str) -> np.ndarray | None:
            """Load a single BEP017 measure matrix by name, or None if absent."""
            pattern = f"*meas-{measure}_relmat*.tsv"
            matches = list(bids_dir.glob(pattern))
            if not matches:
                return None
            # Load TSV (dense format - no header, tab-separated)
            return np.loadtxt(matches[0], delimiter="\t")

        measure_units = _bids_measure_units(bids_dir, [*structural_measures, *observational_measures])

        # Load structural measures
        weights = None
        lengths = None
        for i, measure in enumerate(structural_measures):
            data = load_measure(measure)
            if data is not None:
                if i == 0:
                    weights = data
                elif i == 1:
                    lengths = data

        if weights is None:
            raise ValueError(f"Could not load structural weights from {bids_dir}. Looked for measures: {structural_measures}")

        n_nodes = weights.shape[0]

        # Load observational measures
        observations = {}
        for measure in observational_measures:
            data = load_measure(measure)
            if data is not None:
                observations[measure] = data

        # Build labels if not from nodeindices
        if labels is None:
            labels = [f"node_{i}" for i in range(n_nodes)]

        # Create nodes
        nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]

        # Build network. Declare the observational measures we actually loaded so the `observations` property (which gates on ``observational_measures``) can resolve them.
        length_measure = structural_measures[1] if len(structural_measures) > 1 else None
        declared_length_unit = _declared_length_unit(measure_units, length_measure)
        if declared_length_unit:
            kwargs.setdefault("distance_unit", declared_length_unit)
        instance = cls(
            nodes=nodes,
            edges=[],
            number_of_nodes=n_nodes,
            number_of_regions=n_nodes,
            label=atlas or bids_dir.name,
            observational_measures=list(observations) or None,
            **kwargs,
        )

        instance.set_array("edges/weight", weights)
        if lengths is not None:
            instance.set_array("edges/length", lengths)
        for measure, data in observations.items():
            instance.set_array(f"edges/{measure}", data)
        object.__setattr__(instance, "_bids_dir", str(bids_dir))
        object.__setattr__(instance, "_bids_measure_units", measure_units)

        # Record the parcellation atlas so get_atlas()/get_centers() can resolve region centres from the named atlas's entities by label.
        if atlas and getattr(instance, "parcellation", None) is None:
            parc = tvbo_datamodel.Parcellation(
                label=atlas,
                atlas=tvbo_datamodel.BrainAtlas(name=atlas),
            )
            try:
                instance.parcellation = parc
            except Exception:  # noqa: BLE001
                object.__setattr__(instance, "parcellation", parc)

        return instance

    # NOTE: the canonical `observations` property is defined later in this class (source-agnostic: resolves BIDS-loaded and inline-store data, keyed by `observational_measures`). A duplicate definition here was dead code (shadowed by the later one) and has been removed.

    def load_from_bids(
        self,
        bids_dir: str | Path,
        structural_measures: list[str] | None = None,
        observational_measures: list[str] | None = None,
        atlas: str | None = None,
    ) -> "Network":
        """Load BEP017 data into existing network.

        Allows loading structural connectivity and/or observational targets independently into an already-configured network (preserves coupling).

        Parameters
        ----------
        bids_dir : str or Path
            Path to BEP017-compliant BIDS directory.
        structural_measures : list of str, optional
            Measures for structural connectivity.
            If None, auto-discovered from available ``meas-*`` relmat files.
        observational_measures : list of str, optional
            Measures for observational targets (e.g., ["BoldCorrelation"]).
            If None, does not load observational data.
        atlas : str, optional
            Atlas name to filter files. Auto-detected if not provided.

        Returns:
        -------
        Network
            Self (for method chaining).

        Example:
        -------
        >>> network = Network()
        >>> network.load_from_bids(
        ...     "tvbo/database/networks/bids/dk_average",
        ... )
        >>> network.load_from_bids(
        ...     "tvbo/database/networks/bids/dk_average",
        ...     observational_measures=["BoldCorrelation"],
        ... )
        """
        bids_dir = Path(bids_dir)

        if structural_measures is None:
            structural_measures = _discover_bids_measures(bids_dir)

        # Find relmat files
        relmat_files = list(bids_dir.glob("*_relmat.dense.tsv")) + list(bids_dir.glob("*_relmat.tsv"))

        if not relmat_files:
            raise ValueError(f"No BEP017 relmat files found in {bids_dir}")

        # Parse atlas from filenames if not specified
        if atlas is None:
            for f in relmat_files:
                if "atlas-" in f.name:
                    atlas = f.name.split("atlas-")[1].split("_")[0]
                    break

        # Helper to load a measure
        def load_measure(measure: str) -> np.ndarray | None:
            """Load a single BEP017 measure matrix by name, or None if absent."""
            pattern = f"*meas-{measure}_relmat*.tsv"
            matches = list(bids_dir.glob(pattern))
            if not matches:
                return None
            return np.loadtxt(matches[0], delimiter="\t")

        # Load structural measures if requested
        if structural_measures:
            weights = None
            lengths = None
            for i, measure in enumerate(structural_measures):
                data = load_measure(measure)
                if data is not None:
                    if i == 0:
                        weights = data
                    elif i == 1:
                        lengths = data

            if weights is not None:
                n_nodes = weights.shape[0]
                self.set_array("edges/weight", weights)
                if lengths is not None:
                    self.set_array("edges/length", lengths)
                self.number_of_nodes = n_nodes
                self.number_of_regions = n_nodes

                # Load labels from nodeindices file
                nodeindices_files = list(bids_dir.glob(f"*atlas-{atlas}*_nodeindices.tsv"))
                if nodeindices_files:
                    df = pd.read_csv(nodeindices_files[0], sep="\t")
                    if "label" in df.columns:
                        labels = df["label"].tolist()
                        self.nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]

                # Record the parcellation atlas. BIDS connectomes carry no parcellation, so get_atlas() would default to "wholebrain"; naming the atlas lets get_centers() resolve region centres from the atlas's entities (matched to node labels).
                if atlas and getattr(self, "parcellation", None) is None:
                    parc = tvbo_datamodel.Parcellation(
                        label=atlas,
                        atlas=tvbo_datamodel.BrainAtlas(name=atlas),
                    )
                    try:
                        self.parcellation = parc
                    except Exception:  # noqa: BLE001
                        object.__setattr__(self, "parcellation", parc)

        # Load observational measures if requested
        if observational_measures:
            loaded = []
            for measure in observational_measures:
                data = load_measure(measure)
                if data is not None:
                    self.set_array(f"edges/{measure}", data)
                    loaded.append(measure)
            # Declare loaded measures so the `observations` property (which gates on ``observational_measures``) can resolve them.
            existing = list(self.observational_measures or [])
            self.observational_measures = existing + [m for m in loaded if m not in existing]

        object.__setattr__(self, "_bids_dir", str(bids_dir))
        return self

    def load_matrix(
        self,
        weights: np.ndarray,
        lengths: np.ndarray | None = None,
        labels: list[str] | None = None,
    ) -> "Network":
        """Load weight/length matrices into existing network (preserves coupling).

        Use this instead of from_matrix when you need to update connectivity data while keeping the network's coupling definitions intact.

        Parameters
        ----------
        weights : np.ndarray
            Connection weight matrix (N x N).
        lengths : np.ndarray, optional
            Tract length matrix (N x N).
        labels : list of str, optional
            Node labels. Updates nodes if provided.

        Returns:
        -------
        Network
            Self (for chaining).
        """
        weights = np.asarray(weights)
        n_nodes = weights.shape[0]

        self.set_array("edges/weight", weights)
        if lengths is not None:
            self.set_array("edges/length", lengths)

        # Update node count
        self.number_of_nodes = n_nodes
        self.number_of_regions = n_nodes

        # Update nodes if labels provided
        if labels is not None:
            self.nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]

        return self

    @classmethod
    def from_string(cls, yaml_string: str, **kwargs: Any) -> "Network":
        """Create a Network from a YAML string.

        This is a convenience constructor for creating networks directly from YAML specifications, commonly used in notebooks and scripts.

        Parameters
        ----------
        yaml_string : str
            YAML string defining the network with nodes and edges.
        **kwargs : Any
            Additional keyword arguments passed to Network constructor.

        Returns:
        -------
        Network
            New Network parsed from the YAML string.

        Examples:
        --------
        ```python
        from tvbo import Network

        network = Network.from_string('''
        label: MyNetwork
        nodes:
          - id: 0
            label: NodeA
            dynamics: Oscillator
          - id: 1
            label: NodeB
            dynamics: Excitable
        edges:
          - source: 0
            target: 1
            weight: 0.5
        ''')
        print(network.label)
        ```
        """
        from tvbo.utils import yaml_loader

        return yaml_loader.loads(yaml_string, cls)

    # ── File I/O (§12.4) ─────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "Network":
        """Load from YAML/JSON sidecar with lazy binary companion.

        Supports YAML and JSON sidecars (auto-detected by extension).
        Supports HDF5, Zarr, and CSV companions.
        Arrays are NOT loaded into memory — loaded lazily on first access.

        Parameters
        ----------
        path : str or Path
            Path to YAML or JSON sidecar file.

        Returns:
        -------
        Network
            Network with lazy array references.

        Examples:
        --------
        >>> net = Network.from_db("dk87")
        >>> net.number_of_nodes       # metadata: instant, no I/O
        87
        >>> net.matrix("weight").shape  # arrays: loaded on first access
        (87, 87)
        """
        from tvbo.data.network_io import load_network

        return load_network(path)

    @classmethod
    def from_tvb_zip(cls, zip_path: str | Path) -> "Network":
        """Import from TVB connectivity ZIP (weights.txt + tract_lengths.txt).

        Parameters
        ----------
        zip_path : str or Path
            Path to TVB connectivity ZIP file.

        Returns:
        -------
        Network
            Network with arrays loaded, ready for ``save()``.

        Examples:
        --------
        >>> net = Network.from_tvb_zip("connectivity_76.zip")
        >>> net.number_of_nodes
        76
        """
        from tvbo.adapters.tvb import from_tvb_zip

        return from_tvb_zip(zip_path)

    @classmethod
    def from_tvb(cls, connectivity) -> "Network":
        """Import a live TVB Connectivity object.

        Lossless conversion preserving all TVB fields (weights, lengths, centres, cortical flags, areas, hemispheres, conduction speed).

        Parameters
        ----------
        connectivity : tvb.datatypes.connectivity.Connectivity
            Configured TVB Connectivity instance.

        Returns:
        -------
        Network
            Network with arrays loaded, ready for ``save()``.

        Examples:
        --------
        >>> from tvb.datatypes.connectivity import Connectivity
        >>> conn = Connectivity.from_file()
        >>> net = Network.from_tvb(conn)
        >>> net.number_of_nodes
        76
        """
        from tvbo.adapters.tvb import from_tvb

        return from_tvb(connectivity)

    @classmethod
    def from_tvb_surface(cls, connectivity, surface, region_mapping):
        """Create a multi-level Network from TVB surface simulation data.

        Produces two linked networks:

        1. **Region-level** (parent): from TVB Connectivity
        2. **Vertex-level** (child): mesh + region_mapping linking vertices to regions via hierarchical ``node_mapping``

        Parameters
        ----------
        connectivity : tvb.datatypes.connectivity.Connectivity
            Configured TVB Connectivity (region-level).
        surface : tvb.datatypes.surfaces.Surface
            TVB CorticalSurface with vertices and triangles.
        region_mapping : tvb.datatypes.region_mapping.RegionMapping
            TVB RegionMapping (vertex → region).

        Returns:
        -------
        tuple[Network, Network]
            ``(region_network, surface_network)``

        Examples:
        --------
        >>> from tvb.datatypes.connectivity import Connectivity
        >>> from tvb.datatypes.surfaces import CorticalSurface
        >>> from tvb.datatypes.region_mapping import RegionMapping
        >>> conn = Connectivity.from_file()
        >>> surf = CorticalSurface.from_file()
        >>> rmap = RegionMapping.from_file()
        >>> region_net, surface_net = Network.from_tvb_surface(conn, surf, rmap)
        """
        from tvbo.adapters.tvb import from_tvb_surface

        return from_tvb_surface(connectivity, surface, region_mapping)

    # ── Save / Export ─────────────────────────────────────────────

    def save(
        self,
        path: str | Path,
        binary_format: str = "h5",
        sidecar_format: str = "yaml",
    ):
        """Save as sidecar + binary companion.

        Sidecar is written via LinkML yaml_dumper or json_dumper — always schema-valid output, no manual serialization.

        Parameters
        ----------
        path : str or Path
            Output path for sidecar.
        binary_format : str
            "h5" (default), "zarr", or "csv".
        sidecar_format : str
            "yaml" (default) or "json".

        Examples:
        --------
        >>> net.save("output/")                                     # dir → BIDS filename
        >>> net.save("output/dk87.yaml")                           # YAML + HDF5
        >>> net.save("output/dk87.yaml", sidecar_format="json")     # JSON + HDF5
        >>> net.save("output/dk87.yaml", binary_format="zarr")      # YAML + Zarr
        >>> net.save("output/dk87.yaml", binary_format="csv")       # YAML + CSV
        """
        from tvbo.data.network_io import save_network

        path = Path(path)
        if path.is_dir() or str(path).endswith("/"):
            sidecar_ext = ".json" if sidecar_format == "json" else ".yaml"
            bids_name = self.bids_filename
            if bids_name is None:
                label = getattr(self, "label", None) or "network"
                bids_name = f"{label}{sidecar_ext}"
            bids_stem = Path(bids_name).with_suffix("").with_suffix("")
            path = path / bids_stem.with_suffix(sidecar_ext)

        save_network(self, path, binary_format, sidecar_format)
        # Remember save path so it can be used as a reference string
        object.__setattr__(self, "_save_path", str(Path(path).name))

    def to_bep017(self, output_dir: str | Path):
        """Export to BEP017-compatible per-measure files.

        Each template edge becomes a separate TSV + JSON sidecar.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for BEP017 files.
        """
        from tvbo.data.converters import to_bep017

        to_bep017(self, output_dir)

    # ── BIDS filename ─────────────────────────────────────────────

    @property
    def bids_filename(self) -> str:
        """Generate BIDS-compliant filename using pybids build_path (§6.5).

        Reads entities directly from Network attributes — no YAML serialization round-trip needed.  Sensor networks (descriptor ``"sensors"``) use ``SENSOR_PATTERNS``; all others use ``RELMAT_PATTERNS``.

        Returns:
        -------
        str
            BIDS-compliant filename for this network.

        Examples:
        --------
        >>> net.bids_filename
        'tpl-MNI152NLin2009cAsym_..._desc-SCFC_relmat.h5'
        """
        from bids.layout.writing import build_path

        if getattr(self, "descriptor", None) in ("sensors", "projection"):
            from tvbo.data.converters import sensor_entities
            from tvbo.data.network_io import SENSOR_PATTERNS

            return build_path(sensor_entities(self), SENSOR_PATTERNS)

        from tvbo.data.converters import relmat_entities
        from tvbo.data.network_io import RELMAT_PATTERNS

        return build_path(relmat_entities(self), RELMAT_PATTERNS)

    # ── Platform retrieval (§12.8) ────────────────────────────────

    TVBO_PLATFORM_URL = "https://tvbo.charite.de"

    @classmethod
    def from_platform(
        cls,
        atlas: str,
        tractogram: str = "dTOR",
        base_url: str = TVBO_PLATFORM_URL,
        cache_dir: str | Path | None = None,
    ) -> "Network":
        """Download a normative connectivity network from the tvbo platform.

        Fetches the sidecar (YAML) and companion (HDF5) from the tvbo API and caches locally for subsequent loads.

        Parameters
        ----------
        atlas : str
            Atlas name (e.g., "DesikanKilliany", "Schaefer1000").
        tractogram : str
            Tractogram name (default: "dTOR").
        base_url : str
            Platform base URL.
        cache_dir : str or Path or None
            Local cache directory (default: ``~/.tvbo/networks``).

        Returns:
        -------
        Network
            Network loaded from platform (cached locally).
        """
        import requests

        from tvbo.data.network_io import load_network

        if cache_dir is None:
            cache_dir = Path.home() / ".tvbo" / "networks"
        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        api = f"{base_url.rstrip('/')}/api/v1/networks"

        # Find network by atlas + tractogram
        resp = requests.get(api, params={"atlas": atlas, "tractogram": tractogram})
        resp.raise_for_status()
        matches = resp.json()
        if not matches:
            raise ValueError(f"No network found for atlas={atlas}, tractogram={tractogram}")
        net_id = matches[0]["id"]

        # Check cache
        cached_yaml = cache_dir / f"{net_id}.yaml"
        cached_h5 = cache_dir / f"{net_id}.h5"
        if cached_yaml.exists() and cached_h5.exists():
            return load_network(cached_yaml)

        # Download sidecar
        r = requests.get(f"{api}/{net_id}/sidecar")
        r.raise_for_status()
        cached_yaml.write_text(r.text)

        # Download binary companion
        r = requests.get(f"{api}/{net_id}/data")
        r.raise_for_status()
        cached_h5.write_bytes(r.content)

        # Patch data_file reference to point to local companion
        import yaml as yaml_module

        sidecar = yaml_module.safe_load(cached_yaml.read_text())
        sidecar["data_file"] = cached_h5.name
        cached_yaml.write_text(yaml_module.dump(sidecar, sort_keys=False))

        return load_network(cached_yaml)

    @classmethod
    def list_platform_networks(cls, base_url: str = TVBO_PLATFORM_URL, **filters) -> list[dict]:
        """List available normative networks on the tvbo platform.

        Parameters
        ----------
        base_url : str
            Platform base URL.
        **filters
            Filtering parameters (e.g., atlas="DesikanKilliany").

        Returns:
        -------
        list[dict]
            List of network summaries.
        """
        import requests

        api = f"{base_url.rstrip('/')}/api/v1/networks"
        resp = requests.get(api, params=filters)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def load(cls, source: str | Path | None = None, **entities) -> Union["Network", list["Network"]]:
        """Unified loader: file path, database name, or BIDS entities.

        Accepts any of:

        - **File path** (YAML, JSON, or HDF5): loads from disk.
          For HDF5, automatically finds the companion YAML sidecar.
        - **Short name**: resolves via the tvbo database
          (e.g. ``"Lobar"``, ``"DesikanKilliany"``).
        - **BIDS entities** as keyword arguments
          (e.g. ``atlas="Schaefer2018", scale="100"``).

        Parameters
        ----------
        source : str or Path, optional
            A file path or database name.  When omitted, BIDS entity
            kwargs are used to search the database.
        **entities
            BIDS key-value filters (``atlas``, ``rec``, ``scale``,
            ``desc``, ``seg``, ``cohort``, …).

        Returns:
        -------
        Network or list[Network]
            A single Network, or a list when multiple BIDS matches occur.

        Examples:
        --------
        >>> Network.load("Lobar")                               # database name
        >>> Network.load("networks/my_network.yaml")            # YAML file
        >>> Network.load("networks/my_network.h5")              # HDF5 companion
        >>> Network.load(atlas="Schaefer2018", scale="100")     # BIDS entities
        """
        if source is not None:
            p = Path(source)
            ext = p.suffix.lower()
            # HDF5 companion → itself when self-describing, else its YAML/JSON sidecar
            if ext in (".h5", ".hdf5"):
                from tvbo.data.network_io import read_embedded_metadata

                if read_embedded_metadata(p) is not None:
                    return cls.from_file(str(p))
                sidecar = p.with_suffix(".yaml")
                if not sidecar.exists():
                    sidecar = p.with_suffix(".json")
                if not sidecar.exists():
                    raise FileNotFoundError(f"No YAML/JSON sidecar found for {p}")
                return cls.from_file(str(sidecar))
            # Explicit sidecar path
            if ext in (".yaml", ".yml", ".json") or p.is_file():
                return cls.from_file(str(p))
            # No extension / not a file → treat as database name
            return cls.from_db(str(source), **entities)
        # No source → BIDS entity search
        if entities:
            return cls.from_db(**entities)
        raise ValueError("Provide a file path, database name, or BIDS entities.")

    @classmethod
    def from_db(cls, name: str | None = None, **entities) -> Union["Network", list["Network"]]:
        """Load a Network from the tvbo database by name or BIDS entities.

        Supports two modes:

        1. **By name** (existing): ``Network.from_db("DesikanKilliany")``
        2. **By BIDS key-values**: ``Network.from_db(atlas="DesikanKilliany", rec="dTOR")``

        BIDS entity keys match the ``key-value`` pairs in filenames, e.g.
        ``atlas``, ``rec``, ``scale``, ``seg``, ``desc``, ``cohort``.

        When entities match a single file, returns a Network.
        When multiple match, returns a list of Networks.

        Parameters
        ----------
        name : str, optional
            Short name or atlas name (legacy mode). Ignored when entities
            are given.
        **entities
            BIDS key-value filters. All specified entities must match.

        Returns:
        -------
        Network or list[Network]

        Examples:
        --------
        >>> sc = Network.from_db("DesikanKilliany")        # by name
        >>> sc = Network.from_db(atlas="DesikanKilliany", rec="dTOR")
        >>> scs = Network.from_db(atlas="Schaefer2018", scale="100")  # list
        """
        if not entities:
            from tvbo.data.registry import resolve

            return cls.from_file(str(resolve("Network", name)))

        matches = _filter_networks_by_entities(entities)
        if len(matches) == 0:
            available = [f.stem for f in NETWORK_DIR.glob("*.yaml")]
            raise FileNotFoundError(f"No network matching {entities}. Available: {sorted(available)}")
        if len(matches) == 1:
            return cls.from_file(str(matches[0]))
        return [cls.from_file(str(m)) for m in sorted(matches)]

    @classmethod
    def list_db(cls, **entities) -> list[str]:
        """List available networks in the tvbo database, optionally filtered.

        Parameters
        ----------
        **entities
            BIDS key-value filters (e.g. ``atlas="Schaefer2018"``).

        Returns:
        -------
        list[str]
            Sorted list of matching network stems.

        Examples:
        --------
        >>> Network.list_db()                            # all networks
        >>> Network.list_db(atlas="Schaefer2018")        # only Schaefer
        >>> Network.list_db(rec="dTOR", scale="100")     # dTOR at scale 100
        """
        if not entities:
            from tvbo.data.registry import list_entries

            return list_entries("Network")
        return sorted(m.stem for m in _filter_networks_by_entities(entities))

    # Keep nodes and regions synchronized on assignment
    def __setattr__(self, name: str, value: Any) -> None:
        # Accept Network objects for parent_network
        if name == "parent_network" and value is not None:
            if isinstance(value, Network):
                object.__setattr__(self, "_parent_network_obj", value)
                value = _network_ref_string(value)

        super_setattr = super().__setattr__

        super_setattr(name, value)

    def to_yaml(self, filepath: str | None = None, format: str = "tvbo") -> str:
        """Serialize Network to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to save YAML file. If None, returns YAML string.
        format : str
            Output format: "tvbo" (default) or "pyrates".
            PyRates format generates a complete experiment YAML (network + dynamics).

        Returns:
        -------
        str
            YAML representation of the Network

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        yaml_str = sc.to_yaml()
        sc.to_yaml("connectome.yaml")  # Save to file
        sc.to_yaml("network.yaml", format="pyrates")  # PyRates format
        ```
        """
        if format.lower() == "pyrates":
            from tvbo.codegen.pyrates import to_pyrates_yaml_string

            return to_pyrates_yaml_string(network=self, filepath=filepath)
        else:
            from tvbo.utils import to_yaml as _to_yaml

            kwargs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            clean = self.__class__.__bases__[0](**kwargs)
            return _to_yaml(clean, filepath)

    # ---- JAX pytree: flatten/unflatten ----
    _STATIC_SPEC_HELD_OUT = ("weight", "length", "parcellation", "edges", "data_file", "bids_dir")
    """Slots the static spec omits: the matrices travel as leaves, the edge declarations hold non-deterministic `Parameter` strings, and the loading slots would make `_pytree_build` re-resolve from disk."""

    def _pytree_leaves(self) -> dict:
        """The resident arrays, keyed by companion path: this network's JAX children.

        JAX flattens a dict by sorted key, so the order arrays were materialised in never retraces and the key set is the treedef: `materialize("weight", "gain")` and `materialize("gain", "weight")` compile once, `materialize("weight")` compiles again. Every resident array is a leaf whatever its shape or dtype — a `(276, 16384)` leadfield and a `(327684, 3)` vertex array as much as the connectome — so `jax.grad` returns a cotangent for each float one and, with ``allow_int=True``, a `float0` for integer topology.

        Nothing is read here. A network whose companion offers arrays it has not materialised flattens to an empty set and warns, because the solver would then trace nothing and read the connectome as a constant inside the trace; under ``TVBO_JAX_STRICT`` that is an error. A scipy sparse matrix is refused by name: it is not a JAX type, and converting it here would decide a treedef the caller never asked for.
        """
        from scipy import sparse

        arrays = self.arrays
        for key, value in arrays.items():
            if sparse.issparse(value):
                raise TypeError(
                    f"{self}: arrays[{key!r}] is a scipy sparse matrix, which is not a JAX type. Materialise it as a BCOO or a dense array before entering a JAX transformation."
                )
        if not arrays and self._lazy_paths():
            message = f"{self} flattens with nothing materialised; its companion offers {', '.join(self._lazy_paths())}. Call materialize() before entering a JAX transformation, or the connectome is read as a constant inside the trace."
            if os.environ.get("TVBO_JAX_STRICT"):
                raise RuntimeError(message)
            warnings.warn(message, LazyMaterializationWarning, stacklevel=3)
        return arrays

    def _pytree_static(self) -> str:
        """The spec as canonical JSON, without the slots `_STATIC_SPEC_HELD_OUT` names — compared as a string, which is what keeps treedef equality cheap."""
        return static_spec(self, self._STATIC_SPEC_HELD_OUT)

    @classmethod
    def _pytree_build(cls, static: str, leaves: dict) -> "Network":
        """The spec rebuilt from its JSON, the children installed as the resident arrays.

        The arrays are installed as they arrive — tracers under a transformation — and `matrix` hands a JAX array back untouched, so what a traced computation reads is the leaf JAX gave it and never a pre-trace attribute.
        """
        import json as _json

        obj = cls(**_json.loads(static))
        object.__setattr__(obj, "_arrays", dict(leaves))
        return obj

    # Back-compat pointer
    @property
    def metadata(self) -> "Network":
        """Back-compatible pointer that returns the network itself as its metadata."""
        return self

    # ---- Numeric accessors (compute on demand; no extra attributes) ----
    def _matrix_from_array(self, arr: np.ndarray | JaxArray) -> tvbo_datamodel.Matrix:
        arr = jnp.array(arr)
        N0, N1 = arr.shape
        x = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N0)])
        y = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N1)])
        return tvbo_datamodel.Matrix(x=x, y=y, values=arr.reshape(-1).astype(jnp.float32).tolist())

    def node_index_map(self) -> dict[int, int]:
        """``{Node.id: row/column index}`` for the connectome matrices.

        ``Node.id`` is a *unique identifier* (``dcterms:identifier``), not a position: a network may declare ``[{id: 0}, {id: 2}]`` and its edges then address nodes by those ids, while the matrices are indexed by declaration order. Matrix-only networks (no ``nodes``) address rows directly, so the map is the identity there.
        """
        return {(i if getattr(nd, "id", None) is None else int(nd.id)): i for i, nd in enumerate(self.nodes or [])}

    def _placed_edges(self) -> list:
        """The edges that connect two nodes. An edge with no endpoints is a template edge naming a companion matrix, not a connection."""
        return [e for e in (self.edges or []) if e.source is not None and e.target is not None]

    def _edge_matrix(self, value_of, fill: float = 0.0) -> np.ndarray | None:
        """Build one connectome matrix from the explicit edges.

        The single place where an edge becomes matrix entries: endpoints are resolved from ``Node.id`` through :meth:`node_index_map` and bounds-checked once here, so every connectome matrix reads the same connectome.
        ``value_of(edge, i, j)`` supplies the entry (``None`` skips the edge) and receives the resolved indices; the raw ids stay on ``edge`` for callers that need them. Matrices are target-by-source — an edge source -> target is stored at ``[target, source]``, the coupling convention backends expect — and undirected edges are mirrored. ``None`` when the network declares no edge that actually connects two nodes.
        """
        # Endpointless entries are TEMPLATE edges: they name a matrix carried in the companion file, they are not connections. Select before allocating, or a template-only network (any connectome loaded from `data_file:`) builds a dense N x N of `fill` and then scans it — 8.4 GB and a 1e9-element pass for a 32k-vertex mesh, to return a matrix with nothing in it.
        placed = self._placed_edges()
        if not placed:
            return None
        n = self.number_of_nodes or self.number_of_regions or 1
        index_map = self.node_index_map()
        M = np.full((n, n), fill, dtype=np.float64)
        for edge in placed:
            i, j = edge.source, edge.target
            i, j = index_map.get(i, i), index_map.get(j, j)
            if not (0 <= i < n and 0 <= j < n):
                continue
            value = value_of(edge, i, j)
            if value is None:
                continue
            M[j, i] = value
            if not edge.directed:
                M[i, j] = value
        return M

    def _weights_from_edges(self) -> np.ndarray | None:
        """Connectome weights from the explicit edges; an edge with no weight counts as 1."""
        return self._edge_matrix(lambda edge, i, j: edge_param(edge, "weight", 1.0))

    @property
    def node_parameter_vectors(self) -> dict[str, np.ndarray]:
        """Per-node parameters as ``{name: (n_nodes,) array}``, in declared node order.

        Only parameters that *every* node declares are returned (a partial vector has no well-defined value for the gaps). Consumed by symbolic weight/length transforms, which expose each as an ``(n, 1)`` column so an expression like ``W / roi_size`` normalises per target region.
        """
        nodes = self.nodes or []
        if not nodes:
            return {}
        names = set.intersection(*[set((getattr(nd, "parameters", None) or {}).keys()) for nd in nodes])
        out: dict[str, np.ndarray] = {}
        for name in names:
            vals = [getattr(nd.parameters[name], "value", None) for nd in nodes]
            if all(v is not None for v in vals):
                out[str(name)] = np.asarray([float(v) for v in vals], dtype=np.float64)
        return out

    def node_positions(self) -> np.ndarray:
        """Node coordinates as an ``(n_nodes, 3)`` array, in declared node order.

        A missing ``z`` defaults to 0; a node with no position at all raises, since a partial coordinate matrix (a mesh, a distance calc) is silently wrong rather than merely incomplete. Use ``_get_node_position`` for a tolerant per-node lookup.
        """
        out = []
        for node in self.nodes or []:
            pos = getattr(node, "position", None)
            if pos is None:
                raise ValueError(
                    f"node {getattr(node, 'id', '?')!r} has no position; a full (n_nodes, 3) coordinate array cannot be built."
                )
            out.append([pos.x, pos.y, getattr(pos, "z", 0.0) or 0.0])
        return np.asarray(out, dtype=float)

    def _get_node_position(self, node_id: int) -> tuple[float, float, float] | None:
        """Get (x, y, z) position for a node by ID."""
        if not self.nodes:
            return None
        for node in self.nodes:
            if getattr(node, "id", None) == node_id:
                pos = getattr(node, "position", None)
                if pos is not None:
                    x = getattr(pos, "x", None)
                    y = getattr(pos, "y", None)
                    z = getattr(pos, "z", 0.0)  # default z=0 if not specified
                    if x is not None and y is not None:
                        return (float(x), float(y), float(z) if z else 0.0)
        return None

    def _compute_euclidean_distance(self, i: int, j: int) -> float | None:
        """Compute Euclidean distance between two nodes from their positions."""
        pos_i = self._get_node_position(i)
        pos_j = self._get_node_position(j)
        if pos_i is None or pos_j is None:
            return None
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        dz = pos_j[2] - pos_i[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _lengths_from_edges(self) -> np.ndarray | None:
        """Tract lengths from the explicit edges.

        Reads ``length``, then ``distance``; with neither declared, falls back to the Euclidean distance between the two nodes' positions (in ``distance_unit``).
        """

        def length(edge, i, j):
            d = edge_param(edge, "length")
            if d is None:
                d = edge_param(edge, "distance")
            if d is None or d == 0:
                # Positions are keyed by Node.id, so look up the edge's own endpoints.
                d = self._compute_euclidean_distance(edge.source, edge.target)
            return 0.0 if d is None else d

        return self._edge_matrix(length)

    @property
    def node_labels(self) -> list[str]:
        """Node labels derived from nodes.

        Returns:
        -------
        list of str
            Labels for each node in the network

        Examples:
        --------
        ```python
        net = Network.from_matrix(weights, lengths, labels=["A", "B", "C"])
        print(net.node_labels)  # ['A', 'B', 'C']
        ```
        """
        if not self.nodes:
            return []
        return [n.label for n in self.nodes]  # type: ignore[union-attr]

    def _atlas_terminology_entities(self) -> dict:
        """Parcellation-terminology entities for this network's atlas, or ``{}``.

        Resolves ``parcellation.atlas.name`` (case-insensitively — networks may declare ``HCPMMP1`` while the packaged atlas is keyed ``hcpmmp1``) and reads its SANDS ``terminology.entities``. Returns ``{}`` when the network declares no atlas or the atlas has no terminology. A network that DOES declare an atlas this installation cannot resolve still returns ``{}`` (a custom parcellation is entitled to have no crosswalk) but warns, because the visible symptom is otherwise a reconciliation that silently matches only the labels that happen to agree verbatim.
        """
        parc = getattr(self, "parcellation", None)
        atlas = getattr(parc, "atlas", None) if parc is not None else None
        name = getattr(atlas, "name", None) if atlas is not None else None
        if not name:
            return {}
        try:
            from tvbo.classes.atlas import Atlas, available_atlases

            resolved = (
                name
                if name in available_atlases
                else next((a for a in available_atlases if a.lower() == str(name).lower()), None)
            )
            if not resolved:
                import logging

                logging.getLogger(__name__).warning(
                    "Network declares atlas %r, which is not among the available atlases %s — region aliases unavailable, "
                    "so a by_label reconciliation will match only labels that agree verbatim.",
                    name,
                    sorted(available_atlases),
                )
                return {}
            term = getattr(Atlas(resolved), "terminology", None)
            return getattr(term, "entities", None) or {}
        except Exception as exc:  # noqa: BLE001 — aliases are optional; degrade but surface
            import logging

            logging.getLogger(__name__).warning(
                "Could not load atlas terminology for %r (region aliases unavailable): %s",
                name,
                exc,
            )
            return {}

    def region_alias_map(self) -> dict[str, str]:
        """Map each node's canonical label AND every known alias -> canonical label.

        Aliases are alternative label strings that denote the SAME region under a different nomenclature. They come from two sources, unioned: each node's own ``alternateName`` (inline on the network) and, when the network declares a parcellation atlas, that atlas terminology's names per region. The atlas join is by NAME, never by position, and matches a region to a node whose label is either the region's canonical name or any of its ``alternateName`` entries — so a network that spells a region divergently (``Left-Thalamus-Proper`` where the atlas says ``L_Thalamus``) still inherits that region's whole crosswalk, keyed on the label the network itself uses. The identity ``label -> label`` is always included so exact matches still resolve.

        Used by ``by_label`` node reconciliation so a dataset-sourced target whose nodes carry a divergent convention (e.g. ``THALAMUS_LEFT`` for ``L_Thalamus``) aligns by name. Raises when one alias would map to two different canonical labels, or when one atlas region's names match several nodes — an ambiguous crosswalk must fail loudly rather than silently mis-assign a region (and, in particular, a hemisphere).
        """
        labels = self.node_labels
        canon_set = set(labels)
        index: dict[str, str] = {}

        def _add(alias: str, canonical: str) -> None:
            prev = index.get(alias)
            if prev is not None and prev != canonical:
                raise ValueError(
                    f"Ambiguous node alias {alias!r}: maps to both {prev!r} and "
                    f"{canonical!r}. A region alias must identify exactly one node."
                )
            index[alias] = canonical

        for lbl in labels:
            _add(str(lbl), str(lbl))
        for node in self.nodes or []:
            canonical = str(getattr(node, "label", "") or "")
            for alias in getattr(node, "alternateName", None) or []:
                _add(str(alias), canonical)
        for key, ent in self._atlas_terminology_entities().items():
            names = {str(getattr(ent, "name", None) or key)}
            names.update(str(alias) for alias in getattr(ent, "alternateName", None) or [])
            hits = sorted(names & canon_set)
            if not hits:
                continue  # atlas region not present in this (sub)network
            if len(hits) > 1:
                raise ValueError(
                    f"Ambiguous atlas region {key!r}: its names {hits} match more than "
                    f"one node of this network. A region must identify exactly one node."
                )
            for alias in sorted(names - {hits[0]}):
                _add(alias, hits[0])
        return index

    def as_data_file_reference(self, data_file: str):
        """A compact datamodel ``Network`` that points at *data_file* and still IS this network.

        Freezing a connectome writes its matrices to a companion file and replaces the network in the rendered spec with this reference, so what travels beside ``data_file`` has to be everything the reloaded network needs in order to behave identically: the inline coupling / transforms / parameters, the scalar identity, the measure declarations — ``Network.observations`` and the structural resolution gate on those, so a companion holding ``BoldCorrelation`` data is invisible unless the reference also declares ``observational_measures: [BoldCorrelation]`` — and the ``parcellation``, which names the atlas that a ``by_label`` node crosswalk resolves against. Carrying the parcellation cannot re-expand the node set, because ``data_file`` makes the loader defer connectivity to the companion store.

        Note that ``observations`` is deliberately NOT copied: it is a runtime view over the companion's measures, not a schema slot, and ``observational_measures`` is what reconstructs it.
        """
        from tvbo import datamodel as dm

        ref = dm.Network(data_file=data_file)
        for key, value in keyed_items(getattr(self, "coupling", None), "coupling"):
            ref.coupling[key] = value
        if getattr(self, "transforms", None):
            ref.transforms = list(self.transforms)
        if getattr(self, "parameters", None):
            for key, value in dict(self.parameters).items():
                ref.parameters[key] = value
        for field in (
            "label",
            "descriptor",
            "number_of_nodes",
            "distance_unit",
            "time_unit",
            "structural_measures",
            "observational_measures",
            "parcellation",
        ):
            value = getattr(self, field, None)
            if value is not None:
                setattr(ref, field, list(value) if isinstance(value, (list, tuple)) else value)
        return ref

    @property
    def weights_matrix(self) -> np.ndarray | JaxArray | None:
        """Deprecated: use ``matrix("weight")``."""
        _warn_superseded_accessor("weights_matrix", 'matrix("weight")')
        return self._weights_matrix(apply_transforms=True)

    @property
    def raw_weights_matrix(self) -> np.ndarray | JaxArray | None:
        """Deprecated: use ``matrix("weight", apply_transforms=False)``."""
        _warn_superseded_accessor("raw_weights_matrix", 'matrix("weight", apply_transforms=False)')
        return self._weights_matrix(apply_transforms=False)

    def _weights_matrix(self, apply_transforms: bool = True) -> np.ndarray | JaxArray | None:
        """Connection weights as a dense array — a thin wrapper over ``matrix("weight")``.

        It adds nothing: the pytree payload, the edge-derived fallback and the unconnected zeros all live in `matrix` now. Kept only because the deprecated properties route through it; new callers should use `matrix` directly.
        """
        return self.matrix("weight", format="dense", apply_transforms=apply_transforms)

    @property
    def weights(self):
        """Deprecated: use ``matrix("weight")``."""
        _warn_superseded_accessor("weights", 'matrix("weight")')
        return self._weights_matrix(apply_transforms=True)

    @property
    def lengths_matrix(self) -> np.ndarray | JaxArray | None:
        """Tract length matrix as numpy/JAX array.

        Returns the (N x N) matrix of physical distances (tract lengths) between brain regions in millimeters.

        Returns:
        -------
        np.ndarray or jax.Array, optional
            Tract lengths matrix (N x N) in mm, or None if no matrix/edges

        Examples:
        --------
        ```python
        net = Network.from_matrix(weights, lengths)
        L = net.lengths_matrix
        print(f"Mean length: {L.mean():.1f} mm")
        ```
        """
        if len(self.nodes) == 1:
            return np.zeros((1, 1))
        # None, not zeros: "no tract lengths" is not "every tract has length zero", and standing in a dense N x N of zeros defeats every `is not None` delay guard downstream.
        return self.matrix("length", format="dense")

    @property
    def lengths(self):
        """Tract-length matrix in millimetres (alias for `lengths_matrix`)."""
        return self.lengths_matrix

    @property
    def distances(self):
        """Tract-length matrix in millimetres (alias of `lengths`)."""
        return self.lengths_matrix

    @property
    def observations(self) -> dict[str, np.ndarray]:
        """Observational-measure matrices carried by the network.

        Returns a ``{measure_name: matrix}`` dict for every entry in ``self.observational_measures`` (e.g. ``BoldCorrelation`` — the empirical FC target). Each is one more edge matrix, read through :meth:`array` — resident if `from_bids` or `set_array` put it there, read from the companion otherwise — so experiments consume network observations the same way they consume ``weights``/``distances``.

        Returns:
        -------
        dict of str to np.ndarray
            Empty dict when the network declares no observational measures.
        """
        from scipy import sparse as _sp

        measures = list(self.observational_measures or [])
        if not measures:
            return {}

        def _dense(arr):
            if arr is None:
                return None
            arr = arr.toarray() if _sp.issparse(arr) else np.asarray(arr)
            return arr

        out: dict[str, np.ndarray] = {}
        for name in measures:
            m = _dense(self.array(f"edges/{name}"))
            if m is not None:
                out[name] = m
        return out

    @property
    def labels(self) -> dict[str, str]:
        """Brain region labels from atlas.

        Returns:
        -------
        dict of str to str
            Mapping from region names to lookup labels

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        labels = sc.labels
        print(f"Number of labeled regions: {len(labels)}")
        ```
        """
        atlas = self.get_atlas()
        if atlas.terminology:
            return {e.name: e.lookupLabel for k, e in atlas.terminology.entities.items()}
        return {}

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Build NetworkX MultiDiGraph from network nodes and edges.

        Priority:
        1. Use explicit nodes if available (with their properties)
        2. Use explicit edges if available
        3. Generate edges from weight/length matrices if no explicit edges
        4. Fall back to matrix-only representation if no nodes defined

        Returns:
        -------
        nx.MultiDiGraph
            Graph with node/edge attributes from schema.
            Nodes have: id, label, dynamics, region, position, parameters
            Edges have: weight, delay, distance, directed, source_var, target_var, coupling
        """
        G = nx.MultiDiGraph()

        W = self._weights_matrix()

        # Step 1: Add nodes (prefer explicit nodes, fall back to matrix size)
        if self.nodes:
            for node in self.nodes:
                node_id = node.id if node.id is not None else 0
                node_attrs = {
                    "label": node.label or f"node_{node_id}",
                    "dynamics": node.dynamics,
                    "region": node.region,
                }
                if node.position:
                    x = node.position.x
                    y = node.position.y
                    z = getattr(node.position, "z", 0) or 0
                    node_attrs["x"] = x
                    node_attrs["y"] = y
                    node_attrs["z"] = z
                    node_attrs["pos"] = np.array([x, y, z])
                if node.parameters:
                    for name, param in node.parameters.items():
                        node_attrs[f"param_{name}"] = param.value
                G.add_node(node_id, **node_attrs)
        elif W is not None:
            # No explicit nodes - create from matrix dimensions
            n = W.shape[0]
            for i in range(n):
                G.add_node(i, label=f"node_{i}")

        # Step 2: Add edges (prefer explicit edges, fall back to matrix) Filter out template edges (no source/target) — those represent matrix measures stored in the HDF5 companion, not graph edges.
        explicit_edges = [e for e in (self.edges or []) if getattr(e, "source", None) is not None]
        # Only the matrix fallback needs this: there ``i``/``j`` really are positions, whereas an explicit ``edge.source`` is already a ``Node.id`` (see :meth:`node_index_map`).
        index_to_id = {i: (node.id if node.id is not None else i) for i, node in enumerate(self.nodes or [])}
        if explicit_edges:
            # Use explicit edges
            for edge in explicit_edges:
                edge_attrs = {
                    "directed": edge.directed,
                    "source_var": edge.source_var,
                    "target_var": edge.target_var,
                    "coupling": edge.coupling,
                }
                if edge.parameters:
                    for name, param in edge.parameters.items():
                        edge_attrs[name] = param.value
                        if param.unit:
                            edge_attrs[f"{name}_unit"] = param.unit

                G.add_edge(edge.source, edge.target, **edge_attrs)

                if not edge_attrs["directed"]:
                    G.add_edge(edge.target, edge.source, **edge_attrs)

        else:
            # No explicit edges - generate from stored matrices. Build edges from the union of all stored matrices, attaching each matrix's values as named edge attributes.
            from scipy import sparse as _sp

            # Collect matrix names from in-memory arrays, template-edge metadata, and common aliases accessible via the generic matrix(...) accessor.
            matrix_names = set(self.edge_arrays())
            for e in self.edges or []:
                lbl = getattr(e, "label", None) or getattr(e, "name", None)
                if lbl:
                    matrix_names.add(lbl)
            for name in ("weight", "weights", "length", "lengths", "sc", "fc"):
                if self.matrix(name) is not None:
                    matrix_names.add(name)

            # Materialize all available matrices as dense arrays.
            dense = {}
            for name in matrix_names:
                mat = self.matrix(name)
                if mat is None:
                    continue
                dense[name] = mat.toarray() if _sp.issparse(mat) else np.asarray(mat)

            if dense:
                n = next(iter(dense.values())).shape[0]
                if self.nodes and len(self.nodes) != n:
                    raise ValueError(f"Matrix dimensions ({n}) don't match number of nodes ({len(self.nodes)})")
                for i in range(n):
                    for j in range(n):
                        edge_attrs = {"directed": True}
                        any_nonzero = False
                        for name, mat in dense.items():
                            val = float(mat[i, j])
                            edge_attrs[name] = val
                            if val != 0:
                                any_nonzero = True
                        if any_nonzero:
                            # Ensure "weight" exists for networkx compatibility
                            if "weight" not in edge_attrs:
                                edge_attrs["weight"] = next(
                                    (v for v in edge_attrs.values() if isinstance(v, float) and v != 0),
                                    1.0,
                                )
                            # Receiver-row matrices: emit j -> i (signal direction, matching create_graph); index_to_id maps positions onto node ids.
                            G.add_edge(
                                index_to_id.get(j, j),
                                index_to_id.get(i, i),
                                **edge_attrs,
                            )

        return G

    def __str__(self) -> str:
        parc = getattr(self, "parcellation", None)
        if parc and hasattr(parc, "atlas") and hasattr(parc.atlas, "name"):  # type: ignore[attr-defined]
            return f"Network-{parc.atlas.name}({self.number_of_regions})"  # type: ignore[attr-defined]
        return f"Network(N={self.number_of_regions})"

    def __repr__(self) -> str:
        """The network, and which of its arrays are resident and which are still on disk.

        The difference between the two lists is the answer to "why did this script use 40 GB" — or why it did not.
        """
        resident = sorted(self._resident())
        lazy = sorted(set(self._lazy_paths()) - set(resident))
        parts = [self.__str__()]
        if resident:
            parts.append("resident: " + ", ".join(resident))
        if lazy:
            parts.append("lazy: " + ", ".join(lazy))
        return " | ".join(parts)

    @property
    def atlas(self) -> Any:
        """Brain atlas associated with this connectome.

        Returns:
        -------
        Atlas
            Atlas object containing parcellation metadata

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        atlas = sc.atlas
        print(atlas.region_labels)
        ```
        """
        return self.get_atlas()

    def get_atlas(self) -> Any:
        """Retrieve the Atlas object for this connectome.

        Returns:
        -------
        Atlas
            Atlas instance with parcellation metadata and terminology

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        atlas = sc.get_atlas()
        ```
        """
        from tvbo.classes.atlas import Atlas

        parc = getattr(self, "parcellation", None)
        atlas_data = parc.atlas if parc and hasattr(parc, "atlas") else None  # type: ignore[attr-defined]
        return Atlas(atlas_data)

    def compute_delays(self, output_unit: str | None = None) -> np.ndarray | JaxArray:
        """Deprecated: use :meth:`calculate_delays` instead.

        Parameters
        ----------
        output_unit : str, optional
            Passed through to ``calculate_delays(output_unit=...)``.
        """
        import warnings

        warnings.warn(
            "compute_delays() is deprecated, use calculate_delays() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_delays(output_unit=output_unit)

    def execute(
        self,
        format: str = "tvb",
        target=None,
        threshold_percentile: float = 85,
        **kwargs,
    ) -> Any:
        """Convert connectome to simulator-specific format.

        Parameters
        ----------
        format : str, default="tvb"
            Target format: ``"tvb"``, ``"networkx"``, or ``"tvboptim"``.
        target : Network, optional
            Target network for bipartite projection graphs
            (used with ``format="networkx"`` when a gain matrix exists).
        threshold_percentile : float, default=85
            Keep only gain edges above this percentile (networkx only).
        **kwargs
            Extra arguments forwarded to the adapter. For tvboptim:

            - ``delays=False`` — force a ``DenseGraph`` without delays.
            - ``return_type="graph"`` — return only the graph object.
            - ``return_type="network"`` (default) — return a full tvboptim
              ``Network`` (requires ``dynamics`` and ``coupling`` kwargs).
            - ``dynamics`` — tvboptim dynamics instance.
            - ``coupling`` — tvboptim coupling instance(s).
            - ``noise`` — tvboptim noise instance (optional).

        Returns:
        -------
        Any
            Connectivity object in the specified format.
        """
        if format == "tvb":
            from tvbo.adapters.tvb import to_tvb

            return to_tvb(self)
        elif format == "networkx":
            return self._build_networkx_graph(
                target=target,
                threshold_percentile=threshold_percentile,
            )
        elif format.lower() in ("tvboptim", "tvb-optim"):
            from tvbo.adapters.tvboptim import to_tvboptim

            return to_tvboptim(self, **kwargs)
        raise ValueError(f"Format {format!r} not supported. Valid formats: tvb, networkx, tvboptim.")

    def _build_networkx_graph(
        self,
        target=None,
        threshold_percentile: float = 85,
    ) -> nx.MultiDiGraph:
        """Build a NetworkX graph, optionally bipartite for projection networks.

        When the network contains a rectangular gain matrix and a *target* network is provided, a bipartite graph is built with sensor nodes from ``self`` and region nodes from *target*.

        If *target* is ``None``, the method looks for a ``target_network`` reference on the gain edge and loads it automatically.  Failing that, it falls back to ``dimension_labels`` stored on the edge to create label-only region nodes (no positions).

        Parameters
        ----------
        target : Network, optional
            Target brain network whose nodes receive the projection columns.
        threshold_percentile : float, default=85
            Keep only gain edges above this percentile of nonzero values.
        """
        gain = self.matrix("gain")
        if gain is None:
            return self.graph

        gain = np.asarray(gain)

        # Auto-resolve target from edge metadata when not provided
        if target is None:
            gain_edge = next(
                (e for e in (self.edges or []) if getattr(e, "label", None) == "gain"),
                None,
            )
            if gain_edge is not None:
                ref = getattr(gain_edge, "target_network", None)
                if ref:
                    target = Network.from_file(str(NETWORK_DIR / ref))

        if target is not None:
            return self._build_projection_graph(
                target,
                gain,
                threshold_percentile,
            )

        return self.graph

    def _build_projection_graph(
        self,
        target,
        gain: np.ndarray,
        threshold_percentile: float = 85,
    ) -> nx.MultiDiGraph:
        """Build a bipartite sensor-region projection graph from a gain matrix."""
        G = nx.MultiDiGraph()

        # --- sensor nodes (from self) ---
        sensor_labels = []
        for node in self.nodes:
            lbl = f"S:{node.label}"
            sensor_labels.append(lbl)
            if node.position:
                pos = np.array(
                    [
                        node.position.x,
                        node.position.y,
                        getattr(node.position, "z", 0) or 0,
                    ]
                )
            else:
                pos = np.zeros(3)
            G.add_node(
                lbl,
                pos=pos,
                x=pos[0],
                y=pos[1],
                z=pos[2],
                color="red",
                node_type="sensor",
                label=str(node.label),
            )

        # --- region nodes (from target, cortical subset) ---
        n_sensors, n_cols = gain.shape
        target_nodes = list(target.nodes)
        # If gain columns < total target nodes, select cortical subset
        if len(target_nodes) > n_cols:
            cortical = [n for n in target_nodes if n.label and str(n.label).startswith("ctx-")]
            if len(cortical) == n_cols:
                target_nodes = cortical
            else:
                target_nodes = target_nodes[-n_cols:]

        region_labels = []
        for node in target_nodes[:n_cols]:
            lbl = f"R:{node.label}"
            region_labels.append(lbl)
            if node.position:
                pos = np.array(
                    [
                        node.position.x,
                        node.position.y,
                        getattr(node.position, "z", 0) or 0,
                    ]
                )
            else:
                pos = np.zeros(3)
            G.add_node(
                lbl,
                pos=pos,
                x=pos[0],
                y=pos[1],
                z=pos[2],
                color="blue",
                node_type="region",
                label=str(node.label),
            )

        # --- gain edges (thresholded) ---
        vals = np.abs(gain)
        nonzero = vals[vals > 0]
        threshold = np.percentile(nonzero, threshold_percentile) if nonzero.size > 0 and threshold_percentile > 0 else 0.0
        for i in range(n_sensors):
            for j in range(n_cols):
                v = float(vals[i, j])
                if v > threshold:
                    G.add_edge(
                        region_labels[j],
                        sensor_labels[i],
                        gain=v,
                        weight=v,
                    )

        return G

    def normalize_weights(self, equation_rhs: str | None = None) -> None:
        """Add a normalization transform for connection weights.

        Convenience wrapper for ``add_transform("weight", ...)``.

        Parameters
        ----------
        equation_rhs : str, optional
            Right-hand side of the normalization equation, written over the network's
            edge attributes. Defaults to min-max normalisation of ``weight``.

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize_weights("weight / max(weight)")  # Normalize to [0, 1]
        normalized = sc.matrix("weight")  # Returns normalized weights
        ```

        See Also:
        --------
        add_transform : Add a transform on any edge property
        """
        self.add_transform("weight", equation_rhs)

    def plot_weights(self, ax: Axes, cmap: str = "magma", log: bool = False) -> Any:
        """Plot connection weights matrix as heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        cmap : str, default="magma"
            Matplotlib colormap name
        log : bool, default=False
            If True, use logarithmic color scale

        Returns:
        -------
        matplotlib.image.AxesImage
            Image object for adding colorbar

        Examples:
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        fig, ax = plt.subplots()
        im = sc.plot_weights(ax, log=True)
        plt.colorbar(im, ax=ax)
        ```
        """
        import numpy as np
        from matplotlib.colors import LogNorm

        weights = self._weights_matrix()
        if weights is None:
            weights = np.zeros((1, 1))

        if log:
            # Use LogNorm with vmin set to smallest non-zero value to avoid white holes
            nonzero_weights = weights[weights > 0]  # type: ignore[index,operator]
            vmin = float(nonzero_weights.min()) if nonzero_weights.size > 0 else 1e-10  # type: ignore[attr-defined]
            vmax = float(weights.max()) if weights.max() > 0 else 1.0  # type: ignore[attr-defined]
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        im = ax.imshow(weights, cmap=cmap, interpolation="none", norm=norm)  # type: ignore[arg-type]
        ax.set_title("weight")
        ax.set_box_aspect(1)
        return im

    def plot_lengths(self, ax: Axes, cmap: str = "magma") -> Any:
        """Plot tract lengths matrix as heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        cmap : str, default="magma"
            Matplotlib colormap name

        Returns:
        -------
        matplotlib.image.AxesImage
            Image object for adding colorbar

        Examples:
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        fig, ax = plt.subplots()
        im = sc.plot_lengths(ax)
        plt.colorbar(im, ax=ax, label="mm")
        ```
        """
        lengths = self.lengths_matrix
        if lengths is None:
            lengths = np.zeros((1, 1))
        im = ax.imshow(lengths, cmap=cmap, interpolation="none")  # type: ignore[arg-type]
        ax.set_title("length")
        ax.set_box_aspect(1)
        return im

    def plot_matrix(self, log_weights: bool = False, cmap: str = "magma") -> Figure:
        """Plot both weights and lengths matrices side by side.

        Parameters
        ----------
        log_weights : bool, default=False
            If True, use log scale for weights colormap
        cmap : str, default="magma"
            Matplotlib colormap name

        Returns:
        -------
        matplotlib.figure.Figure
            Figure containing both matrix plots

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.plot_matrix(log_weights=True)
        ```
        """
        fig, axs = plt.subplots(ncols=2, sharey=True)

        w = self.plot_weights(axs[0], cmap=cmap, log=log_weights)
        fig.colorbar(w, ax=axs[0], shrink=0.5)

        lengths_im = self.plot_lengths(axs[1], cmap=cmap)
        fig.colorbar(lengths_im, ax=axs[1], shrink=0.5)

        plt.close()
        return fig

    def calculate_delays(
        self,
        conduction_speed: float | None = None,
        output_unit: str | None = None,
    ) -> np.ndarray | JaxArray:
        """Calculate signal propagation delays between regions.

        Supports two network representations:

        1. **Matrix-based** — delays are ``lengths / conduction_speed``, with optional unit conversion via *output_unit*.
        2. **Edge-based** — delays are extracted from explicit edge objects that carry ``source``, ``target``, and a ``"delay"`` parameter.

        Parameters
        ----------
        conduction_speed : float, optional
            Override conduction speed. If *None*, uses ``self.conduction_speed``.
        output_unit : str, optional
            Desired output time unit (e.g. ``"ms"``, ``"s"``). When given,
            sympy unit conversion is applied. If *None*, the result is in the
            network's native time unit (defaults to ms).

        Returns:
        -------
        np.ndarray or jax.Array
            Delay matrix (N x N). For edge-based networks, entries without an
            edge are ``NaN``.

        Raises:
        ------
        ValueError
            If neither lengths matrix nor edge-based delays are available.

        Examples:
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        delays = sc.calculate_delays(conduction_speed=3.0)
        plt.imshow(delays, cmap='viridis')
        plt.colorbar(label='Delay (ms)')
        ```
        """
        # --- Edge-based path: edges with source/target indices ---
        if hasattr(self, "edges") and self.edges:
            # Only use edge path when edges are actual connections (have source/target)
            edge_delays = self._delays_from_edges()
            if edge_delays is not None:
                return edge_delays

        # --- Matrix-based path: lengths / conduction_speed ---
        if conduction_speed is None:
            cs_param = getattr(self, "conduction_speed", None)
            if cs_param and hasattr(cs_param, "value"):
                conduction_speed = cs_param.value  # type: ignore[attr-defined]
            else:
                conduction_speed = 3.0  # default fallback

        lengths = self.lengths_matrix
        if lengths is None:
            raise ValueError("Lengths matrix is not available")

        delays = lengths / conduction_speed  # type: ignore[operator]

        if output_unit is not None:
            delays = delays * self._unit_conversion_factor(output_unit)

        return delays

    def _delays_from_edges(self) -> np.ndarray | None:
        """Explicit per-edge delays, or ``None`` when no edge declares a positive one.

        Unconnected entries stay ``NaN`` (consumers such as ``adapters.tvboptim._build_graph`` zero them), which is what distinguishes "no edge" from "an edge with delay 0".
        """
        delays = self._edge_matrix(lambda edge, i, j: edge_param(edge, "delay"), fill=np.nan)
        if delays is None or not np.any(np.nan_to_num(delays) > 0):
            return None
        return delays

    def _unit_conversion_factor(self, output_unit: str) -> float:
        """Compute multiplicative factor to convert native delay units to *output_unit*."""
        import sympy.physics.units as u
        from sympy import nsimplify
        from sympy.parsing.sympy_parser import parse_expr

        from tvbo.utils.units import time_unit_of, unit_to_symbol

        unit_ns = dict(vars(u))

        distance_unit_str = unit_to_symbol(getattr(self, "distance_unit", None) or "mm")
        cs_param = self.conduction_speed
        speed_unit_str = (
            unit_to_symbol(cs_param.unit)
            if cs_param and cs_param.unit
            else f"{distance_unit_str}/{unit_to_symbol(time_unit_of(self))}"
        )
        target_time_str = unit_to_symbol(output_unit)

        # Native delay unit: distance / speed  (e.g. mm / (mm/ms) = ms)
        native_delay = parse_expr(distance_unit_str, local_dict=unit_ns) / parse_expr(speed_unit_str, local_dict=unit_ns)
        target_unit = parse_expr(target_time_str, local_dict=unit_ns)

        converted = u.convert_to(native_delay, target_unit)
        return float(nsimplify(converted / target_unit))

    def create_graph(self, weight_threshold: float = 0) -> nx.MultiDiGraph:
        """Create NetworkX graph from network structure.

        Prioritizes explicit nodes/edges representation over weight matrices.
        This allows proper visualization of heterogeneous networks with labeled nodes and typed edges.

        Parameters
        ----------
        weight_threshold : float, default=0
            Minimum weight for including an edge in the graph

        Returns:
        -------
        networkx.MultiDiGraph
            Directed multigraph with 'weight' and 'delay' edge attributes.
            Nodes have 'label' and 'dynamics' attributes when available.
            Edges have 'source_var', 'target_var' attributes when available.
            Edges point in signal direction (source to target). Stored
            matrices are receiver-row (``W[i, j]`` couples node ``j`` into
            node ``i``), so matrix entries are emitted as edges ``j -> i``.
            Explicit pair edges declared with ``directed: false`` (the schema
            default) are mirrored into both directions, matching their
            expansion in ``_edge_matrix``; an explicitly declared reverse
            edge takes precedence over the mirror.

        Examples:
        --------
        ```python
        # From explicit nodes/edges
        network = Network(nodes=[...], edges=[...])
        G = network.create_graph()

        # From weight matrix
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        G = sc.create_graph(weight_threshold=0.1)
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        ```
        """
        G = nx.MultiDiGraph()

        # Priority 1: explicit nodes, unless the edges are only matrix descriptors (no source/target).
        nodes = getattr(self, "nodes", None)
        edges = getattr(self, "edges", None) or []
        pair_edges = [e for e in edges if getattr(e, "source", None) is not None and getattr(e, "target", None) is not None]
        matrix_only_edges = bool(edges) and not pair_edges

        if nodes and len(nodes) > 0 and not matrix_only_edges:
            # Build graph from explicit node/edge representation
            for node in nodes:
                node_id = getattr(node, "id", None)
                if node_id is None:
                    continue
                node_attrs = {
                    "label": getattr(node, "label", None) or f"node_{node_id}",
                    "dynamics": getattr(node, "dynamics", None),
                }
                G.add_node(node_id, **node_attrs)

            # Only surviving edges count as declared: a reverse edge dropped for being too weak must not suppress the mirror and leave the pair one-directional.
            kept = [e for e in pair_edges if (edge_param(e, "weight") or 0.0) > weight_threshold]
            declared_pairs = {(e.source, e.target) for e in kept}
            for edge in kept:
                source = edge.source
                target = edge.target
                weight = edge_param(edge, "weight") or 0.0

                edge_attrs = {
                    "weight": weight,
                    "delay": edge_param(edge, "delay") or 0.0,
                    "distance": edge_param(edge, "distance") or 0.0,
                    "directed": edge.directed,
                    "source_var": edge.source_var,
                    "target_var": edge.target_var,
                }
                G.add_edge(source, target, **edge_attrs)
                if not edge.directed and source != target and (target, source) not in declared_pairs:
                    G.add_edge(target, source, **edge_attrs)

            return G

        # Priority 2: Fall back to weight matrix representation
        W = self._weights_matrix()
        D = self.calculate_delays() if self.lengths_matrix is not None else None
        N_regions = self.number_of_regions

        if N_regions is None or W is None:
            return G

        labels = self.labels if hasattr(self, "labels") and self.labels else None
        label_list = list(labels.values()) if isinstance(labels, dict) else labels

        for i in range(N_regions):
            lab = label_list[i] if label_list and i < len(label_list) else f"node_{i}"
            G.add_node(i, label=lab)

        # Receiver-row W: emit j -> i so edges point in signal direction (see docstring).
        for i in range(N_regions):
            for j in range(N_regions):
                if W[i, j] > weight_threshold:
                    delay = D[i, j] if D is not None else 0.0
                    G.add_edge(j, i, weight=W[i, j], delay=delay)

        return G

    def get_centers(self) -> dict[int, tuple[float, float, float]]:
        """Get 3D spatial coordinates of brain region centers.

        Resolution order:
        1. ``Node.position`` on ``self.nodes`` (in-memory)
        2. ``nodes/coordinates`` dataset in the HDF5/Zarr companion
        3. Atlas metadata (``terminology.entities[*].center``)

        Returns:
        -------
        dict of int to tuple of float
            Mapping from region index to (x, y, z) coordinates in mm

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        centers = sc.get_centers()
        for idx, (x, y, z) in centers.items():
            print(f"Region {idx}: ({x:.1f}, {y:.1f}, {z:.1f})")
        ```
        """
        # --- Source 1: Node.position on self.nodes ---
        nodes = getattr(self, "nodes", None) or []
        if nodes:
            coords = {}
            for i, node in enumerate(nodes):
                pos = getattr(node, "position", None)
                if pos is not None:
                    coords[i] = (float(pos.x), float(pos.y), float(pos.z))
            if coords:
                return coords

        # --- Source 2: nodes/coordinates in companion file ---
        store = getattr(self, "_store", None)
        if store is not None:
            try:
                arr = store.read_dataset("nodes/coordinates")
                return {i: tuple(row) for i, row in enumerate(arr)}
            except (KeyError, FileNotFoundError):
                pass

        # --- Source 3: Atlas metadata ---
        entities = self.get_atlas().terminology.entities
        # Handle both dict and list formats
        if hasattr(entities, "items"):
            entity_items = list(entities.items())
        elif isinstance(entities, list):
            entity_items = list(enumerate(entities))
        else:
            # Empty or unknown format - return default
            return {0: (0, 0, 0)}

        def _entity_coord(entity, region):
            if isinstance(entity, dict):
                center = entity.get("center", {}) or {}
                coord = (center.get("x", 0), center.get("y", 0), center.get("z", 0))
                return coord, entity.get("lookupLabel", region)
            center = getattr(entity, "center", None)
            coord = (center.x, center.y, center.z) if center else (0, 0, 0)
            return coord, getattr(entity, "lookupLabel", region)

        # Build a coord index keyed by every label an entity is known by (name + abbreviation, hemisphere-qualified where two entities share it, + alternateName), plus an ordered list for the lookupLabel fallback. The abbreviation is often a bare region notation both hemispheres carry ("LOG"), so a SHARED one is keyed only qualified ("L.LOG") — keying it bare would let the right hemisphere overwrite the left and hand back the wrong centre. An abbreviation only one entity carries stays reachable bare, because dropping it would lose a connectome's only spelling for that region and drop the match rate below the threshold that keeps this off the positional fallback.
        def _get(entity):
            return entity.get if isinstance(entity, dict) else lambda k, _e=entity: getattr(_e, k, None)

        abbrev_counts: dict[str, int] = {}
        for _, entity in entity_items:
            abbreviation = _get(entity)("abbreviation")
            if abbreviation:
                abbrev_counts[str(abbreviation)] = abbrev_counts.get(str(abbreviation), 0) + 1

        by_label = {}
        ordered = []
        for region, entity in entity_items:
            coord, lookup_label = _entity_coord(entity, region)
            ordered.append((lookup_label if isinstance(lookup_label, int) else region, coord))
            get = _get(entity)
            abbreviation, hemisphere = get("abbreviation"), get("hemisphere")
            side = {"left": "L", "right": "R"}.get(str(hemisphere)) if hemisphere else None
            names = [get("name"), *(get("alternateName") or [])]
            if abbreviation:
                if side:
                    names.append(f"{side}.{abbreviation}")
                if abbrev_counts.get(str(abbreviation), 0) == 1:
                    names.append(abbreviation)
            for nm in names:
                if nm:
                    by_label[str(nm)] = coord

        # (3a) Match this network's node labels to atlas entities by label, so centres line up with node order regardless of the atlas's own ordering.
        if nodes and by_label:
            node_labels = [getattr(node, "label", None) for node in nodes]
            matched = {i: by_label[str(label)] for i, label in enumerate(node_labels) if label and str(label) in by_label}
            n_labelled = sum(1 for label in node_labels if label)
            if matched and len(matched) >= max(1, n_labelled // 2):
                if len(matched) < n_labelled:
                    # A partial dict is indistinguishable from a complete one downstream, and the gap is always a naming-convention mismatch the atlas could absorb as an alternateName. Name the offenders rather than silently dropping them.
                    import warnings

                    unmatched = [str(lbl) for i, lbl in enumerate(node_labels) if lbl and i not in matched]
                    warnings.warn(
                        f"get_centers(): only {len(matched)}/{n_labelled} node labels matched "
                        f"atlas {getattr(self.get_atlas(), 'name', '?')!r}; no centre for "
                        f"{unmatched[:5]}{' …' if len(unmatched) > 5 else ''}. Add these "
                        f"spellings as alternateName on the atlas entities.",
                        stacklevel=2,
                    )
                return matched

        # (3b) Fallback: key centres by the atlas's own lookupLabel order.
        if not ordered:
            return {0: (0, 0, 0)}
        ids = [o[0] for o in ordered]
        centers = np.array([o[1] for o in ordered])
        if all(isinstance(i, (int, float)) for i in ids):
            order = np.argsort(ids)
            center_mapping = {int(ids[j]) - 1: tuple(centers[j]) for j in order}
        else:
            center_mapping = {i: tuple(center) for i, center in enumerate(centers)}

        return center_mapping or {0: (0, 0, 0)}

    def plot_graph(
        self,
        ax: Axes | None = None,
        node_cmap: str | Any = "viridis",
        edge_cmap: str | Any = "viridis",
        node_colors: str = "in-strength",
        node_size: str | float = 8,
        threshold_percentile: float = 0,
        pos_scaling: float = 1,
        node_labels: bool = True,
        edge_labels: bool = True,
        log_in_strength: bool = True,
        node_size_scaling: float = 0,
        edge_color: str = "weight",
        pos: str | dict[int, list[float]] = "spring",
        plot_brain: str | None = None,
        edge_kwargs: dict[str, Any] | None = None,
        node_kwargs: dict[str, Any] | None = None,
        fontsize: float = 12,
        format: str = "networkx",
    ) -> Figure | cm.ScalarMappable:
        """Visualize connectome as network graph.

        Delegates to :func:`tvbo.plot.network_graph.plot_graph_networkx` or :func:`tvbo.plot.network_graph.plot_graph_bsplot` depending on *format*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        node_cmap : str or Colormap, default="viridis"
            Colormap for node colors
        edge_cmap : str or Colormap, default="viridis"
            Colormap for edge colors
        node_colors : str, default="in-strength"
            Node coloring scheme: "in-strength" or "node"
        node_size : str or float, default="in-strength"
            Node size scheme: "in-strength" or numeric value
        threshold_percentile : float, default=0
            Only show edges above this percentile of weights
        pos_scaling : float, default=1
            Scaling factor for spring layout positions
        node_labels : bool, default=True
            Whether to show node index labels
        edge_labels : bool, default=True
            Whether to show edge weight labels
        log_in_strength : bool, default=True
            Use log scale for in-strength calculations
        node_size_scaling : float, default=100
            Scaling factor for node sizes
        edge_color : str, default="weight"
            Edge attribute to use for coloring
        pos : str or dict, default="spring"
            Node positions: "spring" for automatic layout or dict of positions
        plot_brain : str, optional
            Brain view for anatomical layout: "horizontal", "sagittal", or "coronal"
        edge_kwargs : dict, optional
            Additional arguments passed to nx.draw_networkx_edges
        node_kwargs : dict, optional
            Additional arguments passed to nx.draw_networkx_nodes
        fontsize : float, default=8
            Font size for labels
        format : str, default="networkx"
            Plotting format: "networkx" for standard plotting, "bsplot" for fancy
            node/edge plotting with text boxes and curved edges.

        Returns:
        -------
        Figure or ScalarMappable
            Figure if ax is None, otherwise ScalarMappable for colorbar

        Examples:
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})

        # Simple graph
        fig, ax = plt.subplots(figsize=(10, 10))
        mappable = sc.plot_graph(ax, threshold_percentile=75)
        plt.colorbar(mappable, ax=ax)

        # Anatomical layout
        fig, ax = plt.subplots()
        sc.plot_graph(ax, plot_brain="horizontal", node_labels=False)
        ```

        See Also:
        --------
        plot_brain_surface : 3-D brain surface rendering with bsplot tvbo.plot.network.plot_graph_networkx : NetworkX backend tvbo.plot.network.plot_graph_bsplot : bsplot backend
        """
        from tvbo.plot.network import (
            _resolve_positions,
            _threshold_graph,
            plot_graph_bsplot,
            plot_graph_networkx,
        )

        G = self.graph

        # Threshold edges
        if threshold_percentile > 0:
            _threshold_graph(G, self._weights_matrix(), threshold_percentile)

        # Resolve node positions
        resolved_pos = _resolve_positions(
            G,
            pos,
            network=self,
            plot_brain=plot_brain,
        )

        if format == "bsplot":
            return plot_graph_bsplot(
                G,
                ax=ax,
                node_cmap=node_cmap,
                edge_cmap=edge_cmap,
                node_color_by=node_colors,
                log_in_strength=log_in_strength,
                pos=resolved_pos,
                node_labels=node_labels,
                edge_labels=edge_labels,
                fontsize=fontsize,
            )

        else:
            fig = plot_graph_networkx(
                G,
                ax=ax,
                node_cmap=node_cmap,
                edge_cmap=edge_cmap,
                node_color_by=node_colors,
                node_size_by=node_size,
                node_size_scaling=node_size_scaling,
                log_in_strength=log_in_strength,
                edge_color_by=edge_color,
                pos=resolved_pos,
                node_labels=node_labels,
                edge_labels=edge_labels,
                fontsize=fontsize,
                edge_kwargs=edge_kwargs,
                node_kwargs=node_kwargs,
            )
            if ax is not None:
                ax.axis("off")
            return fig

    def plot_brain_surface(self, ax=None, weight_matrix=None, **kwargs):
        """Render the network on the cortical brain surface.

        Nodes are rendered as coloured spheres at their MNI coordinates (from atlas metadata); edges as tubes.  Requires ``bsplot``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If *None*, a new figure is created.
        weight_matrix : ndarray, optional
            Custom matrix for edge colouring.  If *None*, uses the
            default weights matrix.
        **kwargs
            Forwarded to :func:`tvbo.plot.network.plot_graph_brain`.

        Returns:
        -------
        fig : Figure ax : Axes mappables : dict
            ``ScalarMappable`` objects (keys ``"nodes"`` / ``"edges"``).

        See Also:
        --------
        tvbo.plot.network.plot_graph_brain : Full parameter list
        """
        from tvbo.plot.network import plot_graph_brain

        return plot_graph_brain(self, ax=ax, weight_matrix=weight_matrix, **kwargs)

    def _matrix_from_explicit_edges(self, param_name: str) -> np.ndarray | None:
        """Build a dense N×N matrix from explicit (source/target) edge parameters.

        Returns None if no explicit edges carry the requested parameter.
        """
        explicit = [
            e for e in (self.edges or []) if getattr(e, "source", None) is not None and getattr(e, "target", None) is not None
        ]
        if not explicit:
            return None
        n = self.number_of_nodes or 0
        if n == 0:
            return None
        mat = np.zeros((n, n))
        found_any = False
        for e in explicit:
            params = getattr(e, "parameters", None) or {}
            p = params.get(param_name) if hasattr(params, "get") else None
            if p is None:
                continue
            val = getattr(p, "value", None)
            if val is None:
                continue
            mat[int(e.source), int(e.target)] = float(val)
            if not e.directed:
                mat[int(e.target), int(e.source)] = float(val)
            found_any = True
        return mat if found_any else None

    def plot_overview(
        self,
        edge_properties: list[str] | None = None,
        weights_kwargs: dict[str, Any] | None = None,
        lengths_kwargs: dict[str, Any] | None = None,
        graph_kwargs: dict[str, Any] | None = None,
        log_weights: bool = False,
        plot_brain: bool | None = None,
        brain_kwargs: dict[str, Any] | None = None,
        cmap: str = "magma",
        edge_percentile: float = 0,
        show_nodes: bool = True,
        show_edges: bool = True,
        max_edge_labels: int = 15,
    ) -> Figure:
        """Create comprehensive visualization with brain surface and matrices.

        Produces a multi-panel figure with one row per edge property.
        Each row contains either a brain surface + matrix heatmap (when *plot_brain* is True) or just a matrix heatmap, both coloured by the same property.

        Parameters
        ----------
        edge_properties : list of str, optional
            Names of edge matrices to plot (e.g. ``["weight", "length"]``
            or ``["weight", "length", "fc"]``).  Each name must match a
            matrix stored in the network (see ``set_matrix`` /
            ``matrix``).  If *None*, auto-discovers all available edge
            properties.
        weights_kwargs : dict, optional
            *Deprecated* — use ``edge_properties`` instead.
        lengths_kwargs : dict, optional
            *Deprecated* — use ``edge_properties`` instead.
        graph_kwargs : dict, optional
            Keyword arguments passed to `plot_graph`
        log_weights : bool, default=False
            Use logarithmic scale for the ``"weight"`` panel
        plot_brain : bool, optional
            If *True*, render on brain surface (requires ``bsplot``).
            If *False*, use matrix-only layout.  If *None* (default),
            auto-detect: use brain surface when ``bsplot`` is installed
            and atlas coordinates are available.
        brain_kwargs : dict, optional
            Keyword arguments passed to `plot_brain_surface` when
            the brain surface panel is used.
        cmap : str, default="magma"
            Default colormap for matrix heatmaps.
        edge_percentile : float, default=0
            Only show edges above this percentile of weights in the brain
            surface and graph panels.  ``0`` (default) plots all connections.
        show_nodes : bool, default=True
            Show node spheres on the brain surface panel.
        show_edges : bool, default=True
            Show edge tubes on the brain surface panel.
        max_edge_labels : int, default=15
            In graph panels (``plot_brain=False``), automatically hide edge
            labels when the number of visible edges exceeds this value.
            Set to a negative value to always show edge labels.

        Returns:
        -------
        matplotlib.figure.Figure
            Figure with subplots

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.plot_overview(log_weights=True)
        ```

        See Also:
        --------
        plot_graph : Network graph visualization plot_brain_surface : 3-D brain surface rendering plot_matrix : Side-by-side matrix visualization
        """
        from matplotlib.colors import LogNorm

        # Auto-discover all edge properties when not specified
        if edge_properties is None:
            seen = dict()  # preserve insertion order, deduplicate
            for e in self.edges or []:
                # Template edges: keyed by label/name
                lbl = getattr(e, "label", None) or getattr(e, "name", None)
                if lbl:
                    seen[lbl] = True
                # Explicit edges (source/target): collect parameter names
                params = getattr(e, "parameters", None)
                if params and getattr(e, "source", None) is not None:
                    for pname in params.keys() if hasattr(params, "keys") else []:
                        seen[pname] = True
            # Also include stored matrices
            for pname in self.edge_arrays():
                seen[pname] = True
            edge_properties = list(seen.keys())

        # Auto-detect brain surface capability
        if plot_brain is None:
            try:
                import bsplot  # noqa: F401

                centers = self.get_centers()
                has_coords = any(not (c[0] == 0 and c[1] == 0 and c[2] == 0) for c in centers.values())
                plot_brain = has_coords
            except (ImportError, Exception):
                plot_brain = False

        n_props = len(edge_properties)
        n_cols = 2
        fig, axs = plt.subplots(
            nrows=n_props,
            ncols=n_cols,
            layout="tight",
            figsize=(5 * n_cols, 5 * n_props),
            squeeze=False,
        )

        if brain_kwargs is None:
            brain_kwargs = {}
        if graph_kwargs is None:
            graph_kwargs = {}

        # Build edge label → metadata lookup from template edges
        edge_meta = {}
        for e in self.edges or []:
            lbl = getattr(e, "label", None) or getattr(e, "name", None)
            if lbl:
                edge_meta[lbl] = e

        for row, prop in enumerate(edge_properties):
            mat = self.matrix(prop, format="dense")
            if mat is None:
                mat = self._matrix_from_explicit_edges(prop)
            if mat is None:
                mat = np.zeros((1, 1))

            use_log = prop == "weight" and log_weights
            norm = None
            if use_log:
                nonzero = mat[mat > 0]
                vmin = float(nonzero.min()) if nonzero.size > 0 else 1e-10
                vmax = float(mat.max()) if mat.max() > 0 else 1.0
                norm = LogNorm(vmin=vmin, vmax=vmax)

            # Colorbar label from edge metadata unit
            emeta = edge_meta.get(prop)
            unit = getattr(emeta, "unit", None) if emeta else None
            if use_log:
                label = f"log({unit})" if unit else f"log({prop})"
            else:
                label = str(unit) if unit else prop

            col = 0

            # --- Brain surface panel ---
            if plot_brain:
                brain_defaults = {
                    "view": "top",
                    "surface_alpha": 0.25,
                    "node_radius": 1.5,
                    "node_color": "auto",
                    "node_data_key": "strength",
                    "node_scale": {"strength": 2},
                    "edge_radius": 0.12,
                    "edge_color": "auto",
                    "edge_data_key": "weight",
                    "edge_cmap": cmap,
                    "node_cmap": cmap,
                    "edge_scale": {"weight": 6},
                    "threshold_percentile": edge_percentile,
                    "show_nodes": show_nodes,
                    "show_edges": show_edges,
                }
                brain_defaults.update(brain_kwargs)
                _, _, mappables = self.plot_brain_surface(
                    ax=axs[row, 0],
                    weight_matrix=mat,
                    log_weights=use_log,
                    **brain_defaults,
                )
                axs[row, 0].axis("off")

                # Add colorbars for available mappables
                edge_sm = mappables.get("edges")
                if edge_sm is not None:
                    cb = fig.colorbar(
                        edge_sm,
                        ax=axs[row, 0],
                        shrink=0.4,
                        label=label,
                        location="right",
                    )
                    cb.outline.set_visible(False)
                col = 1
            else:
                # --- Graph panel (networkx) ---
                graph_obj = self.graph
                if edge_percentile > 0:
                    edge_weights = np.array(
                        [abs(d.get("weight", 1.0)) for _, _, _, d in graph_obj.edges(keys=True, data=True)],
                        dtype=float,
                    )
                    if edge_weights.size > 0:
                        cutoff = np.percentile(edge_weights, edge_percentile)
                        n_visible_edges = int(np.sum(edge_weights >= cutoff))
                    else:
                        n_visible_edges = 0
                else:
                    n_visible_edges = graph_obj.number_of_edges()

                auto_edge_labels = True if max_edge_labels < 0 else n_visible_edges <= max_edge_labels
                graph_defaults = {
                    "node_labels": True,
                    "edge_labels": auto_edge_labels,
                    "threshold_percentile": edge_percentile,
                    "edge_color": prop,
                }
                graph_defaults.update(graph_kwargs)
                self.plot_graph(ax=axs[row, 0], **graph_defaults)
                axs[row, 0].set_title(prop)
                col = 1

            # --- Matrix panel ---
            ax_mat = axs[row, col]
            im = ax_mat.imshow(mat, cmap=cmap, interpolation="none", norm=norm)
            ax_mat.set_title(prop)
            ax_mat.set_box_aspect(1)
            cb = fig.colorbar(im, ax=ax_mat, shrink=0.5, label=label)
            cb.outline.set_visible(False)

        # --- Font scaling ---
        fontsize_scaler = 1.5
        for ax in axs.flat:
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(lbl.get_fontsize() * fontsize_scaler)
            ax.title.set_fontsize(ax.title.get_fontsize() * fontsize_scaler)
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * fontsize_scaler)
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * fontsize_scaler)

        plt.close()
        return fig

    def normalize(self) -> None:
        """Add min-max normalization of connection weights.

        Appends a transform to scale weights to [0, 1] range.
        Equivalent to ``add_transform("weight")``, whose default is that normalisation.

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize()
        normalized_weights = sc.matrix("weight")  # Now in [0, 1] range
        ```

        See Also:
        --------
        add_transform : Add a transform on any edge property normalize_weights : Set custom normalization equation
        """
        self.add_transform("weight")

    # ── Node mapping (hierarchical composition) ─────────────────────

    def set_node_mapping(
        self,
        mapping,
        parent_network=None,
        dataset_path: str = "/nodes/parent_index",
    ) -> None:
        """Set the node-to-parent mapping array.

        This stores the mapping data internally so that :func:`save` writes it into the HDF5 companion automatically — no manual ``h5py`` code required.

        Parameters
        ----------
        mapping : array-like of int
            Int32 array of shape ``(N,)`` where entry *i* is the parent
            node index that node *i* maps to (e.g. a region mapping
            that assigns each cortical vertex to a parcellation region).
        parent_network : str or Network, optional
            Path/URI of the parent Network YAML sidecar, **or** the
            parent Network object itself.  When a Network is passed
            its reference string is derived automatically (see
            :func:`_network_ref_string`).
        dataset_path : str
            HDF5 dataset path written into ``self.node_mapping``
            (default ``"/nodes/parent_index"``).

        Examples:
        --------
        >>> surface_net.set_node_mapping(region_mapping,
        ...                             parent_network="dk_sc.yaml")
        >>> # or pass the Network object directly:
        >>> surface_net.set_node_mapping(region_mapping,
        ...                             parent_network=sc)
        >>> surface_net.save(tmpdir / "surface_rh.yaml")
        """
        self.set_array(dataset_path.lstrip("/"), np.asarray(mapping, dtype=np.int32))
        self.node_mapping = dataset_path
        if parent_network is not None:
            self.parent_network = parent_network  # __setattr__ handles Network→str

    @property
    def node_mapping_data(self):
        """The node-to-parent mapping array, or ``None``.

        The dataset ``node_mapping`` names, resident if :meth:`set_node_mapping` put it there and read from the companion otherwise.
        """
        path = getattr(self, "node_mapping", None) or "/nodes/parent_index"
        return self.array(path.lstrip("/"))

    # ── Generalized edge / matrix API ─────────────────────────────

    def _fill_produced_matrices(self, arrays: dict) -> None:
        """Resolve every Edge that declares a ``producer:`` into ``arrays``, once.

        A produced matrix is what lets a discrete differential operator or a rule-generated connectome be DECLARED — the recipe states the callable that builds it and stays the single source of truth — where a pre-built file needs a prep step run by hand before the spec can execute at all.

        Filling ``_arrays`` is what makes every accessor see it: they do not agree on one entry point (``matrix`` walks a resolution order, ``weights`` reads ``_arrays`` directly), so resolving in only one of them leaves the other silently falling through to whatever it does when a connectome is missing. A stored array still wins — this never overwrites an existing entry.

        Resolution is the shared one (:mod:`tvbo.data.param_io`), so a matrix producer caches, fingerprints and reports errors exactly as a parameter's does. One consequence differs from a hand-set matrix: the array is a read-only view of the resolve cache, which several Networks may share, so it is transformed by deriving a new array rather than written into.
        """
        from tvbo.data import param_io

        for edge in self.edges or []:
            label = str(getattr(edge, "label", "") or "")
            if not label or _array_key(label) in arrays or getattr(edge, "producer", None) is None:
                continue
            source_dir = getattr(self, "_source_dir", None)
            produced = param_io.resolve(edge, source_dir=Path(source_dir) if source_dir else None, context=self)
            if produced is None:
                raise ValueError(
                    f"network edge {label!r}: its `producer:` returned nothing. A matrix "
                    "producer must return the (n, n) array or sparse matrix itself, or a "
                    "dict from which `output:` names one."
                )
            arrays[_array_key(label)] = produced

    def _resident(self) -> dict:
        """The arrays held in memory, keyed by companion dataset path. Consulting it resolves nothing."""
        try:
            d = object.__getattribute__(self, "_arrays")
            if isinstance(d, dict):
                return d
        except AttributeError:
            pass
        d = {}
        object.__setattr__(self, "_arrays", d)
        return d

    def _lazy_paths(self) -> list[str]:
        """Every dataset path the companion offers, without reading one."""
        store = getattr(self, "_store", None)
        if store is None:
            return []
        # A companion that has moved or gone is a thing `repr` reports, not a thing it raises over.
        try:
            paths = [f"edges/{n}" for n in (getattr(store, "names", None) or [])]
        except (AttributeError, OSError, KeyError):
            paths = []
        for group in ("nodes", "mesh"):
            try:
                paths.extend(store.dataset_keys(group))
            except (AttributeError, OSError, KeyError):
                pass
        return paths

    @property
    def arrays(self) -> dict:
        """The materialised arrays, keyed by companion dataset path — ``edges/weight``, ``mesh/vertices``.

        This and only this is what a JAX transformation traces. What the companion offers but has not been read is listed by `repr` and reached through :meth:`materialize`.
        """
        return self._get_arrays()

    def edge_arrays(self) -> dict:
        """The resident edge matrices alone, keyed by edge name rather than companion path.

        `arrays` is keyed by path and carries meshes and edge parameters besides, so every caller that means "the matrices this network holds" reads this instead of filtering the paths itself.
        """
        return {name: value for key, value in self._get_arrays().items() if (name := _edge_name(key)) is not None}

    def set_array(self, path: str, data) -> None:
        """Hold ``data`` as the array at ``path`` (a bare name means an edge matrix).

        The one write into the resident set. A scipy sparse matrix is kept sparse; anything else becomes an ndarray. Declares no template edge — :meth:`set_matrix` does, for an edge matrix that should reach the sidecar.
        """
        from scipy import sparse

        self._resident()[_array_key(path)] = data if sparse.issparse(data) else np.asarray(data)

    def array(self, path: str):
        """The array at ``path``, resident or read from the companion on first use; ``None`` when there is neither.

        A bare name is an edge matrix; ``mesh/vertices`` or ``nodes/coordinates`` is any dataset the companion carries. What this reads is kept, so the residency `repr` reports is exactly what has been paid for. Reads come back in their stored format.
        """
        key = _array_key(path)
        resident = self._get_arrays()
        if key in resident:
            return resident[key]
        store = getattr(self, "_store", None)
        if store is None:
            return None
        try:
            if key.startswith("edges/") and key.count("/") == 1:
                value = store[key[len("edges/") :]]
            else:
                value = store.read_dataset(key)
        except (KeyError, OSError, AttributeError):
            return None
        resident[key] = value
        return value

    def materialize(self, *paths: str) -> "Network":
        """A copy of this network with ``paths`` resident, and nothing else newly read.

        The explicit point at which arrays are read: ``net.materialize("weight", "length")`` reads two datasets of however many the companion holds, and the copy's `arrays` then holds exactly what a solver or a gradient will see. A bare name is an edge matrix; ``mesh/vertices`` names any dataset. The copy shares the arrays already resident here (references, not copies) and the lazy store; its residency is its own.

        Raises ``KeyError`` for a path neither resident nor in the companion, so a typo is a typo and not an absent leaf.
        """
        import copy as _copy

        out = _copy.copy(self)
        object.__setattr__(out, "_arrays", dict(self._resident()))
        for path in paths:
            if out.array(path) is None:
                raise KeyError(f"{path!r} is neither resident nor in the companion of {self}")
        return out

    def edge_parameter_arrays(self) -> dict[str, dict]:
        """``{edge: {parameter: matrix}}`` for every edge-parameter matrix, resident or in the companion.

        These are kept under ``edges/<edge>/edge_parameters/<parameter>``, the path the companion writes them at.
        """
        out: dict[str, dict] = {}
        for key, value in self._resident().items():
            parts = key.split("/")
            if len(parts) == 4 and parts[0] == "edges" and parts[2] == "edge_parameters":
                out.setdefault(parts[1], {})[parts[3]] = value
        store = getattr(self, "_store", None)
        for name in getattr(store, "names", None) or []:
            if name in out:
                continue
            try:
                params = store.edge_params_of(name)
            except (KeyError, OSError, AttributeError):
                continue
            if params:
                out[name] = dict(params)
        return out

    def _get_arrays(self) -> dict:
        """The resident arrays, with every ``producer:`` edge resolved into them once."""
        d = self._resident()
        try:
            resolved = object.__getattribute__(self, "_producers_resolved")
        except AttributeError:
            resolved = False
        if not resolved:
            object.__setattr__(self, "_producers_resolved", True)
            try:
                self._fill_produced_matrices(d)
            except Exception:
                object.__setattr__(self, "_producers_resolved", False)  # only success latches
                raise
        return d

    def set_matrix(
        self,
        name: str,
        data,
    ) -> None:
        """Set a named edge matrix.

        Accepts dense NumPy arrays, scipy sparse matrices (CSR, COO, etc.), or any array-like that can be converted. The matrix is stored internally and a template edge is created/updated automatically so that ``save()`` writes it to the HDF5 companion.

        Parameters
        ----------
        name : str
            Matrix name (e.g. ``"weight"``, ``"length"``,
            ``"local_connectivity"``). Used as the HDF5 group name
            under ``edges/``.
        data : array-like or scipy.sparse matrix
            The edge matrix to store.

        Examples:
        --------
        >>> net.set_matrix("weight", W_dense)
        >>> net.set_matrix("local_connectivity", LC_sparse_csr)
        """
        self._get_arrays()
        self.set_array(name, data)
        self._ensure_template_edge(name)

    @property
    def matrix_names(self) -> list[str]:
        """Every edge matrix this network can serve, without reading one.

        User-set arrays first, then what the companion file declares. A name here answers ``matrix(name)``; the residency of each is a separate question, which is the point.
        """
        names = list(self.edge_arrays())
        store = getattr(self, "_store", None)
        for n in getattr(store, "names", None) or []:
            if n not in names:
                names.append(n)
        return names

    def carries(self, name: str) -> bool:
        """Whether ``matrix(name)`` would answer, decided from the header and the declaration without reading a value.

        A resident array or a companion dataset under any spelling of *name* answers. So do the explicit edges, from which ``weight`` (an edge with none counts as 1), ``length`` (declared as ``length`` or ``distance``, or the Euclidean distance between two positioned nodes) and a positive ``delay`` are built; any other name is a per-edge parameter the edges declare. ``weight`` also answers on a bare node set, where `matrix` returns zeros, because an unconnected network is a legitimate one — `has_connectome` is the question of whether anything is connected. What a backend asks before it lowers, so a missing attribute is named at the declaration rather than met as a ``None`` in the middle of a build.
        """
        spellings = set(self._matrix_names(name))
        if spellings & {str(n).lower() for n in self.matrix_names}:
            return True
        placed = self._placed_edges()
        if _is_weight_name(name):
            return bool(placed) or bool(self.nodes) or (self.number_of_nodes or 0) >= 1
        if not placed:
            return False
        if _is_length_name(name):
            if any(edge_param(e, "length") or edge_param(e, "distance") for e in placed):
                return True
            return any(self._compute_euclidean_distance(e.source, e.target) is not None for e in placed)
        if str(name).lower() == "delay":
            return any((edge_param(e, "delay") or 0) > 0 for e in placed)
        return any(edge_param(e, name) is not None for e in placed)

    def stored_format(self, name: str) -> str | None:
        """The storage format of the matrix ``matrix(name)`` would serve — ``dense``, ``csr``, ``coo`` — from the header alone, or ``None`` when the network carries none by any spelling."""
        spellings = set(self._matrix_names(name))
        for candidate in self.matrix_names:
            if str(candidate).lower() in spellings:
                return str(self.matrix_info(candidate).format)
        return None

    @property
    def has_connectome(self) -> bool:
        """Whether anything is connected: a weight matrix in hand or on file, or explicit edges between nodes. A node set `matrix("weight")` answers with zeros is not."""
        spellings = set(self._matrix_names("weight"))
        return bool(spellings & {str(n).lower() for n in self.matrix_names}) or bool(self._placed_edges())

    def matrix_info(self, name: str):
        """Shape, dtype and storage format of one array, from the companion's header alone.

        ``name`` is an edge matrix (``"weight"``) or any dataset path the companion carries (``"mesh/vertices"``). A user-set array answers from the object in hand. Raises ``KeyError`` when nothing by that name exists.
        """
        from tvbo.data.matrix_io import ArrayInfo

        key = _array_key(name)
        arrays = self._get_arrays()
        if key in arrays:
            a = arrays[key]
            data = getattr(a, "data", a)
            return ArrayInfo(
                key,
                tuple(a.shape),
                np.dtype(getattr(a, "dtype", float)),
                str(getattr(a, "format", "dense")),
                int(getattr(data, "nbytes", 0)),
            )
        store = getattr(self, "_store", None)
        if store is None or not hasattr(store, "info"):
            raise KeyError(name)
        return store.info(name)

    def _matrix_names(self, name: str) -> tuple:
        """Spellings to try for a named edge matrix, most specific first.

        A declared ``primary_weight`` wins for every weight spelling, so one sidecar can carry several connectivity variants (band-specific reweightings, shuffled controls) and still present one of them as the active weight. Beyond that the order is :func:`_alias_group`'s, which also drives transform selection.
        """
        primary = getattr(self, "primary_weight", None) if _is_weight_name(name) else None
        head = (str(primary).lower(),) if primary else ()
        return tuple(dict.fromkeys(head + _alias_group(name)))

    def matrix(
        self,
        name: str,
        format: str | None = None,
        apply_transforms: bool = True,
    ):
        """Get a named edge matrix, optionally in a specific format.

        The single canonical connectivity accessor. Resolution order: the resident `arrays`, a JAX array among them returned untouched (the live leaf under a transformation) → the companion file, whose matrix is then kept resident → the explicit edges → ``None``. Each SOURCE is exhausted across every alias spelling before the next is consulted — precedence is between sources, and a spelling is not a precedence, so a companion file holding ``weight`` cannot shadow a user-set ``weights``.

        Being canonical means subsuming what the deprecated properties returned, so a WEIGHT target on a node set with no edges yields zeros rather than ``None``: an unconnected network is a legitimate one, and every consumer of this builds an ``(n, n)`` array from the result.

        Parameters
        ----------
        name : str
            Matrix name (e.g. ``"weight"``, ``"length"``). Alias spellings
            (``weights``/``sc``, ``lengths``) resolve to the same matrix.
        format : str, optional
            Return format: ``"dense"``, ``"csr"``, ``"coo"``, ``"lil"``.
            If ``None``, returns the matrix in whatever format it is
            currently stored in.
        apply_transforms : bool
            Apply the declared ``transforms:`` targeting this matrix. Pass
            ``False`` for the raw matrix — the tvboptim codegen path does, so a
            frozen kit keeps raw SC in the network file and the declared op
            visible in the rendered script rather than hidden in this runtime.

        Returns:
        -------
        np.ndarray or scipy.sparse matrix or None
        """
        from scipy import sparse
        from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

        arrays = self._get_arrays()
        store = getattr(self, "_store", None)
        candidates = self._matrix_names(name)

        def _spelled(names):
            """The first candidate among ``names``, exact match before case-folded.

            The case-folded pass is what makes a sidecar spelling lengths ``tractLength`` resolvable at all.
            """
            names = list(names)
            for cand in candidates:
                if cand in names:
                    return cand
            folded = {}
            for k in names:
                folded.setdefault(str(k).lower(), k)
            for cand in candidates:
                if cand in folded:
                    return folded[cand]
            return None

        mat = None
        found = _spelled(self.edge_arrays())
        if found is not None:
            mat = arrays[_array_key(found)]
            if isinstance(mat, JaxArray):
                return mat
        elif store is not None:
            found = _spelled(getattr(store, "names", None) or getattr(store, "arrays", {}).keys())
            if found is not None:
                # A sidecar may declare a template edge the companion does not carry, and a name the store lists is not a promise it can serve one.
                try:
                    mat = store[found]
                except (KeyError, OSError):
                    mat = None
                else:
                    arrays[_array_key(found)] = mat
        if mat is None:
            if _is_weight_name(name):
                mat = self._weights_from_edges()
            elif _is_length_name(name):
                mat = self._lengths_from_edges()

        # No edges is zero weights, not absent ones: a single-node model and an uncoupled ensemble are both legitimate, and every consumer builds an (n, n) array from this.
        _unconnected = False
        if mat is None and _is_weight_name(name):
            n = len(self.nodes) if self.nodes else (self.number_of_nodes or self.number_of_regions or 0)
            if n > 0:
                mat, _unconnected = np.zeros((n, n), dtype=np.float64), True

        if mat is None:
            return None

        # A declared transform describes real connectivity; over an all-zero stand-in a normalisation like `W / mean(W[W > 0])` yields nan rather than zeros.
        if apply_transforms and not _unconnected:
            for t in self.transforms_for(name):
                mat = self._apply_transform(
                    mat.toarray() if sparse.issparse(mat) else np.asarray(mat),
                    t,
                )

        if format is None:
            return mat
        elif format == "dense":
            return mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
        elif format == "csr":
            return csr_matrix(mat)
        elif format == "coo":
            return coo_matrix(mat)
        elif format == "lil":
            return lil_matrix(mat)
        else:
            raise ValueError(f"Unknown format: {format!r}")

    def add_edge(
        self,
        source: int,
        target: int,
        symmetric: bool = True,
        **params: float,
    ) -> None:
        """Add a single edge with named parameter values.

        Convenience wrapper around :meth:`add_edges` for one edge.

        Parameters
        ----------
        source, target : int
            Node indices.
        symmetric : bool
            If ``True`` (default), also adds the reverse edge.
        **params : float
            Named parameter values. Each name becomes a matrix name
            (e.g. ``weight=0.5`` → stored in the ``"weight"`` matrix,
            ``length=30.0`` → stored in ``"length"``).

        Examples:
        --------
        >>> net.add_edge(0, 1, weight=0.5, length=30.0)
        """
        # Map common singular → plural names to match template edge conventions
        _NORMALIZE = {
            "weights": "weight",
            "lengths": "length",
            "delays": "delay",
            "distance": "length",
            "distances": "length",
        }
        mapped = {_NORMALIZE.get(k, k): np.array([v]) for k, v in params.items()}
        self.add_edges(
            np.array([source]),
            np.array([target]),
            symmetric=symmetric,
            **mapped,
        )

    def add_edges(
        self,
        sources,
        targets,
        symmetric: bool = True,
        **matrices,
    ) -> None:
        """Add edges in bulk using COO-style index arrays.

        Each keyword argument is a named matrix (e.g. ``weights=vals``) whose entries are being added at the given ``(source, target)`` positions. Internally the data is kept in COO format for fast incremental building; call :meth:`matrix` with ``format="csr"`` when you need efficient row-slicing.

        Parameters
        ----------
        sources, targets : array-like of int
            Source and target node index arrays (same length).
        symmetric : bool
            If ``True`` (default), each ``(i, j)`` entry is mirrored
            to ``(j, i)``, producing a symmetric matrix.
        **matrices : array-like of float
            Named value arrays, one per matrix to update. Length must
            match ``sources`` and ``targets``.

        Examples:
        --------
        >>> # Build local connectivity from index pairs + kernel weights
        >>> net.add_edges(pairs[:, 0], pairs[:, 1],
        ...              symmetric=True, weight=kernel_vals)
        """
        from scipy import sparse
        from scipy.sparse import coo_matrix

        sources = np.asarray(sources, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.int32)

        arrays = self._get_arrays()

        n = self.number_of_nodes or int(max(sources.max(), targets.max())) + 1

        for name, values in matrices.items():
            values = np.asarray(values, dtype=np.float64)

            if symmetric:
                new_rows = np.concatenate([sources, targets])
                new_cols = np.concatenate([targets, sources])
                new_data = np.concatenate([values, values])
            else:
                new_rows, new_cols, new_data = sources, targets, values

            existing = arrays.get(_array_key(name))

            if existing is None:
                mat = coo_matrix((new_data, (new_rows, new_cols)), shape=(n, n))
            else:
                # Convert existing to COO for fast append
                if not sparse.issparse(existing):
                    existing = coo_matrix(existing)
                elif existing.format != "coo":
                    existing = existing.tocoo()
                rows = np.concatenate([existing.row, new_rows])
                cols = np.concatenate([existing.col, new_cols])
                data = np.concatenate([existing.data, new_data])
                mat = coo_matrix((data, (rows, cols)), shape=existing.shape)

            arrays[_array_key(name)] = mat

            self._ensure_template_edge(name)

    def _get_edge(self, name: str):
        """Return the template Edge with the given label, or None."""
        for e in self.edges or []:
            lbl = getattr(e, "label", None) or getattr(e, "name", None)
            if lbl == name:
                return e
        return None

    def transforms_for(self, target: str):
        """The declared `transforms:` that retarget *target*, in declaration order.

        The one place the target name is matched, so the singular and plural spellings of an edge property stay equivalent everywhere and a new alias is added once. The aliases mirror the ones `matrix` already resolves when it looks the matrix itself up, so `matrix("weights")` and `matrix("weight")` cannot disagree about whether a transform applies. Both the runtime and the emitters that inline a transform into a generated script select through this.

        Args:
            target: Edge property a transform retargets, e.g. `"weight"` or `"length"`.

        Returns:
            List of `Function` transforms declared against *target*.
        """
        names = _alias_group(target)
        return [t for t in (self.transforms or []) if str(transform_target(t)).lower() in names]

    def transform_expression(self, func):
        """A transform's equation as a sympy expression, with its arguments substituted.

        Shared by the runtime and by codegen so a spec resolves to the same expression on both paths. Scalar values come from `Function.arguments`, falling back to `Equation.parameters` for legacy specs.

        An argument declared without a value substitutes nothing and its symbol survives, so the caller reports it as an undeclared name. Substituting the `None` instead raises `SympifyError: None`, which names neither the transform nor the argument.

        Args:
            func: The `Function` transform.

        Returns:
            A `(expression, mask_bindings)` pair; the expression is `None` when *func*
            declares no equation (a callable-based transform) or it does not parse. Each
            mask binding is evaluated once, before the expression that reads it.
        """
        eq = getattr(func, "equation", None)
        if eq is None:
            return None, {}
        from tvbo.codegen.code import parse_eq

        arg_values: dict = {}
        for name, a in (getattr(func, "arguments", {}) or {}).items():
            arg_values[name] = getattr(a, "value", None)
        if getattr(eq, "parameters", None):
            for pname, pval in eq.parameters.items():
                arg_values.setdefault(pname, getattr(pval, "value", pval))

        from tvbo.codegen.transforms import prepare, subscript_locals

        rhs = str(getattr(eq, "rhs", eq) or "")
        exp = parse_eq(eq, local_dict=subscript_locals(rhs))
        if exp is None:
            return None, {}
        subs_map = {s: arg_values[str(s)] for s in exp.free_symbols if arg_values.get(str(s)) is not None}
        exp = exp.subs(subs_map) if subs_map else exp

        return prepare(exp, what=f"transform {transform_target(func) or '?'!r}")

    def _transform_operand(self, func, M):
        """A resolver from a transform symbol to the live array it names.

        The transform's own target binds to *M*, the value flowing through the chain, so a second transform in a chain sees the first one's output — and so a length-target transform does not re-enter itself by asking the network for the matrix it is busy producing. Every other edge attribute binds to the network's stored matrix through the same :func:`tvbo.utils.edge_label` the emitters use, and a declared per-node parameter binds as an ``(n, 1)`` column, so ``weight / roi_size`` divides each target row by that region's size and broadcasts across the source axis.
        """
        from tvbo.utils import edge_label

        target = edge_label(transform_target(func)) or transform_target(func)

        def resolve(name):
            label = edge_label(name) or name
            if label == target:
                return M
            stored = self.matrix(label, format="dense")
            if stored is not None:
                return stored
            vec = self.node_parameter_vectors.get(name)
            if vec is None or vec.shape[0] != M.shape[0]:
                return None
            return jnp.asarray(vec).reshape(-1, 1)

        return resolve

    def _apply_transform(self, M, func):
        """Apply a Function transform to matrix *M*.

        Supports equation-based (symbolic) or callable-based (software) transforms via the Function class.
        """
        # Callable-based transform
        c = getattr(func, "callable", None)
        if c is not None:
            import importlib
            import inspect

            # Make the recipe's source dir importable so a transform callable can live beside the study YAML, mirroring the builder injection in _resolve_from_graph_generator.
            with _source_dir_on_path(getattr(self, "_source_dir", None)):
                mod = importlib.import_module(c.module)
                fn = getattr(mod, c.name)
                kwargs = {}
                for name, arg in func.arguments.items():  # arguments keyed by name
                    kwargs[name] = getattr(arg, "value", None)
                sig = inspect.signature(fn)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if "network" in sig.parameters or accepts_var_kw:
                    kwargs.setdefault("network", self)
                # Inject L only when asked. For a length-target transform ``M`` IS the lengths, so use it directly — reading ``self.lengths_matrix`` here would re-enter this same transform (infinite recursion).
                if "L" in sig.parameters:
                    _is_length = transform_target(func) in _LENGTH_TARGETS
                    kwargs.setdefault("L", M if _is_length else self.lengths_matrix)
                return fn(M, **kwargs)

        # Equation-based transform
        exp, masks = self.transform_expression(func)
        if exp is None:
            return M
        from tvbo.codegen.code import render_expression
        from tvbo.codegen.transforms import edge_symbols, runtime_env

        env = runtime_env(self._transform_operand(func, M), edge_symbols(exp, masks), jnp, jsp)
        for symbol, mask in masks.items():
            env[str(symbol)] = eval(render_expression(mask, format="jax"), env)
        code_str = render_expression(exp, format="jax")
        if isinstance(code_str, str):
            M = eval(code_str, env)
        return M

    def add_transform(self, target: str, equation_rhs: str | None = None) -> None:
        """Append a matrix transform for a named edge property.

        Transforms are applied in order when the matrix is accessed via ``matrix()`` or ``weights_matrix``.

        Parameters
        ----------
        target : str
            Edge property name (e.g. ``"weight"``, ``"length"``, ``"fc"``).
        equation_rhs : str, optional
            Right-hand side of the transform equation, written over the network's own
            edge attributes — ``weight``, ``length``, or ``network.edges.<label>``. The
            attribute named by *target* is the value under transform. Defaults to
            min-max normalisation of *target*. A reduction may be scoped by a boolean
            predicate, written either as a subscript or as a second argument.

        Examples:
        --------
        ```python
        sc = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.add_transform("weight", "weight / max(weight)")
        sc.add_transform("weight", "weight / mean(weight[weight > 0])")
        sc.add_transform("weight", "weight / mean(weight, weight > 0)")
        ```
        """
        if equation_rhs is None:
            equation_rhs = f"({target} - min({target})) / (max({target}) - min({target}))"
        if self.transforms is None:
            self.transforms = []
        self.transforms.append(
            tvbo_datamodel.Function(
                name=target,
                equation=tvbo_datamodel.Equation(rhs=equation_rhs),
            )
        )

    def _ensure_template_edge(self, name: str) -> None:
        """Ensure a template edge (no source/target) exists for *name*.

        Template edges are the link between the ``edges`` list visible in the YAML sidecar and the HDF5 datasets under ``edges/<name>/``.
        """
        from scipy import sparse

        mat = self._get_arrays().get(_array_key(name))

        for e in self.edges or []:
            lbl = getattr(e, "label", None) or getattr(e, "name", None)
            if lbl == name:
                # Update format hint
                if mat is not None:
                    if sparse.issparse(mat):
                        from tvbo.data.matrix_io import auto_format

                        e.format = auto_format(mat)
                    elif isinstance(mat, np.ndarray):
                        from tvbo.data.matrix_io import auto_format

                        e.format = auto_format(mat)
                return

        # Create new template edge
        fmt = "dense"
        if mat is not None:
            from tvbo.data.matrix_io import auto_format

            fmt = auto_format(mat)

        edge = tvbo_datamodel.Edge(label=name, format=fmt, weighted=True)
        if not self.edges:
            self.edges = []
        self.edges.append(edge)


@register_pytree
class Connectome(Network):
    """Deprecated alias for Network. Use Network instead."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "Connectome is deprecated and will be removed in a future version. "
            "Use tvbo.Network (tvbo.classes.network.Network) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
