import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from jax import Array as JaxArray
from jax.tree_util import register_pytree_node_class
from jsonasobj2 import as_dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from tvbo.data.registry import database_dir
from tvbo.datamodel import schema as tvbo_datamodel

# HDF5+YAML network files — resolved via registry (works for pip & editable installs)
NETWORK_DIR = database_dir("Network")

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
    segmentation: Optional[str] = None,
    scale: Optional[str] = None,
) -> Optional[Path]:
    """Find the YAML sidecar for a given atlas + tractogram combination.

    Searches tvbo/database/networks/ for files matching the atlas and rec- entities.
    When ``segmentation`` / ``scale`` are given, also requires ``seg-<segmentation>``
    and ``scale-<scale>``. Falls back to partial matching if exact match fails.
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


def _parse_bids_entities(stem: str) -> Dict[str, str]:
    """Extract BIDS key-value entities from a filename stem.

    E.g. ``"tpl-MNI_atlas-DK_rec-dTOR_scale-100_desc-SC_relmat"``
    → ``{"tpl": "MNI", "atlas": "DK", "rec": "dTOR", "scale": "100", "desc": "SC"}``
    """
    import re

    return dict(re.findall(r"(?:^|_)([a-zA-Z]+)-([^_]+)", stem))


def _filter_networks_by_entities(entities: Dict[str, str]) -> list:
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

    Parses ``meas-<name>`` from filenames and classifies into
    [weight_measure, length_measure] order by matching against known
    naming conventions.  Falls back to file order if names are unknown.
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


def get_normative_connectome_data(
    atlas: str,
    tractogram: str = "dTOR",
    segmentation: Optional[str] = None,
    scale: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    Returns
    -------
    weights : np.ndarray
        Connection strength matrix (N x N)
    lengths : np.ndarray or None
        Tract length matrix (N x N), or None if not available

    Examples
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
            f"No network found for atlas={atlas}, tractogram={tractogram}, "
            f"seg={segmentation}, scale={scale} in {NETWORK_DIR}"
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


def _backfill_name_from_iri(obj: Any, nested_key: str | None = None) -> None:
    """If ``obj`` (or ``obj[nested_key]``) is a dict with ``iri`` but no ``name``,
    derive ``name`` from the IRI's local part. Mutates in place."""
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
    target["name"] = iri.split(":", 1)[-1] if ":" in iri else iri


@register_pytree_node_class
class Network(tvbo_datamodel.Network):
    """A brain network: parcellation, connectome, per-node dynamics, and coupling.

    The spatial substrate of a `SimulationExperiment`. A `Network` ties an
    atlas/parcellation to a tractogram (structural connectivity matrix +
    optional path lengths) and, optionally, per-node `Dynamics` overrides
    and node-level coupling parameters.

    Construct inline, by IRI (resolved against the curated database), or
    from a NumPy / pandas matrix via [`Network.from_array`](#tvbo.classes.network.Network.from_array).

    Examples:
        ```python
        net = Network(
            parcellation={"atlas": {"iri": "tvbo:DesikanKilliany"}},
            tractogram={"iri": "tvbo:dTOR"},
        )
        ```

    See the [Network specification](../../../Specification/Network.qmd) for
    the slot-by-slot reference and the
    [`Connectome`](#tvbo.classes.network.Connectome) subclass for matrix-style
    networks without an explicit parcellation.
    """

    @property
    def number_of_regions(self) -> int:
        """Deprecated alias for number_of_nodes."""
        return self.number_of_nodes

    @number_of_regions.setter
    def number_of_regions(self, value: int) -> None:
        self.number_of_nodes = value

    def __init__(self, **kwargs: Any) -> None:
        # Strip internal-only flags that may leak in from serialised forms.
        for _internal in ("_resolved",):
            kwargs.pop(_internal, None)
        # Resolve deprecated number_of_regions -> number_of_nodes
        if "number_of_regions" in kwargs:
            kwargs.setdefault("number_of_nodes", kwargs.pop("number_of_regions"))
            kwargs.pop("number_of_regions", None)

        # Derive name from iri where missing (so the dataclass post-init
        # doesn't raise MissingRequiredField on iri-only construction).
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

        # Normalise an inline string parcellation -> Parcellation dict so the
        # parent constructor accepts it. Materialisation of normative data
        # happens later in self._resolve().
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
                    f"number_of_nodes={declared} doesn't match len(nodes)={n_nodes}. Using {n_nodes} from nodes list."
                )
            kwargs["number_of_nodes"] = n_nodes
        # Create default nodes if number_of_nodes is set but nodes list is empty
        elif kwargs.get("number_of_nodes") and not kwargs.get("nodes"):
            n_nodes = kwargs["number_of_nodes"]
            kwargs["nodes"] = [tvbo_datamodel.Node(id=i, label=f"node_{i}") for i in range(n_nodes)]

        # Resolve Dynamics slot aliases (components → modes) in network dynamics
        # so the LinkML loader can construct Dynamics objects correctly.
        _net_dynamics = kwargs.get("dynamics")
        if isinstance(_net_dynamics, dict):
            from tvbo.classes.dynamics import _resolve_dynamics_aliases

            for _dk, _dv in _net_dynamics.items():
                if isinstance(_dv, dict):
                    _resolve_dynamics_aliases(_dv)

        super().__init__(**kwargs)

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

        # Materialise connectivity from the declarative spec (parcellation,
        # data_file, bids_dir, graph_generator). Idempotent; safe to call
        # multiple times. See Network._resolve. Pick up the YAML source
        # directory from the SimulationExperiment context (set by from_file)
        # so relative paths resolve correctly even when Network is built as
        # a kwarg inside SimulationExperiment.__init__.
        from tvbo.classes.experiment import SimulationExperiment as _SE
        _source_dir = None
        _pending = getattr(_SE, "_pending_source_file", None)
        if _pending:
            _source_dir = os.path.dirname(_pending)
        self._resolve(source_dir=_source_dir)

    # -------------------------------------------------------------------- #
    # Canonical resolver                                                   #
    # -------------------------------------------------------------------- #
    def _is_materialized(self) -> bool:
        """Return True when this Network already carries connectivity data.

        A Network is "materialized" if it has cached weight matrices, a lazy
        array store (h5 companion), or an explicit edges list. Used by
        ``_resolve`` to short-circuit when no further loading is required.
        """
        if getattr(self, "_cached_weights", None) is not None:
            return True
        if getattr(self, "_store", None) is not None:
            return True
        edges = getattr(self, "edges", None)
        if edges:
            return True
        return False

    def _resolve(self, source_dir: Optional[Union[str, Path]] = None) -> None:
        """Materialise this Network's connectivity from its declarative spec.

        Single source of truth for "given the YAML, populate the matrices".
        Idempotent: a successful resolution sets ``self._resolved = True``
        and subsequent calls are no-ops. Safe to call from
        ``Network.__init__``, ``Network.from_file``, and from
        ``SimulationExperiment.from_datamodel`` via a one-line hook.

        Resolution order (first match wins):

        1. Already materialised (cached weights / store / explicit edges):
           mark resolved and return.
        2. ``data_file`` companion (.h5 / .zarr + .yaml sidecar): load
           lazily via ``tvbo.data.network_io.attach_lazy_store``.
        3. ``bids_dir`` BEP017 directory: route through ``from_bids`` and
           copy matrices onto self.
        4. ``graph_generator.builder`` Callable: invoke (added in A2).
        5. ``parcellation`` (+ optional ``tractogram``, ``bids.segmentation``,
           ``bids.scale``): normative DB load via
           ``get_normative_connectome_data``.
        6. None of the above: no-op (Network must have been constructed via
           explicit ``nodes``/``edges`` or ``Network.from_matrix``).

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
        if self._is_materialized():
            self._resolved = True
            return
        if getattr(self, "data_file", None):
            self._resolve_from_data_file(source_dir)
        elif getattr(self, "bids_dir", None):
            self._resolve_from_bids_dir(source_dir)
        elif self._has_graph_generator_builder():
            self._resolve_from_graph_generator(source_dir)
        elif getattr(self, "parcellation", None):
            self._resolve_from_parcellation()
        self._resolved = True

    def _has_graph_generator_builder(self) -> bool:
        gg = getattr(self, "graph_generator", None)
        if gg is None:
            return False
        return getattr(gg, "builder", None) is not None

    def _resolve_from_graph_generator(self, source_dir: Optional[Union[str, Path]]) -> None:
        """Invoke ``graph_generator.builder`` (a Callable) and copy its result onto self.

        The builder is a TVBO ``Callable`` (the same idiom used for monitor
        class references). Its ``module`` and ``name`` fields locate a
        Python callable; the ``graph_generator.parameters`` dict is passed
        as keyword arguments. The callable must return either a ``Network``
        instance or a tuple ``(weights, lengths)`` / ``(weights, lengths,
        node_params)``.

        ``source_dir`` is forwarded so builders that need it (e.g. for
        loading companion artefacts) can request it as a keyword argument.
        """
        gg = self.graph_generator
        cb = gg.builder
        module_name = getattr(cb, "module", None)
        func_name = getattr(cb, "name", None)
        if not module_name or not func_name:
            raise ValueError(
                "graph_generator.builder must have both `module` and `name` "
                f"set (got module={module_name!r}, name={func_name!r})"
            )

        import importlib

        # Make the YAML source directory importable so builders can live
        # next to the study YAML.
        added_to_path = False
        if source_dir is not None:
            src = str(Path(source_dir).resolve())
            if src not in os.sys.path:
                os.sys.path.insert(0, src)
                added_to_path = True

        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name)

            # Flatten Parameter list to a plain kwargs dict.
            kwargs: Dict[str, Any] = {}
            for p in (gg.parameters or {}).values():
                pname = getattr(p, "name", None)
                if pname is None:
                    continue
                kwargs[pname] = getattr(p, "value", None)
            if getattr(gg, "seed", None) is not None:
                kwargs.setdefault("seed", gg.seed)

            result = fn(**kwargs)
        finally:
            if added_to_path:
                try:
                    os.sys.path.remove(src)
                except ValueError:
                    pass

        # Accept either a Network or a (weights, lengths[, node_params]) tuple.
        if isinstance(result, Network):
            for attr in ("nodes", "edges", "number_of_nodes", "descriptor", "mesh"):
                val = getattr(result, attr, None)
                if val is not None:
                    setattr(self, attr, val)
            for cache in ("_cached_weights", "_cached_lengths", "_store",
                          "_mesh_vertices", "_mesh_elements", "_mesh_normals"):
                v = getattr(result, cache, None)
                if v is not None:
                    setattr(self, cache, v)
        else:
            if not isinstance(result, tuple) or len(result) not in (2, 3):
                raise TypeError(
                    "graph_generator.builder must return a Network or a "
                    "(weights, lengths) / (weights, lengths, node_params) tuple."
                )
            weights = np.asarray(result[0])
            lengths = np.asarray(result[1]) if result[1] is not None else None
            node_params = result[2] if len(result) == 3 else None
            n_nodes = weights.shape[0]
            self.number_of_nodes = n_nodes
            self.nodes = [
                tvbo_datamodel.Node(id=i, label=f"node_{i}") for i in range(n_nodes)
            ]
            self.edges = []
            self._cached_weights = weights
            if lengths is not None:
                self._cached_lengths = lengths
            if node_params:
                # Builder may attach per-node parameters as a dict of
                # {param_name: array of len n_nodes}. Materialise these
                # onto each Node so downstream codegen can consume them.
                for pname, arr in node_params.items():
                    arr = np.asarray(arr)
                    for i in range(n_nodes):
                        if self.nodes[i].parameters is None:
                            self.nodes[i].parameters = {}
                        self.nodes[i].parameters[pname] = tvbo_datamodel.Parameter(
                            name=pname, value=float(arr[i])
                        )

    def _resolve_from_data_file(self, source_dir: Optional[Union[str, Path]]) -> None:
        """Populate self from a companion .h5/.zarr sidecar referenced by ``self.data_file``."""
        data_file = Path(self.data_file)
        if not data_file.is_absolute():
            base = Path(source_dir) if source_dir else Path.cwd()
            data_file = (base / data_file).resolve()
        sidecar = data_file.with_suffix(".yaml") if data_file.suffix in (".h5", ".zarr") else data_file
        if not sidecar.exists():
            raise FileNotFoundError(f"No YAML sidecar found for {data_file}")

        from tvbo.data.network_io import load_network

        loaded = load_network(sidecar)
        # The sidecar is the authoritative source of connectivity. Replace
        # connectivity-bearing fields unconditionally. Inline-authored
        # coupling / transforms / parameters live in slots NOT listed here
        # and are preserved on self.
        for attr in ("nodes", "edges", "number_of_nodes", "descriptor"):
            val = getattr(loaded, attr, None)
            if val is not None:
                setattr(self, attr, val)
        store = getattr(loaded, "_store", None)
        if store is not None:
            self._store = store
        # The Mesh schema slot transfers along with the other LinkML
        # fields below; only the array caches are copied here.
        loaded_mesh = getattr(loaded, "mesh", None)
        if loaded_mesh is not None and getattr(self, "mesh", None) is None:
            self.mesh = loaded_mesh
        for cache in ("_cached_weights", "_cached_lengths",
                       "_mesh_vertices", "_mesh_elements", "_mesh_normals"):
            v = getattr(loaded, cache, None)
            if v is not None:
                setattr(self, cache, v)

    def _resolve_from_bids_dir(self, source_dir: Optional[Union[str, Path]]) -> None:
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
        for cache in ("_cached_weights", "_cached_lengths", "_observations", "_store"):
            v = getattr(loaded, cache, None)
            if v is not None:
                setattr(self, cache, v)

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

        # Optional BIDS disambiguation (seg-, scale-) when the same atlas is
        # published at multiple resolutions or segmentations.
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
            w_arr, l_arr = get_normative_connectome_data(
                atlas_name, trk_name, segmentation=seg, scale=scale
            )
        except FileNotFoundError:
            return

        n_nodes = w_arr.shape[0]
        if l_arr is not None and l_arr.shape[0] != n_nodes:
            import warnings

            warnings.warn(
                f"Weight matrix ({n_nodes}x{n_nodes}) and length matrix "
                f"({l_arr.shape[0]}x{l_arr.shape[1]}) have different sizes. "
                f"Using minimum size."
            )
            n_nodes = min(n_nodes, l_arr.shape[0])
            w_arr = w_arr[:n_nodes, :n_nodes]
            l_arr = l_arr[:n_nodes, :n_nodes]

        if not self.nodes or len(self.nodes) != n_nodes:
            self.nodes = [tvbo_datamodel.Node(id=i, label=f"region_{i}") for i in range(n_nodes)]
        if not self.edges:
            self.edges = []
        self.number_of_nodes = n_nodes
        self._cached_weights = np.asarray(w_arr)
        if l_arr is not None:
            self._cached_lengths = np.asarray(l_arr)

    # -- Backward-compat properties: conduction_speed, global_coupling_strength --
    @property
    def conduction_speed(self):
        """Access conduction_speed from parameters dict."""
        if self.parameters and "conduction_speed" in self.parameters:
            return self.parameters["conduction_speed"]
        return None

    @conduction_speed.setter
    def conduction_speed(self, val):
        self.parameters["conduction_speed"] = val

    @property
    def global_coupling_strength(self):
        """Access global_coupling_strength from parameters dict."""
        if self.parameters and "global_coupling_strength" in self.parameters:
            return self.parameters["global_coupling_strength"]
        return None

    @global_coupling_strength.setter
    def global_coupling_strength(self, val):
        self.parameters["global_coupling_strength"] = val

    # -- Serialization: hide internal cached arrays from LinkML dumpers --
    # JsonObj._items() controls what yaml_dumper / json_dumper / as_dict see.
    # Without this, _cached_weights (numpy arrays) leak into yaml.SafeDumper.
    _INTERNAL_ATTRS = frozenset(
        {
            "_cached_weights",
            "_cached_lengths",
            "_store",
            "_arrays",
            "_edge_params",
            "_pytree_data",
            "_node_mapping_data",
            "_parent_network_obj",
            "_save_path",
            "_orientations",
            "_resolved",
            # _mesh is no longer a private cache — it's the LinkML
            # Network.mesh slot. Keep the array caches that adapters set.
            "_mesh_vertices",
            "_mesh_elements",
            "_mesh_normals",
        }
    )

    # Network.mesh is now a first-class LinkML slot (range Mesh, inlined)
    # — no read-only property wrapper needed. Runtime caches of the
    # underlying arrays (``_mesh_vertices``, ``_mesh_elements``,
    # ``_mesh_normals``) remain in ``_INTERNAL_ATTRS`` and are populated
    # by adapters such as ``from_tvb_surface``.

    @property
    def parent_network_obj(self) -> Optional["Network"]:
        """The parent Network object, if assigned via object reference.

        Returns ``None`` when ``parent_network`` was set as a plain
        string path or was never set.
        """
        try:
            return object.__getattribute__(self, "_parent_network_obj")
        except AttributeError:
            return None

    def _items(self):
        for k, v in super()._items():
            if k not in self._INTERNAL_ATTRS:
                yield k, v

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.Network) -> "Connectome":
        """Create a Connectome from a datamodel instance.

        Parameters
        ----------
        datamodel : tvbo_datamodel.Network
            Source datamodel Connectome instance

        Returns
        -------
        Connectome
            New Connectome with fields copied from datamodel

        Examples
        --------
        ```python
        from tvbo.datamodel import schema as tvbo_datamodel
        dm = tvbo_datamodel.Network(number_of_nodes=10)
        sc = Connectome.from_datamodel(dm)
        ```
        """
        data = as_dict(datamodel)
        # as_dict returns a dict-like object that works with **kwargs
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_matrix(
        cls,
        weights: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        labels: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "Network":
        """Create a Network from named edge-property matrices.

        This is a convenience constructor for creating networks from matrix
        representations. For performance, matrices are stored directly and
        edges are generated lazily only when needed.

        Any keyword argument whose value is array-like (ndarray, sparse
        matrix, or nested sequence) is treated as a named edge-property
        matrix and stored via ``set_matrix``. All other keyword arguments
        are forwarded to the ``Network`` constructor.

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

        Returns
        -------
        Network
            New Network with nodes derived from labels and matrices stored
            for efficient access.

        Examples
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
        bids_dir: Union[str, Path],
        atlas: Optional[str] = None,
        structural_measures: Optional[List[str]] = None,
        observational_measures: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "Network":
        """Create a Network from BEP017-compliant BIDS connectivity data.

        Loads structural connectivity (weights, lengths) and optionally
        observational targets (FC) from a BIDS derivatives directory using
        the BEP017 relationship matrix format.

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

        Returns
        -------
        Network
            Network with matrices loaded from BEP017 files.
            Observational data accessible via network.observations dict.

        Examples
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
        print(network.weights_matrix.shape)  # (84, 84)
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
        def load_measure(measure: str) -> Optional[np.ndarray]:
            pattern = f"*meas-{measure}_relmat*.tsv"
            matches = list(bids_dir.glob(pattern))
            if not matches:
                return None
            # Load TSV (dense format - no header, tab-separated)
            return np.loadtxt(matches[0], delimiter="\t")

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

        # Build network
        instance = cls(
            nodes=nodes,
            edges=[],
            number_of_nodes=n_nodes,
            number_of_regions=n_nodes,
            label=atlas or bids_dir.name,
            **kwargs,
        )

        # Store matrices using object.__setattr__ to bypass LinkML
        object.__setattr__(instance, "_cached_weights", weights)
        object.__setattr__(instance, "_cached_lengths", lengths)
        object.__setattr__(instance, "_bids_dir", str(bids_dir))
        object.__setattr__(instance, "_bids_observations", observations)

        return instance

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        """Observational data loaded from BIDS (e.g., FC for optimization targets)."""
        # Use object.__getattribute__ to bypass LinkML and get plain Python dict
        try:
            return object.__getattribute__(self, "__dict__").get("_bids_observations", {})
        except (AttributeError, KeyError):
            return {}

    def load_from_bids(
        self,
        bids_dir: Union[str, Path],
        structural_measures: Optional[List[str]] = None,
        observational_measures: Optional[List[str]] = None,
        atlas: Optional[str] = None,
    ) -> "Network":
        """Load BEP017 data into existing network.

        Allows loading structural connectivity and/or observational targets
        independently into an already-configured network (preserves coupling).

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

        Returns
        -------
        Network
            Self (for method chaining).

        Example
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
        def load_measure(measure: str) -> Optional[np.ndarray]:
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
                self._cached_weights = weights
                self._cached_lengths = lengths
                self.number_of_nodes = n_nodes
                self.number_of_regions = n_nodes

                # Load labels from nodeindices file
                nodeindices_files = list(bids_dir.glob(f"*atlas-{atlas}*_nodeindices.tsv"))
                if nodeindices_files:
                    df = pd.read_csv(nodeindices_files[0], sep="\t")
                    if "label" in df.columns:
                        labels = df["label"].tolist()
                        self.nodes = [tvbo_datamodel.Node(id=i, label=labels[i]) for i in range(n_nodes)]

        # Load observational measures if requested
        if observational_measures:
            # Use object.__setattr__ to store as plain Python dict, bypassing LinkML
            obs = object.__getattribute__(self, "__dict__").get("_bids_observations", {})
            for measure in observational_measures:
                data = load_measure(measure)
                if data is not None:
                    obs[measure] = data
            object.__setattr__(self, "_bids_observations", obs)

        object.__setattr__(self, "_bids_dir", str(bids_dir))
        return self

    def load_matrix(
        self,
        weights: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        labels: Optional[list[str]] = None,
    ) -> "Network":
        """Load weight/length matrices into existing network (preserves coupling).

        Use this instead of from_matrix when you need to update connectivity
        data while keeping the network's coupling definitions intact.

        Parameters
        ----------
        weights : np.ndarray
            Connection weight matrix (N x N).
        lengths : np.ndarray, optional
            Tract length matrix (N x N).
        labels : list of str, optional
            Node labels. Updates nodes if provided.

        Returns
        -------
        Network
            Self (for chaining).
        """
        weights = np.asarray(weights)
        n_nodes = weights.shape[0]

        # Update cached matrices
        self._cached_weights = weights
        self._cached_lengths = lengths if lengths is not None else None

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

        This is a convenience constructor for creating networks directly from
        YAML specifications, commonly used in notebooks and scripts.

        Parameters
        ----------
        yaml_string : str
            YAML string defining the network with nodes and edges.
        **kwargs : Any
            Additional keyword arguments passed to Network constructor.

        Returns
        -------
        Network
            New Network parsed from the YAML string.

        Examples
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
    def from_file(cls, path: Union[str, Path], **kwargs) -> "Network":
        """Load from YAML/JSON sidecar with lazy binary companion.

        Supports YAML and JSON sidecars (auto-detected by extension).
        Supports HDF5, Zarr, and CSV companions.
        Arrays are NOT loaded into memory — loaded lazily on first access.

        Parameters
        ----------
        path : str or Path
            Path to YAML or JSON sidecar file.

        Returns
        -------
        Network
            Network with lazy array references.

        Examples
        --------
        >>> net = Network.from_db("dk87")
        >>> net.number_of_nodes       # metadata: instant, no I/O
        87
        >>> net.weights_matrix.shape  # arrays: loaded on first access
        (87, 87)
        """
        from tvbo.data.network_io import load_network

        return load_network(path)

    @classmethod
    def from_tvb_zip(cls, zip_path: Union[str, Path]) -> "Network":
        """Import from TVB connectivity ZIP (weights.txt + tract_lengths.txt).

        Parameters
        ----------
        zip_path : str or Path
            Path to TVB connectivity ZIP file.

        Returns
        -------
        Network
            Network with arrays loaded, ready for ``save()``.

        Examples
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

        Lossless conversion preserving all TVB fields (weights, lengths,
        centres, cortical flags, areas, hemispheres, conduction speed).

        Parameters
        ----------
        connectivity : tvb.datatypes.connectivity.Connectivity
            Configured TVB Connectivity instance.

        Returns
        -------
        Network
            Network with arrays loaded, ready for ``save()``.

        Examples
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
        2. **Vertex-level** (child): mesh + region_mapping linking
           vertices to regions via hierarchical ``node_mapping``

        Parameters
        ----------
        connectivity : tvb.datatypes.connectivity.Connectivity
            Configured TVB Connectivity (region-level).
        surface : tvb.datatypes.surfaces.Surface
            TVB CorticalSurface with vertices and triangles.
        region_mapping : tvb.datatypes.region_mapping.RegionMapping
            TVB RegionMapping (vertex → region).

        Returns
        -------
        tuple[Network, Network]
            ``(region_network, surface_network)``

        Examples
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
        path: Union[str, Path],
        binary_format: str = "h5",
        sidecar_format: str = "yaml",
    ):
        """Save as sidecar + binary companion.

        Sidecar is written via LinkML yaml_dumper or json_dumper —
        always schema-valid output, no manual serialization.

        Parameters
        ----------
        path : str or Path
            Output path for sidecar.
        binary_format : str
            "h5" (default), "zarr", or "csv".
        sidecar_format : str
            "yaml" (default) or "json".

        Examples
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

    def to_bep017(self, output_dir: Union[str, Path]):
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

        Reads entities directly from Network attributes — no YAML
        serialization round-trip needed.  Sensor networks (descriptor
        ``"sensors"``) use ``SENSOR_PATTERNS``; all others use
        ``RELMAT_PATTERNS``.

        Returns
        -------
        str
            BIDS-compliant filename for this network.

        Examples
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
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "Network":
        """Download a normative connectivity network from the tvbo platform.

        Fetches the sidecar (YAML) and companion (HDF5) from the tvbo API
        and caches locally for subsequent loads.

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

        Returns
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
    def list_platform_networks(cls, base_url: str = TVBO_PLATFORM_URL, **filters) -> List[Dict]:
        """List available normative networks on the tvbo platform.

        Parameters
        ----------
        base_url : str
            Platform base URL.
        **filters
            Filtering parameters (e.g., atlas="DesikanKilliany").

        Returns
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
    def load(cls, source: Union[str, Path, None] = None, **entities) -> Union["Network", list["Network"]]:
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

        Returns
        -------
        Network or list[Network]
            A single Network, or a list when multiple BIDS matches occur.

        Examples
        --------
        >>> Network.load("Lobar")                               # database name
        >>> Network.load("networks/my_network.yaml")            # YAML file
        >>> Network.load("networks/my_network.h5")              # HDF5 companion
        >>> Network.load(atlas="Schaefer2018", scale="100")     # BIDS entities
        """
        if source is not None:
            p = Path(source)
            ext = p.suffix.lower()
            # HDF5 companion → resolve to YAML sidecar
            if ext in (".h5", ".hdf5"):
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
    def from_db(cls, name: Optional[str] = None, **entities) -> Union["Network", list["Network"]]:
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

        Returns
        -------
        Network or list[Network]

        Examples
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

        Returns
        -------
        list[str]
            Sorted list of matching network stems.

        Examples
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

    def to_yaml(self, filepath: Optional[str] = None, format: str = "tvbo") -> str:
        """Serialize Connectome to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to save YAML file. If None, returns YAML string.
        format : str
            Output format: "tvbo" (default) or "pyrates".
            PyRates format generates a complete experiment YAML (network + dynamics).

        Returns
        -------
        str
            YAML representation of the Connectome

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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
    def tree_flatten(self) -> Tuple[Tuple[JaxArray, JaxArray], Tuple[str]]:
        """Return children and auxiliary data for JAX pytree support.

        Children: (weights, lengths) so JAX can map/transform numerical payloads.
        Aux data: metadata dict WITHOUT the array data to avoid duplication.
        """
        # Convert metadata to a JSON string for stable equality in JAX
        import json as _json

        import numpy as _np

        from linkml_runtime.utils.yamlutils import YAMLRoot

        def _jsonable(o):
            try:
                import jax

                if isinstance(o, jax.Array):
                    o = _np.array(o)
            except Exception:
                pass
            # numpy scalars -> python scalars
            if isinstance(o, _np.generic):
                return o.item()
            # numpy arrays -> lists
            if isinstance(o, _np.ndarray):
                return o.tolist()
            # tuples -> lists for JSON
            if isinstance(o, tuple):
                return list(o)
            # LinkML enums -> plain text string (NOT as_dict which creates
            # a {'_code': {...}} dict that becomes an unparseable JsonObj
            # on round-trip through json.loads → cls(**meta_dict))
            from linkml_runtime.utils.enumerations import EnumDefinitionImpl

            if isinstance(o, EnumDefinitionImpl):
                return str(o)
            # LinkML dataclasses -> dict via as_dict
            if isinstance(o, YAMLRoot):
                return as_dict(o)
            # dataclasses with __dict__
            if hasattr(o, "__dict__"):
                return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            # last resort: stringify
            return str(o)

        # children are the heavy numeric arrays; keep arrays out of aux
        # Always return arrays to maintain consistent tree structure
        # If weights/lengths are None, use empty arrays with proper shape based on number_of_regions

        # Check if we have cached PyTree data (from a previous unflatten)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            weights_arr, lengths_arr = self._pytree_data
        else:
            # Use weights_matrix/lengths_matrix properties which handle edges, Matrix, or defaults
            weights_arr = self.weights_matrix
            lengths_arr = self.lengths_matrix

            # Fallback to zeros if properties return None
            n = self.number_of_nodes or 1
            if weights_arr is None:
                weights_arr = np.zeros((n, n))
            else:
                weights_arr = np.asarray(weights_arr)

            if lengths_arr is None:
                lengths_arr = np.zeros((n, n))
            else:
                lengths_arr = np.asarray(lengths_arr)

        children = (weights_arr, lengths_arr)

        # Get full metadata but exclude weights/lengths to avoid embedding arrays
        meta_dict = as_dict(self)
        # as_dict can return various dict-like structures
        if not isinstance(meta_dict, dict):
            meta_dict = dict(meta_dict) if hasattr(meta_dict, "__iter__") else {}
        # Remove weights, lengths, parcellation, edges, and cache attributes from metadata
        # Parcellation is excluded to prevent reloading data during unflatten
        # Edges are excluded because they contain Parameter objects with non-deterministic
        # string serialization, and the weight/length info is already in the children arrays
        # Also exclude private cached arrays set by from_matrix()
        meta_dict_without_arrays = {
            k: v
            for k, v in meta_dict.items()
            if k
            not in (
                "weight",
                "length",
                "parcellation",
                "edges",
                "_pytree_data",
                "_cached_weights",
                "_cached_lengths",
                "_bids_dir",
                "_bids_observations",
            )
        }

        def _strip_none(obj):
            if isinstance(obj, dict):
                # as_dict() serializes LinkML enums as {'_code': {'text': 'mm', ...}}
                # Flatten back to the plain text key for clean round-trips
                if "_code" in obj and isinstance(obj["_code"], dict) and "text" in obj["_code"]:
                    return obj["_code"]["text"]
                return {k: _strip_none(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [_strip_none(x) for x in obj]
            return obj

        meta_json = _json.dumps(_strip_none(meta_dict_without_arrays), sort_keys=True, default=_jsonable)
        aux = (meta_json,)
        return children, aux  # type: ignore[return-value]

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[str], children: Tuple[JaxArray, JaxArray]) -> "Connectome":
        import json as _json

        (meta_json,) = aux_data
        (weights, lengths) = children
        # Reconstruct from metadata dict (which doesn't include weights/lengths/parcellation)
        meta_dict = _json.loads(meta_json)

        # Don't try to reconstruct Matrix objects from the arrays here
        # because during JAX tracing, we can't convert tracers to Python lists.
        # Instead, we'll create a minimal object and rely on _pytree_data for array access.
        # The weights_matrix and lengths_matrix properties will use _pytree_data if available.

        obj = cls(**meta_dict)

        # Store the array children as a tuple using object.__setattr__
        # This is what weights_matrix and lengths_matrix will use
        object.__setattr__(obj, "_pytree_data", (weights, lengths))

        return obj

    # Back-compat pointer
    @property
    def metadata(self) -> "Connectome":
        return self

    # ---- Numeric accessors (compute on demand; no extra attributes) ----
    def _matrix_from_array(self, arr: Union[np.ndarray, JaxArray]) -> tvbo_datamodel.Matrix:
        arr = jnp.array(arr)
        N0, N1 = arr.shape
        x = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N0)])
        y = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N1)])
        return tvbo_datamodel.Matrix(x=x, y=y, values=arr.reshape(-1).astype(jnp.float32).tolist())

    @staticmethod
    def _get_edge_param(edge, name: str) -> Optional[float]:
        return edge.parameters[name].value if name in edge.parameters else None

    def _weights_from_edges(self) -> Optional[np.ndarray]:
        """Compute weights matrix from edges.

        Looks for 'weight' parameter in edge.parameters.
        Matrices are target-by-source: an edge source -> target is stored at
        [target, source], matching the coupling convention used by backends.
        Undirected edges (directed=False) are mirrored to produce symmetric matrix.
        Returns None if no edges are defined.
        """
        if not self.edges:
            return None
        n = self.number_of_nodes or self.number_of_regions or 1
        W = np.zeros((n, n), dtype=np.float64)
        for edge in self.edges:
            i, j = edge.source, edge.target
            if i is None or j is None:
                continue
            if 0 <= i < n and 0 <= j < n:
                w = self._get_edge_param(edge, "weight")
                if w is None:
                    w = 1.0  # edge exists → default unit weight
                W[j, i] = w
                # Mirror for undirected edges (symmetric)
                if not edge.directed:
                    W[i, j] = w
        return W

    def _get_node_position(self, node_id: int) -> Optional[Tuple[float, float, float]]:
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

    def _compute_euclidean_distance(self, i: int, j: int) -> Optional[float]:
        """Compute Euclidean distance between two nodes from their positions."""
        pos_i = self._get_node_position(i)
        pos_j = self._get_node_position(j)
        if pos_i is None or pos_j is None:
            return None
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        dz = pos_j[2] - pos_i[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _lengths_from_edges(self) -> Optional[np.ndarray]:
        """Compute lengths/distances matrix from edges.

        Looks for 'length' or 'distance' parameter in edge.parameters.
        Matrices are target-by-source: an edge source -> target is stored at
        [target, source], matching the coupling convention used by backends.
        If no distance is specified but nodes have positions, computes
        Euclidean distance from node coordinates (in distance_unit).
        Undirected edges (directed=False) are mirrored to produce symmetric matrix.
        Returns None if no edges are defined.
        """
        if not self.edges:
            return None
        n = self.number_of_nodes or self.number_of_regions or 1
        L = np.zeros((n, n), dtype=np.float64)
        for edge in self.edges:
            i, j = edge.source, edge.target
            if i is None or j is None:
                continue
            if 0 <= i < n and 0 <= j < n:
                d = self._get_edge_param(edge, "length")
                if d is None:
                    d = self._get_edge_param(edge, "distance")
                # If no explicit distance, compute from node positions
                if d is None or d == 0:
                    d = self._compute_euclidean_distance(i, j)
                if d is None:
                    d = 0.0
                L[j, i] = d
                # Mirror for undirected edges (symmetric)
                if not edge.directed:
                    L[i, j] = d
        return L

    def _delays_from_edges(self) -> Optional[np.ndarray]:
        """Compute delays matrix from edges.

        Looks for 'delay' parameter in edge.parameters.
        Undirected edges (directed=False) are mirrored to produce symmetric matrix.
        Returns None if no edges are defined or no delays are set.
        """
        if not self.edges:
            return None
        n = self.number_of_nodes or self.number_of_regions or 1
        D = np.zeros((n, n), dtype=np.float64)
        has_delays = False
        for edge in self.edges:
            i, j = edge.source, edge.target
            if i is None or j is None:
                continue
            if 0 <= i < n and 0 <= j < n:
                delay = self._get_edge_param(edge, "delay")
                D[j, i] = delay
                # Mirror for undirected edges (symmetric)
                if not edge.directed:
                    D[i, j] = delay
                if delay > 0:
                    has_delays = True
        return D if has_delays else None

    @property
    def node_labels(self) -> List[str]:
        """Node labels derived from nodes.

        Returns
        -------
        list of str
            Labels for each node in the network

        Examples
        --------
        ```python
        net = Network.from_matrix(weights, lengths, labels=["A", "B", "C"])
        print(net.node_labels)  # ['A', 'B', 'C']
        ```
        """
        if not self.nodes:
            return []
        return [n.label for n in self.nodes]  # type: ignore[union-attr]

    @property
    def weights_matrix(self) -> Optional[Union[np.ndarray, JaxArray]]:
        """Connection weights matrix as numpy/JAX array.

        Returns cached matrix if available (from from_matrix), otherwise
        computes from edges. If transforms are defined, applies them
        sequentially.

        Returns
        -------
        np.ndarray or jax.Array, optional
            Connection weights matrix (N x N), or None if no edges/matrix

        Examples
        --------
        ```python
        net = Network.from_matrix(weights, lengths)
        W = net.weights_matrix
        print(f"Shape: {W.shape}, Mean: {W.mean():.3f}")
        ```
        """

        if len(self.nodes) == 1:
            return np.zeros((1, 1))
        # Check if we have cached PyTree data from tree_unflatten (during JAX transformations)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            return self._pytree_data[0]

        # Check _arrays (set by set_matrix / add_edges). When the sidecar
        # declares a `primary_weight`, that named edge wins over the default
        # weight / weights / sc lookup — lets one Network file expose
        # several variants under named edges and pick the active one.
        _arrs = self._get_arrays()
        _primary = getattr(self, "primary_weight", None)
        if _primary and _primary in _arrs:
            from scipy import sparse as _sp

            W = _arrs[_primary]
            if _sp.issparse(W):
                W = W.toarray()
        elif "weight" in _arrs:
            from scipy import sparse as _sp

            W = _arrs["weight"]
            if _sp.issparse(W):
                W = W.toarray()
        elif "weights" in _arrs:
            from scipy import sparse as _sp

            W = _arrs["weights"]
            if _sp.issparse(W):
                W = W.toarray()
        elif "sc" in _arrs:
            from scipy import sparse as _sp

            W = _arrs["sc"]
            if _sp.issparse(W):
                W = W.toarray()
        # Check for cached matrix from from_matrix (performance optimization)
        elif hasattr(self, "_cached_weights") and self._cached_weights is not None:
            W = self._cached_weights
            from scipy import sparse as _sp

            if _sp.issparse(W):
                W = W.toarray()
        elif hasattr(self, "_store") and self._store is not None:
            # Lazy load from companion file (set by load_network)
            arrays = self._store.arrays
            if _primary and _primary in arrays:
                self._cached_weights = arrays[_primary]
                W = self._cached_weights
            elif "weight" in arrays:
                self._cached_weights = arrays["weight"]
                W = self._cached_weights
            elif "weights" in arrays:
                self._cached_weights = arrays["weights"]
                W = self._cached_weights
            else:
                W = self._weights_from_edges()
        else:
            # Compute from edges (fallback for networks built from explicit edges)
            W = self._weights_from_edges()

        if W is None:
            n = len(self.nodes) if self.nodes else (self.number_of_nodes or self.number_of_regions or 0)
            if n > 0:
                return np.zeros((n, n), dtype=np.float64)
            return None

        # Apply transforms targeting "weight"
        for t in self.transforms or []:
            if t.name == "weight":
                W = self._apply_transform(W, t)
        return W

    @property
    def weights(self):
        return self.weights_matrix

    @property
    def lengths_matrix(self) -> Optional[Union[np.ndarray, JaxArray]]:
        """Tract length matrix as numpy/JAX array.

        Returns the (N x N) matrix of physical distances (tract lengths)
        between brain regions in millimeters.

        Returns
        -------
        np.ndarray or jax.Array, optional
            Tract lengths matrix (N x N) in mm, or None if no matrix/edges

        Examples
        --------
        ```python
        net = Network.from_matrix(weights, lengths)
        L = net.lengths_matrix
        print(f"Mean length: {L.mean():.1f} mm")
        ```
        """
        if len(self.nodes) == 1:
            return np.zeros((1, 1))
        # Check if we have cached PyTree data from tree_unflatten (during JAX transformations)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            return self._pytree_data[1]

        # Check _arrays (set by set_matrix / add_edges)
        _arrs = self._get_arrays()
        if "length" in _arrs:
            from scipy import sparse as _sp

            L = _arrs["length"]
            return L.toarray() if _sp.issparse(L) else L
        elif "lengths" in _arrs:
            from scipy import sparse as _sp

            L = _arrs["lengths"]
            return L.toarray() if _sp.issparse(L) else L

        # Check for cached matrix from from_matrix (performance optimization)
        if hasattr(self, "_cached_lengths") and self._cached_lengths is not None:
            return self._cached_lengths

        if hasattr(self, "_store") and self._store is not None:
            # Lazy load from companion file (set by load_network)
            arrays = self._store.arrays
            if "length" in arrays:
                self._cached_lengths = arrays["length"]
                return self._cached_lengths
            elif "lengths" in arrays:
                self._cached_lengths = arrays["lengths"]
                return self._cached_lengths

        # Compute from edges (fallback for networks built from explicit edges)
        L = self._lengths_from_edges()
        if L is None:
            n = len(self.nodes) if self.nodes else (self.number_of_nodes or self.number_of_regions or 0)
            if n > 0:
                return np.zeros((n, n), dtype=np.float64)
        return L

    @property
    def lengths(self):
        return self.lengths_matrix

    @property
    def distances(self):
        return self.lengths_matrix

    @property
    def labels(self) -> Dict[str, str]:
        """Brain region labels from atlas.

        Returns
        -------
        dict of str to str
            Mapping from region names to lookup labels

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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

        Returns
        -------
        nx.MultiDiGraph
            Graph with node/edge attributes from schema.
            Nodes have: id, label, dynamics, region, position, parameters
            Edges have: weight, delay, distance, directed, source_var, target_var, coupling
        """
        G = nx.MultiDiGraph()

        W = self.weights_matrix

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

        # Step 2: Add edges (prefer explicit edges, fall back to matrix)
        # Filter out template edges (no source/target) — those represent
        # matrix measures stored in the HDF5 companion, not graph edges.
        explicit_edges = [e for e in (self.edges or []) if getattr(e, "source", None) is not None]
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

                # If undirected, add reverse edge
                if not edge_attrs["directed"]:
                    G.add_edge(edge.target, edge.source, **edge_attrs)

        else:
            # No explicit edges - generate from stored matrices.
            # Build edges from the union of all stored matrices, attaching
            # each matrix's values as named edge attributes.
            from scipy import sparse as _sp

            arrays = self._get_arrays()

            # Collect matrix names from in-memory arrays, template-edge metadata,
            # and common aliases accessible via the generic matrix(...) accessor.
            matrix_names = set(arrays.keys())
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
                            G.add_edge(i, j, **edge_attrs)

        return G

    def __str__(self) -> str:
        parc = getattr(self, "parcellation", None)
        if parc and hasattr(parc, "atlas") and hasattr(parc.atlas, "name"):  # type: ignore[attr-defined]
            return f"Connectome-{parc.atlas.name}({self.number_of_regions})"  # type: ignore[attr-defined]
        return f"Connectome(N={self.number_of_regions})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def atlas(self) -> Any:
        """Brain atlas associated with this connectome.

        Returns
        -------
        Atlas
            Atlas object containing parcellation metadata

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        atlas = sc.atlas
        print(atlas.region_labels)
        ```
        """
        return self.get_atlas()

    def get_atlas(self) -> Any:
        """Retrieve the Atlas object for this connectome.

        Returns
        -------
        Atlas
            Atlas instance with parcellation metadata and terminology

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        atlas = sc.get_atlas()
        ```
        """
        from tvbo.classes.atlas import Atlas

        parc = getattr(self, "parcellation", None)
        atlas_data = parc.atlas if parc and hasattr(parc, "atlas") else None  # type: ignore[attr-defined]
        return Atlas(atlas_data)

    def compute_delays(self, output_unit: Optional[str] = None) -> Union[np.ndarray, JaxArray]:
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

        Returns
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

        When the network contains a rectangular gain matrix and a *target*
        network is provided, a bipartite graph is built with sensor nodes
        from ``self`` and region nodes from *target*.

        If *target* is ``None``, the method looks for a ``target_network``
        reference on the gain edge and loads it automatically.  Failing
        that, it falls back to ``dimension_labels`` stored on the edge to
        create label-only region nodes (no positions).

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

    def normalize_weights(self, equation_rhs: str = "(M - M_min) / (M_max - M_min)") -> None:
        """Add a normalization transform for connection weights.

        Convenience wrapper for ``add_transform("weight", ...)``.

        Parameters
        ----------
        equation_rhs : str, default="(M - M_min) / (M_max - M_min)"
            Right-hand side of normalization equation. Can reference
            M (matrix), M_min, M_max.

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize_weights("M / M_max")  # Normalize to [0, 1]
        normalized = sc.weights_matrix  # Returns normalized weights
        ```

        See Also
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

        Returns
        -------
        matplotlib.image.AxesImage
            Image object for adding colorbar

        Examples
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        fig, ax = plt.subplots()
        im = sc.plot_weights(ax, log=True)
        plt.colorbar(im, ax=ax)
        ```
        """
        import numpy as np
        from matplotlib.colors import LogNorm

        weights = self.weights_matrix
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

        Returns
        -------
        matplotlib.image.AxesImage
            Image object for adding colorbar

        Examples
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing both matrix plots

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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
        conduction_speed: Optional[float] = None,
        output_unit: Optional[str] = None,
    ) -> Union[np.ndarray, JaxArray]:
        """Calculate signal propagation delays between regions.

        Supports two network representations:

        1. **Matrix-based** — delays are ``lengths / conduction_speed``, with
           optional unit conversion via *output_unit*.
        2. **Edge-based** — delays are extracted from explicit edge objects that
           carry ``source``, ``target``, and a ``"delay"`` parameter.

        Parameters
        ----------
        conduction_speed : float, optional
            Override conduction speed. If *None*, uses ``self.conduction_speed``.
        output_unit : str, optional
            Desired output time unit (e.g. ``"ms"``, ``"s"``). When given,
            sympy unit conversion is applied. If *None*, the result is in the
            network's native time unit (defaults to ms).

        Returns
        -------
        np.ndarray or jax.Array
            Delay matrix (N x N). For edge-based networks, entries without an
            edge are ``NaN``.

        Raises
        ------
        ValueError
            If neither lengths matrix nor edge-based delays are available.

        Examples
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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

    def _delays_from_edges(self) -> Optional[np.ndarray]:
        """Build delay matrix from explicit edge objects.

        Returns None if edges don't represent point-to-point connections
        (i.e. they lack source/target attributes).
        """
        edges_with_endpoints = [
            e
            for e in self.edges
            if hasattr(e, "source") and hasattr(e, "target") and e.source is not None and e.target is not None
        ]
        if not edges_with_endpoints:
            return None

        n = self.number_of_nodes or 1
        delays = np.full((n, n), np.nan)
        has_delays = False
        for edge in edges_with_endpoints:
            if hasattr(edge, "parameters") and edge.parameters:
                delay_param = edge.parameters.get("delay")
                if delay_param and hasattr(delay_param, "value"):
                    delay_val = float(delay_param.value)
                    delays[edge.target, edge.source] = delay_val
                    has_delays = has_delays or delay_val > 0
                    if not edge.directed:
                        delays[edge.source, edge.target] = delay_val
        return delays if has_delays else None

    def _unit_conversion_factor(self, output_unit: str) -> float:
        """Compute multiplicative factor to convert native delay units to *output_unit*."""
        import sympy.physics.units as u
        from sympy import nsimplify
        from sympy.parsing.sympy_parser import parse_expr
        from tvbo.utils.units import unit_to_symbol

        unit_ns = dict(vars(u))

        distance_unit_str = unit_to_symbol(getattr(self, "distance_unit", None) or "mm")
        cs_param = self.conduction_speed
        speed_unit_str = (
            unit_to_symbol(cs_param.unit)
            if cs_param and cs_param.unit
            else f"{distance_unit_str}/{unit_to_symbol(getattr(self, 'time_unit', None) or 'ms')}"
        )
        unit_to_symbol(getattr(self, "time_unit", None) or "ms")
        target_time_str = unit_to_symbol(output_unit)

        # Native delay unit: distance / speed  (e.g. mm / (mm/ms) = ms)
        native_delay = parse_expr(distance_unit_str, local_dict=unit_ns) / parse_expr(speed_unit_str, local_dict=unit_ns)
        target_unit = parse_expr(target_time_str, local_dict=unit_ns)

        converted = u.convert_to(native_delay, target_unit)
        return float(nsimplify(converted / target_unit))

    def create_graph(self, weight_threshold: float = 0) -> nx.MultiDiGraph:
        """Create NetworkX graph from network structure.

        Prioritizes explicit nodes/edges representation over weight matrices.
        This allows proper visualization of heterogeneous networks with
        labeled nodes and typed edges.

        Parameters
        ----------
        weight_threshold : float, default=0
            Minimum weight for including an edge in the graph

        Returns
        -------
        networkx.MultiDiGraph
            Directed multigraph with 'weight' and 'delay' edge attributes.
            Nodes have 'label' and 'dynamics' attributes when available.
            Edges have 'source_var', 'target_var' attributes when available.

        Examples
        --------
        ```python
        # From explicit nodes/edges
        network = Network(nodes=[...], edges=[...])
        G = network.create_graph()

        # From weight matrix
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        G = sc.create_graph(weight_threshold=0.1)
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        ```
        """
        G = nx.MultiDiGraph()

        # Priority 1: Use explicit nodes/edges if available
        nodes = getattr(self, "nodes", None)
        edges = getattr(self, "edges", None)

        if nodes and len(nodes) > 0:
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

            if edges:
                for edge in edges:
                    source = getattr(edge, "source", None)
                    target = getattr(edge, "target", None)

                    if source is None or target is None:
                        continue

                    weight = self._get_edge_param(edge, "weight") or 0.0
                    if weight <= weight_threshold:
                        continue

                    edge_attrs = {
                        "weight": weight,
                        "delay": self._get_edge_param(edge, "delay") or 0.0,
                        "distance": self._get_edge_param(edge, "distance") or 0.0,
                        "directed": edge.directed,
                        "source_var": edge.source_var,
                        "target_var": edge.target_var,
                    }
                    G.add_edge(source, target, **edge_attrs)

            return G

        # Priority 2: Fall back to weight matrix representation
        W = self.weights_matrix
        D = self.calculate_delays() if self.lengths_matrix is not None else None
        N_regions = self.number_of_regions

        if N_regions is None or W is None:
            return G

        # Get node labels if available
        labels = self.labels if hasattr(self, "labels") and self.labels else None

        for i in range(N_regions):
            node_attrs = {"label": labels[i] if labels else f"node_{i}"}
            G.add_node(i, **node_attrs)

        for i in range(N_regions):
            for j in range(N_regions):
                if W[i, j] > weight_threshold:
                    delay = D[i, j] if D is not None else 0.0
                    G.add_edge(i, j, weight=W[i, j], delay=delay)

        return G

    def get_centers(self) -> Dict[int, Tuple[float, float, float]]:
        """Get 3D spatial coordinates of brain region centers.

        Resolution order:
        1. ``Node.position`` on ``self.nodes`` (in-memory)
        2. ``nodes/coordinates`` dataset in the HDF5/Zarr companion
        3. Atlas metadata (``terminology.entities[*].center``)

        Returns
        -------
        dict of int to tuple of float
            Mapping from region index to (x, y, z) coordinates in mm

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
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

        # --- Source 3: Atlas metadata (fallback) ---
        labels = []
        ids = []
        centers = []
        entities = self.get_atlas().terminology.entities
        # Handle both dict and list formats
        if hasattr(entities, "items"):
            entity_items = entities.items()
        elif isinstance(entities, list):
            entity_items = enumerate(entities)
        else:
            # Empty or unknown format - return default
            return {0: (0, 0, 0)}

        for region, entity in entity_items:
            # Handle entity being a dict or object
            if isinstance(entity, dict):
                lookup_label = entity.get("lookupLabel", region)
                center = entity.get("center", {})
                coord = (center.get("x", 0), center.get("y", 0), center.get("z", 0))
            else:
                lookup_label = getattr(entity, "lookupLabel", region)
                center = getattr(entity, "center", None)
                if center:
                    coord = (center.x, center.y, center.z)
                else:
                    coord = (0, 0, 0)
            labels.append(region)
            ids.append(lookup_label if isinstance(lookup_label, int) else region)
            centers.append(coord)

        if not centers:
            return {0: (0, 0, 0)}

        centers = np.array(centers)
        # Only sort if ids are numeric
        if all(isinstance(i, (int, float)) for i in ids):
            sort_idx = np.argsort(ids)
            centers = centers[sort_idx]
            labels = np.array(labels)[sort_idx]
            center_mapping = {int(i) - 1: tuple(center) for i, center in zip(sorted(ids), centers)}
        else:
            center_mapping = {i: tuple(center) for i, center in enumerate(centers)}

        if center_mapping == {}:
            return {0: (0, 0, 0)}
        return center_mapping

    def plot_graph(
        self,
        ax: Optional[Axes] = None,
        node_cmap: Union[str, Any] = "viridis",
        edge_cmap: Union[str, Any] = "viridis",
        node_colors: str = "in-strength",
        node_size: Union[str, float] = 8,
        threshold_percentile: float = 0,
        pos_scaling: float = 1,
        node_labels: bool = True,
        edge_labels: bool = True,
        log_in_strength: bool = True,
        node_size_scaling: float = 0,
        edge_color: str = "weight",
        pos: Union[str, Dict[int, List[float]]] = "spring",
        plot_brain: Optional[str] = None,
        edge_kwargs: Optional[Dict[str, Any]] = None,
        node_kwargs: Optional[Dict[str, Any]] = None,
        fontsize: float = 12,
        format: str = "networkx",
    ) -> Union[Figure, cm.ScalarMappable]:
        """Visualize connectome as network graph.

        Delegates to :func:`tvbo.plot.network_graph.plot_graph_networkx` or
        :func:`tvbo.plot.network_graph.plot_graph_bsplot` depending on *format*.

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

        Returns
        -------
        Figure or ScalarMappable
            Figure if ax is None, otherwise ScalarMappable for colorbar

        Examples
        --------
        ```python
        import matplotlib.pyplot as plt
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

        # Simple graph
        fig, ax = plt.subplots(figsize=(10, 10))
        mappable = sc.plot_graph(ax, threshold_percentile=75)
        plt.colorbar(mappable, ax=ax)

        # Anatomical layout
        fig, ax = plt.subplots()
        sc.plot_graph(ax, plot_brain="horizontal", node_labels=False)
        ```

        See Also
        --------
        plot_brain_surface : 3-D brain surface rendering with bsplot
        tvbo.plot.network.plot_graph_networkx : NetworkX backend
        tvbo.plot.network.plot_graph_bsplot : bsplot backend
        """
        from tvbo.plot.network import (
            plot_graph_bsplot,
            plot_graph_networkx,
            _resolve_positions,
            _threshold_graph,
        )

        G = self.graph

        # Threshold edges
        if threshold_percentile > 0:
            _threshold_graph(G, self.weights, threshold_percentile)

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

        Nodes are rendered as coloured spheres at their MNI coordinates
        (from atlas metadata); edges as tubes.  Requires ``bsplot``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If *None*, a new figure is created.
        weight_matrix : ndarray, optional
            Custom matrix for edge colouring.  If *None*, uses the
            default weights matrix.
        **kwargs
            Forwarded to :func:`tvbo.plot.network.plot_graph_brain`.

        Returns
        -------
        fig : Figure
        ax : Axes
        mappables : dict
            ``ScalarMappable`` objects (keys ``"nodes"`` / ``"edges"``).

        See Also
        --------
        tvbo.plot.network.plot_graph_brain : Full parameter list
        """
        from tvbo.plot.network import plot_graph_brain

        return plot_graph_brain(self, ax=ax, weight_matrix=weight_matrix, **kwargs)

    def _matrix_from_explicit_edges(self, param_name: str) -> Optional[np.ndarray]:
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
        edge_properties: Optional[List[str]] = None,
        weights_kwargs: Optional[Dict[str, Any]] = None,
        lengths_kwargs: Optional[Dict[str, Any]] = None,
        graph_kwargs: Optional[Dict[str, Any]] = None,
        log_weights: bool = False,
        plot_brain: Optional[bool] = None,
        brain_kwargs: Optional[Dict[str, Any]] = None,
        cmap: str = "magma",
        edge_percentile: float = 0,
        show_nodes: bool = True,
        show_edges: bool = True,
        max_edge_labels: int = 15,
    ) -> Figure:
        """Create comprehensive visualization with brain surface and matrices.

        Produces a multi-panel figure with one row per edge property.
        Each row contains either a brain surface + matrix heatmap
        (when *plot_brain* is True) or just a matrix heatmap, both
        coloured by the same property.

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

        Returns
        -------
        matplotlib.figure.Figure
            Figure with subplots

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.plot_overview(log_weights=True)
        ```

        See Also
        --------
        plot_graph : Network graph visualization
        plot_brain_surface : 3-D brain surface rendering
        plot_matrix : Side-by-side matrix visualization
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
            for pname in self._get_arrays().keys():
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
            mat = self.matrix(prop)
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
        Equivalent to ``add_transform("weight", "(M - M_min) / (M_max - M_min)")``.

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize()
        normalized_weights = sc.weights_matrix  # Now in [0, 1] range
        ```

        See Also
        --------
        add_transform : Add a transform on any edge property
        normalize_weights : Set custom normalization equation
        """
        self.add_transform("weight", "(M - M_min) / (M_max - M_min)")

    # ── Node mapping (hierarchical composition) ─────────────────────

    def set_node_mapping(
        self,
        mapping,
        parent_network=None,
        dataset_path: str = "/nodes/parent_index",
    ) -> None:
        """Set the node-to-parent mapping array.

        This stores the mapping data internally so that :func:`save`
        writes it into the HDF5 companion automatically — no manual
        ``h5py`` code required.

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

        Examples
        --------
        >>> surface_net.set_node_mapping(region_mapping,
        ...                             parent_network="dk_sc.yaml")
        >>> # or pass the Network object directly:
        >>> surface_net.set_node_mapping(region_mapping,
        ...                             parent_network=sc)
        >>> surface_net.save(tmpdir / "surface_rh.yaml")
        """
        data = np.asarray(mapping, dtype=np.int32)
        object.__setattr__(self, "_node_mapping_data", data)
        self.node_mapping = dataset_path
        if parent_network is not None:
            self.parent_network = parent_network  # __setattr__ handles Network→str

    @property
    def node_mapping_data(self):
        """The node-to-parent mapping array, or ``None``.

        Resolution order: in-memory (set via :meth:`set_node_mapping`)
        → lazy load from HDF5 companion (if ``node_mapping`` is set).
        """
        # 1. In-memory (set by user or by load_network)
        try:
            data = object.__getattribute__(self, "_node_mapping_data")
            if data is not None:
                return data
        except AttributeError:
            pass

        # 2. Lazy load from companion file via _store
        store = getattr(self, "_store", None)
        nm_path = getattr(self, "node_mapping", None)
        if store is not None and nm_path:
            # Convert "/nodes/parent_index" → "nodes/parent_index"
            key = nm_path.lstrip("/")
            try:
                data = store.read_dataset(key)
                object.__setattr__(self, "_node_mapping_data", data)
                return data
            except (KeyError, AttributeError):
                pass

        return None

    # ── Generalized edge / matrix API ─────────────────────────────

    def _get_arrays(self) -> dict:
        """Access the internal ``_arrays`` dict, bypassing JsonObj."""
        try:
            d = object.__getattribute__(self, "_arrays")
            if isinstance(d, dict):
                return d
        except AttributeError:
            pass
        d: dict = {}
        object.__setattr__(self, "_arrays", d)
        return d

    def set_matrix(
        self,
        name: str,
        data,
    ) -> None:
        """Set a named edge matrix.

        Accepts dense NumPy arrays, scipy sparse matrices (CSR, COO, etc.),
        or any array-like that can be converted. The matrix is stored
        internally and a template edge is created/updated automatically
        so that ``save()`` writes it to the HDF5 companion.

        Parameters
        ----------
        name : str
            Matrix name (e.g. ``"weight"``, ``"length"``,
            ``"local_connectivity"``). Used as the HDF5 group name
            under ``edges/``.
        data : array-like or scipy.sparse matrix
            The edge matrix to store.

        Examples
        --------
        >>> net.set_matrix("weight", W_dense)
        >>> net.set_matrix("local_connectivity", LC_sparse_csr)
        """
        from scipy import sparse

        arrays = self._get_arrays()

        if sparse.issparse(data):
            arrays[name] = data
        else:
            arrays[name] = np.asarray(data)

        # Backward compat: sync legacy caches
        if name in ("weight", "weights"):
            self._cached_weights = arrays[name]
        elif name in ("length", "lengths"):
            self._cached_lengths = arrays[name]

        self._ensure_template_edge(name)

    def matrix(
        self,
        name: str,
        format: Optional[str] = None,
    ):
        """Get a named edge matrix, optionally in a specific format.

        Resolution order: ``_arrays`` (user-set) → ``_store`` (lazy file)
        → ``_cached_*`` (legacy) → ``None``.

        Parameters
        ----------
        name : str
            Matrix name (e.g. ``"weight"``, ``"length"``).
        format : str, optional
            Return format: ``"dense"``, ``"csr"``, ``"coo"``, ``"lil"``.
            If ``None``, returns the matrix in whatever format it is
            currently stored in.

        Returns
        -------
        np.ndarray or scipy.sparse matrix or None
        """
        from scipy import sparse
        from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

        mat = None
        # 1. User-set matrices (highest priority)
        arrays = self._get_arrays()
        if name in arrays:
            mat = arrays[name]
        # 2. Lazy store (from file)
        elif hasattr(self, "_store") and self._store is not None and name in self._store:
            mat = self._store[name]
        # 3. Legacy caches
        elif name in ("weight", "weights") and hasattr(self, "_cached_weights") and self._cached_weights is not None:
            mat = self._cached_weights
        elif name in ("length", "lengths") and hasattr(self, "_cached_lengths") and self._cached_lengths is not None:
            mat = self._cached_lengths

        if mat is None:
            return None

        # Apply transforms targeting this matrix
        for t in self.transforms or []:
            if t.name == name:
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

        Examples
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

        Each keyword argument is a named matrix (e.g. ``weights=vals``)
        whose entries are being added at the given ``(source, target)``
        positions. Internally the data is kept in COO format for fast
        incremental building; call :meth:`matrix` with
        ``format="csr"`` when you need efficient row-slicing.

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

        Examples
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

            existing = arrays.get(name)

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

            arrays[name] = mat

            # Backward compat
            if name in ("weight", "weights"):
                self._cached_weights = mat
            elif name in ("length", "lengths"):
                self._cached_lengths = mat

            self._ensure_template_edge(name)

    def _get_edge(self, name: str):
        """Return the template Edge with the given label, or None."""
        for e in self.edges or []:
            lbl = getattr(e, "label", None) or getattr(e, "name", None)
            if lbl == name:
                return e
        return None

    def _apply_transform(self, M, func):
        """Apply a Function transform to matrix *M*.

        Supports equation-based (symbolic) or callable-based (software)
        transforms via the Function class.
        """
        # Callable-based transform
        c = getattr(func, "callable", None)
        if c is not None:
            import importlib
            import inspect

            mod = importlib.import_module(c.module)
            fn = getattr(mod, c.name)
            kwargs = {}
            for arg in (getattr(func, "arguments", None) or []):
                aname = getattr(arg, "name", None)
                if aname is not None:
                    kwargs[aname] = getattr(arg, "value", None)
            available = {"L": self.lengths_matrix, "network": self}
            sig = inspect.signature(fn)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            for key, val in available.items():
                if key in sig.parameters or accepts_var_kw:
                    kwargs.setdefault(key, val)
            return fn(M, **kwargs)

        # Equation-based transform
        eq = getattr(func, "equation", None)
        if eq is None:
            return M
        from tvbo.codegen.code import parse_eq, render_expression

        # Substitute scalar argument values: prefer Function.arguments, fall
        # back to Equation.parameters for legacy specs.
        arg_values: dict = {}
        for a in (getattr(func, "arguments", None) or []):
            aname = getattr(a, "name", None)
            if aname is not None:
                arg_values[aname] = getattr(a, "value", None)
        if hasattr(eq, "parameters") and eq.parameters:
            for pname, pval in eq.parameters.items():
                arg_values.setdefault(pname, getattr(pval, "value", pval))

        exp = parse_eq(eq)
        if exp is not None:
            subs_map = {s: arg_values[str(s)] for s in exp.free_symbols if str(s) in arg_values}
            if subs_map:
                exp = exp.subs(subs_map)

        # Generic primitives available to any equation transform.
        # *_safe variants substitute 1 for zero entries so isolated nodes
        # do not produce NaNs under row/column normalisation.
        L = self.lengths_matrix
        _rs = M.sum(axis=1, keepdims=True)
        _cs = M.sum(axis=0, keepdims=True)
        env = {
            "M": M,
            "W": M,
            "L": L,
            "M_min": jnp.nanmin(M),
            "W_min": jnp.nanmin(M),
            "M_max": jnp.nanmax(M),
            "W_max": jnp.nanmax(M),
            "W_rowsum": _rs,
            "W_colsum": _cs,
            "W_rowsum_safe": jnp.where(_rs > 0, _rs, 1.0),
            "W_colsum_safe": jnp.where(_cs > 0, _cs, 1.0),
            "jnp": jnp,
            "np": jnp,
            "jsp": jsp,
        }
        code_str = render_expression(exp, format="jax")
        if isinstance(code_str, str):
            M = eval(code_str, env)
        return M

    def add_transform(
        self,
        target: str,
        equation_rhs: str = "(M - M_min) / (M_max - M_min)",
    ) -> None:
        """Append a matrix transform for a named edge property.

        Transforms are applied in order when the matrix is accessed
        via ``matrix()`` or ``weights_matrix``.

        Parameters
        ----------
        target : str
            Edge property name (e.g. ``"weight"``, ``"length"``, ``"fc"``).
        equation_rhs : str, default="(M - M_min) / (M_max - M_min)"
            Right-hand side of the transform equation. Can reference
            M (matrix), M_min, M_max.

        Examples
        --------
        ```python
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.add_transform("weight", "M / M_max")
        ```
        """
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

        Template edges are the link between the ``edges`` list visible
        in the YAML sidecar and the HDF5 datasets under ``edges/<name>/``.
        """
        from scipy import sparse

        arrays = self._get_arrays()
        mat = arrays.get(name)

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


@register_pytree_node_class
class Connectome(Network):
    """Deprecated alias for Network. Use Network instead."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "Connectome is deprecated and will be removed in a future version. "
            "Use tvbo.data.tvbo_data.connectomes.Network instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
