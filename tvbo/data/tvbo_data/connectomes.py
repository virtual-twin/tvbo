import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout
import jax.numpy as jnp

from tvbo.data.tvbo_data import CONNECTOME_DIR, bids_utils
from tvbo.datamodel import tvbo_datamodel
from jsonasobj2 import as_dict

connectome_data = BIDSLayout(
    CONNECTOME_DIR,
    validate=False,
    is_derivative=True,
)

available_connectomes = bids_utils.get_unique_entity_values(connectome_data, "desc")


def get_normative_connectome_data(atlas, desc) -> tvbo_datamodel.Matrix:
    fweights = connectome_data.get(
        suffix="weights",
        extension="csv",
        atlas=atlas,
        desc=desc,
        return_type="file",
    )[0]
    flengths = connectome_data.get(
        suffix="lengths",
        extension="csv",
        atlas=atlas,
        desc=desc,
        return_type="file",
    )[0]
    weights = tvbo_datamodel.Matrix(dataLocation=fweights)
    lengths = tvbo_datamodel.Matrix(dataLocation=flengths)
    return weights, lengths


from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Connectome(tvbo_datamodel.Connectome):
    """Direct datamodel Connectome with properties for arrays and helpers.

    - Stores only schema fields; arrays and helpers are properties with light caching.
    - Back-compat: metadata property returns self; labels provided as property.
    """

    def __init__(self, **kwargs):
        """Initialize Connectome with priority order:
        1. weights/lengths from kwargs (if provided)
        2. Load from parcellation/atlas (if provided and weights/lengths not given)
        3. Create default connectome based on number_of_nodes (fallback)
        """
        # Sync number_of_regions and number_of_nodes early
        if "number_of_regions" in kwargs and "number_of_nodes" not in kwargs:
            kwargs["number_of_nodes"] = kwargs["number_of_regions"]
        elif "number_of_nodes" in kwargs and "number_of_regions" not in kwargs:
            kwargs["number_of_regions"] = kwargs["number_of_nodes"]

        # Determine number of nodes for default connectome
        n_nodes = kwargs.get("number_of_nodes") or kwargs.get("number_of_regions") or 1

        # Priority 1: Use weights/lengths from kwargs if provided (already there)
        has_weights = "weights" in kwargs
        has_lengths = "lengths" in kwargs

        # Priority 2: Load normative data if parcellation/atlas specified and no weights/lengths
        if not has_weights and not has_lengths:
            if "parcellation" in kwargs and kwargs["parcellation"].get("atlas"):
                atlas_name = kwargs["parcellation"]["atlas"].get("name")
                tractogram = kwargs.get("tractogram", "dTOR")
                w_in, l_in = get_normative_connectome_data(atlas_name, tractogram)
                kwargs["weights"] = w_in
                kwargs["lengths"] = l_in
                # Infer number of regions from loaded data
                if hasattr(w_in, "dataLocation") and w_in.dataLocation:
                    w_arr = pd.read_csv(w_in.dataLocation, header=None).values
                    n_nodes = w_arr.shape[0]
                    kwargs["number_of_regions"] = n_nodes
                    kwargs["number_of_nodes"] = n_nodes
                has_weights = True
                has_lengths = True

        # Priority 3: Create default matrices if still no weights/lengths
        if not has_weights:
            kwargs["weights"] = tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(n_nodes)]
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(n_nodes)]
                ),
                values=[0.0] * (n_nodes * n_nodes),
            )

        if not has_lengths:
            kwargs["lengths"] = tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(n_nodes)]
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(n_nodes)]
                ),
                values=[1.0] * (n_nodes * n_nodes),
            )

        # Ensure number_of_regions/nodes are set
        if "number_of_regions" not in kwargs:
            kwargs["number_of_regions"] = n_nodes
        if "number_of_nodes" not in kwargs:
            kwargs["number_of_nodes"] = n_nodes

        # Convert numpy arrays to Matrix objects if needed
        if isinstance(kwargs.get("weights"), np.ndarray):
            w_in = kwargs["weights"]
            kwargs["weights"] = tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(w_in.shape[0])]
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(w_in.shape[1])]
                ),
                values=w_in.reshape(-1).astype(float).tolist(),
            )

        if isinstance(kwargs.get("lengths"), np.ndarray):
            l_in = kwargs["lengths"]
            kwargs["lengths"] = tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(l_in.shape[0])]
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=[str(i) for i in range(l_in.shape[1])]
                ),
                values=l_in.reshape(-1).astype(float).tolist(),
            )

        super().__init__(**kwargs)

        if not self.conduction_speed:
            self.conduction_speed = tvbo_datamodel.Parameter(
                name="conduction_speed", label="v", value=3.0, unit="mm/ms"
            )

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.Connectome) -> "Connectome":
        """Create a Connectome instance from a tvbo_datamodel.Connectome object.

        Args:
            datamodel: A tvbo_datamodel.Connectome instance

        Returns:
            A new Connectome instance with all fields copied from the datamodel
        """
        data = as_dict(datamodel)
        return cls(**data)

    # Keep nodes and regions synchronized on assignment
    def __setattr__(self, name, value):
        super_setattr = super().__setattr__

        # Normalize assignments to weights/lengths into Matrix objects
        if name in ("weights", "lengths"):
            try:
                # list -> ndarray
                if isinstance(value, list):
                    value = np.array(value, dtype=float)
                # ndarray -> Matrix with labeled axes
                if isinstance(value, np.ndarray):
                    mat = self._matrix_from_array(value)
                    super_setattr(name, mat)
                    # Keep region/node counts in sync
                    try:
                        n = int(value.shape[0])
                        if getattr(self, "number_of_regions", None) != n:
                            super_setattr("number_of_regions", n)
                    except Exception:
                        pass
                    return
                # path-like str -> Matrix by dataLocation
                if isinstance(value, str):
                    super_setattr(name, tvbo_datamodel.Matrix(dataLocation=value))
                    return
            except Exception:
                # Fallback to raw assignment on any conversion issue
                pass

        super_setattr(name, value)
        if name == "number_of_regions":
            try:
                nodes = getattr(self, "number_of_nodes", None)
                new_val = int(value) if value is not None else None
                if nodes != new_val:
                    super_setattr("number_of_nodes", new_val)
            except Exception:
                # Don't block attribute setting on sync errors
                pass

    def to_yaml(self, filepath: str | None = None):
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    # ---- JAX pytree: flatten/unflatten ----
    def tree_flatten(self):
        """Return children and auxiliary data for JAX pytree support.

        Children: (weights, lengths) so JAX can map/transform numerical payloads.
        Aux data: metadata dict WITHOUT the array data to avoid duplication.
        """
        # Convert metadata to a JSON string for stable equality in JAX
        import json as _json
        import numpy as _np

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
            # last resort: stringify
            return str(o)

        # children are the heavy numeric arrays; keep arrays out of aux
        # Always return arrays to maintain consistent tree structure
        # If weights/lengths are None, use empty arrays with proper shape based on number_of_regions

        # Check if we have cached PyTree data (from a previous unflatten)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            weights_arr, lengths_arr = self._pytree_data
        else:
            # First flatten or normal object - compute arrays from metadata
            if self.weights is not None:
                weights_arr = self.weights_matrix
            elif hasattr(self, "number_of_regions") and self.number_of_regions:
                weights_arr = jnp.zeros(
                    (self.number_of_regions, self.number_of_regions)
                )
            else:
                weights_arr = jnp.zeros((1, 1))

            if self.lengths is not None:
                lengths_arr = self.lengths_matrix
            elif hasattr(self, "number_of_regions") and self.number_of_regions:
                lengths_arr = jnp.zeros(
                    (self.number_of_regions, self.number_of_regions)
                )
            else:
                lengths_arr = jnp.zeros((1, 1))

        children = (weights_arr, lengths_arr)

        # Get full metadata but exclude weights/lengths to avoid embedding arrays
        meta_dict = as_dict(self)
        # Remove weights, lengths, parcellation, and cache attributes from metadata
        # Parcellation is excluded to prevent reloading data during unflatten
        meta_dict_without_arrays = {
            k: v
            for k, v in meta_dict.items()
            if k not in ("weights", "lengths", "parcellation", "_pytree_data")
        }
        meta_json = _json.dumps(
            meta_dict_without_arrays, sort_keys=True, default=_jsonable
        )
        aux = (meta_json,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
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
    def metadata(self):
        return self

    # ---- Numeric accessors (compute on demand; no extra attributes) ----
    def _matrix_from_array(self, arr: np.ndarray) -> tvbo_datamodel.Matrix:
        arr = jnp.array(arr)
        N0, N1 = arr.shape
        x = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N0)])
        y = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N1)])
        return tvbo_datamodel.Matrix(
            x=x, y=y, values=arr.reshape(-1).astype(jnp.float32).tolist()
        )

    @property
    def weights_matrix(self, format="jax"):
        """Return weights matrix as ndarray.

        If a normalization Equation is defined in metadata (self.normalization)
        and provides a callable in `pycode` (e.g., "lambda W, eps=1e-12: ..."),
        the returned matrix is the result of applying that function to the raw
        weights using any provided `parameters` as keyword arguments.
        """
        # Check if we have cached PyTree data from tree_unflatten (during JAX transformations)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            return self._pytree_data[0]

        wm = self.weights
        if wm is not None:
            if isinstance(wm, list):
                W = np.array(wm, dtype=float)
            elif getattr(wm, "values", None):
                x = getattr(wm, "x", None)
                y = getattr(wm, "y", None)
                nx_ = (
                    len(x.values)
                    if x and getattr(x, "values", None)
                    else self.number_of_regions
                )
                ny_ = (
                    len(y.values)
                    if y and getattr(y, "values", None)
                    else self.number_of_regions
                )
                W = np.array(wm.values, dtype=float).reshape(nx_, ny_)
            elif getattr(wm, "dataLocation", None):
                W = pd.read_csv(wm.dataLocation, header=None).values.astype(float)
            else:
                W = None
        else:
            W = None

        if W is None:
            if getattr(self, "number_of_regions", None):
                N = self.number_of_regions
                W = np.ones((N, N))
                np.fill_diagonal(W, 0)
            else:
                return None

        # Apply normalization from metadata if available and executable
        norm = getattr(self, "normalization", None)
        if norm is not None:
            from tvbo.export.code import parse_eq, render_expression
            import jax.numpy as jnp
            import jax.scipy as jsp

            exp = parse_eq(norm)
            # Substitute known parameter values
            subs_map = {}
            for s in exp.free_symbols:
                name = str(s)
                if name in norm.parameters:
                    value = norm.parameters[name].value
                    subs_map[s] = value
            if subs_map:
                exp = exp.subs(subs_map)
            env = {
                "W": W,
                "W_min": jnp.nanmin(W),
                "W_max": jnp.nanmax(W),
                "jnp": jnp,
                "np": jnp,
                "jsp": jsp,
            }
            code_str = render_expression(exp, format=format)
            W = eval(code_str, env)
        return W

    @property
    def lengths_matrix(self):
        # Check if we have cached PyTree data from tree_unflatten (during JAX transformations)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            return self._pytree_data[1]

        lm = self.lengths
        if lm is not None:
            if getattr(lm, "values", None):
                x = getattr(lm, "x", None)
                y = getattr(lm, "y", None)
                nx_ = (
                    len(x.values)
                    if x and getattr(x, "values", None)
                    else self.number_of_regions
                )
                ny_ = (
                    len(y.values)
                    if y and getattr(y, "values", None)
                    else self.number_of_regions
                )
                return np.array(lm.values, dtype=float).reshape(nx_, ny_)
            if getattr(lm, "dataLocation", None):
                return pd.read_csv(lm.dataLocation, header=None).values.astype(float)
        if getattr(self, "number_of_regions", None):
            N = self.number_of_regions
            L = np.ones((N, N))
            np.fill_diagonal(L, 0)
            return L
        return None

    @property
    def labels(self):
        atlas = self.get_atlas()
        if atlas.metadata.terminology:
            return {
                e.name: e.lookupLabel
                for k, e in atlas.metadata.terminology.entities.items()
            }
        return {}

    def __str__(self):
        return (
            f"Connectome-{self.parcellation.atlas.name}({self.number_of_regions})"
            if self.parcellation and self.parcellation.atlas
            else f"Connectome(N={self.number_of_regions})"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def atlas(self):
        return self.get_atlas()

    def get_atlas(self):
        from tvbo.data.tvbo_data.atlases import Atlas

        return Atlas(self.parcellation.atlas if self.parcellation else None)

    def compute_delays(self, conduction_speed="default"):
        if conduction_speed == "default":
            conduction_speed = self.conduction_speed.value
        return self.lengths_matrix / conduction_speed

    def execute(self, format="tvb"):
        if format == "tvb":
            from tvb import datatypes

            # Ensure TVB receives plain NumPy arrays (no JAX tracers)
            _weights = np.asarray(self.weights_matrix, dtype=float)
            _lengths = np.asarray(self.lengths_matrix, dtype=float)
            _centres = np.asarray(list(self.get_centers().values()), dtype=float)
            _speed = np.asarray([self.conduction_speed.value], dtype=float)
            tvb_conn = datatypes.connectivity.Connectivity(
                weights=_weights,
                tract_lengths=_lengths,
                centres=_centres,
                region_labels=self.atlas.region_labels,
                speed=_speed,
            )
            tvb_conn.configure()
            return tvb_conn

    def normalize_weights(self, equation_rhs="(W - W_min) / (W_max - W_min)"):
        from tvbo.datamodel.tvbo_datamodel import Equation

        self.normalization = Equation(rhs=equation_rhs)

    def plot_weights(self, ax, cmap="magma", log=False):
        from matplotlib.colors import LogNorm
        import numpy as np

        weights = self.weights_matrix
        if log:
            # Use LogNorm with vmin set to smallest non-zero value to avoid white holes
            nonzero_weights = weights[weights > 0]
            vmin = nonzero_weights.min() if nonzero_weights.size > 0 else 1e-10
            vmax = weights.max() if weights.max() > 0 else 1.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        im = ax.imshow(weights, cmap=cmap, interpolation="none", norm=norm)
        ax.set_title("weights")
        ax.set_box_aspect(1)
        return im

    def plot_lengths(self, ax, cmap="magma"):
        im = ax.imshow(self.lengths_matrix, cmap=cmap, interpolation="none")
        ax.set_title("lengths")
        ax.set_box_aspect(1)
        return im

    def plot_matrix(self, log_weights=False, cmap="magma"):
        fig, axs = plt.subplots(ncols=2, sharey=True)

        w = self.plot_weights(axs[0], cmap=cmap, log=log_weights)
        fig.colorbar(w, ax=axs[0], shrink=0.5)

        l = self.plot_lengths(axs[1], cmap=cmap)
        fig.colorbar(l, ax=axs[1], shrink=0.5)

        plt.close()
        return fig

    def calculate_delays(self, conduction_speed=None):
        if conduction_speed is None:
            conduction_speed = self.conduction_speed.value
        return self.lengths_matrix / conduction_speed

    def create_graph(self, weight_threshold=0):
        W = self.weights_matrix
        D = self.calculate_delays()
        # Use MultiDiGraph to allow asymmetric and multiple parallel edges
        G = nx.MultiDiGraph()
        N_regions = self.number_of_regions
        for i in range(N_regions):
            G.add_node(i)
        for i in range(N_regions):
            for j in range(N_regions):
                if W[i, j] > weight_threshold:
                    G.add_edge(i, j, weight=W[i, j], delay=D[i, j])
        return G

    def get_centers(self):
        labels = []
        ids = []
        centers = []
        for region, entity in self.get_atlas().metadata.terminology.entities.items():
            labels.append(region)
            ids.append(entity.lookupLabel)
            center = entity.center
            coord = (center.x, center.y, center.z)
            centers.append(coord)

        centers = np.array(centers)
        centers = centers[np.argsort(ids)]
        labels = np.array(labels)[np.argsort(ids)]
        center_mapping = {i - 1: center for i, center in zip(ids, centers)}
        if center_mapping == {}:
            return {0: (0, 0, 0)}
        return center_mapping

    def plot_graph(
        self,
        ax=None,
        node_cmap="viridis",
        edge_cmap="viridis",
        node_colors="in-strength",
        node_size="in-strength",
        threshold_percentile=0,
        pos_scaling=1,
        node_labels=True,
        edge_labels=True,
        log_in_strength=True,
        node_size_scaling=100,
        edge_color="weight",
        pos="spring",
        plot_brain=None,
        edge_kwargs={},
        node_kwargs={},
        fontsize=8,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            return_fig = True
        else:
            return_fig = False

        if isinstance(node_cmap, str):
            node_cmap = plt.get_cmap(node_cmap)
        if isinstance(edge_cmap, str):
            edge_cmap = plt.get_cmap(edge_cmap)

        # Build graph on demand from current weights and delays
        W = self.weights_matrix
        G = self.create_graph(weight_threshold=np.percentile(W, threshold_percentile))

        # Generate positions for nodes
        if pos == "spring":
            pos = nx.spring_layout(
                G,
                k=pos_scaling * (1 / np.sqrt(len(G.nodes))),
                seed=1312,
            )
            ax.set_box_aspect(1)

        if plot_brain:
            view = plot_brain

            if view == "horizontal":
                pos = {
                    i: [center[0], center[1]]
                    for i, center in self.get_centers().items()
                }
            elif view == "sagittal":
                pos = {
                    i: [center[1], center[2]]
                    for i, center in self.get_centers().items()
                }
            elif view == "coronal":
                pos = {
                    i: [center[0], center[2]]
                    for i, center in self.get_centers().items()
                }

            ax.set_aspect("equal")

        # Helper for safe [0,1] normalization that handles empty/constant arrays
        def _safe_norm(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return arr
            vmin = float(np.min(arr))
            vmax = float(np.max(arr))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return np.zeros_like(arr)
            return (arr - vmin) / (vmax - vmin)

        # Materialize edge list once (include keys to distinguish parallel edges)
        edges_list = list(G.edges(keys=True, data=True))  # (u, v, k, data)
        edge_attr_vals = (
            np.array(
                [data.get(edge_color, 0.0) for _, _, _, data in edges_list], dtype=float
            )
            if edges_list
            else np.array([])
        )

        norm_edge_attr = _safe_norm(edge_attr_vals)

        # Choose edge colors: if all equal -> black, else colormap
        if norm_edge_attr.size == 0:
            edge_colors = []
        elif np.all(norm_edge_attr == 0):
            edge_colors = ["black"] * len(edges_list)
        else:
            edge_colors = edge_cmap(norm_edge_attr)

        # Node strengths (incoming)
        node_in_strength = np.sum(W, axis=1).astype(float)
        if log_in_strength:
            node_in_strength = np.log1p(node_in_strength)
        norm_node_in_strength = _safe_norm(node_in_strength)

        if node_size == "in-strength":
            node_sizes = 100 + norm_node_in_strength * node_size_scaling
        else:
            node_sizes = 100 * node_size_scaling

        if node_colors == "in-strength":
            node_coloring = norm_node_in_strength
        elif node_colors == "node":
            nodes = np.array(list(G.nodes), dtype=float)
            node_coloring = _safe_norm(nodes)
        else:
            # constant color fallback
            node_coloring = np.zeros(len(G.nodes)) if len(G.nodes) > 0 else np.array([])

        node_colors = node_cmap(node_coloring)
        # Use explicit edgelist to keep color order aligned with edges_list
        edgelist_draw = [(u, v, k) for (u, v, k, _) in edges_list]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist_draw if edges_list else None,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            ax=ax,
            **edge_kwargs,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,  # Node size
            node_color="white",  # No fill
            edgecolors=node_colors,  # Outline color
            linewidths=1,  # Outline width
            ax=ax,
            **node_kwargs,
        )
        if node_labels:
            nx.draw_networkx_labels(
                G,
                pos,
                labels={i: f"{i}" for i in G.nodes()},
                ax=ax,
                font_size=fontsize,
            )
        if edge_labels:
            if edges_list:
                edge_labels_dict = {}
                for u, v, k, d in edges_list:
                    val = d.get(edge_color, None)
                    try:
                        edge_labels_dict[(u, v, k)] = f"{float(val):.2f}"
                    except (TypeError, ValueError):
                        edge_labels_dict[(u, v, k)] = str(val)
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    edge_labels=edge_labels_dict,
                    ax=ax,
                    font_size=fontsize,
                )
        if return_fig:
            plt.close()
            return fig

        # Build a ScalarMappable for colorbar; guard constant/empty cases
        data = edge_attr_vals if edge_attr_vals.size > 0 else np.array([0.0])
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap=edge_cmap)
        return mappable

    def plot_overview(
        self, weights_kwargs={}, lengths_kwargs={}, graph_kwargs={}, log_weights=False
    ):

        fig, axs = plt.subplots(ncols=3, layout="tight", figsize=(15, 5))
        if "edge_cmap" not in graph_kwargs:
            graph_kwargs["edge_cmap"] = "magma"

        g = self.plot_graph(axs[0], **graph_kwargs)
        axs[0].axis("off")
        w = self.plot_weights(axs[1], log=log_weights, **weights_kwargs)
        l = self.plot_lengths(axs[2], **lengths_kwargs)
        axs[2].sharey(axs[1])

        c1 = fig.colorbar(g, ax=axs[0], shrink=0.5, pad=-0.05)
        c2 = fig.colorbar(w, ax=axs[1], shrink=0.5)
        c3 = fig.colorbar(l, ax=axs[2], shrink=0.5)

        fontsize_scaler = 1.5

        for c in [c1, c2, c3]:
            c.outline.set_visible(False)
            for label in c.ax.get_yticklabels():
                label.set_fontsize(c.ax.yaxis.label.get_fontsize() * fontsize_scaler)

        for ax in axs:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(label.get_fontsize() * fontsize_scaler)
            ax.title.set_fontsize(ax.title.get_fontsize() * fontsize_scaler)
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * fontsize_scaler)
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * fontsize_scaler)

        c1.set_label("ms", fontsize=c1.ax.yaxis.label.get_fontsize() * fontsize_scaler)
        c2.set_label(
            "log1p(weight)" if log_weights else "weight",
            fontsize=c2.ax.yaxis.label.get_fontsize() * fontsize_scaler,
        )
        c3.set_label("mm", fontsize=c3.ax.yaxis.label.get_fontsize() * fontsize_scaler)

        plt.close()
        return fig

    def normalize(self):
        self.normalization = tvbo_datamodel.Equation(
            rhs="(W - W_min) / (W_max - W_min)"
        )
