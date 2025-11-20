from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout
from jax import Array as JaxArray
from jax.tree_util import register_pytree_node_class
from jsonasobj2 import as_dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tvbo.data.tvbo_data import CONNECTOME_DIR, bids_utils
from tvbo.datamodel import tvbo_datamodel

connectome_data = BIDSLayout(
    CONNECTOME_DIR,
    validate=False,
    is_derivative=True,
)

available_connectomes = bids_utils.get_unique_entity_values(connectome_data, "desc")


def get_normative_connectome_data(
    atlas: str, desc: str
) -> Tuple[tvbo_datamodel.Matrix, tvbo_datamodel.Matrix]:
    """Load normative connectivity matrices from BIDS dataset.

    Parameters
    ----------
    atlas : str
        Name of the brain parcellation atlas (e.g., "DesikanKilliany", "Destrieux")
    desc : str
        Description/type of the connectome data (e.g., "dTOR", "dMRT")

    Returns
    -------
    weights : tvbo_datamodel.Matrix
        Connection strength matrix
    lengths : tvbo_datamodel.Matrix
        Tract length matrix

    Examples
    --------
    ```python
    weights, lengths = get_normative_connectome_data("DesikanKilliany", "dTOR")
    ```
    """
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


@register_pytree_node_class
class Connectome(tvbo_datamodel.Connectome):
    """Structural connectivity data with weights, lengths, and visualization tools.

    Represents brain structural connectivity including connection weights, tract lengths,
    and spatial information. Supports loading normative connectomes from atlases or
    custom data, with JAX pytree compatibility.

    Examples
    --------
    ```{python}
    from tvbo import Connectome
    # Load atlas-based connectome
    sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
    sc.plot_matrix()

    # Custom connectome from arrays
    import numpy as np
    sc = Connectome(weights=np.random.rand(10, 10), number_of_regions=10)
    delays = sc.calculate_delays(conduction_speed=3.0)
    ```

    See Also
    --------
    weights_matrix : Access connection weights as array
    lengths_matrix : Access tract lengths as array
    plot_graph : Visualize as network graph
    plot_overview : Complete visualization with matrices and graph
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a Connectome instance.

        Parameters
        ----------
        **kwargs : Any
            Connectome initialization arguments. Common parameters:

            - parcellation : dict
                Parcellation info with atlas name, e.g., {"atlas": {"name": "DesikanKilliany"}}
            - weights : np.ndarray or Matrix
                Connection weights matrix (N x N)
            - lengths : np.ndarray or Matrix
                Tract lengths matrix (N x N)
            - number_of_regions : int
                Number of brain regions
            - conduction_speed : Parameter or float
                Signal propagation speed (default: 3.0 mm/ms)
            - tractogram : str
                Tractography method (default: "dTOR")

        Notes
        -----
        Initialization follows priority order:

        1. Use provided weights/lengths arrays
        2. Load from atlas if parcellation specified
        3. Create default matrices as fallback

        Examples
        --------
        ```{python}
        # From atlas
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

        # From arrays
        import numpy as np
        sc = Connectome(weights=np.random.rand(68, 68), number_of_regions=68)
        ```
        """
        # Sync number_of_regions and number_of_nodes early
        if "number_of_regions" in kwargs and "number_of_nodes" not in kwargs:
            kwargs["number_of_nodes"] = kwargs["number_of_regions"]
        elif "number_of_nodes" in kwargs and "number_of_regions" not in kwargs:
            kwargs["number_of_regions"] = kwargs["number_of_nodes"]

        # Priority 1: Use weights/lengths from kwargs if provided (already there)
        has_weights = "weights" in kwargs
        has_lengths = "lengths" in kwargs

        # Infer n_nodes from numpy arrays if provided
        n_nodes = kwargs.get("number_of_nodes") or kwargs.get("number_of_regions")
        if n_nodes is None:
            if has_weights and isinstance(kwargs["weights"], np.ndarray):
                n_nodes = kwargs["weights"].shape[0]
            elif has_lengths and isinstance(kwargs["lengths"], np.ndarray):
                n_nodes = kwargs["lengths"].shape[0]
            else:
                n_nodes = 1

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
        """Create a Connectome from a datamodel instance.

        Parameters
        ----------
        datamodel : tvbo_datamodel.Connectome
            Source datamodel Connectome instance

        Returns
        -------
        Connectome
            New Connectome with fields copied from datamodel

        Examples
        --------
        ```{python}
        from tvbo.datamodel import tvbo_datamodel
        dm = tvbo_datamodel.Connectome(number_of_nodes=10)
        sc = Connectome.from_datamodel(dm)
        ```
        """
        data = as_dict(datamodel)
        # as_dict returns a dict-like object that works with **kwargs
        return cls(**data)  # type: ignore[arg-type]

    # Keep nodes and regions synchronized on assignment
    def __setattr__(self, name: str, value: Any) -> None:
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
                # Only convert if value is int-like
                if isinstance(value, (int, np.integer)):
                    new_val: Optional[int] = int(value)
                elif value is None:
                    new_val = None
                else:
                    return  # Skip sync for non-numeric values
                if nodes != new_val:
                    super_setattr("number_of_nodes", new_val)
            except Exception:
                # Don't block attribute setting on sync errors
                pass

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Serialize Connectome to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to save YAML file. If None, returns YAML string.

        Returns
        -------
        str
            YAML representation of the Connectome

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        yaml_str = sc.to_yaml()
        sc.to_yaml("connectome.yaml")  # Save to file
        ```
        """
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    # ---- JAX pytree: flatten/unflatten ----
    def tree_flatten(self) -> Tuple[Tuple[JaxArray, JaxArray], Tuple[str]]:
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
        # as_dict can return various dict-like structures
        if not isinstance(meta_dict, dict):
            meta_dict = dict(meta_dict) if hasattr(meta_dict, "__iter__") else {}
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
        return children, aux  # type: ignore[return-value]

    @classmethod
    def tree_unflatten(
        cls, aux_data: Tuple[str], children: Tuple[JaxArray, JaxArray]
    ) -> "Connectome":
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
    def _matrix_from_array(
        self, arr: Union[np.ndarray, JaxArray]
    ) -> tvbo_datamodel.Matrix:
        arr = jnp.array(arr)
        N0, N1 = arr.shape
        x = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N0)])
        y = tvbo_datamodel.BrainRegionSeries(values=[str(i) for i in range(N1)])
        return tvbo_datamodel.Matrix(
            x=x, y=y, values=arr.reshape(-1).astype(jnp.float32).tolist()
        )

    @property
    def weights_matrix(self) -> Optional[Union[np.ndarray, JaxArray]]:
        """Connection weights matrix as numpy/JAX array.

        Returns the (N x N) matrix of connection strengths between regions.
        If normalization is defined, applies the normalization equation.

        Returns
        -------
        np.ndarray or jax.Array, optional
            Connection weights matrix (N x N), or None if unavailable

        Notes
        -----
        If `self.normalization` contains an equation, the weights are
        transformed according to that equation before being returned.

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        W = sc.weights_matrix
        print(f"Shape: {W.shape}, Mean: {W.mean():.3f}")
        ```
        """
        format = "jax"
        # Check if we have cached PyTree data from tree_unflatten (during JAX transformations)
        if hasattr(self, "_pytree_data") and self._pytree_data is not None:
            return self._pytree_data[0]

        wm = self.weights
        if wm is not None:
            if isinstance(wm, list):
                W = np.array(wm, dtype=float)
            if getattr(wm, "values", None):
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
                # Type guard: ensure dimensions are valid integers
                if (
                    nx_ is not None
                    and ny_ is not None
                    and isinstance(nx_, int)
                    and isinstance(ny_, int)
                ):
                    W = np.array(wm.values, dtype=float).reshape(nx_, ny_)
                else:
                    W = None
            elif getattr(wm, "dataLocation", None):
                W = pd.read_csv(wm.dataLocation, header=None).values.astype(float)  # type: ignore[arg-type,attr-defined]
            else:
                W = None
        else:
            W = None

        if W is None:
            N = getattr(self, "number_of_regions", None)
            if N is not None and isinstance(N, int) and N > 0:
                W = np.ones((N, N))
                np.fill_diagonal(W, 0)
            else:
                return None

        # Apply normalization from metadata if available and executable
        norm = getattr(self, "normalization", None)
        if norm is not None:
            import jax.numpy as jnp
            import jax.scipy as jsp

            from tvbo.export.code import parse_eq, render_expression

            exp = parse_eq(norm)
            # Substitute known parameter values
            if exp is not None:
                subs_map = {}
                for s in exp.free_symbols:  # type: ignore[attr-defined]
                    name = str(s)
                    if hasattr(norm, "parameters") and name in norm.parameters:  # type: ignore[attr-defined]
                        value = norm.parameters[name].value  # type: ignore[attr-defined,index]
                        subs_map[s] = value
                if subs_map:
                    exp = exp.subs(subs_map)  # type: ignore[attr-defined]
            env = {
                "W": W,
                "W_min": jnp.nanmin(W),
                "W_max": jnp.nanmax(W),
                "jnp": jnp,
                "np": jnp,
                "jsp": jsp,
            }
            code_str = render_expression(exp, format=format)
            if isinstance(code_str, str):
                W = eval(code_str, env)
        return W

    @property
    def lengths_matrix(self) -> Optional[Union[np.ndarray, JaxArray]]:
        """Tract length matrix as numpy/JAX array.

        Returns the (N x N) matrix of physical distances (tract lengths)
        between brain regions in millimeters.

        Returns
        -------
        np.ndarray or jax.Array, optional
            Tract lengths matrix (N x N) in mm, or None if unavailable

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        L = sc.lengths_matrix
        print(f"Mean length: {L.mean():.1f} mm")
        ```
        """
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
                # Type guard: ensure dimensions are valid integers
                if (
                    nx_ is not None
                    and ny_ is not None
                    and isinstance(nx_, int)
                    and isinstance(ny_, int)
                ):
                    return np.array(lm.values, dtype=float).reshape(nx_, ny_)
            if getattr(lm, "dataLocation", None):
                return pd.read_csv(lm.dataLocation, header=None).values.astype(float)  # type: ignore[arg-type,attr-defined]
        N = getattr(self, "number_of_regions", None)
        if N is not None and isinstance(N, int) and N > 0:
            L = np.ones((N, N))
            np.fill_diagonal(L, 0)
            return L
        return None

    @property
    def labels(self) -> Dict[str, str]:
        """Brain region labels from atlas.

        Returns
        -------
        dict of str to str
            Mapping from region names to lookup labels

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        labels = sc.labels
        print(f"Number of labeled regions: {len(labels)}")
        ```
        """
        atlas = self.get_atlas()
        if atlas.metadata.terminology:
            return {
                e.name: e.lookupLabel
                for k, e in atlas.metadata.terminology.entities.items()
            }
        return {}

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
        ```{python}
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
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        atlas = sc.get_atlas()
        ```
        """
        from tvbo.data.tvbo_data.atlases import Atlas

        parc = getattr(self, "parcellation", None)
        atlas_data = parc.atlas if parc and hasattr(parc, "atlas") else None  # type: ignore[attr-defined]
        return Atlas(atlas_data)

    def compute_delays(
        self, conduction_speed: Union[str, float] = "default"
    ) -> Union[np.ndarray, JaxArray]:
        """Calculate signal propagation delays between regions.

        Parameters
        ----------
        conduction_speed : str or float, default="default"
            Signal propagation speed in mm/ms. If "default", uses
            `self.conduction_speed.value` (typically 3.0 mm/ms)

        Returns
        -------
        np.ndarray or jax.Array
            Delay matrix (N x N) in milliseconds

        Raises
        ------
        ValueError
            If lengths matrix is not available

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        delays = sc.compute_delays(conduction_speed=4.0)
        print(f"Mean delay: {delays.mean():.2f} ms")
        ```
        """
        if conduction_speed == "default":
            cs_param = getattr(self, "conduction_speed", None)
            if cs_param and hasattr(cs_param, "value"):
                conduction_speed = cs_param.value  # type: ignore[attr-defined]
            else:
                conduction_speed = 3.0  # default fallback
        lengths = self.lengths_matrix
        if lengths is None:
            raise ValueError("Lengths matrix is not available")
        return lengths / conduction_speed  # type: ignore[operator]

    def execute(self, format: str = "tvb") -> Any:
        """Convert connectome to simulator-specific format.

        Parameters
        ----------
        format : str, default="tvb"
            Target format. Currently supports "tvb" (The Virtual Brain)

        Returns
        -------
        Any
            Connectivity object in the specified format

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        tvb_conn = sc.execute(format="tvb")
        # Use with TVB simulator
        ```
        """
        if format == "tvb":
            from tvb.datatypes.connectivity import Connectivity  # type: ignore[import-not-found]

            # Ensure TVB receives plain NumPy arrays (no JAX tracers)
            _weights = np.asarray(self.weights_matrix, dtype=float)
            _lengths = np.asarray(self.lengths_matrix, dtype=float)
            _centres = np.asarray(list(self.get_centers().values()), dtype=float)
            cs_param = getattr(self, "conduction_speed", None)
            cs_value = cs_param.value if cs_param and hasattr(cs_param, "value") else 3.0  # type: ignore[attr-defined]
            _speed = np.asarray([cs_value], dtype=float)
            tvb_conn = Connectivity(  # type: ignore[attr-defined]
                weights=_weights,
                tract_lengths=_lengths,
                centres=_centres,
                region_labels=self.atlas.region_labels,
                speed=_speed,
            )
            tvb_conn.configure()
            return tvb_conn

    def normalize_weights(
        self, equation_rhs: str = "(W - W_min) / (W_max - W_min)"
    ) -> None:
        """Set normalization equation for connection weights.

        Parameters
        ----------
        equation_rhs : str, default="(W - W_min) / (W_max - W_min)"
            Right-hand side of normalization equation. Can reference W, W_min, W_max

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize_weights("W / W_max")  # Normalize to [0, 1]
        normalized = sc.weights_matrix  # Returns normalized weights
        ```

        Notes
        -----
        The normalization is applied when accessing `weights_matrix` property.
        """
        from tvbo.datamodel.tvbo_datamodel import Equation

        self.normalization = Equation(rhs=equation_rhs)

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
        ```{python}
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
        ax.set_title("weights")
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
        ```{python}
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
        ax.set_title("lengths")
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
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.plot_matrix(log_weights=True)
        ```
        """
        fig, axs = plt.subplots(ncols=2, sharey=True)

        w = self.plot_weights(axs[0], cmap=cmap, log=log_weights)
        fig.colorbar(w, ax=axs[0], shrink=0.5)

        l = self.plot_lengths(axs[1], cmap=cmap)
        fig.colorbar(l, ax=axs[1], shrink=0.5)

        plt.close()
        return fig

    def calculate_delays(
        self, conduction_speed: Optional[float] = None
    ) -> Union[np.ndarray, JaxArray]:
        """Calculate signal propagation delays between regions.

        Parameters
        ----------
        conduction_speed : float, optional
            Conduction speed in mm/ms. If None, uses `self.conduction_speed.value`

        Returns
        -------
        np.ndarray or jax.Array
            Delay matrix (N x N) in milliseconds

        Raises
        ------
        ValueError
            If lengths matrix is not available

        Examples
        --------
        ```{python}
        import matplotlib.pyplot as plt
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        delays = sc.calculate_delays(conduction_speed=3.0)
        plt.imshow(delays, cmap='viridis')
        plt.colorbar(label='Delay (ms)')
        ```

        See Also
        --------
        compute_delays : Alternative method with string "default" option
        """
        if conduction_speed is None:
            cs_param = getattr(self, "conduction_speed", None)
            if cs_param and hasattr(cs_param, "value"):
                conduction_speed = cs_param.value  # type: ignore[attr-defined]
            else:
                conduction_speed = 3.0  # default fallback
        lengths = self.lengths_matrix
        if lengths is None:
            raise ValueError("Lengths matrix is not available")
        return lengths / conduction_speed  # type: ignore[operator]

    def create_graph(self, weight_threshold: float = 0) -> nx.MultiDiGraph:
        """Create NetworkX graph from connectivity matrices.

        Parameters
        ----------
        weight_threshold : float, default=0
            Minimum weight for including an edge in the graph

        Returns
        -------
        networkx.MultiDiGraph
            Directed multigraph with 'weight' and 'delay' edge attributes

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        G = sc.create_graph(weight_threshold=0.1)
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        ```
        """
        W = self.weights_matrix
        D = self.calculate_delays()
        # Use MultiDiGraph to allow asymmetric and multiple parallel edges
        G = nx.MultiDiGraph()
        N_regions = self.number_of_regions
        if N_regions is None or W is None:
            return G
        for i in range(N_regions):
            G.add_node(i)
        for i in range(N_regions):
            for j in range(N_regions):
                if W[i, j] > weight_threshold:
                    G.add_edge(i, j, weight=W[i, j], delay=D[i, j])
        return G

    def get_centers(self) -> Dict[int, Tuple[float, float, float]]:
        """Get 3D spatial coordinates of brain region centers.

        Returns
        -------
        dict of int to tuple of float
            Mapping from region index to (x, y, z) coordinates in mm

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        centers = sc.get_centers()
        for idx, (x, y, z) in centers.items():
            print(f"Region {idx}: ({x:.1f}, {y:.1f}, {z:.1f})")
        ```
        """
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
        ax: Optional[Axes] = None,
        node_cmap: Union[str, Any] = "viridis",
        edge_cmap: Union[str, Any] = "viridis",
        node_colors: str = "in-strength",
        node_size: Union[str, float] = "in-strength",
        threshold_percentile: float = 0,
        pos_scaling: float = 1,
        node_labels: bool = True,
        edge_labels: bool = True,
        log_in_strength: bool = True,
        node_size_scaling: float = 100,
        edge_color: str = "weight",
        pos: Union[str, Dict[int, List[float]]] = "spring",
        plot_brain: Optional[str] = None,
        edge_kwargs: Optional[Dict[str, Any]] = None,
        node_kwargs: Optional[Dict[str, Any]] = None,
        fontsize: float = 8,
    ) -> Union[Figure, cm.ScalarMappable]:
        """Visualize connectome as network graph.

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

        Returns
        -------
        Figure or ScalarMappable
            Figure if ax is None, otherwise ScalarMappable for colorbar

        Examples
        --------
        ```{python}
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
        """

        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}

        fig: Optional[Figure] = None
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
        if W is None:
            W = np.zeros((1, 1))
        G = self.create_graph(
            weight_threshold=float(np.percentile(W, threshold_percentile))
        )

        # Generate positions for nodes
        if pos == "spring":
            pos = nx.spring_layout(  # type: ignore[assignment]
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
        def _safe_norm(arr: Union[np.ndarray, JaxArray]) -> np.ndarray:
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
        nx.draw_networkx_edges(  # type: ignore[call-overload]
            G,
            pos,  # type: ignore[arg-type]
            edgelist=edgelist_draw if edges_list else None,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            ax=ax,
            **edge_kwargs,
        )
        nx.draw_networkx_nodes(  # type: ignore[call-overload]
            G,
            pos,  # type: ignore[arg-type]
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
                pos,  # type: ignore[arg-type]
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
                        edge_labels_dict[(u, v, k)] = f"{float(val):.2f}"  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        edge_labels_dict[(u, v, k)] = str(val)
                nx.draw_networkx_edge_labels(
                    G,
                    pos,  # type: ignore[arg-type]
                    edge_labels=edge_labels_dict,
                    ax=ax,
                    font_size=fontsize,
                )
        if return_fig:
            assert fig is not None
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
        self,
        weights_kwargs: Optional[Dict[str, Any]] = None,
        lengths_kwargs: Optional[Dict[str, Any]] = None,
        graph_kwargs: Optional[Dict[str, Any]] = None,
        log_weights: bool = False,
    ) -> Figure:
        """Create comprehensive visualization with graph and matrices.

        Produces a three-panel figure showing network graph, weights matrix,
        and lengths matrix with synchronized colorbars and formatting.

        Parameters
        ----------
        weights_kwargs : dict, optional
            Keyword arguments passed to `plot_weights`
        lengths_kwargs : dict, optional
            Keyword arguments passed to `plot_lengths`
        graph_kwargs : dict, optional
            Keyword arguments passed to `plot_graph`
        log_weights : bool, default=False
            Use logarithmic scale for weights

        Returns
        -------
        matplotlib.figure.Figure
            Figure with three subplots (graph, weights, lengths)

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.plot_overview(
            log_weights=True)
        ```

        See Also
        --------
        plot_graph : Network graph visualization
        plot_matrix : Side-by-side matrix visualization
        """

        fig, axs = plt.subplots(ncols=3, layout="tight", figsize=(15, 5))

        # Ensure kwargs are not None before unpacking
        if weights_kwargs is None:
            weights_kwargs = {}
        if lengths_kwargs is None:
            lengths_kwargs = {}
        if graph_kwargs is None:
            graph_kwargs = {}

        if "edge_cmap" not in graph_kwargs:
            graph_kwargs["edge_cmap"] = "magma"

        g = self.plot_graph(axs[0], **graph_kwargs)
        axs[0].axis("off")
        w = self.plot_weights(axs[1], log=log_weights, **weights_kwargs)
        l = self.plot_lengths(axs[2], **lengths_kwargs)
        axs[2].sharey(axs[1])

        c1 = fig.colorbar(g, ax=axs[0], shrink=0.5, pad=-0.05)  # type: ignore[arg-type]
        c2 = fig.colorbar(w, ax=axs[1], shrink=0.5)
        c3 = fig.colorbar(l, ax=axs[2], shrink=0.5)

        fontsize_scaler = 1.5

        for c in [c1, c2, c3]:
            c.outline.set_visible(False)  # type: ignore[misc]
            for label in c.ax.get_yticklabels():
                label.set_fontsize(float(c.ax.yaxis.label.get_fontsize()) * fontsize_scaler)  # type: ignore[arg-type]

        for ax in axs:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(label.get_fontsize() * fontsize_scaler)
            ax.title.set_fontsize(ax.title.get_fontsize() * fontsize_scaler)
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * fontsize_scaler)
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * fontsize_scaler)

        c1.set_label("ms", fontsize=float(c1.ax.yaxis.label.get_fontsize()) * fontsize_scaler)  # type: ignore[arg-type]
        c2.set_label(
            "log1p(weight)" if log_weights else "weight",
            fontsize=float(c2.ax.yaxis.label.get_fontsize()) * fontsize_scaler,  # type: ignore[arg-type]
        )
        c3.set_label("mm", fontsize=float(c3.ax.yaxis.label.get_fontsize()) * fontsize_scaler)  # type: ignore[arg-type]

        plt.close()
        return fig

    def normalize(self) -> None:
        """Apply min-max normalization to connection weights.

        Sets normalization equation to scale weights to [0, 1] range.
        Equivalent to `normalize_weights("(W - W_min) / (W_max - W_min)")`.

        Examples
        --------
        ```{python}
        sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})
        sc.normalize()
        normalized_weights = sc.weights_matrix  # Now in [0, 1] range
        ```

        See Also
        --------
        normalize_weights : Set custom normalization equation
        """
        self.normalization = tvbo_datamodel.Equation(
            rhs="(W - W_min) / (W_max - W_min)"
        )
