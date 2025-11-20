#  utils.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
Utilities Module for TVB-O
==========================

This module provides a set of utility functions for various tasks related to TVB-O simulations.
It includes functions to:

- Retrieve the TVB-O logo.
- Convert color formats.
- Compute postsynaptic potential (PSP) for the Jansen-Rit model.
- Define hierarchical positions for nodes in a graph.
- Generate specific colormaps.
- Display multiple views of brain regions.

Usage:
------
    >>> from utils import get_logo
    >>> logo = get_logo()

Author:
    Leon K. Martin (2023)

Copyright:
    Copyright (c) 2023 Charité Universitätsmedizin Berlin
"""

from os.path import abspath, dirname, join

import jax
import jax.numpy as jnp
import numpy as np
import owlready2
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
from scipy import stats

cm = 1 / 2.54
ROOT_DIR = abspath(dirname(__file__))


def get_logo() -> np.ndarray:
    """
    Retrieve the TVB-O logo.

    Returns:
        np.ndarray: Image array of the TVB-O logo.
    """
    return plt.imread(join(ROOT_DIR, "../tvbo_logo.png"))


def hex2rgba(hex: str, alpha: int = 1, max: int = 1) -> tuple:
    """
    Convert a hex color code to RGBA.

    Args:
        hex (str): Hexadecimal color code.
        alpha (float, optional): Alpha value for the color. Defaults to 1.
        max (int, optional): Scaling factor. Defaults to 1.

    Returns:
        tuple: RGBA values.
    """
    hex = hex.replace("#", "")
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal / (255 / max))
    if max == 255:
        rgb = [int(c) for c in rgb]
    rgb.append(alpha)
    return tuple(rgb)


def is_class(search_string: str, ontology: owlready2.Ontology) -> bool:
    """Check if a search string is in any of the labels of classes in the given ontology.

    Args:
        search_string: The string to search for within the ontology labels.
        ontology: The loaded ontology where to perform the search.

    Returns:
        A boolean indicating if the string is found in any label.
    """
    for cls in ontology.classes():
        for label in cls.label:
            if search_string in label:
                return True
    return False


tvb_colors = ["#2E9795", "#935495", "#4EA8E5", "#E58221"]
cmap = ListedColormap(tvb_colors)


def get_cmap(colors):
    """
    Get a colormap based on the given colors.

    Parameters
    ----------
    colors : list
        List of colors.

    Returns
    -------
    ListedColormap
        Colormap based on the input colors.
    """
    return ListedColormap(colors)


def multiview(data, cortex, suptitle="", figsize=(15, 10), **kwds):
    """
    Display multiple views of brain regions.
    Copied from https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/utils.py

    Parameters
    ----------
    data : ndarray
        Data for the regions.
    cortex : Cortex
        Brain cortex information.
    suptitle : str, optional
        Super title for the plots, by default "".
    figsize : tuple, optional
        Figure size, by default (15, 10).
    **kwds
        Additional keyword arguments.
    """

    vtx = cortex.vertices
    tri = cortex.triangles
    rm = cortex.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[(rm[tri] < 38).any(axis=1)]
    lh_vtx = vtx[rm < 38]
    lh_x, lh_y, lh_z = lh_vtx.T  # TODO: not used, remove?
    lh_tx, lh_ty, lh_tz = lh_vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[(rm[tri] >= 38).any(axis=1)]
    rh_vtx = vtx[rm < 38]
    rh_x, rh_y, rh_z = rh_vtx.T  # TODO: not used, remove?
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    views = {
        "lh-lateral": Triangulation(-x, z, lh_tri[np.argsort(lh_ty)[::-1]]),
        "lh-medial": Triangulation(x, z, lh_tri[np.argsort(lh_ty)]),
        "rh-medial": Triangulation(-x, z, rh_tri[np.argsort(rh_ty)[::-1]]),
        "rh-lateral": Triangulation(x, z, rh_tri[np.argsort(rh_ty)]),
        "both-superior": Triangulation(y, x, tri[np.argsort(tz)]),
    }

    def plotview(
        i,
        j,
        k,
        viewkey,
        z=None,
        zlim=None,
        zthresh=None,
        suptitle="",
        shaded=True,
        cmap=plt.cm.coolwarm,
        viewlabel=False,
    ):
        v = views[viewkey]
        ax = plt.subplot(i, j, k)
        if z is None:
            z = np.rand(v.x.shape[0])
        if not viewlabel:
            plt.axis("off")
        kwargs = (
            {"shading": "gouraud"} if shaded else {"edgecolors": "k", "linewidth": 0.1}
        )
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        ax.set_aspect("equal")
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            plt.xlabel(viewkey)

    plt.figure(figsize=figsize)
    plotview(2, 3, 1, "lh-lateral", data, **kwds)
    plotview(2, 3, 4, "lh-medial", data, **kwds)
    plotview(2, 3, 3, "rh-lateral", data, **kwds)
    plotview(2, 3, 6, "rh-medial", data, **kwds)
    plotview(1, 3, 2, "both-superior", data, suptitle=suptitle, **kwds)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)


def get_continuous_cmap(hex_list, float_list=None):
    """
    Get a continuous colormap based on a list of hex codes.

    Parameters
    ----------
    hex_list : list
        List of hex code strings.
    float_list : list, optional
        List of floats between 0 and 1 of the same length as hex_list, by default None.

    Returns
    -------
    LinearSegmentedColormap
        A colormap based on the input hex codes.
    """
    import matplotlib as mpl
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    def rgb_to_dec(value):
        """Converts rgb to decimal colours (i.e. divides each value by 256)

        :param value: rgb values
        :type value: tuple
        :return: decimal rgb values
        :rtype: tuple
        """
        return [v / 256 for v in value]

    def hex_to_rgb(value):
        """Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values

        :param value: hexcode
        :type value: str
        :return: rgba value
        :rtype: tuple
        """
        value = value.strip("#")  # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))

    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    :param hex_list: list of hex code strings
    :type hex_list: list
    :param float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and
    end with 1., defaults to None
    :type float_list: list, optional
    :return: color map
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def flatten_list(nested_list):
    """
    Flatten a nested list.

    Parameters:
    -----------
    nested_list : list
        The nested list to be flattened.

    Returns:
    --------
    list
        The flattened list.

    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def custom_get(d, key, default=None):
    return d.get(key, default) if d.get(key, default) is not None else default


from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Bunch(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def tree_flatten(self):
        return (tuple(self.values()), tuple(self.keys()))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))


def numbered_print(text):
    lines = text.splitlines()
    max_line_num = len(str(len(lines)))
    for i, line in enumerate(lines, start=1):
        print(f"{i:0{max_line_num}} {line}")


import equinox as eqx


def format_pytree_as_string(
    pytree,
    name: str = "root",
    prefix: str = "",
    is_last: bool = False,
    show_numerical_only: bool = False,
    is_root: bool = True,
    hide_none: bool = False,
    show_array_values: bool = False,
) -> str:
    """
    Recursively formats a JAX pytree structure as a string with Unicode box-drawing characters.

    Args:
        pytree (Any): The pytree to format.
        name (str): The name of the current node.
        prefix (str): Current line prefix.
        is_last (bool): Whether the current node is the last child of its parent.
        show_numerical_only (bool): If True, only show arrays and numerical types (float, int, etc.).
        is_root (bool): Whether this node is the root of the tree.
        hide_none (bool): If True, fields with None values will be hidden.
        show_array_values (bool): If True, print full array values instead of summaries.

    Returns:
        str: The formatted string representation of the pytree.
    """
    # Unicode box-drawing characters for the tree structure
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    # Initialize the result string
    result = []

    # Special handling for root element
    if is_root:
        current_prefix = ""  # Root has no prefix
        next_prefix = ""  # Children of root start without vertical bars
    else:
        # Determine the current line prefix
        current_prefix = prefix + (last if is_last else tee)
        # Determine the prefix for children
        next_prefix = prefix + (space if is_last else branch)

    # Check if the object is a string
    if isinstance(pytree, str):
        if not show_numerical_only:
            result.append(f'{current_prefix}{name}: "{pytree}"')
        return "\n".join(result)

    # Check if the object is None
    if pytree is None:
        if not hide_none and not show_numerical_only:
            result.append(f"{current_prefix}{name}: NoneType")
        return "\n".join(result)

    # Check if the object is a JAX array
    if isinstance(pytree, (jnp.ndarray, np.ndarray)):
        shape_str = f"shape={pytree.shape}"
        dtype_str = f"dtype={pytree.dtype}"
        # result.append(f"{current_prefix}{name}: Array({shape_str}, {dtype_str})")
        if show_array_values:
            # result.append(f"{current_prefix}{name}: Array({shape_str}, {dtype_str})")
            # result.append(f"{current_prefix}{name}: No({shape_str}, {dtype_str})")
            result.append(f"{current_prefix}{name}: {pytree}")
        else:
            result.append(f"{current_prefix}{name}: {eqx.tree_pformat(pytree)}")
        return "\n".join(result)

    # Try to flatten the pytree
    try:
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        # If it's a leaf (i.e., it has no children), format its type
        if not leaves or (len(leaves) == 1 and pytree is leaves[0]):
            # For numerical types, always display the value
            if isinstance(pytree, (int, float, bool, complex)):
                result.append(f"{current_prefix}{name}: {pytree}")
            # For other types, check filter setting
            elif not show_numerical_only:
                result.append(f"{current_prefix}{name}: {type(pytree).__name__}")
            return "\n".join(result)

        # Otherwise, format it as a container and process its children
        result.append(f"{current_prefix}{name}")

        # If it's a dictionary, iterate through its key-value pairs
        if isinstance(pytree, dict):
            items = list(pytree.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    str(key),
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if (
                    child_result
                ):  # Only append if there's content (might be empty with show_numerical_only)
                    result.append(child_result)

        # If it's a dataclass or a custom class with __dict__ attribute
        elif hasattr(pytree, "__dict__"):
            items = list(pytree.__dict__.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    key,
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # If it's a sequence (like list or tuple)
        elif hasattr(pytree, "__len__") and not isinstance(
            pytree, (str, bytes, bytearray)
        ):
            for i, item in enumerate(pytree):
                child_result = format_pytree_as_string(
                    item,
                    f"[{i}]",
                    next_prefix,
                    i == len(pytree) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # For other types of containers
        else:
            result.append(
                f"{current_prefix}{name}: {type(pytree).__name__} (unknown structure)"
            )

    except Exception:
        # If we can't flatten it as a pytree, treat it as a leaf
        # For strings, display the string value if not filtering
        if isinstance(pytree, str):
            if not show_numerical_only:
                result.append(f'{current_prefix}{name}: "{pytree}"')
        # For numerical types, always display the value
        elif isinstance(pytree, (int, float, bool, complex)):
            result.append(f"{current_prefix}{name}: {pytree}")
        # For other types, check filter setting
        elif not show_numerical_only:
            result.append(f"{current_prefix}{name}: {type(pytree).__name__}")

    return "\n".join(result)


def pretty_print_pytree(
    pytree,
    name: str = "root",
    prefix: str = "",
    show_numerical_only: bool = False,
    hide_none: bool = False,
) -> None:
    """
    Prints a pretty formatted representation of a JAX pytree structure.

    Args:
        pytree (Any): The pytree to print.
        name (str): The name of the current node.
        prefix (str): Current line prefix.
        show_numerical_only (bool): If True, only show arrays and numerical types (float, int, etc.).
        hide_none (bool): If True, fields with None values will be hidden.

    Returns:
        None
    """
    formatted_string = format_pytree_as_string(
        pytree, name, prefix, False, show_numerical_only, True, hide_none
    )
    print(formatted_string)


def per_window_fc(tv, xv, window=1e3):
    """
    Calculate per-window functional connectivity.

    Parameters
    ----------
    tv : ndarray
        Time vector.
    xv : ndarray
        Data vector.
    window : float, optional
        Time window for calculation. Default is 1e3.

    Returns
    -------
    ndarray
        Correlation coefficients for each window.
    """
    cs = []
    for i in range(int(tv[-1] / window)):
        cs.append(np.corrcoef(xv[(tv > (i * 1e3)) * (tv < (1e3 * (i + 1)))].T))
    cs = np.array(cs)
    return cs


def ttest_correlation_strength(cs):
    """
    Perform a t-test on the strength of the correlation.

    Parameters
    ----------
    cs : ndarray
        Correlation coefficients.

    Returns
    -------
    ndarray
        P-values of the t-test for each correlation coefficient.
    """
    cs_z = np.arctanh(cs)
    for i in range(cs.shape[1]):
        cs_z[:, i, i] = 0.0
    _, p = stats.ttest_1samp(cs, 0.0)

    return p


# ---- YAML utilities ----
def to_yaml(obj, filepath: str | None = None) -> str:
    """Dump a LinkML datamodel object to YAML.

    - If filepath is provided, write YAML to that file and return the path.
    - If filepath is None, return the YAML string.

    Args:
        obj (object): Datamodel object to serialize.
        filepath (str | None): Optional path to write YAML.

    Returns:
        str: File path when written to disk, otherwise the YAML string.
    """
    try:
        from linkml_runtime.dumpers import yaml_dumper
    except Exception as e:
        raise RuntimeError("linkml_runtime is required for YAML dumping") from e

    if filepath:
        yaml_dumper.dump(obj, filepath)
        return filepath
    return yaml_dumper.dumps(obj)


def from_yaml(filepath: str, cls) -> object:
    """Load a LinkML datamodel object from a YAML file.

    Parameters:
        filepath (str): Path to the YAML file.
        cls (type): The datamodel class to instantiate.

    Returns:
        object: An instance of the datamodel class populated with data from the YAML file.
    """
    try:
        from linkml_runtime.loaders import yaml_loader
    except Exception as e:
        raise RuntimeError("linkml_runtime is required for YAML loading") from e
    md = yaml_loader.load(filepath, target_class=cls)
    return md
