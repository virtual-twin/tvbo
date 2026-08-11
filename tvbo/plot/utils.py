#
# Module: utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Plotting helpers for TVB-O figures.

Provides the shared matplotlib style, the TVB brand color palette derived from the
TVB logo SVG, colormap and color-conversion utilities, and a multi-view brain surface renderer.
"""

from os.path import join, abspath, dirname
from xml.etree import ElementTree as ET

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tvbo.ontology import constants

ROOT = abspath(dirname(__file__))


def use_tvbo_style():
    """Activate the bundled TVB-O matplotlib style sheet."""
    plt.style.use(join(ROOT, "tvbo.mplstyle"))


def extract_svg_colors(svg_path):
    """
    Extracts colors from an SVG file and returns them as a list of hex codes.
    """
    # Parse the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Define namespaces to search for style attributes within the SVG
    namespaces = {"svg": "http://www.w3.org/2000/svg"}

    # Find all elements with a 'style' attribute or fill/stroke attributes
    style_elems = root.findall(".//*[@style]", namespaces)
    fill_elems = root.findall(".//*[@fill]", namespaces)
    stroke_elems = root.findall(".//*[@stroke]", namespaces)

    # Extract color values
    color_values = set()  # Use a set to avoid duplicates

    # Helper function to extract color values from a style string
    def add_color_from_style(style_str):
        """Parse `fill:` and `stroke:` colors from a CSS style string into `color_values`.

        Args:
            style_str: Value of an SVG element's `style` attribute.
        """
        color_attrs = ["fill:", "stroke:"]
        for attr in color_attrs:
            start = style_str.find(attr)
            if start != -1:
                start += len(attr)
                end = style_str.find(";", start)
                end = end if end != -1 else len(style_str)
                color = style_str[start:end].strip()
                if color not in ["none", "transparent"]:
                    color_values.add(color)

    # Extract colors from style attributes
    for elem in style_elems:
        style_str = elem.get("style")
        add_color_from_style(style_str)

    # Extract colors from fill and stroke attributes
    for elem in fill_elems + stroke_elems:
        color = elem.get("fill") or elem.get("stroke")
        if color and color not in ["none", "transparent"]:
            color_values.add(color)

    return color_values


# Convert hex to RGB, then RGB to HSV, and return a sortable representation
def hex_to_sortable_hsv(hex_color):
    """Convert a hex color to a sortable HSV tuple.

    Args:
        hex_color: Hexadecimal color string (e.g. `"#2E9795"`).

    Returns:
        The `(hue, saturation, value)` components as a tuple, suitable as a
        sort key for ordering colors by hue.
    """
    # Convert hex color to RGB and then to HSV
    rgb_color = mcolors.hex2color(hex_color)
    hsv_color = mcolors.rgb_to_hsv(np.array([rgb_color]))
    # Flatten the HSV array to get a sortable tuple
    return tuple(hsv_color.flatten())


# TVB-Colors
formatted_hex_colors = ["#" + color.strip("#") for color in extract_svg_colors(f"{constants.DATA_DIR}/tvb_logo.svg")]

# Sort the colors using the new conversion function
_tvb_colors = sorted(formatted_hex_colors, key=hex_to_sortable_hsv)
tvb_colors = list(np.array(_tvb_colors)[[0, 1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26]])

# Simple 4-color palette for quick use
tvb_colors_simple = ["#2E9795", "#935495", "#4EA8E5", "#E58221"]


def get_logo() -> "np.ndarray":
    """Retrieve the TVB-O logo as a NumPy image array."""
    logo_path = join(abspath(dirname(dirname(__file__))), "tvbo_logo.png")
    return plt.imread(logo_path)


def hex2rgba(hex: str, alpha: int = 1, max: int = 1) -> tuple:
    """Convert a hex color code to RGBA.

    Args:
        hex: Hexadecimal color code.
        alpha: Alpha value for the color. Defaults to 1.
        max: Scaling factor. Defaults to 1.

    Returns:
        RGBA tuple.
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


def get_cmap(colors=None):
    """Get a ListedColormap from given colors (defaults to ``tvb_colors_simple``).

    Parameters
    ----------
    colors : list, optional
        List of colors.

    Returns
    -------
    ListedColormap"""
    from matplotlib.colors import ListedColormap

    return ListedColormap(colors or tvb_colors_simple)


def get_continuous_cmap(hex_list, float_list=None):
    """Get a continuous LinearSegmentedColormap from a list of hex codes.

    Parameters
    ----------
    hex_list : list
        List of hex code strings.
    float_list : list, optional
        List of floats between 0 and 1 of the same length as hex_list.

    Returns
    -------
    LinearSegmentedColormap"""
    import matplotlib as mpl

    def _hex_to_rgb(value):
        value = value.strip("#")
        lv = len(value)
        return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))

    rgb_list = [[v / 256 for v in _hex_to_rgb(h)] for h in hex_list]
    if not float_list:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = {}
    for num, col in enumerate(["red", "green", "blue"]):
        cdict[col] = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
    return mpl.colors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)


def multiview(data, cortex, suptitle="", figsize=(15, 10), **kwds):
    """Display multiple views of brain regions.

    Adapted from https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/utils.py

    Parameters
    ----------
    data : ndarray
        Data for the regions.
    cortex : Cortex
        Brain cortex information.
    suptitle : str, optional
        Super title for the plots.
    figsize : tuple, optional
        Figure size.
    """
    from matplotlib.tri import Triangulation

    vtx = cortex.vertices
    tri = cortex.triangles
    rm = cortex.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[(rm[tri] < 38).any(axis=1)]
    lh_vtx = vtx[rm < 38]
    lh_tx, lh_ty, lh_tz = lh_vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[(rm[tri] >= 38).any(axis=1)]
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
        i, j, k, viewkey, z=None, zlim=None, zthresh=None, suptitle="", shaded=True, cmap=plt.cm.coolwarm, viewlabel=False
    ):
        """Draw a single brain surface view into subplot `(i, j, k)`.

        Args:
            i: Number of subplot rows.
            j: Number of subplot columns.
            k: Index of the subplot to draw into.
            viewkey: Key selecting the triangulation from `views`.
            z: Per-vertex scalar values to color the surface; random values are
                used when omitted.
            zlim: Symmetric color limit; sets the colormap range to `[-zlim, zlim]`.
            zthresh: Threshold below which `z` magnitudes are zeroed out.
            suptitle: Title drawn above this subplot.
            shaded: If `True`, use gouraud shading; otherwise draw mesh edges.
            cmap: Matplotlib colormap for the surface.
            viewlabel: If `True`, keep the axes and label them with `viewkey`.
        """
        v = views[viewkey]
        ax = plt.subplot(i, j, k)
        if z is None:
            z = np.random.rand(v.x.shape[0])
        if not viewlabel:
            plt.axis("off")
        kwargs = {"shading": "gouraud"} if shaded else {"edgecolors": "k", "linewidth": 0.1}
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
