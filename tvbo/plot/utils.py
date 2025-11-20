#
# Module: utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
from os.path import join, abspath, dirname
from xml.etree import ElementTree as ET

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tvbo.knowledge import constants

ROOT = abspath(dirname(__file__))


def use_tvbo_style():
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
    # Convert hex color to RGB and then to HSV
    rgb_color = mcolors.hex2color(hex_color)
    hsv_color = mcolors.rgb_to_hsv(np.array([rgb_color]))
    # Flatten the HSV array to get a sortable tuple
    return tuple(hsv_color.flatten())


# TVB-Colors
formatted_hex_colors = [
    "#" + color.strip("#")
    for color in extract_svg_colors(f"{constants.DATA_DIR}/tvb_logo.svg")
]

# Sort the colors using the new conversion function
_tvb_colors = sorted(formatted_hex_colors, key=hex_to_sortable_hsv)
tvb_colors = list(
    np.array(_tvb_colors)[
        [0, 1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26]
    ]
)
