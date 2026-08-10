#  freesurfer.py
#
# Created on Tue Oct 24 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""FreeSurfer atlas utilities for mapping between region indices and labels.

Loads the FreeSurfer color lookup table and provides helpers to translate between numeric FreeSurfer indices and region labels, exposes the FS86 and
`aparc` label lists, and converts HCP-MMP1 region names to FreeSurfer label conventions.
"""

from os.path import join
from typing import Dict, List, Union, cast

import numpy as np
import pandas as pd

from tvbo.ontology import constants

lut = pd.read_csv(
    join(constants.DATA_DIR, "freesurfer", "FreeSurferColorLUT.txt"),
    comment="#",
    sep="\\s+",
    names=["id", "name", "r", "g", "b", "a"],
)


def fs_mapper(output: str = "label") -> Dict[Union[int, str], Union[int, str]]:
    """Create a mapping between FreeSurfer indices and labels.

    Args:
        output (str, optional): If "label", return a mapping id → label. If "index",
            return a mapping label → id. Defaults to "label".

    Returns:
        dict: Mapping between indices and labels according to the selected output mode.
    """
    lut = pd.read_csv(
        join(constants.DATA_DIR, "freesurfer", "FreeSurferColorLUT.txt"),
        comment="#",
        sep="\\s+",
        names=["id", "name", "r", "g", "b", "a"],
    )

    lut.name = lut.name.str.lower()

    # FS index-name mapper
    mapper = dict()

    # Create index-name pairs.
    for i, r in lut.iterrows():
        if output.lower() in ["label"]:
            mapper[r.id] = r["name"]
        else:
            mapper[r["name"]] = r.id

    return mapper


def idx2label(idx: Union[int, List[int]]) -> Union[str, List[str]]:
    """Convert one or many FreeSurfer indices to labels.

    Args:
        idx (int | list[int]): One index or a list of indices.

    Returns:
        str | list[str]: The corresponding label(s).
    """
    if isinstance(idx, list):
        return [cast(str, fs_mapper(output="label")[i]) for i in idx]
    else:
        return cast(str, fs_mapper(output="label")[idx])


def label2idx(label: Union[str, List[str]]) -> Union[int, List[int]]:
    """Convert one or many FreeSurfer labels to indices.

    Args:
        label (str | list[str]): One label or a list of labels.

    Returns:
        int | list[int]: The corresponding index/indices.
    """
    if isinstance(label, list):
        return [cast(int, fs_mapper(output="index")[lbl]) for lbl in label]
    else:
        return cast(int, fs_mapper(output="index")[label])


fs_aparcaseg86_labels = np.genfromtxt(join(constants.DATA_DIR, "freesurfer", "FS86_labels.txt"), dtype="str")

fs_aparc_labels = np.genfromtxt(join(constants.DATA_DIR, "freesurfer", "FS_aparc_labels.txt"), dtype="str")


def hcp2fs_labels(hcp_labels: List[str]) -> List[str]:
    """Convert HCP-MMP1 region names to FreeSurfer cortical label conventions.

    Each label is lower-cased and its hemisphere prefix rewritten, mapping
    `l_` to `ctx-lh-` and `r_` to `ctx-rh-`.

    Args:
        hcp_labels: HCP-MMP1 region names to convert.

    Returns:
        The region names rewritten in FreeSurfer label form.
    """
    fs_labels: List[str] = list()
    for lbl in hcp_labels:
        lbl = lbl.lower()
        fs_labels.append(lbl.replace("l_", "ctx-lh-").replace("r_", "ctx-rh-"))

    return fs_labels
