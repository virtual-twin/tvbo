#
# Module: computation_utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module contains utility functions used for computations for TVB Simulations, simulation results,
simulation components, etc.
"""
import numpy as np

def average_rois(data: np.ndarray) -> np.ndarray:
    """
    Compute the mean value across regions of interest in given data.

    Parameters:
    - data: Data from regions of interest.

    Returns:
        Averaged values for regions of interest (mean along axis=2).
    """
    return data.mean(axis=2)


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distance matrix.

    Parameters:
        coords: Array of shape (n, d) with n points in d-dimensional space.

    Returns:
        A (n, n) ndarray of pairwise Euclidean distances.
    """
    # Using numpy to calculate the pairwise Euclidean distance
    from scipy.spatial import distance

    return distance.squareform(distance.pdist(coords, "euclidean"))
