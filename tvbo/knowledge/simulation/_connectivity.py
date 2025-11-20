#  connectivity.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
TVB-O wrapper for Global and Local Connectivity functions
=========================================================

"""
import numpy as np
# from tvb.datatypes.connectivity import Connectivity


# def create_custom_connectivity(weights, lengths, centers, labels=None):
#     """
#     Creates a custom connectivity object based on the weights, lengths and centers
#     provided by the user.

#     Replaces tvbo.tvbosimulator.create_sc()

#         Parameters:
#             weights (ndarray): Array representing connectivity weights between regions.
#             lengths (ndarray): Array representing the tract lengths between regions.
#             centers (ndarray): Array representing the center coordinates for each region.
#             labels (list, optional): List of region labels.

#         Returns:
#             tvb.datatypes.connectivity.Connectivity: TVB Connectivity object.
#     """
#     conn = Connectivity()
#     conn.weights = weights
#     conn.centres = centers
#     conn.tract_lengths = lengths
#     if not isinstance(labels, type(None)):
#         conn.region_labels = labels

#     conn.configure()

#     return conn


# def create_connectivity_with_n_regions(nor=1):
#     """
#     Creates a placeholder connectivty object with the specified number of regions.

#     Replaces tvbosimulator.dummy_connectivity()
#        Parameters:
#        - nor (int): Number of regions. Default is 1.

#        Returns:
#            tvb.datatypes.connectivity.Connectivity: Placeholder connectivity instance.
#     """

#     # Create empty connectivity
#     conn = Connectivity()
#     # First weights and distances
#     conn.motif_all_to_all(number_of_regions=nor)
#     # Centers, specify the number of regions, otherwise it'll use a default value.
#     conn.centres_spherical(number_of_regions=nor)
#     # By default, the new regions labels are numeric characters, ie [0, 1, ...]
#     conn.create_region_labels(mode="alphabetic")
#     # But custom region labels can be used
#     conn.region_labels = np.array([str(i) for i in range(nor)])
#     conn.configure()

#     return conn


# def get_single_node():
#     # Create 1-Node Connectivity
#     sc = Connectivity()
#     sc.weights = np.zeros((1, 1))
#     sc.tract_lengths = np.zeros((1, 1))
#     sc.centres = np.zeros((1))
#     sc.create_region_labels(mode="alphabetic")
#     return sc


# # TODO: typo?
# def sinlge_node():
#     nor = 1
#     weights = np.zeros((nor, nor))
#     lengths = np.full_like(weights, 0)
#     centers = np.zeros((nor))
#     labels = np.arange(nor)

#     SC = tvbosimulator.create_sc(weights, lengths, centers, labels)
#     SC.create_region_labels(mode="alphabetic")
#     SC.configure()
#     return SC


# # TODO: add all params to docstring
# def simulated_random_sc(
#     n_nodes,
#     mean=0.5,
#     std_dev=0.15,
#     intra_hemisphere_density=0.8,
#     general_inter_hemisphere_density=0.2,
#     diagonal_inter_density=1,
# ):
#     """
#     Generate a simulated random structural connectivity matrix for a brain
#     with 'n_nodes' nodes, using a Gaussian distribution for connection strengths.

#     Parameters:
#         n_nodes (int): Number of nodes in the brain model.
#         mean (float): Mean of the Gaussian distribution for weights.
#         std_dev (float): Standard deviation of the Gaussian distribution for weights.

#     Returns:
#         np.ndarray: A symmetric matrix representing the brain's structural connectivity.
#     """
#     connectivity_matrix = np.zeros((n_nodes, n_nodes))

#     # Density definitions
#     split_index = n_nodes // 2

#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             if i != j:  # Exclude self-connections
#                 density = 0
#                 if (i < split_index and j < split_index) or (
#                     i >= split_index and j >= split_index
#                 ):
#                     density = intra_hemisphere_density
#                 elif i == j - split_index or j == i - split_index:
#                     density = diagonal_inter_density
#                 else:
#                     density = general_inter_hemisphere_density

#                 if np.random.rand() < density:
#                     connectivity_matrix[i, j] = np.random.normal(mean, std_dev)

#     # Symmetrize the matrix and normalize
#     connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
#     connectivity_matrix = (connectivity_matrix - np.min(connectivity_matrix)) / (
#         np.max(connectivity_matrix) - np.min(connectivity_matrix)
#     )

#     return connectivity_matrix


# def load_tvb_connectivity(
#     connectivity="default",
#     weights=None,
#     tract_lengths=None,
#     speed=3.0,
#     centers=None,
#     nnodes=None,
#     region_labels=None,
#     orientations=None,
#     areas=None,
# ):
#     """
#     Loads a The Virtual Brain (TVB) connectivity.

#     This function creates a Connectivity object from TVB, and sets its attributes
#     based on the provided arguments.

#     Args:
#         connectivity (str, optional): The type of connectivity to load.
#                                       If "default", a Connectivity object is loaded from file.
#                                       Defaults to "default".
#         weights (numpy.ndarray, optional): The weights of the connectivity.
#         tract_lengths (numpy.ndarray, optional): The tract lengths of the connectivity.
#         speed (float, optional): The speed of signal transmission. Defaults to 3.0.
#         centers (numpy.ndarray, optional): The centers of the regions.
#         nnodes (int, optional): The number of nodes in the connectivity.
#                                 If not provided, it's inferred from the shape of weights.
#         region_labels (list of str, optional): The labels of the regions.
#         orientations (numpy.ndarray, optional): The orientations of the regions.
#         areas (numpy.ndarray, optional): The areas of the regions.

#     Returns:
#         Connectivity: A TVB Connectivity object with the specified attributes.
#     """
#     if connectivity == "default":
#         return Connectivity.from_file()

#     SC = Connectivity()

#     if isinstance(nnodes, type(None)):
#         nnodes = weights.shape[0]

#     SC.set_weights(weights, nnodes)
#     SC.set_tract_lengths(tract_lengths, nnodes)
#     SC.speed = np.array([speed])

#     if isinstance(centers, type(None)):
#         centers = np.zeros((nnodes, 3))

#     SC.set_centres(centers, nnodes)

#     if not isinstance(areas, type(None)):
#         SC.set_areas(areas, nnodes)
#     if not isinstance(orientations, type(None)):
#         SC.set_orientations(orientations, nnodes)

#     if isinstance(region_labels, type(None)):
#         SC.create_region_labels(mode="alphabetic")

#     SC.set_region_labels(region_labels)

#     return SC


# Calculate the Euclidean distance matrix
def euclidean_distance_matrix(coords):
    # Using numpy to calculate the pairwise Euclidean distance
    from scipy.spatial import distance

    return distance.squareform(distance.pdist(coords, "euclidean"))
