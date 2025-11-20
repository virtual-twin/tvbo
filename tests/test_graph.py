#
# Module: test_graph.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
import unittest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tvbo.plot.network import (
    get_categories_from_graph,
    get_category_from_graph,
    get_default_params,
    get_layout,
    reverse_edges,
    set_axis_limits,
)


# Example graph used for testing
def create_test_graph():
    G = nx.DiGraph()
    G.add_node("A", category="alpha")
    G.add_node("B", category="beta")
    G.add_edge("A", "B", type="is_a")
    return G


class TestGraphFunctions(unittest.TestCase):
    def test_get_default_params(self):
        params = get_default_params()
        self.assertIsInstance(params, dict)
        self.assertIn("node_size_factor", params)
        # Assert other expected params...

    def test_reverse_edges(self):
        test_pos = {"A": (0, 1), "B": (1, 0)}
        expected_pos = {"A": np.array([1, 0]), "B": np.array([0, 1])}
        reversed_pos = reverse_edges(test_pos)
        for key in expected_pos:
            np.testing.assert_array_equal(reversed_pos[key], expected_pos[key])

    def test_get_category_from_graph(self):
        G = create_test_graph()
        self.assertEqual(get_category_from_graph(G, "A"), "alpha")
        self.assertEqual(get_category_from_graph(G, "B"), "beta")

    def test_get_categories_from_graph(self):
        G = create_test_graph()
        self.assertListEqual(get_categories_from_graph(G), ["alpha", "beta"])

    def test_set_axis_limits(self):
        G = create_test_graph()
        pos = {"A": (0, 0), "B": (1, 1)}
        ax = plt.figure().add_subplot(111)
        set_axis_limits(pos, ax)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        self.assertEqual(xlim[0], -0.15)
        self.assertEqual(ylim[0], -0.15)
        # Assert other limits...

    # Add other test cases...


if __name__ == "__main__":
    unittest.main()
