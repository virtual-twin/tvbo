"""Runners that execute TVBO simulations on different backends.

This package collects the execution helpers used once a model and connectome
are assembled: graph-based integration of a connectome as a network of coupled
local models ([`GraphRunner`](graph.qmd#GraphRunner)), the SciPy reference
helpers it builds on ([`tvbo.run.compgraph`](compgraph.qmd)), and low-level
utilities for running TVBO-generated Julia code ([`tvbo.run.julia`](julia.qmd)).
"""
