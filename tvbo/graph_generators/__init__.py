"""Python materialisers for TVBO graph generators.

Each curated `GraphGenerator` entry in `tvbo/database/graph_generators/`
declares its `bindings.python.callable` pointing at a function in this
package. `Network.resolve()` invokes the callable with the generator's
parameters at YAML load time and stores the returned adjacency.

The mapping (generator name → materialiser) is the database, not Python.
Adding a new generator means adding a YAML file plus (optionally) a
materialiser function — no edits to dispatch tables.
"""

from .builtins import random_reservoir, weight_shuffle

__all__ = ["random_reservoir", "weight_shuffle"]
