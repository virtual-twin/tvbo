"""Analysis subpackage.

Houses analysis result container classes (e.g., BifurcationResult) and related
APIs that are logically distinct from plotting utilities or simulation drivers.
"""
from .bifurcation import BifurcationResult  # re-export

__all__ = ["BifurcationResult"]
