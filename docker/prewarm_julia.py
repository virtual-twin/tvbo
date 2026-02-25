"""Pre-warm juliacall during Docker build.

The Dockerfile installs tvbo from PyPI, which does not yet carry
tvbo/juliapkg.json. This script explicitly adds all Julia packages
into the juliacall-managed environment so they are baked into the
image layer — avoiding re-download and re-compilation at run time.

At run time (CI / user install from source), tvbo/juliapkg.json
declares the same packages declaratively and juliapkg resolves them
against the already-precompiled depot.
"""

from juliacall import Main as jl  # bootstraps Julia binary via juliapkg

jl.seval("""
import Pkg
for p in ["BifurcationKit", "DiffEqCallbacks", "Graphs", "ModelingToolkit",
          "NetworkDynamics", "OrdinaryDiffEqSDIRK", "OrdinaryDiffEqTsit5",
          "Plots", "SimpleWeightedGraphs", "StochasticDiffEq"]
    Pkg.add(p)
end
Pkg.precompile()
""")
