"""Pre-warm juliacall during Docker build.

juliapkg downloads a compatible Julia binary on first import of juliacall
and sets up an isolated project environment. This script triggers that
full initialization so the Julia binary + all scientific packages are baked
into the image layer — avoiding re-download/re-compilation on every CI run.

Without a juliapkg.json in tvbo, scientific packages (NetworkDynamics etc.)
are not auto-declared, so we add them explicitly here.
"""

from juliacall import Main as jl  # bootstraps Julia binary + PythonCall via juliapkg

# Install all scientific packages used by tvbo Julia backends into the
# active juliapkg-managed environment, then precompile.
jl.seval("""
import Pkg
pkgs = ["Graphs", "NetworkDynamics", "OrdinaryDiffEqTsit5",
        "OrdinaryDiffEqSDIRK", "SimpleWeightedGraphs", "StochasticDiffEq",
        "BifurcationKit", "ModelingToolkit", "Plots"]
for p in pkgs
    Pkg.add(p)
end
Pkg.precompile()
""")
