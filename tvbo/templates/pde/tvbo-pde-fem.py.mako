"""
Minimal FEM diffusion solver (implicit Euler) generated from a TVB-O experiment.
Relies only on attributes of the 'experiment' datamodel passed to render_template(experiment=...).

Requirements:
    pip install scikit-fem meshio nibabel numpy scipy matplotlib
"""

from typing import Optional, Callable
import os

import numpy as np
import meshio
import nibabel as nib

from skfem import InteriorBasis, asm, condense, solve
from skfem.element import ElementTriP1, ElementTetP1
from skfem.io import from_meshio as skfem_from_meshio
from skfem.models.poisson import mass, laplace

# Extract values directly from the experiment datamodel
<%
fd = experiment.field_dynamics
primary = fd.field
meshinfo = primary.mesh
solver = fd.solver
bcs = fd.boundary_conditions or []
integ = experiment.integration

# Dirichlet value (first matching BC), default 0.0
dir_val = 0.0
for bc in bcs:
    if str(bc.bc_type).lower() == 'dirichlet':
        try:
            dir_val = float(getattr(bc.value, 'righthandside', bc.value))
        except Exception:
            dir_val = 0.0
        break

# Sum diffusion coefficients of all operators (kept simple)
coeffs = []
for op in (fd.operators or []):
    try:
        coeffs.append(float(op.coefficient.value))
    except Exception:
        coeffs.append(1.0)
diff_coeff = sum(coeffs) if coeffs else 1.0
%>

UNKNOWN: str = ${repr(primary.label)}
ELEMENT_TYPE: str = ${repr(str(meshinfo.element_type))}
DATA_LOCATION: str = ${repr(str(meshinfo.dataLocation))}
DT: float = ${float(solver.dt)}
STEPS: int = ${int(round(integ.duration / solver.dt))}
U0: float = ${float(experiment.field_dynamics.field.initial_value)}
DIRICHLET_VALUE: float = ${dir_val}
DIFF_COEFF: float = ${diff_coeff}


def _load_mesh(data_location: str, element_type: str):
    """Load triangle/tetra mesh from a file path; if a prefix is present (e.g., 'gifti:path'), use the part after ':'.

    Notes for GIFTI:
      - Surface meshes are typically in files like *.surf.gii (contain coordinates + triangles).
      - Files like *.shape.gii hold scalar data on a reference surface and do NOT contain a mesh.
    """
    path = data_location.split(":", 1)[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    if path.lower().endswith(".gii"):
        gi = nib.load(path)
        triangles, vertices = gi.darrays
        faces = triangles.data
        coords = vertices.data
        m = meshio.Mesh(points=coords, cells=[("triangle", faces)])
    else:
        m = meshio.read(path)

    # Validate that we have triangle/tetra cells
    has_tri = any(c.type.startswith("triangle") for c in m.cells)
    has_tet = any(c.type.startswith("tetra") for c in m.cells)
    if not (has_tri or has_tet):
        raise RuntimeError(
            "Loaded file contains no triangle/tetra cells; if this is a GIFTI shape file (*.shape.gii), "
            "please provide a surface mesh (*.surf.gii) instead."
        )

    # Convert via skfem's meshio bridge
    sk_mesh = skfem_from_meshio(m)
    et = (element_type or "").lower()
    if et.startswith("tet") or (not et and has_tet):
        return sk_mesh, ElementTetP1()
    return sk_mesh, ElementTriP1()


def build():
    """Build solver from baked experiment values. Returns (solve_pde, visualize, meta)."""
    mesh, element = _load_mesh(DATA_LOCATION, ELEMENT_TYPE)
    basis = InteriorBasis(mesh, element)

    # Assemble once
    M = asm(mass, basis)
    K = asm(laplace, basis)
    A_full = M + DT * DIFF_COEFF * K

    # Boundary dofs: full boundary with constant Dirichlet value
    bdofs = basis.get_dofs(mesh.boundary_facets())
    x0_template = np.full(basis.N, float(DIRICHLET_VALUE))

    u0 = np.full(basis.N, float(U0))

    def solve_pde(
        steps: int = STEPS,
        save_timeseries: bool = False,
        outpath: Optional[str] = None,
        u0_override: Optional[np.ndarray] = None,
        source: Optional['np.ndarray | Callable[[int, float, np.ndarray], np.ndarray]'] = None,
    ):
        """Advance u by implicit Euler.

        Parameters
        - steps: number of timesteps
        - save_timeseries: store all states
        - outpath: optional .npz save path
        - u0_override: per-node initial condition overriding the default u0
        - source: either a constant per-node vector f, or a callable f(n, t, u) returning per-node forcing
                 (used as b_full = M@u + DT * M@f)
        Note: With U0=0, Dirichlet=0, and no source, the solution remains identically zero by design.
        """
        u = (u0_override.astype(float).copy() if u0_override is not None else u0.copy())
        U = np.zeros((steps + 1, basis.N), dtype=float) if save_timeseries else None
        if U is not None:
            U[0] = u

        for n in range(steps):
            if source is None:
                b_full = M @ u
            else:
                f = source(n, (n + 1) * DT, u) if callable(source) else source
                b_full = M @ u + DT * (M @ f)

            _out = condense(A_full, b_full, D=bdofs, x=x0_template)
            if isinstance(_out, tuple):
                if len(_out) == 2:
                    A, b = _out
                    u = solve(A, b)
                elif len(_out) == 3:
                    A, b, xc = _out
                    u = solve(A, b, xc)
                else:
                    A, b = _out[0], _out[1]
                    if len(_out) >= 3:
                        xc = _out[2]
                        u = solve(A, b, xc)
                    else:
                        u = solve(A, b)
            else:
                # Fallback to using as A, with original right-hand side
                A, b = _out, b_full
                u = solve(A, b)
            if U is not None:
                U[n + 1] = u

        if outpath:
            np.savez(outpath, u=u, U=U, dt=DT, nodes=mesh.p, cells=getattr(mesh, 't', None), unknown=UNKNOWN, steps=steps)
        return u, U

    def visualize(u: np.ndarray):
        import matplotlib.pyplot as plt
        from skfem.visuals.matplotlib import draw, plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        draw(mesh, ax=ax)
        plot(mesh, u, ax=ax, colorbar=True)
        ax.set_title(f"Solution {UNKNOWN}")
        plt.show()

    meta = dict(dt=DT, ndofs=basis.N, unknown=UNKNOWN, nodes=mesh.p, cells=getattr(mesh, 't', None))
    return solve_pde, visualize, meta


# Convenience: export a ready-to-use solver using the baked experiment
solve_pde, visualize, meta = build()
