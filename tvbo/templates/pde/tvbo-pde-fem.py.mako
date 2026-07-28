"""
Minimal FEM diffusion solver (implicit Euler) generated from a TVB-O experiment.
Relies only on attributes of the 'experiment' datamodel passed to render_template(experiment=...).

Requirements:
    pip install "tvbo[pde]"   # scikit-fem, meshio, nibabel
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

# Support both new (state_variables + fd.mesh) and old (fd.field) paths
if fd.state_variables:
    primary = fd.state_variables[0]
    meshinfo = fd.mesh or getattr(primary, 'mesh', None)
    primary_label = primary.label or primary.name
    u0_val = float(primary.initial_value) if primary.initial_value is not None else 0.0
    bcs = primary.boundary_conditions or fd.boundary_conditions or []
else:
    primary = fd.field
    meshinfo = primary.mesh or fd.mesh
    primary_label = primary.label
    u0_val = float(primary.initial_value)
    bcs = fd.boundary_conditions or []

# Resolve mesh file: prefer mesh_file, fall back to dataLocation
mesh_loc = getattr(meshinfo, 'mesh_file', None) or getattr(meshinfo, 'dataLocation', None) or ''
mesh_fmt = getattr(meshinfo, 'mesh_format', None) or ''

solver = fd.solver
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

# Sum diffusion coefficients of all operators.
# fd.parameters is a dict {name: Parameter}; op.coefficient may be a ParameterName ref.
# Build name->value lookup from fd.parameters.values().
param_values = {}
for p in (fd.parameters or {}).values() if hasattr(fd.parameters, 'values') else (fd.parameters or []):
    pname = getattr(p, 'name', None) or str(p)
    pval = getattr(p, 'value', None)
    if pval is not None:
        param_values[pname] = float(pval)

coeffs = []
for op in (fd.operators or []):
    c = op.coefficient
    # Try direct .value first (full Parameter object)
    val = getattr(c, 'value', None)
    if val is not None:
        coeffs.append(float(val))
    else:
        # Resolve by name from fd.parameters
        cname = getattr(c, 'name', None) or str(c)
        coeffs.append(param_values.get(cname, 1.0))
diff_coeff = sum(coeffs) if coeffs else 1.0
%>

UNKNOWN: str = ${repr(primary_label)}
ELEMENT_TYPE: str = ${repr(str(meshinfo.element_type))}
DATA_LOCATION: str = ${repr(mesh_loc)}
MESH_FORMAT: str = ${repr(mesh_fmt)}
DT: float = ${float(solver.step_size)}
STEPS: int = ${int(round(integ.duration / solver.step_size))}
U0: float = ${u0_val}
DIRICHLET_VALUE: float = ${dir_val}
DIFF_COEFF: float = ${diff_coeff}


def _load_mesh(data_location: str, element_type: str, mesh_format: str = ""):
    """Load triangle/tetra mesh from a file path.

    Supports prefix syntax (e.g., 'gifti:path/to/file.gii').
    Uses mesh_format for explicit format override; otherwise auto-detects from extension.
    """
    # Strip format prefix if present (e.g., 'gifti:path')
    if ":" in data_location and not os.path.isabs(data_location.split(":", 1)[-1]):
        prefix, path = data_location.split(":", 1)
        if not mesh_format:
            mesh_format = prefix
    else:
        path = data_location.split(":", 1)[-1] if ":" in data_location else data_location

    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    fmt = mesh_format.lower() if mesh_format else ""

    # Use explicit format or auto-detect from extension
    is_gifti = fmt in ("gifti", "gii") or path.lower().endswith(".gii")
    is_freesurfer = fmt == "freesurfer" or any(
        path.lower().endswith(s) for s in (".pial", ".white", ".inflated")
    )

    if is_gifti:
        gi = nib.load(path)
        # GIFTI standard: POINTSET (vertices) first, TRIANGLE second.
        # Use intent codes for robustness.
        coords, faces = None, None
        for da in gi.darrays:
            if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                coords = da.data
            elif da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
                faces = da.data
        if coords is None or faces is None:
            # Fallback: assume [vertices, triangles] order
            coords = gi.darrays[0].data
            faces = gi.darrays[1].data
        m = meshio.Mesh(points=coords, cells=[("triangle", faces)])
    elif is_freesurfer:
        coords, faces = nib.freesurfer.read_geometry(path)
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
    mesh, element = _load_mesh(DATA_LOCATION, ELEMENT_TYPE, MESH_FORMAT)
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

            u = solve(*condense(A_full, b_full, D=bdofs, x=x0_template))
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


# ---------------------------------------------------------------------------
# Standalone entry point (executed when this script is run directly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    _parser = argparse.ArgumentParser(description="Run PDE-FEM TVBO simulation")
    _parser.add_argument("-t", "--t-end", type=float, default=None,
                         help="End time of simulation (defaults to dt only)")
    _parser.add_argument("-o", "--output", type=_Path, default=None,
                         help="Output .npz file (or directory)")
    _args = _parser.parse_args()

    _outpath = None
    if _args.output is not None:
        _out = _args.output
        if _out.suffix != ".npz":
            _out.mkdir(parents=True, exist_ok=True)
            _outpath = str(_out / "result.npz")
        else:
            _out.parent.mkdir(parents=True, exist_ok=True)
            _outpath = str(_out)

    _u, _U = solve_pde(t_end=_args.t_end, outpath=_outpath)
    print(f"Done: u.shape={getattr(_u, 'shape', None)}, "
          f"steps={'-' if _U is None else _U.shape[0]}")
    if _outpath:
        print(f"Wrote results to {_outpath}")
