"""
FEM field solver generated from a TVB-O experiment.

The operator assembled here is the one declared in `field_dynamics`: each state
variable's equation is parsed into mass and stiffness blocks, and the block system

    M du/dt = A u + M f(u, t)

is advanced implicitly. Terms linear in the state go into A and are solved implicitly
(unconditionally stable, so the step follows accuracy rather than a stability limit);
anything else is evaluated at the current state each step.

Requirements:
    pip install "tvbo[pde]"   # scikit-fem, meshio, nibabel
"""

from typing import Callable, Optional
import os

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from tvbo.data.mesh_fem import boundary_vertices, p1_mass, p1_stiffness
from tvbo.data.mesh_io import read_mesh
from tvbo.data.param_io import read_artifact

<%
from tvbo.templates.pde.utils import field_assembly_plan

plan = field_assembly_plan(experiment)
%>

VARIABLES: list = ${repr(plan['variables'])}
LABELS: list = ${repr(plan['labels'])}
N_VARS: int = ${len(plan['variables'])}
ELEMENT_TYPE: str = ${repr(plan['mesh']['element_type'])}
DATA_LOCATION: str = ${repr(plan['mesh']['path'])}
MESH_FORMAT: str = ${repr(plan['mesh']['format'])}
DT: float = ${plan['dt']}
STEPS: int = ${plan['steps']}
METHOD: str = ${repr(plan['method'])}
BLOCKS: list = ${repr(plan['blocks'])}
EXPLICIT: dict = ${repr(plan['explicit'])}
EVENTS: list = ${repr(plan['events'])}
PARAMETERS: dict = ${repr(plan['parameters'])}
FIELD_PARAMETERS: list = ${repr(plan['field_parameters'])}
FIELD_SOURCES: dict = ${repr(plan['field_sources'])}
DIRICHLET: list = ${repr(plan['boundary_conditions'])}
INITIAL: list = ${repr(plan['initial_values'])}


class _Discretisation:
    """A mesh's P1 operators and the vertices its boundary conditions can hold.

    ``mass`` and ``stiffness`` take an optional coefficient — ``None``, a scalar, or a
    per-vertex array — so a spatially varying propagation scale assembles as
    ``div(c grad(u))`` through the same call that builds the plain operator.
    """

    def __init__(self, nodes, cells, boundary, mass, stiffness):
        self.nodes = nodes
        self.cells = cells
        self.boundary = boundary
        self.mass = mass
        self.stiffness = stiffness
        self.n = int(nodes.shape[1])


def _mesh_location(data_location: str, mesh_format: str = ""):
    """``(path, format)`` from a mesh location that may carry a legacy ``format:`` prefix.

    ``dataLocation`` accepts ``gifti:some/file.gii``. A Windows drive letter or a URL scheme
    must not be read as one, so a prefix counts only when it is a plain word of more than one
    character — and an explicit ``mesh_format`` always wins over it.
    """
    path, prefix = data_location, ""
    head, separator, tail = data_location.partition(":")
    if separator and head.isalpha() and len(head) > 1:
        path, prefix = tail, head
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    return path, (mesh_format or prefix or None)


def _discretise(data_location: str, element_type: str, mesh_format: str = ""):
    """Read the declared mesh and assemble the P1 operators of its element type.

    Triangles are assembled in closed form, which is what makes a cortical surface usable:
    it is a 2-manifold carrying three coordinates, and a general FEM library inverts a
    square element Jacobian that such a triangle does not have. Tetrahedra, where element
    and coordinate dimension agree, go through scikit-fem.
    """
    if (element_type or "").lower().startswith("tet"):
        import meshio
        from skfem import Basis, BilinearForm, asm
        from skfem.element import ElementTetP1
        from skfem.helpers import dot, grad
        from skfem.io import from_meshio
        from skfem.models.poisson import laplace, mass

        basis = Basis(from_meshio(meshio.read(_mesh_location(data_location)[0])), ElementTetP1())

        @BilinearForm
        def weighted_mass(u, v, w):
            return w["c"] * u * v

        @BilinearForm
        def weighted_laplace(u, v, w):
            return w["c"] * dot(grad(u), grad(v))

        def _mass(coefficient=None):
            if coefficient is None:
                return asm(mass, basis)
            return weighted_mass.assemble(basis, c=basis.interpolate(coefficient))

        def _stiffness(coefficient=None):
            if coefficient is None:
                return asm(laplace, basis)
            return weighted_laplace.assemble(basis, c=basis.interpolate(coefficient))

        mesh = basis.mesh
        return _Discretisation(
            mesh.p, mesh.t,
            np.asarray(basis.get_dofs(mesh.boundary_facets()).flatten(), dtype=int),
            _mass, _stiffness,
        )

    vertices, faces = read_mesh(*_mesh_location(data_location, mesh_format))
    return _Discretisation(
        vertices.T, faces.T, boundary_vertices(faces),
        lambda coefficient=None: p1_mass(vertices, faces, coefficient),
        lambda coefficient=None: p1_stiffness(vertices, faces, coefficient),
    )


def _as_field(value, n_dofs):
    """A per-vertex array from a scalar or an array, with its length checked."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_dofs, float(arr))
    arr = arr.ravel()
    if arr.size != n_dofs:
        raise ValueError(f"coefficient field has {arr.size} values, expected {n_dofs}")
    return arr


def build(fields: Optional[dict] = None):
    """Assemble the block system and return ``(solve_pde, visualize, meta)``.

    The constraint set and the operator are both fixed, so the reduced left-hand side is
    factored once here and every timestep is a triangular solve. A closed surface has no
    boundary facets at all, so a declared Dirichlet condition constrains nothing and the
    unconstrained path is taken — that is the cortical case, and why mass is conserved
    there.

    ``meta`` carries the assembled operators back to the caller: checking a conservation
    law needs the mass matrix, since the conserved quantity of pure diffusion is
    ``1.M.u`` rather than the nodal sum, and validating the discretisation needs ``A``.

    Args:
        fields: values for every per-vertex coefficient named in ``FIELD_PARAMETERS``,
            as ``{name: array}``. A scalar is broadcast. Any coefficient that declared a
            ``source:``/``producer:`` is read from ``FIELD_SOURCES`` instead and need not be
            passed; an explicit entry still wins. Arrays are read at run time rather than
            baked into this file, because an operator the size of a cortical surface does
            not belong in generated source.
    """
    disc = _discretise(DATA_LOCATION, ELEMENT_TYPE, MESH_FORMAT)
    n = disc.n

    fields = dict(fields or {})
    for name, (path, key) in FIELD_SOURCES.items():
        if name not in fields:
            fields[name] = read_artifact(path, key)
    missing = [f for f in FIELD_PARAMETERS if f not in fields]
    if missing:
        raise ValueError(
            f"per-vertex coefficient(s) {missing} were declared without a scalar value and "
            f"without a `source:`/`producer:`, so they must be supplied to "
            f"build(fields=...); got {sorted(fields)}"
        )
    fields = {k: _as_field(v, n) for k, v in fields.items()}

    M = disc.mass()
    K = disc.stiffness()

    def _block(spec):
        """One term's contribution. Stiffness enters NEGATED: the weak form of the
        Laplacian is -integral(grad u . grad v), so `a*laplacian(u)` contributes -a*K."""
        scale = 1.0 if spec["scalar"] is None else float(spec["scalar"])
        if spec["kind"] == "mass":
            if spec["coefficient_field"]:
                return scale * disc.mass(fields[spec["coefficient_field"]])
            return scale * M
        if spec["weight_field"]:
            weight = fields.get(spec["weight_field"])
            if weight is None:
                weight = _as_field(PARAMETERS[spec["weight_field"]], n) \
                    if spec["weight_field"] in PARAMETERS else None
            if weight is None:
                raise ValueError(f"no value for coefficient {spec['weight_field']!r}")
            return -scale * disc.stiffness(weight)
        return -scale * K

    rows = [[None] * N_VARS for _ in range(N_VARS)]
    for spec in BLOCKS:
        contribution = _block(spec)
        current = rows[spec["row"]][spec["col"]]
        rows[spec["row"]][spec["col"]] = contribution if current is None else current + contribution
    for i in range(N_VARS):
        if rows[i][i] is None:
            rows[i][i] = sps.csr_matrix((n, n))
    A = sps.bmat(rows, format="csr")
    M_blk = sps.block_diag([M] * N_VARS, format="csr")

    theta = 0.5 if METHOD == "crank-nicolson" else 1.0
    lhs = (M_blk - theta * DT * A).tocsc()

    boundary = disc.boundary
    constrained = np.zeros(N_VARS * n)
    held = None
    if DIRICHLET and boundary.size:
        blocks = []
        for bc in DIRICHLET:
            blocks.append(boundary + bc["row"] * n)
            constrained[boundary + bc["row"] * n] = float(bc["value"])
        held = np.unique(np.concatenate(blocks))
    free = np.setdiff1d(np.arange(N_VARS * n), held) if held is not None \
        else np.arange(N_VARS * n)
    lhs_free = lhs[free][:, free].tocsc()
    coupling = lhs[free][:, held].tocsr() if held is not None else None
    factor = spla.splu(lhs_free)

    u_init = np.concatenate([np.full(n, float(v)) for v in INITIAL])

    explicit = None  # the remainder the block assembly could not take
    if EXPLICIT:
        import sympy as sp

        symbols = list(VARIABLES) + sorted(PARAMETERS) + list(FIELD_PARAMETERS) + ["t"]
        args = [sp.Symbol(s) for s in symbols]
        compiled = {
            VARIABLES.index(var): sp.lambdify(
                args, sp.sympify(expr, locals={s: sp.Symbol(s) for s in symbols}), "numpy")
            for var, expr in EXPLICIT.items()
        }

        def explicit(state, time):
            values = {v: state[i * n:(i + 1) * n] for i, v in enumerate(VARIABLES)}
            values.update({k: PARAMETERS[k] for k in sorted(PARAMETERS)})
            values.update({k: fields[k] for k in FIELD_PARAMETERS})
            values["t"] = time
            out = np.zeros(N_VARS * n)
            for row, fn in compiled.items():
                out[row * n:(row + 1) * n] += np.broadcast_to(
                    np.asarray(fn(*[values[s] for s in symbols]), dtype=float), (n,))
            return out

    def solve_pde(
        steps: int = STEPS,
        save_timeseries: bool = False,
        outpath: Optional[str] = None,
        u0_override: Optional[np.ndarray] = None,
        source: Optional['np.ndarray | Callable[[int, float, np.ndarray], np.ndarray]'] = None,
        t0: float = 0.0,
    ):
        """Advance the field system.

        Args:
            steps: number of timesteps.
            save_timeseries: keep every state, returned as ``(steps+1, n_vars, n_dofs)``.
            outpath: optional ``.npz`` destination.
            u0_override: initial condition, either ``(n_dofs,)`` for the first variable
                or ``(n_vars, n_dofs)`` for all of them.
            source: per-node forcing ``f``, constant or ``f(step, t, u)``, shaped like the
                state; enters as ``M @ f`` so it is a source on the equation's own scale.
            t0: simulated time the first step starts from. A long run is advanced in
                chunks — feeding each chunk's final state back as ``u0_override`` — and a
                declared stimulus is a function of absolute time, so a chunk that restarted
                the clock at zero would replay the drive once per chunk.
        """
        u = u_init.copy()
        if u0_override is not None:
            arr = np.asarray(u0_override, dtype=float)
            if arr.size == n:
                u[:n] = arr.ravel()
            elif arr.size == N_VARS * n:
                u = arr.ravel().copy()
            else:
                raise ValueError(
                    f"u0 has {arr.size} values; expected {n} or {N_VARS * n}")

        history = np.zeros((steps + 1, N_VARS, n)) if save_timeseries else None
        if history is not None:
            history[0] = u.reshape(N_VARS, n)

        for step in range(steps):
            rhs = M_blk @ u + (1.0 - theta) * DT * (A @ u)
            forcing = None
            if explicit is not None:
                forcing = explicit(u, t0 + step * DT)
            if source is not None:
                f = source(step, t0 + (step + 1) * DT, u) if callable(source) else source
                f = np.asarray(f, dtype=float).ravel()
                if f.size == n and N_VARS > 1:
                    f = np.concatenate([f, np.zeros((N_VARS - 1) * n)])
                forcing = f if forcing is None else forcing + f
            if forcing is not None:
                rhs = rhs + DT * (M_blk @ forcing)
            if held is None:
                u = factor.solve(rhs)
            else:
                u = constrained.copy()
                u[free] = factor.solve(rhs[free] - coupling @ constrained[held])
            if history is not None:
                history[step + 1] = u.reshape(N_VARS, n)

        if outpath:
            np.savez(outpath, u=u, U=history, dt=DT, nodes=disc.nodes,
                     cells=disc.cells, variables=np.array(VARIABLES), steps=steps)
        return u, history

    def visualize(u: np.ndarray):
        """Draw the first variable on the mesh, in the plane or on the surface itself."""
        import matplotlib.pyplot as plt

        values = np.asarray(u).ravel()[:n]
        x, y = disc.nodes[0], disc.nodes[1]
        triangles = disc.cells.T
        if disc.nodes.shape[0] > 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            art = ax.plot_trisurf(x, y, disc.nodes[2], triangles=triangles,
                                  cmap="viridis", linewidth=0, antialiased=False)
            art.set_array(values[triangles].mean(axis=1))
        else:
            fig, ax = plt.subplots()
            art = ax.tripcolor(x, y, triangles, values, shading="gouraud")
            ax.set_aspect("equal")
        fig.colorbar(art, ax=ax)
        ax.set_title(f"Solution {VARIABLES[0]}")
        plt.show()

    meta = dict(dt=DT, ndofs=n, n_vars=N_VARS, variables=list(VARIABLES),
                labels=list(LABELS), unknown=VARIABLES[0], method=METHOD, events=list(EVENTS),
                nodes=disc.nodes, cells=disc.cells,
                mass_matrix=M, stiffness_matrix=K, operator=A, block_mass=M_blk)
    return solve_pde, visualize, meta


_UNRESOLVED = [f for f in FIELD_PARAMETERS if f not in FIELD_SOURCES]

if _UNRESOLVED:
    solve_pde = visualize = None
    meta = dict(dt=DT, ndofs=None, n_vars=N_VARS, variables=list(VARIABLES),
                labels=list(LABELS), unknown=VARIABLES[0], method=METHOD,
                requires_fields=list(_UNRESOLVED))
else:
    solve_pde, visualize, meta = build()


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    _parser = argparse.ArgumentParser(description="Run PDE-FEM TVBO simulation")
    _parser.add_argument("-n", "--steps", type=int, default=STEPS,
                         help=f"number of timesteps (default {STEPS})")
    _parser.add_argument("-o", "--output", type=_Path, default=None,
                         help="Output .npz file (or directory)")
    _args = _parser.parse_args()

    if solve_pde is None:
        raise SystemExit(
            f"this field system needs per-vertex coefficients {_UNRESOLVED} that declare no "
            f"`source:`/`producer:`; import build(fields=...) instead of running the script "
            f"directly"
        )

    _outpath = None
    if _args.output is not None:
        _out = _args.output
        if _out.suffix != ".npz":
            _out.mkdir(parents=True, exist_ok=True)
            _outpath = str(_out / "result.npz")
        else:
            _out.parent.mkdir(parents=True, exist_ok=True)
            _outpath = str(_out)

    _u, _U = solve_pde(steps=_args.steps, save_timeseries=_outpath is not None,
                       outpath=_outpath)
    print(f"Done: u.shape={getattr(_u, 'shape', None)}, steps={_args.steps}")
    if _outpath:
        print(f"Wrote results to {_outpath}")
