"""The PDE/FEM backend integrates the equation that was declared.

The backend it replaces never parsed ``equation.rhs``: it summed ``operators[].coefficient``
into one scalar and assembled ``M + dt*D*K``, so a spec could print one equation and run
another — the exact failure a declarative framework exists to prevent. These tests pin the
operator to the declaration, and check the physics rather than the emitted text: a grep
over generated source cannot tell a Laplacian from its negative.
"""

import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("meshio")

from tvbo import SimulationExperiment
from tvbo.templates.pde.utils import FieldPlanError, field_assembly_plan


@pytest.fixture(scope="module")
def mesh_path(tmp_path_factory):
    import meshio
    from skfem import MeshTri

    m = MeshTri.init_symmetric().refined(3)
    path = tmp_path_factory.mktemp("mesh") / "unit_square.msh"
    meshio.Mesh(points=m.p.T, cells=[("triangle", m.t.T)]).write(str(path))
    return str(path)


def _experiment(mesh_path, state_variables, parameters, method="crank-nicolson", dt=0.01, duration=1.0, events=None):
    return SimulationExperiment(
        **{
            "label": "field test",
            "events": events or {},
            "field_dynamics": {
                "label": "field",
                "mesh": {"label": "sq", "element_type": "triangle", "mesh_file": mesh_path},
                "parameters": parameters,
                "state_variables": state_variables,
                "solver": {"label": "s", "discretization": "FEM", "method": method, "dt": dt},
            },
            "integration": {"duration": duration},
        }
    )


def _sv(name, rhs, initial_value=0.0, bcs=None):
    return {
        "name": name,
        "label": name,
        "initial_value": initial_value,
        "boundary_conditions": bcs or [],
        "equation": {"lhs": f"{name}_t", "rhs": rhs},
    }


def _gaussian(nodes):
    x, y = nodes[0], nodes[1]
    return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)


def test_diffusion_conserves_its_integral_on_a_closed_domain(mesh_path):
    """With no boundary constraint the natural BC is zero-flux, so ``int(u)`` is exactly
    invariant. The conserved quantity is ``1.M.u`` — the nodal sum is not conserved and
    checking it instead would report a spurious 2 % drift."""
    exp = _experiment(mesh_path, [_sv("u", "D * laplacian(u)")], {"D": {"name": "D", "value": 0.01}})
    ns = exp.execute("pde")
    meta = ns["meta"]
    u0 = _gaussian(meta["nodes"])
    _, U = ns["solve_pde"](steps=50, save_timeseries=True, u0_override=u0)

    M, one = meta["mass_matrix"], np.ones(meta["ndofs"])
    before, after = one @ (M @ U[0, 0]), one @ (M @ U[-1, 0])
    assert abs(after - before) / before < 1e-10
    assert U[-1, 0].max() < U[0, 0].max(), "the bump must spread"


def test_dirichlet_boundary_drains_the_field(mesh_path):
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u)", bcs=[{"label": "zero", "bc_type": "Dirichlet", "value": {"rhs": "0"}}])],
        {"D": {"name": "D", "value": 0.01}},
    )
    ns = exp.execute("pde")
    u0 = _gaussian(ns["meta"]["nodes"])
    _, U = ns["solve_pde"](steps=100, save_timeseries=True, u0_override=u0)
    assert U[-1, 0].max() < 0.5 * U[0, 0].max()
    assert np.isfinite(U).all()


def test_reaction_term_decays_at_the_declared_rate(mesh_path):
    """``- k*u`` is a term the old backend could not express at all: it only assembled a
    diffusion operator, so this equation would have run as pure diffusion."""
    k = 2.0
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u) - k * u")],
        {"D": {"name": "D", "value": 0.01}, "k": {"name": "k", "value": k}},
        dt=0.005,
        duration=1.0,
    )
    ns = exp.execute("pde")
    meta = ns["meta"]
    u0 = _gaussian(meta["nodes"])
    _, U = ns["solve_pde"](steps=200, save_timeseries=True, u0_override=u0)

    M, one = meta["mass_matrix"], np.ones(meta["ndofs"])
    ratio = (one @ (M @ U[-1, 0])) / (one @ (M @ U[0, 0]))
    assert ratio == pytest.approx(np.exp(-k), rel=2e-3)


DAMPED_WAVE = [
    _sv("phi", "w", initial_value=1.0),
    _sv("w", "g**2 * (-(2/g)*w - phi + laplacian(phi))"),
]
"""Pang2023 eq (9) in first-order form. For the spatially uniform mode the Laplacian
vanishes and the system is exactly critically damped: ``phi = phi0 (1 + g t) exp(-g t)``."""


@pytest.mark.parametrize("method,expected_order", [("crank-nicolson", 2.0), ("implicit Euler", 1.0)])
def test_damped_wave_matches_the_analytic_solution_at_the_stated_order(mesh_path, method, expected_order):
    """A second-order-in-time system, which the previous backend could not express: it
    read only ``state_variables[0]`` and assembled a single first-order equation."""
    g = 2.0
    errors = []
    for dt in (0.02, 0.01, 0.005):
        exp = _experiment(mesh_path, DAMPED_WAVE, {"g": {"name": "g", "value": g}}, method=method, dt=dt, duration=1.0)
        ns = exp.execute("pde")
        assert ns["meta"]["n_vars"] == 2
        _, U = ns["solve_pde"](steps=int(1.0 / dt), save_timeseries=True)
        t = np.arange(U.shape[0]) * dt
        errors.append(np.abs(U[:, 0, :].mean(axis=1) - (1 + g * t) * np.exp(-g * t)).max())

    order = np.log2(errors[1] / errors[2])
    assert order == pytest.approx(expected_order, abs=0.15), f"orders from {errors}"


def test_divergence_form_with_constant_coefficient_equals_a_plain_laplacian(mesh_path):
    """``div(c*grad(u))`` at constant ``c`` and ``c*laplacian(u)`` are the same operator,
    so they must agree to round-off — the check that the weighted assembly is not merely
    plausible but identical where the two forms coincide."""
    varying = _experiment(mesh_path, [_sv("u", "div(c * grad(u))")], {"c": {"name": "c", "value": None}})
    ns = varying.execute("pde")
    assert ns["meta"]["requires_fields"] == ["c"]
    solve_v, _, meta_v = ns["build"](fields={"c": 0.01})

    plain = _experiment(mesh_path, [_sv("u", "D * laplacian(u)")], {"D": {"name": "D", "value": 0.01}})
    ns_p = plain.execute("pde")

    u0 = _gaussian(meta_v["nodes"])
    _, U_v = solve_v(steps=50, save_timeseries=True, u0_override=u0)
    _, U_p = ns_p["solve_pde"](steps=50, save_timeseries=True, u0_override=u0)
    assert np.abs(U_v - U_p).max() < 1e-12


def test_spatially_varying_coefficient_still_conserves(mesh_path):
    """The divergence form is conservative whatever the coefficient does — that is the
    reason to prefer it over ``c(x)*laplacian(u)`` for a heterogeneous medium."""
    varying = _experiment(mesh_path, [_sv("u", "div(c * grad(u))")], {"c": {"name": "c", "value": None}})
    build = varying.execute("pde")["build"]
    _, _, meta = build(fields={"c": 0.01})
    x = meta["nodes"][0]
    solve, _, meta = build(fields={"c": 0.002 + 0.02 * (x < 0.5)})

    u0 = _gaussian(meta["nodes"])
    _, U = solve(steps=50, save_timeseries=True, u0_override=u0)
    M, one = meta["mass_matrix"], np.ones(meta["ndofs"])
    assert abs(one @ (M @ U[-1, 0]) - one @ (M @ U[0, 0])) < 1e-10
    assert U[-1, 0][x < 0.5].sum() > U[0, 0][x < 0.5].sum(), "fast side should gain"


@pytest.fixture(scope="module")
def producer_module(tmp_path_factory):
    """A study-side callable deriving a per-vertex coefficient, importable by bare name."""
    import sys

    directory = tmp_path_factory.mktemp("code")
    (directory / "pde_field_producer.py").write_text(
        "import numpy as np\n"
        "\n"
        "\n"
        "def split_medium(n_nodes, fast, slow):\n"
        '    """Coefficient field whose first half propagates faster than its second."""\n'
        "    out = np.full(int(n_nodes), float(slow))\n"
        "    out[: int(n_nodes) // 2] = float(fast)\n"
        "    return out\n"
    )
    sys.path.insert(0, str(directory))
    yield
    sys.path.remove(str(directory))


def test_a_produced_coefficient_field_needs_no_hand_off(mesh_path, producer_module):
    """A coefficient declaring a ``producer:`` is resolved at codegen, so the experiment runs
    unattended — ``run("pde")`` with nothing passed in.

    Without this the PDE backend could express a heterogeneous medium but not *declare* one:
    ``solve_pde`` was None whenever a field parameter existed, so every such experiment
    needed a Python caller to hand the array to ``build()``, which is the driver a
    declarative recipe exists to remove. The produced values are checked to reach the
    operator rather than merely to exist: an array that resolved but was dropped would leave
    this identical to the constant-coefficient run.
    """
    from tvbo.data.mesh_io import read_mesh

    n_nodes = len(read_mesh(mesh_path)[0])
    fast, slow = 0.02, 0.002
    produced = _experiment(
        mesh_path,
        [_sv("u", "div(c * grad(u))")],
        {
            "c": {
                "name": "c",
                "producer": {
                    "callable": {"name": "split_medium", "module": "pde_field_producer"},
                    "arguments": {"n_nodes": {"value": n_nodes}, "fast": {"value": fast}, "slow": {"value": slow}},
                },
            }
        },
    )
    ns = produced.execute("pde")
    assert ns["solve_pde"] is not None, "a produced field must not need build(fields=...)"

    u0 = _gaussian(ns["meta"]["nodes"])
    _, U = ns["solve_pde"](steps=50, save_timeseries=True, u0_override=u0)

    handed = _experiment(mesh_path, [_sv("u", "div(c * grad(u))")], {"c": {"name": "c", "value": None}}).execute("pde")
    coefficient = np.full(n_nodes, slow)
    coefficient[: n_nodes // 2] = fast
    solve_handed, _, _ = handed["build"](fields={"c": coefficient})
    _, U_handed = solve_handed(steps=50, save_timeseries=True, u0_override=u0)
    assert np.abs(U - U_handed).max() < 1e-14

    solve_flat, _, _ = handed["build"](fields={"c": slow})
    _, U_flat = solve_flat(steps=50, save_timeseries=True, u0_override=u0)
    assert np.abs(U - U_flat).max() > 1e-6, "the produced heterogeneity never reached the operator"


def test_a_field_coefficient_on_a_bare_laplacian_is_refused(mesh_path):
    """``c(x)*laplacian(u)`` has no exact FEM assembly. Refusing it names the divergence
    form instead of silently assembling something adjacent."""
    exp = _experiment(mesh_path, [_sv("u", "c * laplacian(u)")], {"c": {"name": "c", "value": None}})
    with pytest.raises(FieldPlanError, match="divergence form"):
        field_assembly_plan(exp)


def test_an_unimplemented_boundary_condition_raises(mesh_path):
    """A declared Neumann condition must not be silently dropped."""
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u)", bcs=[{"label": "flux", "bc_type": "Neumann", "value": {"rhs": "0"}}])],
        {"D": {"name": "D", "value": 0.01}},
    )
    with pytest.raises(FieldPlanError, match="not implemented"):
        field_assembly_plan(exp)


def test_the_equation_drives_the_operator_not_the_operators_list(mesh_path):
    """The regression that motivated the rewrite: changing only the coefficient in the
    EQUATION must change the result, even when ``operators:`` says something else."""
    plans = []
    for value in (0.01, 0.05):
        exp = _experiment(mesh_path, [_sv("u", "D * laplacian(u)")], {"D": {"name": "D", "value": value}})
        exp.field_dynamics.operators = []
        plans.append(field_assembly_plan(exp)["blocks"][0]["scalar"])
    assert plans == [0.01, 0.05]


def test_run_returns_a_timeseries_with_one_entry_per_field_variable(mesh_path):
    exp = _experiment(mesh_path, DAMPED_WAVE, {"g": {"name": "g", "value": 2.0}}, dt=0.01, duration=0.2)
    result = exp.run("pde")
    data = result.integration.data
    assert list(data.coords["variable"].values) == ["phi", "w"]
    assert data.sizes["time"] == 21


def test_a_constant_source_term_in_the_equation_is_actually_applied(mesh_path):
    """A term the block assembly cannot take (here a constant drive) is evaluated each
    step rather than dropped. Steady state of ``u_t = D*lap(u) - k*u + S`` is ``S/k``."""
    k, S = 2.0, 6.0
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u) - k * u + S")],
        {"D": {"name": "D", "value": 0.01}, "k": {"name": "k", "value": k}, "S": {"name": "S", "value": S}},
        dt=0.005,
        duration=5.0,
    )
    ns = exp.execute("pde")
    assert ns["meta"]["n_vars"] == 1
    _, U = ns["solve_pde"](steps=1000, save_timeseries=True)
    assert U[-1, 0].mean() == pytest.approx((S / k) * (1 - np.exp(-k * 5.0)), rel=1e-6)


V1_PULSE = {
    "Q": {
        "name": "Q",
        "event_type": "stimulus",
        "label": "1 ms pulse",
        "equation": {"rhs": "Piecewise((amplitude, (t >= t_on) & (t < t_off)), (0, True))"},
        "parameters": {
            "amplitude": {"name": "amplitude", "value": 20.0},
            "t_on": {"name": "t_on", "value": 0.001},
            "t_off": {"name": "t_off", "value": 0.002},
        },
    }
}
"""The paper's evoked drive: a 1 ms pulse, declared as an `events:` entry."""


def test_a_declared_stimulus_event_actually_drives_the_field(mesh_path):
    """`events:` is how a recipe declares a drive, so the backend must substitute it into
    the equation that names it. Dropping it leaves a field that is quiet, plausible and
    wrong — the failure mode that motivated parsing the RHS in the first place."""
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u) - k * u + Q")],
        {"D": {"name": "D", "value": 0.01}, "k": {"name": "k", "value": 2.0}},
        dt=0.0005,
        duration=0.01,
        events=V1_PULSE,
    )
    ns = exp.execute("pde")
    assert ns["meta"]["events"] == ["Q"]
    _, U = ns["solve_pde"](steps=20, save_timeseries=True)

    before, during, after = U[2, 0].max(), U[4, 0].max(), U[-1, 0].max()
    assert before == 0.0, "nothing may happen before the pulse opens"
    assert during == pytest.approx(20.0 * 0.001, rel=0.1), "the pulse integrates to A*(t_off-t_on)"
    assert 0 < after < during, "and decays once the pulse closes"


def test_an_event_redefining_a_model_parameter_is_refused(mesh_path):
    """Two declarations of one symbol is exactly the drift a single spec exists to prevent."""
    clashing = {"Q": {**V1_PULSE["Q"], "parameters": {"k": {"name": "k", "value": 99.0}}}}
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u) - k * u + Q")],
        {"D": {"name": "D", "value": 0.01}, "k": {"name": "k", "value": 2.0}},
        events=clashing,
    )
    with pytest.raises(FieldPlanError, match="redefines parameter"):
        field_assembly_plan(exp)


@pytest.fixture(scope="module")
def sphere_path(tmp_path_factory, icosphere):
    """A closed surface written out as a mesh file — the cortical case in miniature.

    Closed means no boundary facets at all, which is the geometry every brain-surface field
    equation is posed on and the one a planar test cannot exercise.
    """
    import meshio

    vertices, faces = icosphere(3, 50.0)
    path = tmp_path_factory.mktemp("sphere") / "sphere.msh"
    meshio.Mesh(points=vertices, cells=[("triangle", faces)]).write(str(path))
    return str(path)


def _wave_experiment(path, gamma, r_s, dt, duration):
    """Pang2023 eq (9) as a first-order system on a surface."""
    return _experiment(
        path,
        [_sv("phi", "w"), _sv("w", "gamma_s**2 * (-(2/gamma_s)*w - phi + r_s**2 * laplacian(phi))")],
        {"gamma_s": {"name": "gamma_s", "value": gamma}, "r_s": {"name": "r_s", "value": r_s}},
        dt=dt,
        duration=duration,
    )


def _modal_system(meta, gamma, r_s, n_modes=None):
    """The eigenbasis of the assembled operators, and each mode's damped-oscillator matrix.

    Solving ``K psi = lambda M psi`` with ``psi`` mass-orthonormal is what turns eq (9) into
    one ODE per mode: the Laplacian becomes multiplication by ``-lambda`` and the equations
    stop referring to each other.
    """
    from scipy.linalg import eigh

    evals, modes = eigh(meta["stiffness_matrix"].toarray(), meta["mass_matrix"].toarray())
    if n_modes is not None:
        evals, modes = evals[:n_modes], modes[:, :n_modes]
    jacobians = np.zeros((len(evals), 2, 2))
    jacobians[:, 0, 1] = 1.0
    jacobians[:, 1, 0] = -(gamma**2) * (1.0 + r_s**2 * evals)
    jacobians[:, 1, 1] = -2.0 * gamma
    return modes, jacobians


def _integrate_modes(modes, jacobians, meta, phi0, steps, theta=0.5):
    """Step every modal ODE with the same theta-scheme the field solver uses."""
    dt = meta["dt"]
    state = np.zeros((len(jacobians), 2))
    state[:, 0] = modes.T @ (meta["mass_matrix"] @ phi0)

    eye = np.eye(2)
    step = np.linalg.solve(eye - theta * dt * jacobians, eye + (1 - theta) * dt * jacobians)
    for _ in range(steps):
        state = np.einsum("kij,kj->ki", step, state)
    return modes @ state[:, 0]


def test_the_field_solution_is_exactly_its_modal_ode_system(sphere_path):
    """Pang2023 integrates ODEs, not the PDE it prints — and that is not an approximation.

    Expanding the field in the Laplace-Beltrami eigenbasis turns eq (9) into one damped
    oscillator per mode; with the FULL discrete basis the reformulation is exact, so the two
    solutions must agree to round-off rather than merely closely. Pinning it as an equality
    is what licenses reading the study's residual as truncation and nothing else: a
    reformulation that were only approximate would put its own error in that number.
    """
    gamma, r_s, dt, steps = 116.0, 28.9, 1e-4, 200
    ns = _wave_experiment(sphere_path, gamma, r_s, dt, steps * dt).execute("pde")
    meta = ns["meta"]

    nodes = meta["nodes"]
    phi0 = np.exp(-((nodes[0] - 50.0) ** 2 + nodes[1] ** 2 + nodes[2] ** 2) / 200.0)
    _, U = ns["solve_pde"](steps=steps, save_timeseries=True, u0_override=phi0)

    modes, jacobians = _modal_system(meta, gamma, r_s)
    reconstructed = _integrate_modes(modes, jacobians, meta, phi0, steps)

    scale = np.abs(U[-1, 0]).max()
    assert scale > 1e-3, "the field must actually have evolved"
    assert np.abs(U[-1, 0] - reconstructed).max() / scale < 1e-9


def test_truncating_the_mode_basis_is_what_makes_the_two_disagree(sphere_path):
    """The paper keeps 200 modes of a 32k-vertex surface and states no error for it. Here
    the same truncation is applied to a surface small enough to hold its whole basis, so
    the residual against the field solution is attributable: it falls monotonically as
    modes are added and reaches round-off only when the basis is complete."""
    gamma, r_s, dt, steps = 116.0, 28.9, 1e-4, 200
    ns = _wave_experiment(sphere_path, gamma, r_s, dt, steps * dt).execute("pde")
    meta = ns["meta"]

    nodes = meta["nodes"]
    phi0 = np.exp(-((nodes[0] - 50.0) ** 2 + nodes[1] ** 2 + nodes[2] ** 2) / 200.0)
    _, U = ns["solve_pde"](steps=steps, save_timeseries=True, u0_override=phi0)
    field = U[-1, 0]
    scale = np.abs(field).max()

    errors = []
    for n_modes in (10, 50, 200, meta["ndofs"]):
        modes, jacobians = _modal_system(meta, gamma, r_s, n_modes=n_modes)
        approx = _integrate_modes(modes, jacobians, meta, phi0, steps)
        errors.append(np.abs(field - approx).max() / scale)

    assert errors == sorted(errors, reverse=True), f"more modes must not be worse: {errors}"
    assert errors[0] > 1e-3, "10 modes should visibly miss a localised bump"
    assert errors[-1] < 1e-9, "the complete basis is the field solution itself"


def test_a_state_dependent_nonlinear_term_is_integrated_explicitly(mesh_path):
    """Logistic growth is nonlinear, so it cannot enter the implicit block; it is carried
    as the explicit remainder and must still reach the declared carrying capacity."""
    exp = _experiment(
        mesh_path,
        [_sv("u", "D * laplacian(u) + r*u*(1 - u)", initial_value=0.1)],
        {"D": {"name": "D", "value": 0.001}, "r": {"name": "r", "value": 3.0}},
        dt=0.002,
        duration=6.0,
    )
    ns = exp.execute("pde")
    _, U = ns["solve_pde"](steps=3000, save_timeseries=True)
    assert U[-1, 0].mean() == pytest.approx(1.0, rel=1e-3)
    assert U[0, 0].mean() == pytest.approx(0.1, rel=1e-9)
