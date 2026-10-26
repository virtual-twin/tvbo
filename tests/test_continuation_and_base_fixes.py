"""What the shared adapter layer hands the Julia and continuation templates: the connectome's orientation, the dynamics a continuation renders on, and the solver an integration method lowers to.

`BaseAdapter.build_weight_matrix` emitted an explicit-edge network source-by-target but a matrix-only one as `Network.matrix` stores it, target-by-source, so a directed asymmetric connectome ran transposed on NetworkDynamics.jl. `ContinuationAdapter.render_code` rendered the experiment's dynamics while `run` continued the one the continuation names. A canonical integration method reached the Julia templates under its tvbo name, which is not a DifferentialEquations.jl solver, and a fixed-step method was integrated adaptively. An integration slot written ``None`` failed on ``float(None)``, and an absent integration fell back to a step and method the schema does not declare. NumCont read a branch as two-parameter only when its nested continuation listed two parameters, while BifurcationKit and PyRates read any nested parameter as the second, so one spec ran as a periodic-orbit branch on one backend and a codim-2 curve on the others.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from tvbo import SimulationExperiment
from tvbo.adapters.base import BaseAdapter, ContinuationAdapter
from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

DYNAMICS = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}
COUPLING = {"Linear": {"name": "Linear", "iri": "tvbo:Linear"}}


def _experiment(network, method="Heun"):
    return SimulationExperiment(
        dynamics=DYNAMICS, network=network, integration={"method": method, "step_size": 0.1, "duration": 1.0}
    )


def _emitted_weights(experiment) -> np.ndarray:
    """The ``W = [...]`` literal the NetworkDynamics.jl source carries, as rows."""
    line = next(line for line in experiment.render_code(format="networkdynamics").splitlines() if line.startswith("W = ["))
    rows = line.removeprefix("W = [").removesuffix("]").split("; ")
    return np.array([[float(v) for v in row.split()] for row in rows])


# ── Weight-matrix orientation ─────────────────────────────────────────────


def test_a_matrix_only_network_is_emitted_source_by_target():
    """`Network.matrix` stores an edge ``i → j`` at ``[j, i]``; ``SimpleWeightedDiGraph(W)`` reads it at ``[i, j]``."""
    stored = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
    experiment = _experiment({"number_of_nodes": 3, "coupling": COUPLING})
    experiment.network.set_matrix("weight", stored)

    emitted = _emitted_weights(experiment)

    for source in range(3):
        for target in range(3):
            assert emitted[source, target] == stored[target, source]


def _explicit_network(n_nodes=10, ids=None):
    """A directed network with more explicit edges than the listing threshold, weighted through both the `weight` slot and `parameters`, over nodes whose ids are not their positions."""
    ids = ids or [10 * k + 3 for k in range(n_nodes)]
    pairs = [(s, t) for s in range(n_nodes) for t in range(n_nodes) if s != t and (s + 2 * t) % 3]
    edges = []
    for k, (s, t) in enumerate(pairs):
        edge = {"source": ids[s], "target": ids[t], "directed": True}
        weight = float(1 + (7 * s + 3 * t) % 5)
        if k % 2:
            edge["weight"] = weight
        else:
            edge["parameters"] = {"weight": {"name": "weight", "value": weight}}
        edges.append(edge)
    network = {"number_of_nodes": n_nodes, "nodes": [{"id": i} for i in ids], "edges": edges, "coupling": COUPLING}
    return network, {(s, t): float(1 + (7 * s + 3 * t) % 5) for s, t in pairs}


def test_explicit_edges_over_the_threshold_are_emitted_by_position_and_declared_weight():
    network, weights = _explicit_network()
    assert len(weights) > 50
    emitted = _emitted_weights(_experiment(network))

    expected = np.zeros((10, 10))
    for (s, t), w in weights.items():
        expected[s, t] = w
    np.testing.assert_array_equal(emitted, expected)


def test_explicit_edges_and_their_matrix_form_emit_the_same_weights():
    network, _ = _explicit_network()
    explicit = _experiment(network)
    matrix_form = _experiment({"number_of_nodes": 10, "coupling": COUPLING})
    matrix_form.network.set_matrix("weight", explicit.network.matrix("weight", format="dense"))

    np.testing.assert_array_equal(_emitted_weights(explicit), _emitted_weights(matrix_form))


def test_listed_edges_address_rows_not_node_ids():
    """Under the threshold a template lists the edges, by the rows the matrix would use."""
    network = {
        "number_of_nodes": 3,
        "nodes": [{"id": 5}, {"id": 7}, {"id": 9}],
        "edges": [{"source": 5, "target": 7, "directed": True}, {"source": 9, "target": 5, "directed": True}],
        "coupling": COUPLING,
    }
    code = _experiment(network).render_code(format="networkdynamics")
    assert re.findall(r"add_edge!\(g, (\d+), (\d+)\)", code) == [("1", "2"), ("3", "1")]


def test_a_template_edge_is_not_listed_as_a_connection():
    """An edge without endpoints names a companion matrix; the matrix is emitted, not the edge."""
    stored = np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    network = {"number_of_nodes": 3, "edges": [{"label": "weight", "format": "dense", "directed": True}], "coupling": COUPLING}
    experiment = _experiment(network)
    experiment.network.set_matrix("weight", stored)

    assert NetworkDynamicsAdapter(experiment).listed_edges() == []
    np.testing.assert_array_equal(_emitted_weights(experiment), stored.T)
    assert "add_edge!" not in experiment.render_code(format="networkdynamics")


# ── Continuation dynamics ─────────────────────────────────────────────────


class _Recorder(ContinuationAdapter):
    """A continuation backend that records the pair it is handed instead of rendering or running it."""

    def render_continuation(self, model, continuation, **kwargs):
        return model

    def run_one(self, model, continuation, name, **kwargs):
        return model


def _dynamics(name):
    """A dynamics stand-in declaring the one parameter ``x`` a `_continuation` frees."""
    return SimpleNamespace(name=name, parameters={"x": SimpleNamespace(domain=SimpleNamespace(lo=0.0, hi=1.0))})


def _continuation(dynamics=None):
    """A continuation stand-in freeing ``x`` on *dynamics*."""
    return SimpleNamespace(dynamics=dynamics, free_parameters=[SimpleNamespace(name="x", domain=None)])


def test_render_code_renders_the_dynamics_run_continues():
    experiment_dynamics, named = _dynamics("A"), _dynamics("B")
    experiment = SimpleNamespace(
        dynamics=experiment_dynamics,
        network=SimpleNamespace(dynamics={"B": named}),
        continuations={"c": _continuation("B")},
    )
    adapter = _Recorder(experiment)
    assert adapter.render_code() is named
    assert adapter.run() is named


def test_an_explicit_model_still_wins():
    experiment = SimpleNamespace(dynamics=_dynamics("A"), network=None, continuations={"c": _continuation()})
    explicit = _dynamics("X")
    assert _Recorder(experiment).render_code(model=explicit) is explicit


# ── Integration method → Julia solver ─────────────────────────────────────


def test_the_adapter_canonicalises_through_the_one_table(monkeypatch):
    """No registry scan: `tvbo.utils.integration_method` is the one canonicaliser."""
    import tvbo.data.registry

    monkeypatch.setattr(tvbo.data.registry, "list_entries", lambda *a, **k: pytest.fail("scanned the registry"))
    assert BaseAdapter.canonical_integration_method("rk4") == "RungeKutta4thOrder"
    assert BaseAdapter.canonical_integration_method("tsit5") == "Tsit5"
    assert BaseAdapter.canonical_integration_method("AutoTsit5") == "AutoTsit5"


def test_the_fixed_step_methods_are_canonical_names():
    from tvbo.utils import INTEGRATION_METHODS

    assert BaseAdapter.FIXED_STEP_METHODS <= set(INTEGRATION_METHODS)
    assert BaseAdapter.is_fixed_step("rk4") and BaseAdapter.is_fixed_step("euler")
    assert not BaseAdapter.is_fixed_step("Dopri5")


def test_every_canonical_method_has_a_julia_solver_and_its_package():
    from tvbo.adapters.julia_model import JULIA_SOLVER_PACKAGES, JULIA_SOLVERS
    from tvbo.utils import INTEGRATION_METHODS

    assert set(JULIA_SOLVERS) == set(INTEGRATION_METHODS)
    assert set(JULIA_SOLVERS.values()) <= set(JULIA_SOLVER_PACKAGES)


@pytest.mark.parametrize(
    ("declared", "solver", "fixed"),
    [
        ("rk4", "RK4", True),
        ("RungeKutta4thOrder", "RK4", True),
        ("Heun", "Heun", True),
        ("euler", "Euler", True),
        ("Identity", "FunctionMap", True),
        ("Dopri5", "DP5", False),
        ("Dopri853", "DP8", False),
        ("VODE", "VCABM", False),
        ("Tsit5", "Tsit5", False),
        ("AutoTsit5", "AutoTsit5", False),
    ],
)
@pytest.mark.parametrize("fmt", ["networkdynamics", "modelingtoolkit"])
def test_the_julia_solve_call_names_a_differentialequations_solver(declared, solver, fixed, fmt):
    """A fixed-step method is integrated at the declared step: DifferentialEquations.jl's `Heun` and `RK4` adapt unless told not to."""
    network = (
        {"number_of_nodes": 2, "edges": [{"source": 0, "target": 1}], "coupling": COUPLING}
        if fmt == "networkdynamics"
        else None
    )
    experiment = SimulationExperiment(
        dynamics=DYNAMICS, network=network, integration={"method": declared, "step_size": 0.1, "duration": 1.0}
    )
    solve = next(line for line in experiment.render_code(format=fmt).splitlines() if line.startswith("sol = solve("))
    assert re.search(rf"\b{solver}\(", solve), solve
    assert ("dt=0.1, adaptive=false" in solve) == fixed, solve


# ── Integration defaults ──────────────────────────────────────────────────


def _schema_window():
    """The window the schema's `Integrator` defaults (`ifabsent`) describe."""
    from tvbo.datamodel.schema import Integrator

    schema = Integrator()
    return schema.step_size, schema.duration, schema.transient_time, BaseAdapter.canonical_integration_method(schema.method)


def test_an_integration_slot_written_none_reads_as_the_schema_default():
    experiment = _experiment({"number_of_nodes": 2, "coupling": COUPLING})
    for slot in ("step_size", "duration", "transient_time", "method"):
        setattr(experiment.integration, slot, None)
    window = BaseAdapter(experiment).get_integration_info()
    assert (window["dt"], window["duration"], window["transient_time"], window["method"]) == _schema_window()


def test_an_experiment_without_an_integration_integrates_the_schema_default():
    window = BaseAdapter(SimpleNamespace(integration=None)).get_integration_info()
    dt, duration, transient, method = _schema_window()
    assert (window["dt"], window["duration"], window["transient_time"], window["method"]) == (dt, duration, transient, method)
    assert window["total_duration"] == transient + duration


def test_an_undeclared_method_is_the_schema_default():
    assert BaseAdapter.canonical_integration_method(None) == _schema_window()[3] == "Heun"


# ── One codim-2 rule ──────────────────────────────────────────────────────


def _fp(name):
    return SimpleNamespace(name=name, domain=None)


@pytest.mark.parametrize(
    ("nested", "second"),
    [(["C"], "C"), (["p", "C"], "C"), ({"C": _fp("C")}, "C"), (["p"], None), ([], None), (None, None)],
)
def test_a_branch_is_codim2_when_its_continuation_frees_a_parameter_besides_the_primary(nested, second):
    """The second parameter alone or after the inherited primary; nothing, or the primary alone, is a periodic-orbit branch."""
    parent = SimpleNamespace(free_parameters=[_fp("p")])
    if isinstance(nested, list):
        nested = [_fp(name) for name in nested]
    branch = SimpleNamespace(continuation=None if nested is None else SimpleNamespace(free_parameters=nested))
    found = ContinuationAdapter.codim2_parameter(branch, parent)
    assert (None if found is None else found.name) == second
    assert ContinuationAdapter.is_codim2(branch, parent) is (second is not None)


def _codim2_experiment(nested):
    """Generic2dOscillator continued in ``I``, with a periodic-orbit branch and a Hopf curve whose continuation frees *nested*."""
    domains = {"I": {"lo": -10, "hi": 20}, "b": {"lo": -20, "hi": 0}}
    spec = {
        "name": "eq_in_I",
        "free_parameters": [{"name": "I", "domain": domains["I"]}],
        "branches": {
            "po_from_hopf": {"source_point": "hopf:all"},
            "hopf_curve_in_b": {
                "source_point": "hopf:all",
                "bothside": True,
                "continuation": {
                    "name": "c2_hopf_in_b",
                    "free_parameters": [{"name": n, "domain": domains[n]} for n in nested],
                    "max_steps": 500,
                    "ds": 0.01,
                },
            },
        },
    }
    return SimulationExperiment(dynamics=DYNAMICS, continuations={"eq_in_I": spec})


def test_bifurcationkit_renders_both_spellings_of_a_codim2_branch_alike():
    from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

    alone, after_primary = (BifurcationKitAdapter(_codim2_experiment(n)).render_code() for n in (["b"], ["I", "b"]))
    assert alone == after_primary
    assert "# Codim-2 branch: hopf_curve_in_b (hopf → b)" in alone
    assert "Codim-2 branch" not in BifurcationKitAdapter(_codim2_experiment(["I"])).render_code()


class _Bundle(list):
    """An AUTO result bundle stand-in: its branches' special-point labels, and a label lookup that returns the label."""

    def __call__(self, label):
        return label


@pytest.mark.parametrize("nested", [["b"], ["I", "b"]])
def test_auto_continues_both_spellings_in_the_primary_and_the_second(nested):
    from tvbo.adapters import numcont

    experiment = _codim2_experiment(nested)
    cont = experiment.continuations["eq_in_I"]
    assert numcont._po_branches(cont) == {"po_from_hopf": cont.branches["po_from_hopf"]}

    started = []
    auto = SimpleNamespace(run=lambda **kw: started.append(kw) or ["R"], sv=lambda *a: None, merge=lambda r: r)
    bundle = _Bundle([SimpleNamespace(labels=SimpleNamespace(by_label={"HB": {3: None}}))])
    kwargs_eq = {
        "EPSL": 1e-7,
        "EPSU": 1e-7,
        "ITNW": 5,
        "ITMX": 9,
        "EPSS": 1e-5,
        "RL0": -10.0,
        "RL1": 20.0,
        "DSMAX": 0.1,
        "DSMIN": 1e-6,
        "IADS": 1,
    }
    out = numcont.NumContAdapter(experiment)._run_codim2_branches(
        auto=auto, R_eq=bundle, cont=cont, fp_name="I", kwargs_eq=kwargs_eq, model=experiment.dynamics
    )
    # AUTO bounds the principal parameter by RL0/RL1 and the second by UZSTOP; the branch's `bothside` runs both directions.
    assert [(kw["data"], kw["ICP"], kw["RL0"], kw["RL1"], kw["UZSTOP"], kw["DS"]) for kw in started] == [
        ("HB1", ["I", "b"], -10.0, 20.0, {"b": [-20.0, 0.0]}, 0.01),
        ("HB1", ["I", "b"], -10.0, 20.0, {"b": [-20.0, 0.0]}, -0.01),
    ]
    assert [entry[:4] for entry in out] == [("hopf_curve_in_b_HB1", "hopf", "I", "b")]


def test_auto_leaves_a_branch_freeing_only_the_primary_to_the_periodic_orbits():
    from tvbo.adapters import numcont

    cont = _codim2_experiment(["I"]).continuations["eq_in_I"]
    auto = SimpleNamespace(run=lambda **kw: pytest.fail("restarted a one-parameter branch as codim-2"))
    numcont.NumContAdapter(SimpleNamespace())._run_codim2_branches(
        auto=auto, R_eq=_Bundle(), cont=cont, fp_name="I", kwargs_eq={}, model=None
    )


@pytest.mark.parametrize("nested", [["b"], ["I", "b"]])
def test_pyrates_continues_both_spellings_in_the_primary_and_the_second(nested):
    from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter
    from tvbo.codegen.pyrates import pyrates_names

    experiment = _codim2_experiment(nested)
    cont = experiment.continuations["eq_in_I"]
    branch = cont.branches["hopf_curve_in_b"]
    assert PyRatesBifurcationAdapter.is_codim2(branch, cont)
    assert not PyRatesBifurcationAdapter.is_codim2(cont.branches["po_from_hopf"], cont)

    started = []
    adapter = PyRatesBifurcationAdapter(experiment)
    adapter._find_special_points = lambda ode, name, kind: [f"{kind}1"]
    ode = SimpleNamespace(run=lambda **kw: started.append(kw) or (None, None))
    names = pyrates_names(experiment.dynamics, fortran=True)
    adapter._run_codim2_branch(
        ode,
        "param",
        branch,
        cont,
        names["I"],
        -10.0,
        20.0,
        names=names,
        model=experiment.dynamics,
        param_idx={names["b"]: 3},
    )
    assert [(kw["starting_point"], kw["ICP"], kw["RL0"], kw["RL1"], kw["UZSTOP"]) for kw in started] == [
        ("HB1", [names["I"], names["b"]], -10.0, 20.0, {names["b"]: [-20.0, 0.0]})
    ]
