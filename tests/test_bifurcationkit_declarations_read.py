"""Every setting a BifurcationKit continuation declares is read or refused.

The adapter read a discretization's ``mesh_intervals`` and ``degree`` only as attributes, so the same two declared as parameters, the slot ``mesh_adaptation`` is read from, were dropped: the curated Generic2dOscillator reference continuation declared a mesh of 20 and rendered 50, and the COModel docs page declared degree 3 and rendered 4. An ``options`` key the adapter does not read was dropped the same way: Cortes2013 declared ``mesh_adaptation`` as a discretization option and ran on a fixed mesh, and the reference continuation declared its ``linear_solver`` as a branch option. The adapter now reads both mesh settings from the parameters, else from the attribute, and refuses an ``options`` key it does not read, naming it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tvbo
from tvbo import SimulationExperiment
from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

DATABASE = Path(tvbo.__file__).parent / "database"
JANSENRIT = DATABASE / "experiments" / "bifurcation" / "JansenRit-bifurcation.yaml"
GENERIC2D_FULL = DATABASE / "continuations" / "Generic2dOscillator-bifurcation-full.yaml"


def _jansenrit(tmp_path, edit) -> str:
    """The curated Jansen-Rit continuation with *edit* applied to its parsed ``eq_in_p``, rendered for BifurcationKit."""
    spec = yaml.safe_load(JANSENRIT.read_text())
    edit(spec["continuations"]["eq_in_p"])
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return BifurcationKitAdapter(SimulationExperiment.from_file(str(path))).render_code()


def _orbit(cont) -> dict:
    return cont["branches"]["po_alpha_rhythm"]


def _discretization(cont) -> dict:
    return _orbit(cont)["discretization"]


@pytest.mark.parametrize(
    ("method", "rendered"),
    [("collocation", "PeriodicOrbitOCollProblem(60, 5)"), ("trapezoid", "PeriodicOrbitTrapProblem(M = 60)")],
)
def test_a_mesh_declared_as_parameters_reaches_the_rendered_problem(tmp_path, method, rendered):
    def edit(cont):
        _discretization(cont).update(method=method, parameters={"mesh_intervals": {"value": 60}, "degree": {"value": 5}})

    assert rendered in _jansenrit(tmp_path, edit)


def test_an_attribute_sets_the_mesh_where_no_parameter_does(tmp_path):
    def edit(cont):
        _discretization(cont).update(mesh_intervals=30, degree=3, parameters={})

    assert "PeriodicOrbitOCollProblem(30, 3)" in _jansenrit(tmp_path, edit)


def test_the_comodel_docs_list_of_parameters_reaches_the_rendered_problem(tmp_path):
    def edit(cont):
        _discretization(cont)["parameters"] = [
            {"name": "mesh_intervals", "value": 50},
            {"name": "degree", "value": 3},
            {"name": "mesh_adaptation", "value": True},
        ]

    assert "PeriodicOrbitOCollProblem(50, 3, meshadapt = true)" in _jansenrit(tmp_path, edit)


def test_mesh_adaptation_moved_into_the_parameters_renders_an_adaptive_mesh(tmp_path):
    """Cortes2013's discretization as it reads after the move."""

    def edit(cont):
        _orbit(cont)["discretization"] = {
            "method": "collocation",
            "mesh_intervals": 50,
            "degree": 4,
            "parameters": {"mesh_adaptation": {"value": True}},
        }

    assert "PeriodicOrbitOCollProblem(50, 4, meshadapt = true)" in _jansenrit(tmp_path, edit)


UNREAD = {
    "continuation": (lambda cont: cont, "linear_solver", "continuation 'eq_in_p'", "'tangent'"),
    "branch": (_orbit, "linear_solver", "branch 'po_alpha_rhythm'", "no option"),
    "branch continuation": (
        lambda cont: _orbit(cont)["continuation"],
        "linear_solver",
        "the continuation of branch 'po_alpha_rhythm'",
        "'tangent'",
    ),
    "discretization": (_discretization, "mesh_adaptation", "the discretization of branch 'po_alpha_rhythm'", "'jacobian'"),
    "codim-2 branch": (lambda cont: cont["branches"]["fold_p_C"], "linear_solver", "branch 'fold_p_C'", "no option"),
}
"""Per declaration: where the option goes, its key, how the refusal names the declaration, and the keys read there."""


@pytest.mark.parametrize("slot", list(UNREAD))
def test_an_option_the_backend_does_not_read_is_refused_by_name(tmp_path, slot):
    target, key, label, read = UNREAD[slot]

    def edit(cont):
        target(cont).setdefault("options", []).append({"name": key, "value": "true"})

    with pytest.raises(ValueError) as refused:
        _jansenrit(tmp_path, edit)
    assert f"{label} declares the option {key!r}, which the bifurcationkit backend" in str(refused.value)
    assert f"it reads {read} there" in str(refused.value)


def test_the_reference_continuation_renders_every_setting_it_declares():
    spec = yaml.safe_load(GENERIC2D_FULL.read_text())
    experiment = SimulationExperiment(
        dynamics={"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}, continuations={spec["name"]: spec}
    )
    code = BifurcationKitAdapter(experiment).render_code()
    assert "PeriodicOrbitOCollProblem(20, 4, meshadapt = false, jacobian = BifurcationKit.DenseAnalyticalInplace())" in code
    assert "linear_algo = COPBLS()" in code
