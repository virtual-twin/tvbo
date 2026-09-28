"""The curated Jansen-Rit continuation, and the continuation defects it exposed on all three backends.

The spec named ``p`` and ``C``, parameters of ``tvbo:JansenRit1995``, while resolving to ``tvbo:JansenRit``, so BifurcationKit raised ``KeyError: 'p'`` and AUTO-07p ``Name 'p' not found``. Its warm-up ran at the model's own input, where the column oscillates, so Newton had no equilibrium to start from. PyRates compiled ``A`` and ``a`` into one symbol of case-insensitive Fortran. Its two-parameter curves used ``initial_state.method: from_branch``, which no backend implements, and ran as one-parameter continuations. Found alongside: a bound of ``0`` read as no bound, ``auto``/``auto-07p`` rendered PyRates code while running AUTO-07p, the PyRates script reloaded the model from the ontology and lost the experiment's overrides, and no backend started from the value the continuation declares for its parameter.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import tvbo
from tvbo import SimulationExperiment
from tvbo.adapters.base import ContinuationAdapter
from tvbo.adapters.bifurcationkit import BifurcationKitAdapter
from tvbo.adapters.numcont import NumContAdapter
from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter, _pyrates_param_name
from tvbo.codegen.pyrates import PYRATES_REPL, operator_template, pyrates_names, to_pyrates_yaml_string

SPEC = Path(tvbo.__file__).parent / "database" / "experiments" / "bifurcation" / "JansenRit-bifurcation.yaml"
BACKENDS = (BifurcationKitAdapter, NumContAdapter, PyRatesBifurcationAdapter)
PARAMETERS = "A, B, C, a, b, v0, e0, r, p"


@pytest.fixture(scope="module")
def experiment():
    return SimulationExperiment.from_file(str(SPEC))


def _variant(tmp_path, edit) -> SimulationExperiment:
    """The curated spec with *edit* applied to its parsed YAML, loaded as the curated file is."""
    spec = yaml.safe_load(SPEC.read_text())
    edit(spec)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return SimulationExperiment.from_file(str(path))


def _eq_in_p(spec) -> dict:
    return spec["continuations"]["eq_in_p"]


# ── The spec ──────────────────────────────────────────────────────────────


def test_the_spec_continues_the_model_whose_parameters_it_frees(experiment):
    cont = experiment.continuations["eq_in_p"]
    assert experiment.dynamics.name == "JansenRit1995"
    assert list(experiment.continuations) == ["eq_in_p"]
    ContinuationAdapter.refuse_unrunnable(experiment.dynamics, cont, "eq_in_p")


def test_the_two_parameter_curves_are_branches_of_the_equilibrium_continuation(experiment):
    cont = experiment.continuations["eq_in_p"]
    read = {
        name: (
            ContinuationAdapter.source_point(branch, cont),
            getattr(ContinuationAdapter.codim2_parameter(branch, cont), "name", None),
        )
        for name, branch in cont.branches.items()
    }
    assert read == {"po_alpha_rhythm": (("HB", None), None), "fold_p_C": (("LP", 1), "C"), "hopf_p_C": (("HB", 1), "C")}


def test_the_warm_up_settles_on_an_equilibrium_at_the_declared_start(experiment):
    """The model's rates are per second, so ten time units settle it; at its own input p = 220 the column oscillates and no warm-up ends on an equilibrium."""
    from tvbo.adapters.numcont import _warmup_to_steady_state

    cont = experiment.continuations["eq_in_p"]
    for model, settles in (
        (ContinuationAdapter.start_dynamics(experiment.dynamics, cont), True),
        (experiment.dynamics, False),
    ):
        state = _warmup_to_steady_state(model, cont)
        rate = max(abs(float(v)) for v in model.execute(format="python")(state, 0.0))
        assert (rate < 1e-6) is settles, (model.parameters["p"].value, rate)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_every_backend_renders_the_spec(experiment, backend):
    assert backend(experiment).render_code()


def test_bifurcationkit_renders_the_two_curves_as_codim2(experiment):
    code = BifurcationKitAdapter(experiment).render_code()
    assert "# Codim-2 branch: fold_p_C (fold → C)" in code
    assert "# Codim-2 branch: hopf_p_C (hopf → C)" in code


def test_bifurcationkit_continues_every_parameter_of_a_branch_as_a_float(experiment):
    """The model declares ``C: 135``, an integer, and BifurcationKit's codim-2 step cannot mix it with a Float64 ``C``."""
    assert "p = merge(p, (p = 50.0, C = 135.0,))" in BifurcationKitAdapter(experiment).render_code()


def test_a_bifurcationkit_codim2_branch_takes_every_setting_it_leaves_unset_from_its_parent(experiment):
    code = BifurcationKitAdapter(experiment).render_code()
    expected = "ContinuationPar(p_min = 10.0, p_max = 500.0, ds = 0.5, dsmax = 5.0, max_steps = 500, dsmin = 0.0001, nev = 6, detect_bifurcation = 2, newton_options = NewtonPar(tol = 1e-09))"
    assert code.count(expected) == 2


def _hopf_curve_declares_newton_tol(spec):
    _eq_in_p(spec)["branches"]["hopf_p_C"]["continuation"]["newton_tol"] = 1e-8


def test_the_pyrates_continuation_uses_autos_own_newton_constants():
    """The time integration PyRates starts from leaves ITMX = 2 and NWTN = 2, with which AUTO labels neither fold."""
    adapter = PyRatesBifurcationAdapter(SimpleNamespace())
    fields = ("bothside", "ds", "ds_min", "ds_max", "max_steps", "tol_stability", "newton_tol", "newton_max_iterations")
    undeclared = SimpleNamespace(**dict.fromkeys(fields))
    keys = ("ITNW", "ITMX", "NWTN", "EPSL", "EPSU")
    assert {k: adapter._cont_to_auto_kwargs(undeclared, "p", 0, 1, bothside=False)[k] for k in keys} == {
        "ITNW": 5,
        "ITMX": 9,
        "NWTN": 3,
        "EPSL": 1e-7,
        "EPSU": 1e-7,
    }
    declared = SimpleNamespace(**{**vars(undeclared), "newton_max_iterations": 12})
    assert {k: adapter._cont_to_auto_kwargs(declared, "p", 0, 1, bothside=False)[k] for k in keys} == {
        "ITNW": 12,
        "ITMX": 12,
        "NWTN": 3,
        "EPSL": 1e-7,
        "EPSU": 1e-7,
    }


def test_the_pyrates_continuation_locates_special_points_whatever_the_stability_tolerance():
    """AUTO's EPSS is the accuracy a special point is located to, and ``tol_stability`` bounds an eigenvalue's real part; one does not set the other."""
    adapter = PyRatesBifurcationAdapter(SimpleNamespace())
    fields = ("bothside", "ds", "ds_min", "ds_max", "max_steps", "tol_stability", "newton_tol", "newton_max_iterations")
    undeclared = SimpleNamespace(**dict.fromkeys(fields))
    declared = SimpleNamespace(**{**vars(undeclared), "tol_stability": 1e-10})
    assert adapter._cont_to_auto_kwargs(declared, "p", 0, 1, bothside=False)["EPSS"] == 1e-6


# ── One meaning of newton_tol ─────────────────────────────────────────────


def _auto_runs(monkeypatch, experiment) -> list[dict]:
    """The keywords of every ``auto.run`` NumCont makes for *experiment*, against a stand-in AUTO whose equilibrium branch labels three Hopf points and two folds."""
    import sys

    from tvbo.adapters import numcont
    from tvbo.analysis.bifurcation import BifurcationResult

    class Bundle(list):
        """An AUTO result: its labelled special points, a label lookup returning the label, and no periodic solutions."""

        def __call__(self, label=None):
            return [] if label is None else label

    bundle = Bundle(
        [SimpleNamespace(labels=SimpleNamespace(by_label={"HB": dict.fromkeys((1, 2, 3)), "LP": dict.fromkeys((4, 5))}))]
    )
    calls = []
    fake = SimpleNamespace(
        run=lambda **kw: calls.append(kw) or bundle,
        sv=lambda *a: None,
        merge=lambda r: bundle,
        parseC=SimpleNamespace(parseC=dict),
    )
    monkeypatch.setitem(sys.modules, "auto", fake)
    monkeypatch.setattr("tvbo.utils.auto.check_auto_dir", lambda: None)
    monkeypatch.setattr(numcont, "_initial_state_dict", lambda model, cont: {1: 0.0})
    monkeypatch.setattr(BifurcationResult, "from_auto", classmethod(lambda cls, *a, **k: SimpleNamespace(periodic_orbits=[])))
    NumContAdapter(experiment).run()
    return calls


def test_newton_tol_sets_autos_convergence_criteria_on_every_auto_run(tmp_path, monkeypatch):
    """AUTO-07p ignored `newton_tol`: every run took EPSL = EPSU = 1e-7. Each branch now takes its own, else its parent's, as BifurcationKit's `NewtonPar` does."""
    calls = _auto_runs(monkeypatch, _variant(tmp_path, _hopf_curve_declares_newton_tol))
    runs = [(kw.get("data", "eq"), kw["EPSL"], kw["EPSU"], kw["ITNW"], kw["ITMX"]) for kw in calls]
    # Both directions of the equilibrium, the orbit from every Hopf point, and both directions of each curve.
    assert runs == [
        *[("eq", 1e-9, 1e-9, 5, 9)] * 2,
        *[(f"HB{k}", 1e-7, 1e-7, 5, 9) for k in (1, 2, 3)],
        *[("LP1", 1e-9, 1e-9, 5, 9)] * 2,
        *[("HB1", 1e-8, 1e-8, 5, 9)] * 2,
    ]
    assert [kw.get("JAC") for kw in calls[:2]] == [-1, -1]


def test_every_backend_hands_a_branch_its_own_newton_tol_else_its_parents(tmp_path):
    experiment = _variant(tmp_path, _hopf_curve_declares_newton_tol)
    cont = experiment.continuations["eq_in_p"]
    orbit, fold, hopf = (cont.branches[name] for name in ("po_alpha_rhythm", "fold_p_C", "hopf_p_C"))
    assert "NewtonPar(tol = 1e-07)" in BifurcationKitAdapter._prepare_branch(orbit, cont)["po_cp_args_str"]
    for branch, tol in ((fold, "1e-09"), (hopf, "1e-08")):
        codim2 = BifurcationKitAdapter._prepare_codim2_branch(branch, cont, experiment.dynamics, (-50.0, 400.0))
        assert f"NewtonPar(tol = {tol})" in codim2["codim2_cp_str"]
    pyrates = PyRatesBifurcationAdapter(experiment)
    for branch, tol in ((orbit, 1e-7), (fold, 1e-9), (hopf, 1e-8)):
        kwargs = pyrates._cont_to_auto_kwargs(branch.continuation, "p", 0, 1, parent=cont, bothside=False)
        assert (kwargs["EPSL"], kwargs["EPSU"]) == (tol, tol)
        assert ContinuationAdapter.auto_newton_constants(branch.continuation, cont)["EPSL"] == tol


BRANCH_STEPS = {
    "po_alpha_rhythm": {"ds": 0.5, "ds_min": 1e-4, "ds_max": 1.0, "max_steps": 200},
    "fold_p_C": {"ds": 0.5, "ds_min": 1e-4, "ds_max": 5.0, "max_steps": 500},
    "hopf_p_C": {"ds": 0.5, "ds_min": 1e-4, "ds_max": 5.0, "max_steps": 500},
}
"""Each branch's steps: its nested continuation's where it declares them, else the parent's ``ds: 0.5``, ``ds_min: 1e-4``, ``ds_max: 1.0``."""


def test_every_backend_hands_a_branch_its_parents_steps_where_it_declares_none(experiment, monkeypatch):
    """The orbit branch declares only ``max_steps`` and each curve leaves ``ds_min`` unset. AUTO-07p ran the orbit at 0.01, 1e-6 and 0.1 and PyRates at 0.01, 1e-8 and 0.1, BifurcationKit ran the curves at its own ``dsmin``, and only BifurcationKit's orbit, built on the parent's `ContinuationPar`, took the parent's."""
    cont = experiment.continuations["eq_in_p"]
    branches = {name: cont.branches[name] for name in BRANCH_STEPS}
    orbit = BifurcationKitAdapter._prepare_branch(branches["po_alpha_rhythm"], cont)["po_cp_args_str"]
    assert "ds = 0.5, dsmin = 0.0001, dsmax = 1.0, max_steps = 200" in orbit
    for name in ("fold_p_C", "hopf_p_C"):
        curve = BifurcationKitAdapter._prepare_codim2_branch(branches[name], cont, experiment.dynamics, (-50.0, 400.0))
        assert "ds = 0.5, dsmax = 5.0, max_steps = 500, dsmin = 0.0001" in curve["codim2_cp_str"]

    pyrates = PyRatesBifurcationAdapter(experiment)
    keys = {"ds": "DS", "ds_min": "DSMIN", "ds_max": "DSMAX", "max_steps": "NMX"}
    for name, steps in BRANCH_STEPS.items():
        kwargs = pyrates._cont_to_auto_kwargs(branches[name].continuation, "p", 0, 1, parent=cont, bothside=False)
        assert {slot: kwargs[key] for slot, key in keys.items()} == steps

    runs = _auto_runs(monkeypatch, experiment)
    orbit = next(kw for kw in runs if kw["IPS"] == 2)
    curve = next(kw for kw in runs if kw["ISW"] == 2 and kw["data"] == "LP1")
    for run, name in ((orbit, "po_alpha_rhythm"), (curve, "fold_p_C")):
        assert {slot: run[key] for slot, key in keys.items()} == BRANCH_STEPS[name]


# ── An analytic Jacobian where AUTO-07p would difference twice ────────────


def test_numcont_carries_the_jacobian_where_a_two_parameter_curve_needs_it(experiment):
    """AUTO-07p differences ``FUNC`` for its Jacobian, and a two-parameter curve differences that again: at ``newton_tol`` 1e-9 this spec's Hopf curve stopped within its first steps and its fold curve short of the Bogdanov-Takens point."""
    model = experiment.dynamics
    entry = re.compile(r"^ +DFDU\(\d \+ \d\*NDIM\) = ", re.M)
    source = NumContAdapter(experiment).render_continuation(model, experiment.continuations["eq_in_p"])
    assert "IF (IJAC == 0) RETURN" in source and len(entry.findall(source)) == 13
    assert "IJAC ==" not in NumContAdapter(experiment).render_continuation(model, None)


def test_the_numcont_jacobian_is_the_derivative_of_the_right_hand_side(experiment):
    import numpy as np
    import sympy as sp

    from tvbo.adapters.numcont import _param_values, _state_jacobian

    model = experiment.dynamics
    entries = _state_jacobian(model, experiment.continuations["eq_in_p"])
    values = {sp.Symbol(name): value for name, value in _param_values(model).items()}
    for name, derived in model.in_dependency_order("derived_parameters").items():
        values[sp.Symbol(name)] = model.symbolic_rhs(derived).xreplace(values)
    n = len(model.state_variables)
    matrix = sp.Matrix(n, n, lambda i, j: entries.get((i, j), 0)).xreplace(values)
    analytic = sp.lambdify([[sp.Symbol(name) for name in model.state_variables]], matrix)
    dfun = model.execute(format="python")
    x = np.array([0.1, 1.0, 12.0, 20.0, 15.0, -3.0])
    steps = 1e-6 * np.maximum(1.0, np.abs(x))
    differenced = np.column_stack(
        [
            (np.asarray(dfun(x + dx, 0.0)) - np.asarray(dfun(x - dx, 0.0))) / (2 * h)
            for h, dx in zip(steps, np.diag(steps), strict=True)
        ]
    )
    np.testing.assert_allclose(np.asarray(analytic(x), dtype=float), differenced, rtol=1e-6, atol=1e-4)


# ── One Fortran rename ────────────────────────────────────────────────────


def test_fortran_names_separate_case_twins_and_reserved_arguments():
    names = ContinuationAdapter.fortran_names(["A", "a", "B", "b", "u", "C1", "a"], reserved=("U",))
    assert names == {"A": "A", "a": "alow", "B": "B", "b": "blow", "u": "u_par", "C1": "C1"}


@pytest.mark.parametrize("names", [["AB", "Ab"], ["A", "a", "alow"]])
def test_fortran_names_refuse_twins_no_rename_separates(names):
    with pytest.raises(ValueError, match="case-insensitive Fortran"):
        ContinuationAdapter.fortran_names(names)


def test_pyrates_renames_the_case_twins_on_the_fortran_path_only(experiment):
    model = experiment.dynamics
    assert pyrates_names(model) is PYRATES_REPL
    names = pyrates_names(model, fortran=True)
    assert {key: names[key] for key in ("A", "a", "B", "b", "p", "C", "c_intercolumn")} == {
        "A": "A",
        "a": "alow",
        "B": "B",
        "b": "blow",
        "p": "p",
        "C": "C",
        "c_intercolumn": "c_intercolumn",
    }
    fortran, plain = operator_template(model, fortran=True), operator_template(model)
    assert {"alow", "blow"} <= set(fortran["variables"]) and not {"a", "b"} & set(fortran["variables"])
    assert {"a", "b"} <= set(plain["variables"]) and not {"alow", "blow"} & set(plain["variables"])
    assert "alow" in to_pyrates_yaml_string(model, fortran=True)
    assert "alow" not in to_pyrates_yaml_string(model)


def test_numcont_and_pyrates_rename_through_the_one_helper(experiment):
    model, cont = experiment.dynamics, experiment.continuations["eq_in_p"]
    replace = NumContAdapter._prepare_context(model, cont)["replace"]
    assert replace["a"] == pyrates_names(model, fortran=True)["a"] == _pyrates_param_name(model, "a") == "alow"
    assert _pyrates_param_name(model, "p") == "p"


def test_the_numcont_template_declares_the_renamed_symbols(experiment):
    code = NumContAdapter(experiment).render_code()
    assert "DOUBLE PRECISION A, B, C, alow, blow, v0, e0, r, p" in code


# ── Refusals, by name, before anything renders or runs ────────────────────


def _free(*names):
    return [{"name": name, "domain": {"lo": 0, "hi": 1}} for name in names]


def _unbounded(spec):
    _eq_in_p(spec)["branches"]["fold_p_C"]["continuation"]["free_parameters"][0].pop("domain")


REFUSALS = {
    "no free parameter": (lambda spec: _eq_in_p(spec).pop("free_parameters"), "frees no parameter"),
    "unbounded second": (_unbounded, "branch 'fold_p_C' continues 'C' with no lo or hi bound"),
    "codim-2 without source point": (
        lambda spec: _eq_in_p(spec)["branches"]["hopf_p_C"].pop("source_point"),
        "two-parameter branch 'hopf_p_C' declares no source_point",
    ),
    "periodic orbit from a fold": (
        lambda spec: _eq_in_p(spec)["branches"]["po_alpha_rhythm"].update(source_point="fold:1"),
        "periodic-orbit branch 'po_alpha_rhythm' declares source_point 'fold:1'",
    ),
    "bothside declared twice": (
        lambda spec: _eq_in_p(spec)["branches"]["fold_p_C"]["continuation"].update(bothside=False),
        "branch 'fold_p_C' declares bothside True and its continuation bothside False",
    ),
    "unknown primary": (
        lambda spec: _eq_in_p(spec).update(free_parameters=_free("mu")),
        rf"frees 'mu'.*its parameters are {PARAMETERS}\.",
    ),
    "unknown second": (
        lambda spec: _eq_in_p(spec)["branches"]["fold_p_C"]["continuation"].update(free_parameters=_free("J")),
        rf"frees 'J'.*its parameters are {PARAMETERS}\.",
    ),
    "from_branch": (
        lambda spec: _eq_in_p(spec)["initial_state"].update(method="from_branch"),
        "'from_branch', which no continuation backend implements",
    ),
    "a simulation's seed": (
        lambda spec: _eq_in_p(spec)["initial_state"].update(method="from_working_point"),
        "'from_working_point', which seeds a simulation, and a continuation starts by one of time_integration, given, newton",
    ),
    "given off an equilibrium": (
        lambda spec: _eq_in_p(spec)["initial_state"].update(method="given"),
        "starts by initial_state.method 'given' from a state of 'JansenRit1995' that is not an equilibrium",
    ),
    "two free parameters": (
        lambda spec: _eq_in_p(spec).update(free_parameters=_free("p", "C")),
        r"frees 2 parameters \(p, C\)",
    ),
}


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_spec_no_backend_runs_is_refused_by_name_before_rendering(tmp_path, backend, case):
    edit, message = REFUSALS[case]
    adapter = backend(_variant(tmp_path, edit))
    adapter.render_continuation = lambda *a, **k: pytest.fail("rendered a refused spec")
    adapter.run_one = lambda *a, **k: pytest.fail("ran a refused spec")
    for call in (adapter.render_code, adapter.run):
        with pytest.raises(ValueError, match=f"continuation 'eq_in_p'(?s:.*){message}"):
            call()


# ── Bounds, start value, alias ────────────────────────────────────────────


def _dynamics_bounding_p(lo, hi):
    return SimpleNamespace(parameters={"p": SimpleNamespace(domain=SimpleNamespace(lo=lo, hi=hi))})


@pytest.mark.parametrize(
    ("lo", "hi", "own", "bounds"),
    [
        (0, 0.5, (None, None), (0.0, 0.5)),
        (-1, 0, (-5, 5), (-1.0, 0.0)),
        (None, 0.5, (-3, None), (-3.0, 0.5)),
        (None, None, (0, 9), (0.0, 9.0)),
    ],
)
def test_a_bound_is_the_free_parameters_else_the_dynamics_and_zero_is_one(lo, hi, own, bounds):
    """BifurcationKit, AUTO-07p and PyRates filled a missing bound with nothing, ±10 and ±20."""
    fp = SimpleNamespace(name="p", domain=SimpleNamespace(lo=lo, hi=hi))
    assert ContinuationAdapter.parameter_bounds(fp, _dynamics_bounding_p(*own), "c") == bounds
    cont = SimpleNamespace(name="c", free_parameters=[fp])
    found = PyRatesBifurcationAdapter(SimpleNamespace())._get_free_parameter(cont, _dynamics_bounding_p(*own))
    assert (found["p_min"], found["p_max"]) == bounds


def test_a_bound_neither_declares_is_refused_by_name():
    fp = SimpleNamespace(name="p", domain=SimpleNamespace(lo=None, hi=4))
    with pytest.raises(ValueError, match=r"continuation 'c' continues 'p' with no lo bound"):
        ContinuationAdapter.parameter_bounds(fp, _dynamics_bounding_p(None, 9), "continuation 'c'")


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_every_backend_starts_from_the_declared_value(experiment, backend):
    adapter = backend(experiment)
    adapter.run_one = lambda model, cont, name, **kwargs: model
    assert adapter.run().parameters["p"].value == 50.0
    assert experiment.dynamics.parameters["p"].value == 220


def test_the_rendered_source_starts_from_the_declared_value(experiment, tmp_path):
    assert "p = merge(p, (p = 50.0, C = 135.0,))" in BifurcationKitAdapter(experiment).render_code()
    assert "    p: 50.0\n" in PyRatesBifurcationAdapter(experiment).render_code()
    undeclared = _variant(tmp_path, lambda spec: _eq_in_p(spec)["free_parameters"][0].pop("value"))
    assert "p = merge(p, (p = 220.0, C = 135.0,))" in BifurcationKitAdapter(undeclared).render_code()


def test_the_pyrates_script_carries_the_experiments_dynamics(tmp_path):
    overridden = _variant(tmp_path, lambda spec: spec["dynamics"].update(parameters={"C": {"value": 200.0}}))
    code = PyRatesBifurcationAdapter(overridden).render_code()
    assert "    C: 200.0\n" in code
    assert "from_ontology" not in code


def test_auto_renders_the_numcont_source_it_runs(experiment):
    from tvbo.export import resolve

    assert {resolve(key).key for key in ("auto", "auto-07p", "auto07p", "numcont")} == {"numcont"}
    assert not {"auto", "auto-07p"} & set(resolve("pyrates-bifurcation").aliases)
    code = experiment.render_code("auto-07p")
    assert code == NumContAdapter(experiment).render_code()
    assert code.lstrip().startswith("SUBROUTINE FUNC")


def test_the_formats_page_lists_every_built_in_format_with_its_aliases():
    """The page listed `auto` and `auto-07p` under PyRates after they moved to the `numcont` format."""
    import re

    from tvbo.export.formats import _BUILTINS

    page = (Path(tvbo.__file__).parents[1] / "docs" / "CLI" / "formats.qmd").read_text()
    listed = {}
    for key, notes in re.findall(r"^\| `([^`]+)` \| `[^`]+` \| [^|]+ \|([^|]*)\|$", page, re.MULTILINE):
        listed[key] = set(re.findall(r"`([^`]+)`", notes.split(";")[0]))
    assert listed == {fmt.key: set(fmt.aliases) for fmt in _BUILTINS}


# ── A BifurcationKit fold is a fold ───────────────────────────────────────


def _labelled(params, row, label="bp"):
    """The labels `_reclassify_folds` leaves on an equilibrium branch through *params* whose row *row* BifurcationKit labelled *label*."""
    import pandas as pd

    from tvbo.analysis.bifurcation import BifurcationResult

    df = pd.DataFrame({"param": params, "specialpoint": [label if i == row else None for i in range(len(params))]})
    result = BifurcationResult(df=df)
    result._reclassify_folds()
    return [label for label in result.df["specialpoint"] if label is not None]


@pytest.mark.parametrize(
    ("params", "row"),
    [
        ([110.0, 112.0, 113.58, 113.556, 112.0, 110.0], 3),
        ([110.0, 112.0, 113.556, 113.58, 112.0, 110.0], 2),
        ([110.0, 112.0, 113.58, 112.0, 110.0], 2),
    ],
    ids=["step past the turn", "step before it, on a reversed half", "at the turn"],
)
def test_a_bifurcationkit_branch_point_beside_a_parameter_turn_is_a_fold(params, row):
    """BifurcationKit labels the fold of JansenRit at p = 113.556 `bp`, the step past the turn at 113.586, where the parameter itself does not turn."""
    assert _labelled(params, row) == ["fold"]


def test_a_branch_point_the_parameter_passes_through_stays_one():
    assert _labelled([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3) == ["bp"]
    assert _labelled([1.0, 2.0, 3.0, 4.0, 3.0, 2.0], 1) == ["bp"]


# ── The continuation, run ─────────────────────────────────────────────────


def _special(result, kind, column="param") -> list[float]:
    """The *column* values of every special point of *kind* on *result*'s branch."""
    labels = result.df["specialpoint"].astype(str)
    return [float(p) for label, p in zip(labels, result.df[column], strict=True) if kind in label.split(",")]


def _assert_near(found, expected, tol=0.05):
    assert all(any(abs(f - e) < tol for f in found) for e in expected), (found, expected)


def _needs_auto():
    if not os.environ.get("AUTO_DIR"):
        pytest.skip("AUTO_DIR is not set")


def _curves(result) -> dict:
    """*result*'s codim-2 curves by source kind, each checked to carry the primary ``p`` in ``param`` and the second ``C`` in ``param2`` within its domain."""
    curves = {curve._source_type: curve for curve in result.codim2_curves}
    assert set(curves) == {"fold", "hopf"} and all(len(curve.df) > 10 for curve in curves.values())
    for curve in curves.values():
        assert (curve._ics_name, curve._fp2_name) == ("p", "C")
        assert 10 - 1e-6 <= curve.df["param2"].min() and curve.df["param2"].max() <= 500 + 1e-6
    return curves


@pytest.mark.slow
def test_auto07p_finds_both_folds_and_three_hopf_points(experiment, monkeypatch, tmp_path):
    _needs_auto()
    monkeypatch.chdir(tmp_path)
    result = experiment.run("auto-07p").continuations["eq_in_p"]
    _assert_near(_special(result, "fold"), [113.586, -41.301])
    _assert_near(_special(result, "hopf"), [-12.148, 89.829, 315.696])
    assert len(result.periodic_orbits) == 3
    curves = _curves(result)
    # The fold curve runs through its cusp and stops where C leaves its domain, not where p would.
    _assert_near(_special(curves["fold"], "cusp"), [168.705])
    assert curves["fold"].df["param2"].max() == pytest.approx(500.0)
    # With the analytic Jacobian both curves reach their Bogdanov-Takens point at the equilibrium's newton_tol.
    for curve in curves.values():
        _assert_near(_special(curve, "bt"), [15.937])


@pytest.mark.slow
@pytest.mark.backend_pyrates
def test_pyrates_finds_both_folds_and_three_hopf_points(experiment, monkeypatch, tmp_path):
    _needs_auto()
    pytest.importorskip("pycobi")
    monkeypatch.chdir(tmp_path)
    result = experiment.run("pyrates-bifurcation").continuations["eq_in_p"]
    _assert_near(_special(result, "fold"), [113.586, -41.301])
    _assert_near(_special(result, "hopf"), [-12.148, 89.829, 315.696])
    assert len(result.periodic_orbits) == 3
    curves = _curves(result)
    _assert_near(_special(curves["fold"], "cusp"), [168.705])
    assert curves["fold"].df["param2"].max() == pytest.approx(500.0)
    for curve in curves.values():
        _assert_near(_special(curve, "bt"), [15.937])


@pytest.mark.slow
@pytest.mark.julia
@pytest.mark.backend_julia
def test_bifurcationkit_finds_the_folds_the_hopf_points_and_the_orbit_between_two_of_them(experiment, monkeypatch, tmp_path):
    pytest.importorskip("juliacall")
    monkeypatch.chdir(tmp_path)
    result = experiment.run("bifurcationkit").continuations["eq_in_p"]
    # BifurcationKit labels the fold at p = 113.6 a branch point, at the step past the turn of p.
    _assert_near(_special(result, "fold"), [113.586, -41.301])
    assert not _special(result, "bp")
    # Detection level 2 flags a Hopf point at the step past its crossing, without bisecting to it.
    _assert_near(_special(result, "hopf"), [-12.148, 89.829, 315.696], tol=1.5)
    assert any(orbit.df["param"].min() < 95 and orbit.df["param"].max() > 310 for orbit in result.periodic_orbits)
    curves = _curves(result)
    _assert_near(_special(curves["fold"], "cusp"), [168.705])
    _assert_near(_special(curves["hopf"], "bt"), [15.937])
    # BifurcationKit continues a curve in C, so the finaliser stops it at the first step where p leaves its domain.
    assert all(curve.df["param"].iloc[1:-1].between(-50, 400).all() for curve in curves.values())
