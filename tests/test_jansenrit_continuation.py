"""The curated Jansen-Rit continuation, and the continuation defects it exposed on all three backends.

The spec named ``p`` and ``C``, parameters of ``tvbo:JansenRit1995``, while resolving to ``tvbo:JansenRit``, so BifurcationKit raised ``KeyError: 'p'`` and AUTO-07p ``Name 'p' not found``. Its warm-up ran at the model's own input, where the column oscillates, so Newton had no equilibrium to start from. PyRates compiled ``A`` and ``a`` into one symbol of case-insensitive Fortran. Its two-parameter curves used ``initial_state.method: from_branch``, which no backend implements, and ran as one-parameter continuations. Found alongside: a bound of ``0`` read as no bound, ``auto``/``auto-07p`` rendered PyRates code while running AUTO-07p, the PyRates script reloaded the model from the ontology and lost the experiment's overrides, and no backend started from the value the continuation declares for its parameter.
"""

from __future__ import annotations

import os
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
            ContinuationAdapter.source_point(branch, ""),
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


def test_a_bifurcationkit_codim2_branch_takes_the_parents_newton_options_and_nev(experiment):
    code = BifurcationKitAdapter(experiment).render_code()
    expected = "ContinuationPar(p_min = 10.0, p_max = 500.0, ds = 0.5, dsmax = 5.0, max_steps = 500, nev = 6, newton_options = NewtonPar(tol = 1e-09))"
    assert code.count(expected) == 2


def test_the_pyrates_continuation_uses_autos_own_newton_constants():
    """The time integration PyRates starts from leaves ITMX = 2 and NWTN = 2, with which AUTO labels neither fold."""
    adapter = PyRatesBifurcationAdapter(SimpleNamespace())
    fields = ("bothside", "ds", "ds_min", "ds_max", "max_steps", "tol_stability", "newton_tol", "newton_max_iterations")
    undeclared = SimpleNamespace(**dict.fromkeys(fields))
    assert {k: adapter._cont_to_auto_kwargs(undeclared, "p", 0, 1)[k] for k in ("ITNW", "ITMX", "NWTN")} == {
        "ITNW": 5,
        "ITMX": 9,
        "NWTN": 3,
    }
    declared = SimpleNamespace(**{**vars(undeclared), "newton_max_iterations": 12})
    assert {k: adapter._cont_to_auto_kwargs(declared, "p", 0, 1)[k] for k in ("ITNW", "ITMX", "NWTN")} == {
        "ITNW": 12,
        "ITMX": 12,
        "NWTN": 3,
    }


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


REFUSALS = {
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
        with pytest.raises(ValueError, match=f"continuation 'eq_in_p' (?s:.*){message}"):
            call()


# ── Bounds, start value, alias ────────────────────────────────────────────


@pytest.mark.parametrize(("lo", "hi", "bounds"), [(0, 0.5, (0.0, 0.5)), (-1, 0, (-1.0, 0.0)), (None, 0.5, (-20.0, 0.5))])
def test_a_bound_of_zero_is_a_bound(lo, hi, bounds):
    cont = SimpleNamespace(free_parameters=[SimpleNamespace(name="p", domain=SimpleNamespace(lo=lo, hi=hi))])
    found = PyRatesBifurcationAdapter(SimpleNamespace())._get_free_parameter(cont, SimpleNamespace(parameters={}))
    assert (found["p_min"], found["p_max"]) == bounds


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


@pytest.mark.slow
def test_auto07p_finds_both_folds_and_three_hopf_points(experiment, monkeypatch, tmp_path):
    _needs_auto()
    monkeypatch.chdir(tmp_path)
    result = experiment.run("auto-07p").continuations["eq_in_p"]
    _assert_near(_special(result, "fold"), [113.586, -41.301])
    _assert_near(_special(result, "hopf"), [-12.148, 89.829, 315.696])
    assert len(result.periodic_orbits) == 3
    curves = {curve._source_type: curve for curve in result.codim2_curves}
    assert set(curves) == {"fold", "hopf"} and all(len(curve.df) > 10 for curve in curves.values())
    # The fold curve runs through its cusp and stops where C leaves its domain, not where p would.
    _assert_near(_special(curves["fold"], "cusp"), [168.705])
    assert curves["fold"].df["param2"].max() == pytest.approx(500.0)


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
    curves = {curve._source_type: curve for curve in result.codim2_curves}
    assert set(curves) == {"fold", "hopf"} and all(len(curve.df) > 10 for curve in curves.values())


@pytest.mark.slow
@pytest.mark.julia
@pytest.mark.backend_julia
def test_bifurcationkit_finds_the_folds_the_hopf_points_and_the_orbit_between_two_of_them(experiment, monkeypatch, tmp_path):
    pytest.importorskip("juliacall")
    monkeypatch.chdir(tmp_path)
    result = experiment.run("bifurcationkit").continuations["eq_in_p"]
    # BifurcationKit labels the fold at p = 113.6 a branch point.
    _assert_near(_special(result, "fold") + _special(result, "bp"), [113.586, -41.301])
    # Detection level 2 flags a Hopf point at the step past its crossing, without bisecting to it.
    _assert_near(_special(result, "hopf"), [-12.148, 89.829, 315.696], tol=1.5)
    assert any(orbit.df["param"].min() < 95 and orbit.df["param"].max() > 310 for orbit in result.periodic_orbits)
    curves = {curve._source_type: curve for curve in result.codim2_curves}
    # A BifurcationKit curve holds the second parameter in param and the primary in param2.
    _assert_near(_special(curves["fold"], "cusp", "param2"), [168.705])
    _assert_near(_special(curves["hopf"], "bt", "param2"), [15.937])
