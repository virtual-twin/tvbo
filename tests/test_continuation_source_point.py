"""A branch declares where it starts, and a backend that cannot honour that says so.

`Continuation.branches` accepts `source_point: "<kind>:<n>"`. The BifurcationKit codim-1 path emits a periodic-orbit continuation, which starts from a Hopf point; it has no equilibrium branch switching. Before, a `bp:` or `fold:` source was parsed for its index and its KIND was dropped, so the emitted Julia looked for Hopf points, found none where the declaration meant a pitchfork, and wrote a result with the branch simply missing — a spec that validates, a run that succeeds, and an output that is quietly short one branch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tvbo.adapters.bifurcationkit import BifurcationKitAdapter


def _branch(source_point, name="po"):
    """A minimal periodic-orbit BranchSwitch stand-in: only what `_prepare_branch` reads."""
    return SimpleNamespace(
        name=name, source_point=source_point, continuation=None, discretization=None, delta_p=None, bothside=None
    )


@pytest.mark.parametrize("source", ["hopf:all", "hopf:1", "hopf:-1", None])
def test_a_hopf_source_is_accepted(source):
    """The kinds this path actually emits, including the unset default."""
    assert BifurcationKitAdapter._prepare_branch(_branch(source))


@pytest.mark.parametrize("source", ["bp:1", "fold:2", "bp:all"])
def test_a_source_the_backend_cannot_start_from_is_refused(source):
    """Silently emitting a Hopf switch for a declared branch point is the failure this guards."""
    with pytest.raises(ValueError, match="source_point"):
        BifurcationKitAdapter._prepare_branch(_branch(source))


def test_the_refusal_names_the_branch_and_what_to_declare_instead():
    """A spec author has to be able to act on it without reading the adapter."""
    with pytest.raises(ValueError) as excinfo:
        BifurcationKitAdapter._prepare_branch(_branch("bp:1", name="nontrivial"))
    message = str(excinfo.value)
    assert "nontrivial" in message
    assert "hopf" in message
    assert "initial_state" in message


# ── One reading of `source_point` for every backend ───────────────────────


@pytest.mark.parametrize(
    ("declared", "parsed"),
    [
        ("hopf:all", ("HB", None)),
        ("hopf", ("HB", None)),
        ("HB:ALL", ("HB", None)),
        ("hopf:2", ("HB", 2)),
        ("hopf:-1", ("HB", -1)),
        ("fold:1", ("LP", 1)),
        ("saddle-node:3", ("LP", 3)),
        ("bp:-2", ("BP", -2)),
        ("branch_point:1", ("BP", 1)),
    ],
)
def test_every_spelling_reads_as_one_kind_and_index(declared, parsed):
    from tvbo.adapters.base import ContinuationAdapter

    assert ContinuationAdapter.source_point(_branch(declared), "fold:all") == parsed


def test_an_undeclared_source_is_the_default():
    from tvbo.adapters.base import ContinuationAdapter

    assert ContinuationAdapter.source_point(_branch(None), "fold:all") == ("LP", None)


@pytest.mark.parametrize("declared", ["hopf:x", "hopf:0", "hopf:1.5", "hopf:*", "pd:1", "hopfish:1", "3"])
def test_anything_else_is_refused_by_naming_the_accepted_forms(declared):
    """A non-numeric index read as "all" on one backend, as "last" on another and as nothing on a third."""
    from tvbo.adapters.base import ContinuationAdapter

    with pytest.raises(ValueError, match=r"'<kind>:<n>'"):
        ContinuationAdapter.source_point(_branch(declared), "hopf:all")


@pytest.mark.parametrize(("index", "selected"), [(None, ["a", "b", "c"]), (1, ["a"]), (3, ["c"]), (-1, ["c"]), (-3, ["a"])])
def test_an_index_is_one_based_and_counts_back_from_the_last(index, selected):
    from tvbo.adapters.base import ContinuationAdapter

    assert ContinuationAdapter.select_points(["a", "b", "c"], index) == selected


def test_an_index_past_the_points_found_is_refused_and_none_found_selects_none():
    from tvbo.adapters.base import ContinuationAdapter

    with pytest.raises(ValueError, match="out of range"):
        ContinuationAdapter.select_points(["a", "b"], 3)
    assert ContinuationAdapter.select_points([], 2) == []


@pytest.mark.parametrize(
    ("source", "julia"),
    [("hopf:all", None), ("hopf", None), ("hopf:2", "2"), ("hopf:-1", "end"), ("hopf:-2", "end-1"), (None, "end")],
)
def test_bifurcationkit_emits_the_parsed_hopf_point(source, julia):
    """A bare kind is every point of it; an undeclared source is the last Hopf point."""
    assert BifurcationKitAdapter._prepare_branch(_branch(source))["hopf_idx_jl"] == julia


@pytest.mark.parametrize(
    ("source", "kind", "julia"), [("fold:2", "fold", "2"), ("bp", "bp", None), ("HB:-1", "hopf", "end"), (None, "hopf", None)]
)
def test_bifurcationkit_codim2_emits_the_parsed_source(source, kind, julia):
    fp = SimpleNamespace(name="C", domain=None)
    branch = _branch(source, name="c2")
    branch.continuation = SimpleNamespace(
        free_parameters=[fp],
        **{
            k: None
            for k in (
                "ds",
                "ds_min",
                "ds_max",
                "max_steps",
                "tol_stability",
                "nev",
                "n_inversion",
                "max_bisection_steps",
                "detect_bifurcation",
            )
        },
    )
    context = BifurcationKitAdapter._prepare_codim2_branch(branch, None)
    assert (context["source_type"], context["source_idx_jl"], context["is_fold"]) == (kind, julia, kind in ("fold", "bp"))


def test_pyrates_counts_a_hopf_index_from_one():
    """PyRates read `hopf:1` as the second Hopf point, BifurcationKit and AUTO as the first."""
    from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter

    started = []
    ode = SimpleNamespace(run=lambda **kw: started.append(kw["starting_point"]) or (None, None))
    adapter = PyRatesBifurcationAdapter(SimpleNamespace())
    adapter._find_special_points = lambda ode, cont, kind: [f"{kind}{k}" for k in (1, 2, 3)]
    cont = SimpleNamespace(
        bothside=False,
        ds=None,
        ds_min=None,
        ds_max=None,
        max_steps=None,
        tol_stability=None,
        newton_tol=None,
        newton_max_iterations=None,
    )
    for source, expected in (("hopf:1", ["HB1"]), ("hopf:-1", ["HB3"]), ("hopf:all", ["HB1", "HB2", "HB3"])):
        started.clear()
        adapter._run_branch(ode, "param", _branch(source), cont, "p", 0.0, 1.0)
        assert started == expected, source


def test_auto_restarts_codim2_from_the_parsed_points():
    from tvbo.adapters import numcont

    class Bundle(list):
        def __call__(self, label):
            return label

    bundle = Bundle([SimpleNamespace(labels=SimpleNamespace(by_label={"LP": {5: None, 9: None}, "HB": {7: None}}))])
    restarted = []
    auto = SimpleNamespace(run=lambda **kw: restarted.append(kw["data"]) or "R", sv=lambda *a: None, merge=lambda r: r)
    fps = [SimpleNamespace(name=n, domain=None) for n in ("p", "C")]
    kwargs_eq = {
        "EPSL": 1e-7,
        "EPSU": 1e-7,
        "EPSS": 1e-5,
        "RL0": 0.0,
        "RL1": 1.0,
        "DSMAX": 0.1,
        "DSMIN": 1e-6,
        "IADS": 1,
    }
    for source, expected in (("fold:all", ["LP1", "LP2"]), ("fold:-1", ["LP2"]), ("hopf:1", ["HB1"])):
        restarted.clear()
        branch = _branch(source, name="c2")
        branch.continuation = SimpleNamespace(free_parameters=fps, parameters=None, bothside=False)
        cont = SimpleNamespace(branches={"c2": branch}, free_parameters=fps[:1])
        numcont.NumContAdapter(SimpleNamespace())._run_codim2_branches(
            auto=auto, R_eq=bundle, cont=cont, fp_name="p", kwargs_eq=kwargs_eq
        )
        assert restarted == expected, source
    with pytest.raises(ValueError, match="out of range"):
        branch.source_point = "fold:3"
        numcont.NumContAdapter(SimpleNamespace())._run_codim2_branches(
            auto=auto, R_eq=bundle, cont=cont, fp_name="p", kwargs_eq=kwargs_eq
        )


def test_auto_continues_periodic_orbits_from_the_hopf_points_the_branch_selects():
    from tvbo.adapters import numcont

    po = _branch("hopf:2", name="po")
    codim2 = _branch("hopf:1", name="c2")
    codim2.continuation = SimpleNamespace(free_parameters=[SimpleNamespace(name="C")])
    parent = SimpleNamespace(branches={"c2": codim2, "po": po}, free_parameters=[SimpleNamespace(name="p")])
    branch, index = numcont._po_branch(parent)
    assert (branch, index) == (po, 2)
    assert numcont._po_branch(SimpleNamespace(branches={})) == (None, None)
