"""Probes for G174's Bogdanov-Takens point on BifurcationKit and AUTO-07p: print each codim-2 curve's steps, labels and step budget, asserting nothing."""

from __future__ import annotations

import pytest

from tvbo.adapters.base import ContinuationAdapter
from tvbo.adapters.numcont import _branch_par
from tvbo.analysis.bifurcation import _extract_special_points

from .test_jansenrit_continuation import _needs_auto, _special, experiment  # noqa: F401

COLUMNS = ("step", "branch_id", "param", "param2", "ωₕ", "BT", "GH", "specialpoint", "sp_status", "sp_precision")


def _print_rows(backend, curve, rows):
    """A `PROBE <backend> <kind>` line with *curve*'s step count and p and C extent, then one `row` line per index in *rows* with the `COLUMNS` its branch carries."""
    kind, branch = curve._source_type, curve.branch
    columns = [c for c in COLUMNS if c in branch.variables]
    p, c = branch["param"].values, branch["param2"].values
    print(f"PROBE {backend} {kind} steps={curve.n_steps} p=[{p.min()}, {p.max()}] C=[{c.min()}, {c.max()}] columns={columns}")
    for i in rows:
        print(f"PROBE {backend} {kind} row {i}", " ".join(f"{name}={branch[name].values[i]}" for name in columns))


@pytest.mark.slow
@pytest.mark.julia
@pytest.mark.backend_julia
def test_print_each_curve_s_steps_and_special_points(experiment, monkeypatch, tmp_path):  # noqa: F811
    """Every step of each BifurcationKit curve (p in param, C in param2, the Hopf frequency in ωₕ), and one `sp` line per point it kept, with its status, interval width and recorded p and C."""
    pytest.importorskip("juliacall")
    monkeypatch.chdir(tmp_path)
    result = experiment.run("bifurcationkit").continuations["eq_in_p"]
    print("PROBE hopf points p:", _special(result, "hopf"))
    for curve in result.codim2_curves:
        _print_rows("bifurcationkit", curve, range(curve.n_steps))
        for point in _extract_special_points(curve.br):
            print(f"PROBE bifurcationkit {curve._source_type} sp", point)


@pytest.mark.slow
def test_print_each_auto_curve_s_extent_labels_and_step_budget(experiment, monkeypatch, tmp_path):  # noqa: F811
    """The NMX, DS and DSMAX each AUTO-07p curve is declared with, then each curve's first and last three steps, both ends of each sub-branch and every labelled step (`endpoint`, `mx` and `uz` say where and why a direction stopped)."""
    _needs_auto()
    monkeypatch.chdir(tmp_path)
    cont = experiment.continuations["eq_in_p"]
    for name, branch in ContinuationAdapter.branches_of(cont).items():
        if ContinuationAdapter.is_codim2(branch, cont):
            print(
                f"PROBE auto-07p {name} declared",
                {key: _branch_par(cont, branch.continuation, key, None) for key in ("NMX", "DS", "DSMAX")},
            )
    result = experiment.run("auto-07p").continuations["eq_in_p"]
    for curve in result.codim2_curves:
        n, branch = curve.n_steps, curve.branch
        ids = branch["branch_id"].values if "branch_id" in branch.variables else [0] * n
        starts = [i for i in range(1, n) if ids[i] != ids[i - 1]]
        labelled = [i for i, label in enumerate(branch["specialpoint"].values) if label]
        rows = {0, 1, 2, n - 3, n - 2, n - 1, *starts, *(i - 1 for i in starts), *labelled}
        _print_rows("auto-07p", curve, sorted(i for i in rows if 0 <= i < n))
