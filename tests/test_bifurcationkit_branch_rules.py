"""What a BifurcationKit continuation keeps as a branch, and where a fold curve starts.

BifurcationKit built the vector field's ForwardDiff Jacobian at iterates where the Jansen-Rit sigmoid's `exp` overflows, so its entries were NaN while the residual stayed finite, and its periodic-orbit continuation went on through hundreds of them; a branch built that way was returned as an orbit (G260). The same continuation returned stubs whose period fell to -22.2 and -0.0006. A fold curve started from the point detection without bisection stores, a step past the fold, whose eigenvalue nearest zero is 1.763, and lost the whole curve to a NaN Jacobian (G174). Its Hopf curve reported a Bogdanov-Takens point at p = 13.131, C = 138.003, whose bisection had not converged, far from the one AUTO-07p locates at p = 15.937, and four unconverged copies of its one Bautin point. With shorter steps it passed through the Bogdanov-Takens point, frequency 1.8e-5, which BifurcationKit labels `zh`, and met its Bautin point twice, once on each sign of the frequency.

The rendered script raises at a non-finite Jacobian in every continuation, fails an orbit whose period reaches zero, and starts every fold curve from the fold `newton_fold` locates. Each of these is a failed continuation: the `catch` around it warns, naming its source point, and the branch is not returned. Each codim-2 curve reports its special points by their location and values, whatever BifurcationKit labels them: a point whose bisection did not converge is warned about and dropped, a converged point on a curve that records the Hopf frequency is a Bogdanov-Takens point exactly when that frequency is small against the curve's own, and points of one type that coincide within their location precision are one. The curated curves bisect to the tenth sign inversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tvbo
from tvbo import SimulationExperiment
from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

BIFURCATION = Path(tvbo.__file__).parent / "database" / "experiments" / "bifurcation"
JANSENRIT = BIFURCATION / "JansenRit-bifurcation.yaml"
PERIOD_CHECK = "minimum(br_po.branch.period) > 0 || error("
REPORT = "function _report_codim2_points!(br, ics, source)"


def _render(path) -> str:
    return BifurcationKitAdapter(SimulationExperiment.from_file(str(path))).render_code()


def _jansenrit_with_orbits_by(tmp_path, method) -> str:
    """The curated Jansen-Rit continuation, its orbits discretised by *method*, rendered for BifurcationKit."""
    spec = yaml.safe_load(JANSENRIT.read_text())
    spec["continuations"]["eq_in_p"]["branches"]["po_alpha_rhythm"]["discretization"]["method"] = method
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return _render(path)


def _codim2_blocks(code) -> dict[str, str]:
    """Each codim-2 block of *code* by its branch name."""
    return {block.split(" ", 1)[0]: block for block in code.split("# Codim-2 branch: ")[1:]}


def _function(code, signature) -> str:
    """The Julia function of *code* that *signature* opens, through its closing `end`."""
    start = code.index(signature)
    return code[start : code.index("\nend\n", start)]


def _in_order(text, markers) -> bool:
    return [text.index(marker) for marker in markers] == sorted(text.index(marker) for marker in markers)


@pytest.mark.parametrize("spec", ["JansenRit-bifurcation.yaml", "QIF-bifurcation.yaml"])
def test_every_continuation_raises_at_a_non_finite_jacobian(spec):
    """The checked Jacobian replaces BifurcationKit's before the equilibrium branch, so that branch, every orbit and every codim-2 curve build through it."""
    code = _render(BIFURCATION / spec)
    checked = "re_make(prob; J = (x, p) -> _finite_jacobian(BifurcationKit.jacobian(vf, x, p)))"
    assert "all(isfinite, J) || error(" in code
    assert code.index(checked) < code.index("br = continuation(prob, ")


@pytest.mark.parametrize("method", ["collocation", "trapezoid"])
def test_an_orbit_whose_period_reaches_zero_is_a_failed_continuation(tmp_path, method):
    """The check follows the orbit's continuation and precedes keeping it, inside the `try` whose `catch` warns that the continuation from that Hopf point failed."""
    code = _jansenrit_with_orbits_by(tmp_path, method)
    order = ["br_po = continuation(", PERIOD_CHECK, "push!(po_branches, br_po)", '@warn "PO continuation from Hopf']
    assert _in_order(code, order)


def test_a_fold_curve_starts_from_the_fold_newton_fold_locates_and_a_hopf_curve_from_its_point():
    blocks = _codim2_blocks(_render(JANSENRIT))
    fold, hopf = blocks["fold_p_C"], blocks["hopf_p_C"]
    locate = "BifurcationKit.newton_fold(br, _bif_idx; normN = norminf, options = _opts_c2.newton_options,"
    assert fold.index(locate) < fold.index("_fold.converged || error(") < fold.index("_source, _bif_idx, (@optic _.C)")
    assert "newton_fold" not in hopf
    assert "local _source = br\n" in hopf and "_source, _bif_idx, (@optic _.C)" in hopf


def test_the_jansenrit_orbits_adapt_their_collocation_mesh():
    assert "PeriodicOrbitOCollProblem(50, 4, meshadapt = true)" in _render(JANSENRIT)


def test_a_codim2_point_whose_bisection_did_not_converge_is_not_reported():
    """`_report_codim2_points!` warns about such points and deletes them by their status alone, and every codim-2 curve passes through it between its continuation and being kept."""
    report = _function(_render(JANSENRIT), REPORT)
    unconverged = "unlocated = findall(sp -> sp.status != :converged, br.specialpoint)"
    assert _in_order(report, [unconverged, "whose bisection did not converge", "deleteat!(br.specialpoint, unlocated)"])
    for block in _codim2_blocks(_render(JANSENRIT)).values():
        assert _in_order(block, ["local _br_c2 = continuation(", "push!(codim2_results, _report_codim2_points!(_br_c2, :p, "])


def test_on_a_curve_that_records_the_hopf_frequency_a_bogdanov_takens_point_is_read_from_that_frequency():
    """After the unconverged points go, a converged point whose |ωₕ| is below 1e-3 of the curve's median |ωₕ| becomes `bt`, and a `bt` at or above it is warned about and deleted, whatever BifurcationKit labelled it."""
    report = _function(_render(JANSENRIT), REPORT)
    markers = [
        "deleteat!(br.specialpoint, unlocated)",
        "if hasproperty(first(br.branch), :ωₕ)",
        "ω_bt = 1e-3 * ω[cld(length(ω), 2)]",
        "not_bt = findall(sp -> sp.type == :bt && abs(sp.printsol.ωₕ) >= ω_bt, br.specialpoint)",
        "deleteat!(br.specialpoint, not_bt)",
        "abs(sp.printsol.ωₕ) < ω_bt && (br.specialpoint[i] = setproperties(sp; type = :bt))",
    ]
    assert _in_order(report, markers)


def test_codim2_points_of_one_type_that_coincide_within_their_location_precision_are_reported_once():
    """Points are compared in order of precision, so the one kept is the most precisely located, after the Bogdanov-Takens points are read from the frequency."""
    code = _render(JANSENRIT)
    coincide = _function(code, "function _coincide(a, b, br, ics)")
    assert "δ = a.precision + b.precision" in coincide
    assert "a.type == b.type && abs(a.param - b.param) <= δ && " in coincide
    markers = [
        "setproperties(sp; type = :bt)",
        "sortperm(br.specialpoint; by = sp -> sp.precision)",
        "_coincide(br.specialpoint[i], br.specialpoint[j], br, ics) && push!(copies, i)",
        "deleteat!(br.specialpoint, sort(copies))",
    ]
    assert _in_order(_function(code, REPORT), markers)


def test_the_jansenrit_curves_bisect_to_the_tenth_sign_inversion():
    blocks = _codim2_blocks(_render(JANSENRIT))
    assert set(blocks) == {"fold_p_C", "hopf_p_C"}
    assert all("n_inversion = 10, max_bisection_steps = 60" in block for block in blocks.values())
