"""A Procedural GraphGenerator's typed DAG resolves to backend-independent expressions.

The DAG is metadata: every option is a schema field, and the resolver builds the SymPy
tree directly. Nothing round-trips through an expression string except an `equation`
step's author-written `rhs`, so the parser's limits (no keyword arguments, `!=` collapsing
to `True`, an unregistered head silently becoming multiplication) are unreachable by
construction rather than avoided by convention.

The fixture is Koller2024's 2-D sheet — the construction that motivated Tier 2 — so these
also pin that a real paper's network is expressible without per-generator Python.
"""

import numpy as np
import pytest
import sympy as sp

from tvbo.codegen.code import render_expression
from tvbo.graph_generators.procedural import (
    ProceduralError,
    build,
    partition,
    seeded_steps,
)

# Koller2024 2-D sheet, exactly the construction in koller2024_networks.build_2d_sheet:
# distance kernel -> stochastic connection mask -> column-normalise -> in-strength
# gradient from two opposing Gaussians.
def _normal_field(mean):
    """A Normal spatial field: vector mean + isotropic cov, as a Distribution."""
    return {"name": "Normal", "parameters": {"mean": mean, "cov": 300.0}}


KOLLER_SHEET = {
    "parameters": {"sigma": 10.0, "alpha": 2.0, "beta": 4.0},
    "derived": {
        "d_ij": {"type": "pairwise_distance", "of": "layout", "diagonal": "inf"},
        "a_ij": {"equation": {"rhs": "(1 / (2*sigma)) * exp(-d_ij / sigma)"}},
        "mask_ij": {
            "type": "stochastic_mask",
            "of": "d_ij",
            "comparison": "le",
            "distribution": {"name": "Exponential", "parameters": {"scale": 17.0}},
        },
        "a_masked": {"equation": {"rhs": "a_ij * mask_ij"}},
        "a_normalized": {"type": "normalize", "of": "a_masked", "axis": 0},
        "sink_pdf": {"type": "distribution_pdf", "of": "layout",
                     "distribution": _normal_field([40.0, 40.0])},
        "source_pdf": {"type": "distribution_pdf", "of": "layout",
                       "distribution": _normal_field([100.0, 100.0])},
        "grad_raw": {"equation": {"rhs": "sink_pdf - source_pdf"}},
        "gradient_template": {"type": "minmax_rescale", "of": "grad_raw",
                              "target_range": {"lo": -1, "hi": 1}},
    },
}


def _named(spec):
    return dict(build(spec))


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #
def test_every_step_resolves_to_a_sympy_expression():
    resolved = build(KOLLER_SHEET)
    assert [n for n, _ in resolved] == list(KOLLER_SHEET["derived"])
    for name, expr in resolved:
        assert isinstance(expr, sp.Basic), f"{name} did not resolve to a SymPy object"


def test_self_distance_becomes_a_fill_diagonal_node():
    """`self: inf` is a field, and lowers onto the fill_diagonal primitive."""
    d = _named(KOLLER_SHEET)["d_ij"]
    assert d.func.__name__ == "fill_diagonal"
    assert d.args[1] is sp.oo
    assert d.args[0].func.__name__ == "pairwise_distance"


def test_stochastic_mask_is_a_relational_over_a_sampler():
    mask = _named(KOLLER_SHEET)["mask_ij"]
    assert isinstance(mask, sp.LessThan)
    sampler = mask.args[1]
    assert sampler.func.__name__ == "sample_exponential"
    # key first (JAX is pure), then the scale, then the sample shape.
    assert str(sampler.args[0]) == "key"
    assert sampler.args[1] == sp.Float(17.0)


def test_no_abs_wrapper_is_needed_around_an_exponential_draw():
    """Koller's source writes `abs(np.random.exponential(...))`; the DAG omits the abs.

    This is a deliberate, provable simplification rather than a dropped detail: the
    exponential distribution has support [0, inf), so `abs` is the identity on every
    value it can return. Omitting it keeps the schema minimal — no `transform` field
    exists solely to express a no-op — and leaves the mask numerically identical. A
    distribution that can go negative (e.g. Normal) would need an explicit `equation`
    step, which is what that step type is for.
    """
    sampler = _named(KOLLER_SHEET)["mask_ij"].args[1]
    assert sampler.func.__name__ == "sample_exponential"
    assert "abs" not in str(sampler)


def test_distribution_parameters_are_fields_not_keywords():
    """The whole point: a kwarg would be unrepresentable, a field is not."""
    sampler = _named(KOLLER_SHEET)["mask_ij"].args[1]
    assert sp.Float(17.0) in sampler.args


# --------------------------------------------------------------------------- #
# Backend rendering                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["numpy", "jax"])
def test_whole_dag_renders_on_every_backend(fmt):
    for name, expr in build(KOLLER_SHEET):
        rendered = render_expression(expr, format=fmt)
        assert rendered and "Not supported" not in rendered, f"{name} failed on {fmt}"


def test_jax_render_uses_pure_prng():
    src = render_expression(_named(KOLLER_SHEET)["mask_ij"], format="jax")
    assert "jax.random.exponential" in src
    assert "key" in src


# --------------------------------------------------------------------------- #
# The derived deterministic / stochastic split                                 #
# --------------------------------------------------------------------------- #
def test_seed_dependence_is_transitive():
    """a_masked is not itself a draw, but it references the mask, so it is seeded."""
    seeded = seeded_steps(KOLLER_SHEET)
    assert "mask_ij" in seeded
    assert "a_masked" in seeded          # via rhs `a_ij * mask_ij`
    assert "a_normalized" in seeded      # via `of: a_masked`


def test_geometry_stays_deterministic():
    """The expensive construction must be hoistable out of a per-realisation loop."""
    deterministic, stochastic = partition(KOLLER_SHEET)
    for name in ("d_ij", "a_ij", "sink_pdf", "source_pdf", "grad_raw", "gradient_template"):
        assert name in deterministic, f"{name} should not depend on the seed"
    assert stochastic == ["mask_ij", "a_masked", "a_normalized"]


def test_partition_covers_the_dag_exactly_once():
    deterministic, stochastic = partition(KOLLER_SHEET)
    assert sorted(deterministic + stochastic) == sorted(KOLLER_SHEET["derived"])
    assert not set(deterministic) & set(stochastic)


def test_partition_preserves_dag_order():
    deterministic, stochastic = partition(KOLLER_SHEET)
    order = list(KOLLER_SHEET["derived"])
    for part in (deterministic, stochastic):
        assert part == [n for n in order if n in part]


def test_a_dag_with_no_randomness_has_an_empty_suffix():
    spec = {"derived": {k: v for k, v in KOLLER_SHEET["derived"].items()
                        if k in ("d_ij", "a_ij")}, "parameters": KOLLER_SHEET["parameters"]}
    deterministic, stochastic = partition(spec)
    assert stochastic == []
    assert deterministic == ["d_ij", "a_ij"]


# --------------------------------------------------------------------------- #
# Malformed DAGs fail loudly                                                   #
# --------------------------------------------------------------------------- #
def test_unknown_step_type_is_rejected():
    with pytest.raises(ProceduralError, match="unknown type"):
        build({"derived": {"x": {"type": "no_such_step"}}})


def test_forward_reference_is_rejected():
    """Steps are ordered; referencing a later name is a recipe error, not a silent nan."""
    with pytest.raises(ProceduralError, match="not a previously-defined"):
        build({"derived": {"a": {"type": "normalize", "of": "b", "axis": 0},
                           "b": {"equation": {"rhs": "1"}}}})


def test_unknown_distribution_is_rejected():
    with pytest.raises(ProceduralError, match="no sampler"):
        build({"derived": {"m": {"type": "stochastic_mask", "of": "n_nodes",
                                 "distribution": {"name": "Cauchy", "parameters": {"scale": 1.0}}}}})


def test_missing_distribution_parameter_is_rejected():
    with pytest.raises(ProceduralError, match="requires parameter"):
        build({"derived": {"m": {"type": "stochastic_mask", "of": "n_nodes",
                                 "distribution": {"name": "Exponential", "parameters": {}}}}})


def test_distribution_pdf_rejects_a_non_normal_family():
    with pytest.raises(ProceduralError, match="defined for Normal"):
        build({"derived": {"f": {"type": "distribution_pdf", "of": "layout",
                                 "distribution": {"name": "Exponential",
                                                  "parameters": {"scale": 1.0}}}}})


def test_minmax_rescale_requires_a_target_range():
    with pytest.raises(ProceduralError, match="target_range"):
        build({"derived": {"g": {"type": "equation", "equation": {"rhs": "1"}},
                           "r": {"type": "minmax_rescale", "of": "g"}}})
