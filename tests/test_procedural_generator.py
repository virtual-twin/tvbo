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
    "parameters": {"sigma": 10.0, "alpha": 2.0, "beta": 4.0,
                   "nx": 30, "ny": 30, "x_extent": 140.0, "y_extent": 140.0},
    "steps": {
        # The layout is an ordinary named intermediate, not a special generator slot:
        # positions are produced by a primitive and referenced by name like anything else.
        "layout": {"equation": {"rhs": "grid_positions(nx, ny, x_extent, y_extent)"}},
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
    assert [n for n, _ in resolved] == list(KOLLER_SHEET["steps"])
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
    # key first (JAX is pure), then the sub-stream index, the scale, then the shape.
    assert str(sampler.args[0]) == "_prng_key"
    assert sampler.args[1] == sp.Integer(0)
    assert sampler.args[2] == sp.Float(17.0)


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
# The inferred deterministic / stochastic split                                 #
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
    assert sorted(deterministic + stochastic) == sorted(KOLLER_SHEET["steps"])
    assert not set(deterministic) & set(stochastic)


def test_partition_preserves_dag_order():
    deterministic, stochastic = partition(KOLLER_SHEET)
    order = list(KOLLER_SHEET["steps"])
    for part in (deterministic, stochastic):
        assert part == [n for n in order if n in part]


def test_a_dag_with_no_randomness_has_an_empty_suffix():
    spec = {"steps": {k: v for k, v in KOLLER_SHEET["steps"].items()
                        if k in ("layout", "d_ij", "a_ij")},
            "parameters": KOLLER_SHEET["parameters"]}
    deterministic, stochastic = partition(spec)
    assert stochastic == []
    assert deterministic == ["layout", "d_ij", "a_ij"]


def test_a_sampler_inside_an_equation_step_is_still_seeded():
    """Seededness comes from the resolved expression, not the declared step type.

    An `equation` step may call a sampler head directly. Trusting `type` would classify it
    as deterministic and hoist it out of the per-realisation loop, so every "independent"
    realisation would silently share one draw — the worst failure this module can have,
    because the ensemble still runs and still produces plausible numbers.

    Detection keys off the sampler HEAD, not the PRNG symbol's name, so the step is caught
    however its key argument is spelled (here a plain `anykey`).
    """
    spec = {"steps": {
        "raw": {"equation": {"rhs": "sample_normal(anykey, 0, 1, n_nodes, n_nodes)"}},
        "scaled": {"equation": {"rhs": "raw * 2"}},
        "geom": {"equation": {"rhs": "grid_positions(2, 2, 1.0, 1.0)"}},
    }}
    deterministic, stochastic = partition(spec)
    assert stochastic == ["raw", "scaled"]
    assert deterministic == ["geom"]


def test_sample_step_yields_the_draw_not_a_mask():
    """`sample` is the draw itself; `stochastic_mask` is a comparison over one."""
    spec = {"steps": {"raw": {"type": "sample",
                                "distribution": {"name": "Normal",
                                                 "parameters": {"mean": 0.0, "std": 1.0}}}}}
    expr = dict(build(spec))["raw"]
    assert expr.func.__name__ == "sample_normal"
    assert not isinstance(expr, sp.Rel)


def test_sample_selects_its_distribution_by_parameter_name():
    """`of` on a `sample` step names the Distribution-valued parameter to draw from.

    That is how a curated generator exposes its randomness as a choice
    (RandomReservoir's `weight_distribution`) while the inline `distribution` states the
    family it falls back to.
    """
    fields = {"type": "sample", "of": "wd",
              "distribution": {"name": "Normal",
                               "parameters": {"mean": 0.0, "std": 1.0}}}
    supplied = {"name": "Uniform", "parameters": {"lo": -1.0, "hi": 1.0}}
    assert dict(build({"steps": {"raw": fields}}))["raw"].func.__name__ == "sample_normal"
    chosen = dict(build({"parameters": {"wd": supplied}, "steps": {"raw": fields}}))["raw"]
    assert chosen.func.__name__ == "sample_uniform"


def test_sample_rejects_an_of_that_names_an_intermediate():
    """On every other step type `of` is an array, so reaching for one here is the natural
    mistake — and silently falling back to the default family would build a plausible
    network from the wrong distribution without saying so."""
    with pytest.raises(ProceduralError, match="not from an array"):
        build({"steps": {
            "geom": {"equation": {"rhs": "grid_positions(2, 2, 1.0, 1.0)"}},
            "raw": {"type": "sample", "of": "geom",
                    "distribution": {"name": "Normal",
                                     "parameters": {"mean": 0.0, "std": 1.0}}}}})


def test_an_undefined_layout_is_a_dangling_reference():
    """No name is magically pre-bound: a layout must be defined like any other step."""
    with pytest.raises(ProceduralError, match="not a previously-defined"):
        build({"steps": {"d": {"type": "pairwise_distance", "of": "layout"}}})


# --------------------------------------------------------------------------- #
# Malformed DAGs fail loudly                                                   #
# --------------------------------------------------------------------------- #
def test_unknown_step_type_is_rejected():
    with pytest.raises(ProceduralError, match="unknown type"):
        build({"steps": {"x": {"type": "no_such_step"}}})


def test_forward_reference_is_rejected():
    """Steps are ordered; referencing a later name is a recipe error, not a silent nan."""
    with pytest.raises(ProceduralError, match="not a previously-defined"):
        build({"steps": {"a": {"type": "normalize", "of": "b", "axis": 0},
                           "b": {"equation": {"rhs": "1"}}}})


def test_unknown_distribution_is_rejected():
    with pytest.raises(ProceduralError, match="no sampler"):
        build({"steps": {"m": {"type": "stochastic_mask", "of": "n_nodes",
                                 "distribution": {"name": "Cauchy", "parameters": {"scale": 1.0}}}}})


def test_an_omitted_parameter_falls_back_to_the_families_standard_form():
    """`Normal` without a mean means Normal(0, 1) — the family's own standard form.

    Erroring instead would reject the ordinary way these are written, and the deleted
    engine supplied exactly these defaults through its distribution constructors.
    """
    expr = dict(build({"steps": {"m": {"type": "sample",
                                       "distribution": {"name": "Normal",
                                                        "parameters": {"std": 0.5}}}}}))["m"]
    assert expr.args[2] == sp.Float(0.0)   # mean, defaulted
    assert expr.args[3] == sp.Float(0.5)   # std, as given


def test_an_unknown_distribution_parameter_is_rejected():
    """A typo must not be dropped: the parameter it meant would take its standard form,
    so the draw would silently come from a different distribution than the spec states."""
    with pytest.raises(ProceduralError, match="has no parameter"):
        build({"steps": {"m": {"type": "sample",
                               "distribution": {"name": "Normal",
                                                "parameters": {"mena": 0.5, "std": 1.0}}}}})


def test_gaussian_is_the_same_family_as_normal_for_sampling():
    """`distribution_pdf` already accepts `Gaussian`; one spelling must not resolve on
    one step type and fail on another."""
    for family in ("Normal", "Gaussian"):
        expr = dict(build({"steps": {"m": {"type": "sample",
                                           "distribution": {"name": family}}}}))["m"]
        assert expr.func.__name__ == "sample_normal"


def test_distribution_pdf_rejects_a_non_normal_family():
    with pytest.raises(ProceduralError, match="defined for Normal"):
        build({"steps": {
            "layout": {"equation": {"rhs": "grid_positions(2, 2, 1.0, 1.0)"}},
            "f": {"type": "distribution_pdf", "of": "layout",
                  "distribution": {"name": "Exponential", "parameters": {"scale": 1.0}}}}})


def test_minmax_rescale_requires_a_target_range():
    with pytest.raises(ProceduralError, match="target_range"):
        build({"steps": {"g": {"type": "equation", "equation": {"rhs": "1"}},
                           "r": {"type": "minmax_rescale", "of": "g"}}})


def test_reserved_prng_parameter_name_is_rejected():
    """The PRNG symbol is reserved; a parameter of that name would be silently clobbered.

    The resolver binds the generator's PRNG state into the evaluation namespace under this
    name and detects seededness partly by looking for it. A parameter sharing it would be
    overwritten by the seed (wrong values, no error) and would make every step look seeded,
    disabling the deterministic-prefix hoisting.
    """
    with pytest.raises(ProceduralError, match="reserved"):
        build({"parameters": {"_prng_key": 1.0},
               "steps": {"a": {"equation": {"rhs": "_prng_key * 2"}}}})


@pytest.mark.parametrize(
    "rhs, signature",
    [
        ("gaussian_pdf(p, m)", "gaussian_pdf"),
        ("minmax_rescale(x, 0)", "minmax_rescale"),
        ("pairwise_distance(p, q)", "pairwise_distance"),
        ("fill_diagonal(M)", "fill_diagonal"),
        ("eigvals(M, 1)", "eigvals"),
        ("grid_positions(1, 2, 3.0)", "grid_positions"),
    ],
)
def test_wrong_arity_names_the_primitive(rhs, signature):
    """A miscounted argument must name the primitive, not raise a bare IndexError."""
    with pytest.raises(ValueError, match=signature):
        render_expression(rhs, format="numpy")
