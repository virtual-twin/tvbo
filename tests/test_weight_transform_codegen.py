"""Weight `transforms:` are inlined in the generated tvboptim code (self-contained kit).

A declared connectome transform (e.g. ``log(weight+1)/max(log(weight+1))``) is applied at
runtime by ``Network.weights_matrix``. A frozen/standalone kit must not depend on that: the
codegen renders the transform to pure ``jnp`` inside ``create_network`` and is handed the RAW
weights, so the transform stays declared in the spec, the raw SC stays in the network file,
and the exact op is visible in the script rather than hidden in tvbo runtime. These freeze the
raw/transformed accessor split, byte-identity of the inlined op against ``weights_matrix``,
and that the emitted network builder carries the transform as pure ``jnp``.

A transform equation is written over the network's own edge attributes — ``weight``,
``length``, ``network.edges.<label>`` — and any reduction in it may be masked, in either of the
two spellings. These also freeze that both mean the same thing on both paths.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from tvbo.classes.network import Network
from tvbo.templates.tvboptim.utils import weight_transform_codegen

RHS = "log(weight + 1) / max(log(weight + 1))"


def _net_with_transform():
    W = np.array([[0, 4, 16, 0],
                  [4, 0, 0, 100],
                  [16, 0, 0, 1],
                  [0, 100, 1, 0]], float)
    net = Network.from_matrix(weights=W, lengths=np.zeros_like(W))
    net.add_transform("weight", RHS)
    return net, W


def _apply_emitted(net, weights, distances=None):
    """Run the emitted const_env + per-transform env exactly as `create_network` does."""
    transforms, const_env, _ = weight_transform_codegen(net)
    scope = {"jnp": jnp, "weights": jnp.asarray(weights),
             "distances": None if distances is None else jnp.asarray(distances)}
    for line in const_env:
        exec(line, scope)
    for expr, chained_env in transforms:
        for line in chained_env:
            exec(line, scope)
        scope["weights"] = eval(expr, scope)
    return np.asarray(scope["weights"])


def test_raw_accessor_is_untouched_transformed_is_normalised():
    net, W = _net_with_transform()
    raw = np.asarray(net.raw_weights_matrix)
    transformed = np.asarray(net.weights_matrix)
    assert np.array_equal(raw, W)                    # raw_weights_matrix skips transforms
    assert not np.allclose(raw, transformed)         # the transform actually changed it
    assert transformed.max() <= 1.0 + 1e-6           # log(weight+1)/max normalises to <= 1


def test_inline_transform_is_byte_identical_to_weights_matrix():
    """The rendered jax expr applied to the RAW weights reproduces ``weights_matrix`` exactly,
    so ``experiment.py`` passing ``raw_weights_matrix`` while ``create_network`` inlines the
    transform is a no-op for every working run."""
    net, _ = _net_with_transform()
    transforms, const_env, needs_lengths = weight_transform_codegen(net)
    assert len(transforms) == 1
    assert const_env == []
    assert needs_lengths is False
    expr, chained_env = transforms[0]
    assert expr.startswith("jnp.") and "weight" in expr
    assert chained_env == ["weight = weights"]       # the target rebinds from the running value
    assert np.array_equal(_apply_emitted(net, net.raw_weights_matrix),
                          np.asarray(net.weights_matrix))


def test_network_without_transform_emits_nothing():
    net = Network.from_matrix(weights=np.array([[0, 1.0], [1.0, 0]]),
                              lengths=np.zeros((2, 2)))
    transforms, const_env, needs_lengths = weight_transform_codegen(net)
    assert transforms == []
    assert const_env == []
    assert needs_lengths is False
    assert np.array_equal(np.asarray(net.raw_weights_matrix),
                          np.asarray(net.weights_matrix))


def test_rendered_tvboptim_source_inlines_the_transform():
    """The generated network builder applies the declared transform as pure jnp — the kit is
    self-contained (no reliance on tvbo runtime re-deriving the weights)."""
    from tvbo import SimulationExperiment

    exp = SimulationExperiment(
        id=1, label="weight-transform",
        dynamics={"name": "Osc", "system_type": "continuous", "output": ["x"],
                  "parameters": {"a": {"value": 1.0}},
                  "state_variables": {"x": {"equation": {"rhs": "-a*x"},
                                            "initial_value": 0.1}}},
        network={"number_of_nodes": 4,
                 "nodes": [{"id": i, "label": f"n{i}"} for i in range(4)]},
        integration={"method": "heun", "step_size": 0.1, "duration": 1.0,
                     "transient_time": 0.0, "unit": "s"},
    )
    net, _ = _net_with_transform()
    exp.network = net

    code = exp.render_code("tvboptim")
    builder = code.split("def create_network", 1)[-1].split("\ndef ", 1)[0]
    assert "jnp.log(1 + weight)" in builder             # transform inlined in the builder
    assert "weights_matrix" not in builder              # not delegated to tvbo runtime
    assert "_apply_transform" not in builder


# ── A transform is a Function: callable and equation are both lowered ───────────────────


def test_a_callable_transform_reaches_the_kit():
    """`Function.callable` lowers to an import and a call, matching the runtime.

    Hopf_Pareto_ParallelOpt declares `normalized_graph_laplacian` this way. The codegen
    used to skip any transform without an `equation:`, so with `experiment.py` handing over
    raw weights the kit integrated the un-normalised SC — wrong numbers, no error.
    """
    from tvbo.datamodel.schema import Callable as CallableRef, Function

    W = np.array([[0, 2.0, 1.0], [4.0, 0, 3.0], [1.0, 5.0, 0]])
    net = Network.from_matrix(weights=W, lengths=np.zeros_like(W))
    net.transforms = [Function(name="weight", callable=CallableRef(
        module="tvbo.classes.network", name="normalized_graph_laplacian"))]

    transforms, const_env, _ = weight_transform_codegen(net)
    assert len(transforms) == 1
    assert any(line.startswith("from tvbo.classes.network import") for line in const_env)
    assert np.allclose(_apply_emitted(net, W), np.asarray(net.weights_matrix))


def test_equation_parameters_are_substituted_like_the_runtime():
    """A scalar declared under `Equation.parameters` folds in, as `_apply_transform` does.

    Reading only `Function.arguments` emitted the bare name into the kit, where it is
    undefined.
    """
    from tvbo.datamodel.schema import Equation, Function

    W = np.array([[0, 2.0], [4.0, 0]])
    net = Network.from_matrix(weights=W, lengths=np.zeros_like(W))
    net.transforms = [Function(name="weight",
                                    equation=Equation(rhs="weight / k",
                                                 parameters={"k": {"value": 4.0}}))]

    expr, _ = weight_transform_codegen(net)[0][0]
    assert "k" not in expr
    assert np.allclose(_apply_emitted(net, W), np.asarray(net.weights_matrix))


def test_an_undeclared_symbol_fails_at_render_time():
    """Better a render-time error than an undefined name in a kit that dies on the cluster."""
    net = Network.from_matrix(weights=np.array([[0, 1.0], [1.0, 0]]), lengths=np.zeros((2, 2)))
    net.add_transform("weight", "weight / k_undeclared")
    with pytest.raises(ValueError, match="k_undeclared"):
        weight_transform_codegen(net)


def test_transforms_for_matches_the_length_alias():
    """One selector owns the `length`/`lengths` spelling for runtime and codegen alike."""
    from tvbo.datamodel.schema import Equation, Function

    net = Network.from_matrix(weights=np.zeros((2, 2)), lengths=np.zeros((2, 2)))
    net.transforms = [
        Function(name="lengths", equation=Equation(rhs="length * 2")),
        Function(name="weight", equation=Equation(rhs="weight * 3")),
    ]
    assert len(net.transforms_for("length")) == 1
    assert len(net.transforms_for("lengths")) == 1
    assert len(net.transforms_for("weight")) == 1


# ── The vocabulary is the network's own edge attributes ─────────────────────────────────


def test_an_edge_attribute_is_the_vocabulary_not_an_invented_alias():
    """A transform names `weight`/`length`, resolved by the resolver observations use.

    The equation body used to be written against a private table of `W`/`L`/`W_max`
    primitives, declared once for the runtime and again as JAX source for the emitter —
    two lists that had already drifted.
    """
    W = np.array([[0, 2.0], [4.0, 0]])
    L = np.array([[0, 10.0], [10.0, 0]])
    net = Network.from_matrix(weights=W, lengths=L)
    net.add_transform("weight", "weight / (length + 1)")

    transforms, const_env, needs_lengths = weight_transform_codegen(net)
    assert needs_lengths is True                       # create_network must be handed lengths
    assert "length = distances" in const_env           # bound once, not per transform
    assert transforms[0][1] == ["weight = weights"]    # the target rebinds per transform
    assert np.allclose(_apply_emitted(net, W, L), np.asarray(net.weights_matrix))


def test_a_third_edge_attribute_resolves_at_runtime_but_not_in_a_kit():
    """The runtime reads any label through `matrix()`; a kit holds only what it is handed.

    Refusing at render time beats emitting a name the kit does not define and only
    discovering it once the job reaches a cluster.
    """
    W = np.array([[0, 2.0], [4.0, 0]])
    net = Network.from_matrix(weights=W, lengths=np.zeros((2, 2)))
    net.set_matrix("fc", np.array([[0, 0.5], [0.5, 0]]))
    net.add_transform("weight", "weight * fc")

    assert np.allclose(np.asarray(net.weights_matrix), W * np.array([[0, 0.5], [0.5, 0]]))
    with pytest.raises(ValueError, match="fc"):
        weight_transform_codegen(net)


def test_a_per_node_vector_is_guarded_by_node_count():
    """The runtime skips a mismatched vector; a kit would embed it and broadcast it wrongly.

    A cohort kit is rendered once from one subject, so an unguarded vector reaches every
    other subject's connectome at the wrong length.
    """
    from tvbo.datamodel.schema import Node

    net = Network.from_matrix(weights=np.zeros((3, 3)), lengths=np.zeros((3, 3)))
    net.nodes = [Node(id=i, label=f"n{i}", parameters={"roi_size": {"value": float(i + 1)}})
                 for i in range(2)]
    net.add_transform("weight", "weight / roi_size")
    with pytest.raises(ValueError, match="2 values, but the network has 3 nodes"):
        weight_transform_codegen(net)


def test_the_default_transform_is_written_over_its_own_target():
    """`add_transform` with no equation min-max normalises whatever it targets."""
    net = Network.from_matrix(weights=np.array([[0, 2.0], [4.0, 0]]), lengths=np.zeros((2, 2)))
    net.add_transform("weight")
    assert net.transforms[0].equation.rhs == "(weight - min(weight)) / (max(weight) - min(weight))"
    transformed = np.asarray(net.weights_matrix)
    assert transformed.min() == 0.0 and transformed.max() == 1.0


# ── A mask scopes a reduction, in whichever of the three spellings ──────────────────────


SPARSE = np.array([[0.0, 2, 1], [4, 0, 3], [1, 5, 0]])
MASKED_MEAN = SPARSE / SPARSE[SPARSE > 0].mean()

SPELLINGS = {
    "subscript": {"equation": {"rhs": "weight / mean(weight[weight > 0])"}},
    "argument": {"equation": {"rhs": "weight / mean(weight, weight > 0)"}},
}


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_every_mask_spelling_scopes_the_reduction_at_runtime(spelling):
    """A mask makes `mean(weight)` the mean of the existing edges, not of every cell.

    The whole point: a sparse connectome normalised by the all-entry mean is scaled by
    roughly 1/density — ~5x on a 20%-dense matrix — and nothing reports it.
    """
    net = Network.from_matrix(SPARSE, transforms=[{"name": "weight", **SPELLINGS[spelling]}])
    assert np.allclose(np.asarray(net.weights_matrix), MASKED_MEAN)


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_every_mask_spelling_reaches_the_kit_unchanged(spelling):
    """Codegen and runtime resolve one expression, so a kit cannot normalise differently.

    A masked expression is also a use of its *base* symbol: `mean(weight[weight > 0])`
    binds `weight`, not a symbol literally named `weight[weight > 0]`. Checking the raw
    free-symbol text rejected that as undeclared and refused to render the recipe.
    """
    net = Network.from_matrix(SPARSE, transforms=[{"name": "weight", **SPELLINGS[spelling]}])
    expr, chained = weight_transform_codegen(net)[0][0]
    assert "jnp.where" in expr, expr
    assert chained[0] == "weight = weights"
    assert chained[1:] == ["_mask0 = jnp.greater(weight, 0)"]   # bound once, not per mention
    assert expr.count("jnp.greater") == 0                       # the predicate is not re-emitted
    assert np.allclose(_apply_emitted(net, SPARSE), MASKED_MEAN)


def test_both_spellings_lower_to_the_same_source():
    """They are two notations for one node, so they cannot drift apart at the printer."""
    rendered = {
        name: weight_transform_codegen(
            Network.from_matrix(SPARSE, transforms=[{"name": "weight", **spec}])
        )[0][0][0]
        for name, spec in SPELLINGS.items()
    }
    assert len(set(rendered.values())) == 1, rendered


def test_two_reductions_carry_their_own_predicates():
    """A predicate scopes the reduction it is written on, so one equation can mix them."""
    net = Network.from_matrix(
        SPARSE,
        transforms=[{"name": "weight",
                     "equation": {"rhs": "mean(weight[weight > 0]) / max(weight, weight < 4)"}}],
    )
    expected = SPARSE[SPARSE > 0].mean() / SPARSE[SPARSE < 4].max()
    assert np.allclose(np.asarray(net.weights_matrix), expected)


def test_a_masked_extremum_fills_with_its_identity():
    """max over the empty set is -inf: an all-false mask must poison, not look plausible."""
    net = Network.from_matrix(SPARSE, transforms=[
        {"name": "weight", "equation": {"rhs": "weight / max(weight[weight < 4])"}}])
    assert np.allclose(np.asarray(net.weights_matrix), SPARSE / SPARSE[SPARSE < 4].max())

    empty = Network.from_matrix(SPARSE, transforms=[
        {"name": "weight", "equation": {"rhs": "max(weight[weight > 1e9])"}}])
    assert np.isneginf(np.asarray(empty.weights_matrix)).all()


def test_a_boolean_subscript_outside_a_reduction_is_refused():
    """`weight[weight > 0]` alone selects a data-dependent count, so it has no static shape.

    Inside a reduction it does — which is why the subscript spelling is accepted there and
    only there.
    """
    net = Network.from_matrix(
        np.array([[0.0, 2], [3, 0]]),
        transforms=[{"name": "weight", "equation": {"rhs": "weight[weight > 0] / 2"}}],
    )
    with pytest.raises(ValueError, match="outside a reduction"):
        _ = net.weights_matrix
