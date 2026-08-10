"""Weight `transforms:` are inlined in the generated tvboptim code (self-contained kit).

A declared connectome transform (e.g. ``log(W+1)/max(log(W+1))``) is applied at runtime by
``Network.weights_matrix``. A frozen/standalone kit must not depend on that: the codegen
renders the transform to pure ``jnp`` inside ``create_network`` and is handed the RAW weights,
so the transform stays declared in the spec, the raw SC stays in the network file, and the
exact op is visible in the script rather than hidden in tvbo runtime. These freeze the
raw/transformed accessor split, byte-identity of the inlined op against ``weights_matrix``,
and that the emitted network builder carries the transform as pure ``jnp``.
"""
import jax.numpy as jnp
import numpy as np

from tvbo.classes.network import Network
from tvbo.templates.tvboptim.utils import weight_transform_codegen

RHS = "log(W + 1) / max(log(W + 1))"


def _net_with_transform():
    W = np.array([[0, 4, 16, 0],
                  [4, 0, 0, 100],
                  [16, 0, 0, 1],
                  [0, 100, 1, 0]], float)
    net = Network.from_matrix(weights=W, lengths=np.zeros_like(W))
    net.add_transform("weight", RHS)
    return net, W


def test_raw_accessor_is_untouched_transformed_is_normalised():
    net, W = _net_with_transform()
    raw = np.asarray(net.raw_weights_matrix)
    transformed = np.asarray(net.weights_matrix)
    assert np.array_equal(raw, W)                    # raw_weights_matrix skips transforms
    assert not np.allclose(raw, transformed)         # the transform actually changed it
    assert transformed.max() <= 1.0 + 1e-6           # log(W+1)/max normalises to <= 1


def test_inline_transform_is_byte_identical_to_weights_matrix():
    """The rendered jax expr applied to the RAW weights reproduces ``weights_matrix`` exactly,
    so ``experiment.py`` passing ``raw_weights_matrix`` while ``create_network`` inlines the
    transform is a no-op for every working run."""
    net, _ = _net_with_transform()
    transforms, const_env = weight_transform_codegen(net)
    assert len(transforms) == 1
    assert const_env == []
    expr, matrix_env = transforms[0]
    assert expr.startswith("jnp.") and "W" in expr   # pure jnp, references the raw matrix

    weights = jnp.asarray(net.raw_weights_matrix)
    env = {"jnp": jnp, "distances": None, "weights": weights}
    for line in matrix_env:
        exec(line, env)
    inlined = np.asarray(eval(expr, env))
    assert np.array_equal(inlined, np.asarray(net.weights_matrix))


def test_network_without_transform_emits_nothing():
    net = Network.from_matrix(weights=np.array([[0, 1.0], [1.0, 0]]),
                              lengths=np.zeros((2, 2)))
    transforms, const_env = weight_transform_codegen(net)
    assert transforms == []
    assert const_env == []
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
    assert "jnp.log(1 + W)" in builder                 # transform inlined in the builder
    assert "weights_matrix" not in builder             # not delegated to tvbo runtime
    assert "_apply_transform" not in builder
