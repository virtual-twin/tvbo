"""A declared initial-state distribution draws by one definition of its family, spread and bounds on tvb, jax and tvboptim.

The host draw that opens a tvb or jax run (`collect_initial_conditions`) and the draw the tvboptim module emits (`initial_condition_draw`) read the same declaration: a normal centred between the bounds with a quarter of their width as spread or a uniform between them, the bounds the distribution's own `domain`, else the state variable's, else (-1, 1). They read it through `initial_state_distribution`, which refuses by name what neither draw would honour: another family, a declared `mean` or `std`, and bounds that are not finite or not in order.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest

from tvbo import SimulationExperiment

N_NODES = 5
SEED = 7

ONE_VARIABLE = {
    "id": 1,
    "label": "drawn opening",
    "dynamics": {
        "name": "Drawn",
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": N_NODES, "nodes": [{"id": i, "label": f"r{i}"} for i in range(N_NODES)]},
    "execution": {"random_seed": SEED},
    "integration": {"method": "Euler", "step_size": 0.1, "duration": 1.0},
}


def _experiment(distribution, domain=None):
    """The one-variable recipe whose ``x`` declares *distribution* and, if given, the state-variable *domain*."""
    spec = copy.deepcopy(ONE_VARIABLE)
    variable = spec["dynamics"]["state_variables"]["x"]
    variable["distribution"] = distribution
    if domain is not None:
        variable["domain"] = domain
    return SimulationExperiment(**spec)


def _host_draw(experiment):
    """``x`` as the tvb and jax runs open on it."""
    return experiment.collect_initial_conditions().data[-1, 0, :, 0]


def _emitted_draw(experiment):
    """The family and parameters of the draw the tvboptim module emits for ``x``, read across whitespace since black lays the module out."""
    code = experiment.render_code("tvboptim")
    uniform = re.search(r"jax\.random\.uniform\(\s*_k0,\s*\(n_nodes,\),\s*minval=([-\d.e]+),\s*maxval=([-\d.e]+)\s*\)", code)
    if uniform:
        return "uniform", float(uniform[1]), float(uniform[2])
    normal = re.search(r"([-\d.e]+)\s*\+\s*([-\d.e]+)\s*\*\s*jax\.random\.normal\(\s*_k0,", code)
    return "normal", float(normal[1]), float(normal[2])


@pytest.mark.parametrize(
    "distribution, domain, expected",
    [
        ({"name": "Normal", "domain": {"lo": 1.0, "hi": 2.0}}, None, ("normal", 1.5, 0.25)),
        ({"name": "Uniform", "domain": {"lo": 0.5, "hi": 3.0}}, None, ("uniform", 0.5, 3.0)),
        ({"name": "Uniform"}, {"lo": -2.0, "hi": 4.0}, ("uniform", -2.0, 4.0)),
        ({"name": "Uniform"}, None, ("uniform", -1.0, 1.0)),
    ],
    ids=["normal-own-domain", "uniform-own-domain", "state-variable-domain", "default-bounds"],
)
def test_both_paths_draw_one_definition(distribution, domain, expected):
    """The module tvboptim runs emits the family and parameters, and the host draw is that family with those parameters on the generator the run's seed keys."""
    family, first, second = expected
    assert _emitted_draw(_experiment(distribution, domain)) == expected
    generator = np.random.default_rng((SEED, 0))
    reference = generator.normal(first, second, N_NODES) if family == "normal" else generator.uniform(first, second, N_NODES)
    np.testing.assert_array_equal(_host_draw(_experiment(distribution, domain)), reference)


REFUSED = [
    ({"name": "Uniform"}, {"lo": 0.0, "hi": float("inf")}, r"x: .*Declare `distribution\.domain` with finite bounds"),
    ({"name": "Uniform", "domain": {"lo": 3.0, "hi": 1.0}}, None, r"x: .*Declare `distribution\.domain` with finite bounds"),
    (
        {"name": "TruncatedNormal", "domain": {"lo": 0.0, "hi": 1.0}},
        None,
        r"x: an initial-state distribution draws uniform or normal; `TruncatedNormal` is not drawn here",
    ),
    (
        {"name": "Normal", "domain": {"lo": 0.0, "hi": 1.0}, "parameters": {"mean": {"value": 0.5}, "std": {"value": 0.1}}},
        None,
        r"x: an initial-state distribution draws from its bounds; `mean`/`std` are not read here; declare `domain`",
    ),
]


@pytest.mark.parametrize("path", [_host_draw, _emitted_draw], ids=["host", "tvboptim"])
@pytest.mark.parametrize(
    "distribution, domain, message", REFUSED, ids=["unbounded", "inverted", "other-family", "declared-moments"]
)
def test_declarations_no_draw_reads_are_refused(path, distribution, domain, message):
    """What neither draw would honour is refused by name on both paths: bounds from which a draw would be infinite (a half-open state variable under a distribution without a domain) or empty (inverted), a family other than uniform or normal, and a `mean`/`std` the bounds stand in for."""
    with pytest.raises(ValueError, match=message):
        path(_experiment(distribution, domain))
