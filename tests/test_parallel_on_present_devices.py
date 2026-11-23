"""A Pareto search and its refinement run on the devices a process has, and give every seed the same result on any device count.

The tvboptim backend pmapped an ``nsga2`` population and the refinement from its Pareto seeds over a width baked into the generated code (the refinement's ``execution.n_workers``), whatever the device count, so a one-device run of ``Hopf_Pareto_ParallelOpt`` stopped inside ``ParallelExecution.run`` with ``cannot select an axis to squeeze out … shape=(2, 4)``. The population's own ``n_parallel`` was accepted and then ignored. Both now take ``(n_pmap, n_vmap)`` from ``device_split`` at run time, the population at its exploration's ``n_parallel`` and the refinement at ``execution.n_workers``: at most the devices present, the rest of the width vectorised on each.

XLA fixes the host device count when JAX initialises, so each device count runs the experiment in its own interpreter with ``--xla_force_host_platform_device_count``; the reference is the one-device run, compared seed by seed.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tvboptim")

import tvbo
from tvbo import SimulationExperiment

WIDTH = 4
"""The refinement's declared width, ``execution.n_workers``, and the population size: two devices divide it and three do not."""

PROBE = {
    "id": 1,
    "label": "Pareto refinement probe",
    "dynamics": {
        "name": "Probe",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"p": {"value": 0.5}},
        "coupling_inputs": {"c": {}},
        "state_variables": {"x": {"equation": {"rhs": "p - x + c"}, "initial_value": 0.0, "coupling_variable": True}},
    },
    "network": {
        "number_of_nodes": 2,
        "edges": [{"source": 0, "target": 1, "weight": 0.5}, {"source": 1, "target": 0, "weight": 0.25}],
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["x"],
                "pre_expression": {"rhs": "x"},
                "post_expression": {"rhs": "gx"},
            }
        },
    },
    "functions": {
        "mean_x": {"source_code": "jnp.mean(data[:, 0, :])", "arguments": {"data": {}}},
        "minus_mean_x": {"source_code": "-jnp.mean(data[:, 0, :])", "arguments": {"data": {}}},
        "combined_loss_fn": {"source_code": "fc_level ** 2 + freq_level", "arguments": {"fc_level": {}, "freq_level": {}}},
    },
    "observations": {
        name: {
            "source": ["x"],
            "pipeline": [{"output": name, "function": fn, "arguments": {"data": {"value": "integration.result"}}}],
        }
        for name, fn in (("fc_level", "mean_x"), ("freq_level", "minus_mean_x"))
    },
    "explorations": {
        "ga": {
            "name": "ga",
            "strategy": "nsga2",
            "objectives": ["fc_level", "freq_level"],
            "space": [{"parameter": "Probe.p", "domain": {"lo": 0.0, "hi": 2.0}}],
            "parameters": {
                "population_size": {"value": WIDTH},
                "num_generations": {"value": 2},
                "seed": {"value": 42},
                "reference_point": {"value": [3.0, 1.0]},
            },
        }
    },
    "optimizations": {
        "refine": {
            "name": "refine",
            "depends_on": "ga",
            "loss": {
                "function": "combined_loss_fn",
                "arguments": {name: {"value": f"observations.{name}.data"} for name in ("fc_level", "freq_level")},
            },
            "free_parameters": [{"parameter": "Probe.p"}],
            "algorithm": "adam",
            "learning_rate": 0.01,
            "max_iterations": 3,
        }
    },
    "integration": {"method": "Heun", "step_size": 1.0, "duration": 20.0, "transient_time": 0.0},
    "execution": {"n_workers": WIDTH, "precision": "float64", "random_seed": 0},
}
"""Two undelayed coupled nodes, each relaxing towards ``p`` plus what the other sends. The two objectives are the mean of ``x`` and its negative, so every distinct individual is on the Pareto front, and the refinement descends ``m ** 2 - m`` towards a mean of one half from each seed. The exploration leaves ``n_parallel`` at ``auto``, so every process evaluates one individual per device, each as a batch of one, and draws the same front."""

RUN = """
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from tvbo import SimulationExperiment
res = SimulationExperiment(**json.loads(sys.stdin.read())).run("tvboptim", mode="all")
front = np.atleast_2d(np.asarray(res.explorations.ga.pareto_X))
seeds = {f"{x[0]:.12g}": [float(np.asarray(s.p_final)), float(np.asarray(s.final_fc_level))] for x, s in zip(front, res.optimizations.refine.seeds, strict=True)}
print("RESULT " + json.dumps({"devices": jax.local_device_count(), "seeds": seeds}))
"""
"""Runs the probe on tvboptim and prints, keyed by each Pareto seed's ``p``, its refined ``p`` and the mean of ``x`` there."""


def _run_on(devices):
    """The probe run in a fresh interpreter with *devices* host devices, the finished process."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices} --xla_cpu_multi_thread_eigen=false"
    env["PYTHONPATH"] = os.pathsep.join([str(Path(tvbo.__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", RUN], input=json.dumps(PROBE), env=env, capture_output=True, text=True, timeout=900
    )


def _printed(done):
    """What the run *done* printed, ``{"devices": n, "seeds": {...}}``, once it is asserted to have finished."""
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert done.returncode == 0 and lines, f"the run failed:\n{done.stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def one_device():
    """The reference run, on one device; asserted in the tests, so a failed run reports as their failure."""
    return _run_on(1)


@pytest.mark.parametrize("devices", [2, 3], ids=["divides-the-width", "pads-the-width"])
def test_each_seed_refines_the_same_on_any_device_count(one_device, devices):
    """Every Pareto seed refines to the same ``p`` and mean on one device as on *devices*, the two runs matched by seed."""
    one, several = _printed(one_device), _printed(_run_on(devices))
    assert (one["devices"], several["devices"]) == (1, devices)
    reference, seeds = one["seeds"], several["seeds"]
    assert len(reference) > 1, "the front must hold several seeds for the comparison to mean anything"
    assert all(abs(p_final - float(seed)) > 1e-6 for seed, (p_final, _) in reference.items()), (
        "the refinement must move every seed"
    )
    assert sorted(seeds) == sorted(reference)
    for seed, values in reference.items():
        np.testing.assert_allclose(seeds[seed], values, rtol=1e-9, atol=1e-12, err_msg=f"seed p = {seed}")


@pytest.mark.parametrize(
    ("devices", "width", "items", "split"),
    [(1, 4, 8, (1, 4)), (8, 2, 8, (2, 1)), (3, 4, 4, (3, 2)), (8, 8, 3, (3, 1)), (2, 1, 5, (1, 1))],
    ids=[
        "one-device-vectorises-the-width",
        "width-below-the-devices",
        "width-the-devices-do-not-divide",
        "fewer-items-than-the-width",
        "width-one-stays-serial",
    ],
)
def test_the_split_never_pmaps_past_the_devices_present(monkeypatch, devices, width, items, split):
    """``device_split`` pmaps over at most the devices present and the items there are, and vectorises the rest of the width."""
    import jax

    from tvbo.templates.tvboptim.callbacks import device_split

    monkeypatch.setattr(jax, "local_device_count", lambda *args, **kwargs: devices)
    assert device_split(width, items) == split


def test_the_rendered_search_fixes_no_pmap_width():
    """The emitted population and refinement read their pmap width from the devices at run time, never a literal fixed at render time."""
    code = SimulationExperiment(**copy.deepcopy(PROBE)).render_code("tvboptim")
    assert re.findall(r"\bn_pmap=\d+", code) == []
    assert f"device_split({WIDTH}, len(_pareto_X))" in code


def _population_widths(spec):
    """The widths, unquoted, that *spec*'s rendered population split is called with."""
    code = SimulationExperiment(**spec).render_code("tvboptim")
    return [width.strip("'\"") for width in re.findall(r"device_split\(([^,]+), _X\.shape\[0\]\)", code)]


@pytest.fixture(scope="module")
def population_widths():
    """The rendered population split's width for the probe as written, whose exploration leaves ``n_parallel`` at ``auto``, and with ``n_parallel: 3``."""
    declared = copy.deepcopy(PROBE)
    declared["explorations"]["ga"]["n_parallel"] = 3
    return {"auto": _population_widths(copy.deepcopy(PROBE)), 3: _population_widths(declared)}


@pytest.mark.parametrize(
    ("n_parallel", "devices", "split"),
    [(3, 1, (1, 3)), (3, 2, (2, 2)), (3, 8, (3, 1)), ("auto", 1, (1, 1)), ("auto", 2, (2, 1)), ("auto", 8, (4, 1))],
    ids=["3-one-device", "3-two-devices", "3-eight-devices", "auto-one-device", "auto-two-devices", "auto-eight-devices"],
)
def test_a_declared_n_parallel_reaches_the_population_split(monkeypatch, population_widths, n_parallel, devices, split):
    """The exploration's ``n_parallel`` is what the emitted population split is called with, and on *devices* it spreads the population of four as ``device_split`` rules: a number is the exact width, ``auto`` one individual per device, capped by the population on eight."""
    import jax

    from tvbo.templates.tvboptim.callbacks import device_split

    assert population_widths[n_parallel] == [str(n_parallel)]
    monkeypatch.setattr(jax, "local_device_count", lambda *args, **kwargs: devices)
    assert device_split(n_parallel, WIDTH) == split
