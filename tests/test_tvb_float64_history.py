"""A tvb run at the declared ``execution.precision: float64`` keeps its delay history, and the weights its coupling reads, in float64.

TVB declares those arrays float32 (``NDArray(..., 'f')`` in ``tvb/simulator/history.py``), and an ``NDArray`` allocates once and casts every assignment, so the coupling of a float64 run read its sources and weights rounded to about 6e-8. The probe is the one of ``test_delay_history_from_opening``: ``x`` holds a drawn opening and is sent over a 4 ms tract, ``y`` integrates what arrives, so ``y(t) = t a W x(0)`` in closed form. A weight of 0.3 and an opening drawn from [1, 2] are not dyadic, so float32 rounds both.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvb")

from tvb.simulator.descriptors import NDArray
from tvb.simulator.history import SparseHistory
from tvb.simulator.simulator import Simulator

from tvbo import SimulationExperiment

from .test_delay_history_from_opening import W, _assert_trace_reads_its_opening, _declared_per_node, _drawn, _trace

pytestmark = pytest.mark.backend_tvb

WEIGHT = 0.3
FLOAT32_ARRAYS = ("weights", "delays", "es_weights", "buffer", "current_state", "delayed_state", "nnz_weights")
"""The arrays TVB's ``SparseHistory`` and its bases declare float32."""


def _float_arrays(history):
    """The floating-point arrays *history*'s class declares, its bases' included, by name."""
    attrs = {name: getattr(type(history), name) for name in dir(type(history))}
    return {name for name, attr in attrs.items() if isinstance(attr, NDArray) and np.dtype(attr.dtype).kind == "f"}


def test_a_float64_run_keeps_its_delay_history_in_float64():
    """Every floating-point array of the configured history is float64, the seven TVB declares float32 among them."""
    history = SimulationExperiment(**_drawn()).execute("tvb").history
    declared = _float_arrays(history)
    assert set(FLOAT32_ARRAYS) <= declared
    for name in declared:
        assert getattr(history, name).dtype == np.float64, name


@pytest.mark.parametrize("opening", [_drawn, _declared_per_node], ids=["drawn", "declared"])
def test_an_opening_on_a_non_dyadic_weight_reads_the_closed_form(opening):
    """``y`` is ``t a W x(0)`` at 1e-8 with every weight 0.3.

    At float32 the drawn opening is off by float32's rounding of the weight, the buffer and their product, which may partly cancel; the declared opening (1.5 and -0.5, exact in float32, their products with float32's 0.3 exact as well) is off by float32's rounding of 0.3 alone, 4e-8 relative.
    """
    spec = opening()
    for edge in spec["network"]["edges"]:
        edge["parameters"]["weight"]["value"] = WEIGHT
    _assert_trace_reads_its_opening(_trace(spec, "tvb"), weights=np.where(W != 0, WEIGHT, 0.0))


def test_a_float32_run_is_tvb_s_own():
    """Under ``precision: float32`` the simulator and its history are TVB's own classes, and the run is the one TVB's ``Simulator`` gives on the same components, bit for bit."""
    spec = _drawn()
    spec["execution"]["precision"] = "float32"
    sim = SimulationExperiment(**spec).execute("tvb")
    assert type(sim) is Simulator
    assert type(sim.history) is SparseHistory
    stock = Simulator(
        model=sim.model,
        connectivity=sim.connectivity,
        coupling=sim.coupling,
        conduction_speed=sim.conduction_speed,
        integrator=sim.integrator,
        monitors=[type(sim.monitors[0])()],
        simulation_length=sim.simulation_length,
        initial_conditions=sim.initial_conditions,
    ).configure()
    (_, ours), *_ = sim.run()
    ((_, theirs),) = stock.run()
    np.testing.assert_array_equal(ours, theirs)
