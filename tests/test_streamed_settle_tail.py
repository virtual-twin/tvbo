"""A streaming reducer is handed only the tail of the settle its kernel can reach.

Every settle row passed to ``_stream_reduction`` is captured as a constant of the compiled fold. The base run trimmed the settle to ``_STREAMING_WARMUP_STEPS`` (the longest support any streamed kernel warms), while each exploration and the post-fit fold handed over the whole of it: a 60 s settle at 0.1 ms on 379 nodes put 3.69 GB of constants into every sweep's observable and a 31 GB footprint on the Glasser sweep. ``_settle_rows`` now does the trimming for every caller, which changes no value, since a kernel's ring reads no further back than its support.
"""

import re

import jax.numpy as jnp
import pytest
import xarray as xr

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.utils import Bunch  # noqa: E402

SPEC = """
label: Settled BOLD sweep
dynamics:
  name: Relax
  parameters: {k: {value: 0.5}}
  state_variables:
    x: {equation: {rhs: "k - x"}, initial_value: 0.0}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Relax}, {id: 1, label: B, dynamics: Relax}]
integration: {method: euler, step_size: 1.0, duration: 40000.0, transient_time: 60000.0, unit: ms}
observations:
  bold: {source: [x], OBSERVATION}
explorations:
  sweep:
    mode: product
    space:
      - parameter: Relax.k
        domain: {lo: 0.5, hi: 0.9, n: 2}
"""

KERNEL = "iri: 'tvbo:BOLD_HRF_strided', reduce: streaming"
SETTLE_STEPS = 60000


def _experiment(observation: str = KERNEL) -> SimulationExperiment:
    return SimulationExperiment.from_string(SPEC.replace("OBSERVATION", observation))


@pytest.fixture(scope="module")
def generated():
    return _experiment().execute("tvboptim")


def test_the_settle_handed_over_is_the_kernels_tail(generated):
    rows = jnp.arange(SETTLE_STEPS * 2.0).reshape(SETTLE_STEPS, 1, 2)
    tail = generated._settle_rows(Bunch(ys=rows))
    assert 0 < generated._STREAMING_WARMUP_STEPS < SETTLE_STEPS
    assert tail.shape == (generated._STREAMING_WARMUP_STEPS, 1, 2)
    assert bool((tail == rows[-generated._STREAMING_WARMUP_STEPS :]).all())


def test_no_settle_hands_over_nothing(generated):
    assert generated._settle_rows(None) is None


def test_a_kernel_free_stream_is_handed_no_settle():
    generated = _experiment("aggregation: mean, reduce: streaming").execute("tvboptim")
    assert generated._settle_rows(Bunch(ys=jnp.zeros((SETTLE_STEPS, 1, 2)))) is None


def test_every_reducer_takes_its_settle_through_the_one_trim():
    handed = re.findall(
        r"(?<!def )_stream_reduction\((?:[^()]|\([^()]*\))*?settle=([^,)]+)", _experiment().render_code("tvboptim")
    )
    assert handed, "the base run and the sweep both stream this experiment"
    assert all(s.startswith("_settle_rows(") for s in handed), handed


def test_a_swept_cell_at_the_base_parameters_reproduces_the_base_run():
    result = _experiment().run("tvboptim")
    base = result.observations.bold
    cell = result.explorations["sweep"].observations["bold"].sel({"Relax.k": 0.5}, drop=True).transpose(*base.dims)
    xr.testing.assert_allclose(cell, base.drop_vars([c for c in base.coords if c not in cell.coords]), rtol=1e-12)
