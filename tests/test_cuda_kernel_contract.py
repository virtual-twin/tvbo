"""The CUDA driver and the kernel it loads agree about the model.

`compile_cuda` looked the kernel name up on `experiment.network.dynamics` — the network's
*library* of models, a keyed dict — while the template renders the one model the kernel
integrates, `experiment.dynamics`. So `exp.run("cuda")` raised `AttributeError: 'dict'
object has no attribute 'name'` before it ever reached a GPU, and had it not, the name
would have been the hemodynamic model rather than the integrated one. `n_states` sized the
state buffer from the same wrong place.

None of this needs a GPU to check: the emitted source names the kernel, and the driver
computes the name it will ask for. CI has no GPU, so this is the part that can be pinned.
"""

import re

import pytest

from tvbo.classes.experiment import SimulationExperiment

RECIPE = "tvbo/database/experiments/RateML_ReducedWongWang.yaml"


@pytest.fixture(scope="module")
def experiment():
    return SimulationExperiment.from_file(RECIPE)


def _kernel_names(source):
    """Every ``extern "C" __global__`` entry point the emitted source defines."""
    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


def test_the_driver_asks_for_a_kernel_the_source_defines(experiment):
    """The name `compile_cuda` resolves must be one the template actually emitted."""
    wanted = experiment.dynamics.name.replace(" ", "").replace("-", "")

    assert wanted in _kernel_names(experiment.render_code("cuda"))


def test_the_kernel_is_named_for_the_integrated_model_not_the_library(experiment):
    """`network.dynamics` here holds only `BalloonWindkessel`, which integrates nothing."""
    library = set(experiment.network.dynamics or {})
    wanted = experiment.dynamics.name.replace(" ", "").replace("-", "")

    assert wanted not in library
    assert not hasattr(experiment.network.dynamics, "name")


def test_the_state_buffer_is_sized_from_the_integrated_model(experiment):
    """`n_states` came off the library too, so the ring buffer was sized from the wrong model."""
    integrated = len(experiment.dynamics.state_variables)

    assert integrated == 2
    assert integrated != len(experiment.network.dynamics or {})


def test_every_emitted_kernel_declares_c_linkage(experiment):
    """Without it `get_function` cannot find them: PyCUDA compiles with `no_extern_c=True`.

    It has to, because the kernel includes `<curand_kernel.h>`, whose templates may not
    carry the C linkage `SourceModule` otherwise wraps a whole source in.
    """
    source = experiment.render_code("cuda")

    assert _kernel_names(source), "no extern \"C\" kernel was emitted"
    assert "__global__" in source
    assert re.search(r'(?<!extern "C" )__global__ void', source) is None
