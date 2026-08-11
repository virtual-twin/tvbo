# -*- coding: utf-8 -*-
"""
CUDA Adapter for TVBO
=====================

Run generated CUDA kernels using PyCUDA.

Usage:
    from tvbo import SimulationExperiment

    exp = SimulationExperiment.from_file("experiment.yaml")
    result = exp.run('cuda')
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.classes.experiment import SimulationExperiment


def compile_cuda(experiment: SimulationExperiment) -> Tuple[Any, Any]:
    """Compile CUDA kernel for experiment.

    Compiled with `no_extern_c=True`. The kernel includes `<curand_kernel.h>` for its
    noise, and the block `SourceModule` otherwise wraps a whole source in gives those
    headers C linkage — which their templates may not have, so every compile failed with
    33 errors out of curand rather than anything to do with the model. The kernels declare
    `extern "C"` themselves instead, which is what keeps `get_function` able to find them
    under their unmangled names.

    Args:
        experiment: SimulationExperiment instance

    Returns:
        Tuple of (module, kernel_func)
    """
    try:
        import pycuda.autoinit  # noqa: F401
        from pycuda.compiler import SourceModule
    except ImportError:
        raise ImportError(
            "PyCUDA not installed. Install with: pip install pycuda\nNote: Requires NVIDIA GPU and CUDA toolkit."
        )

    cuda_source = experiment.render_code("cuda")
    module = SourceModule(cuda_source, no_extern_c=True)

    # The kernel is named for the model it integrates, which is what the template renders.
    model_name = experiment.dynamics.name.replace(" ", "").replace("-", "")
    kernel = module.get_function(model_name)

    return module, kernel


def run_cuda(
    experiment: SimulationExperiment,
    n_steps: Optional[int] = None,
    dt: Optional[float] = None,
    n_work_items: int = 1,
    global_speed: float = 1.0,
    global_coupling: float = 0.1,
    buffer_length: Optional[int] = None,
    swept_params: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Run CUDA simulation for experiment.

    All configuration comes from experiment metadata.

    The kernel integrates in the model's own time unit — it multiplies *dt* straight into
    the model's equations — while indexing the delay ring as ``length / speed / dt``, which
    is millimetres over metres-per-second and so is milliseconds. The two agree only for a
    model whose time unit is ``ms``; anything else gets the right trajectory on the wrong
    delays, so *dt* is passed through in model units rather than converted onto either.

    Args:
        experiment: SimulationExperiment instance
        n_steps: Number of integration steps (default from experiment.integration)
        dt: Integration time step in the model's own time unit (default: its `step_size`)
        n_work_items: Number of parallel parameter configurations
        global_speed: Conduction speed (m/s)
        global_coupling: Global coupling strength
        buffer_length: History buffer length (auto-calculated if None)
        swept_params: Dict mapping param names to arrays of values per work item

    Returns:
        Dict with 'tavg', 'n_node', 'n_steps', 'dt'
    """
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    # Compile kernel
    module, kernel = compile_cuda(experiment)

    # Get from experiment metadata
    if dt is None:
        dt = float(experiment.integration.step_size)
    if n_steps is None:
        n_steps = int(float(experiment.integration.duration) / dt)

    # Network data from experiment
    weights = np.asarray(experiment.network.weights, dtype=np.float32)
    lengths = np.asarray(experiment.network.lengths, dtype=np.float32)
    n_node = weights.shape[0]
    n_states = len(experiment.dynamics.state_variables)

    # The ring must hold the LONGEST delay any work item sees, so the slowest speed sizes it.
    if buffer_length is None:
        swept_speed = (swept_params or {}).get("global_speed")
        slowest = float(np.min(swept_speed)) if swept_speed is not None else float(global_speed)
        max_delay = np.max(lengths) / slowest / dt
        buffer_length = max(int(max_delay) + 10, 100)
    nh = np.uint32(buffer_length)

    # Transfer to GPU
    weights_gpu = gpuarray.to_gpu(weights.flatten())
    lengths_gpu = gpuarray.to_gpu(lengths.flatten())

    # Swept parameters
    if swept_params is None:
        params_host = np.array(
            [
                [global_speed] * n_work_items,
                [global_coupling] * n_work_items,
            ],
            dtype=np.float32,
        )
    else:
        params_list = []
        for name in ["global_speed", "global_coupling"]:
            if name in swept_params:
                params_list.append(swept_params[name].astype(np.float32))
            else:
                val = global_speed if name == "global_speed" else global_coupling
                params_list.append(np.full(n_work_items, val, dtype=np.float32))
        params_host = np.array(params_list, dtype=np.float32)

    params_gpu = gpuarray.to_gpu(params_host.flatten())

    # State buffer
    state_size = int(nh) * n_states * n_node * n_work_items
    state_gpu = gpuarray.zeros(state_size, dtype=np.float32)

    # Time-averaged observable
    tavg_size = n_states * n_node * n_work_items
    tavg_gpu = gpuarray.zeros(tavg_size, dtype=np.float32)

    # Grid/block dimensions
    block = (32, 1, 1)
    grid = ((n_work_items + 31) // 32, 1)

    # Run kernel
    kernel(
        np.uint32(0),
        np.uint32(n_node),
        nh,
        np.uint32(n_steps),
        np.uint32(n_work_items),
        np.float32(dt),
        weights_gpu.gpudata,
        lengths_gpu.gpudata,
        params_gpu.gpudata,
        state_gpu.gpudata,
        tavg_gpu.gpudata,
        block=block,
        grid=grid,
    )
    cuda.Context.synchronize()

    # Get results
    tavg_host = tavg_gpu.get().reshape(n_states, n_node, n_work_items)
    tavg_host = np.transpose(tavg_host, (2, 0, 1))  # [n_work_items, n_states, n_node]

    return {
        "tavg": tavg_host,
        "n_node": n_node,
        "n_steps": n_steps,
        "dt": dt,
    }


def save_cuda(experiment: SimulationExperiment, path: str) -> str:
    """Save CUDA source to file.

    Args:
        experiment: SimulationExperiment instance
        path: Output file path (.cu)

    Returns:
        Path to saved file
    """
    cuda_source = experiment.render_code("cuda")
    Path(path).write_text(cuda_source)
    return path
