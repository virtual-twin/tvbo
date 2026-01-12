#!/usr/bin/env python
"""
CUDA Simulation Runner for BIH Cluster
=======================================

Usage on cluster:
    srun --partition=gpu --gres=gpu:1 --time=1:00:00 python run_cuda_cluster.py rwongwang_kernel.cu

Or submit as batch job:
    sbatch run_cuda.sh
"""
import sys
import numpy as np


def run_simulation(kernel_file, n_nodes=68, n_steps=10000, dt=0.1):
    """Run CUDA simulation from kernel file."""
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    
    # Load kernel source
    with open(kernel_file, 'r') as f:
        kernel_code = f.read()
    
    # Extract kernel name from source (assumes __global__ void NAME(...))
    import re
    match = re.search(r'__global__\s+void\s+(\w+)\s*\(', kernel_code)
    kernel_name = match.group(1) if match else 'model_dfun'
    
    # Compile
    mod = SourceModule(kernel_code)
    kernel = mod.get_function(kernel_name)
    
    # Simulation parameters
    n_state_vars = 2  # V, W for rwongwang
    nh = 512  # history buffer length
    n_work_items = 1  # single simulation
    
    # Allocate host arrays
    state = np.zeros((nh, n_state_vars * n_nodes), dtype=np.float32)
    weights = np.random.rand(n_nodes, n_nodes).astype(np.float32) * 0.01
    lengths = np.ones((n_nodes, n_nodes), dtype=np.float32) * 10.0  # delays in ms
    params = np.array([1.0, 1.0], dtype=np.float32)  # global_speed, global_coupling
    tavg = np.zeros(n_state_vars * n_nodes, dtype=np.float32)
    
    # Initialize state (V=0.998, W=0.121 from model defaults)
    state[0, :n_nodes] = 0.998  # V
    state[0, n_nodes:] = 0.121  # W
    
    # Allocate device arrays
    state_gpu = cuda.mem_alloc(state.nbytes)
    weights_gpu = cuda.mem_alloc(weights.nbytes)
    lengths_gpu = cuda.mem_alloc(lengths.nbytes)
    params_gpu = cuda.mem_alloc(params.nbytes)
    tavg_gpu = cuda.mem_alloc(tavg.nbytes)
    
    # Copy to device
    cuda.memcpy_htod(state_gpu, state)
    cuda.memcpy_htod(weights_gpu, weights.flatten())
    cuda.memcpy_htod(lengths_gpu, lengths.flatten())
    cuda.memcpy_htod(params_gpu, params)
    cuda.memcpy_htod(tavg_gpu, tavg)
    
    # Run kernel
    print(f"Running {kernel_name} on GPU...")
    print(f"  n_nodes: {n_nodes}")
    print(f"  n_steps: {n_steps}")
    print(f"  dt: {dt}")
    
    kernel(
        np.uint32(0),           # i_step
        np.uint32(n_nodes),     # n_node
        np.uint32(nh),          # nh
        np.uint32(n_steps),     # n_step
        np.uint32(n_work_items),# n_work_items
        np.float32(dt),         # dt
        weights_gpu,            # weights
        lengths_gpu,            # lengths
        params_gpu,             # params
        state_gpu,              # state
        tavg_gpu,               # tavg
        block=(1, 1, 1),
        grid=(1, 1)
    )
    
    # Get results
    cuda.memcpy_dtoh(tavg, tavg_gpu)
    cuda.memcpy_dtoh(state, state_gpu)
    
    print("\nTime-averaged state:")
    print(f"  V mean: {tavg[:n_nodes].mean():.6f}")
    print(f"  W mean: {tavg[n_nodes:].mean():.6f}")
    
    return tavg, state


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_cuda_cluster.py <kernel.cu> [n_nodes] [n_steps]")
        sys.exit(1)
    
    kernel_file = sys.argv[1]
    n_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 68
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    
    tavg, state = run_simulation(kernel_file, n_nodes, n_steps)
