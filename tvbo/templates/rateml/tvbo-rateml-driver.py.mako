## -*- coding: utf-8 -*-
##
## RateML-style CUDA Driver Template
## ==================================
##
## Generates PyCUDA driver code for running GPU simulations.
##
<%doc>
Context Variables:
- model: Dynamics instance (required)
- experiment: SimulationExperiment (optional)
- swept_params: list of parameter names to sweep

Output:
- Python driver script for PyCUDA-based simulation
</%doc>
<%
# Model info
model_name = model.name.replace(' ', '').replace('-', '')
model_name_lower = model_name.lower()

# State variables
state_vars = list(model.state_variables.items())
n_states = len(state_vars)

# Exposures
exposures = [name for name, sv in state_vars if getattr(sv, 'record', True)]
n_exposures = len(exposures)

# Swept parameters
if 'swept_params' not in context.keys():
    swept_params = ['global_speed', 'global_coupling']
n_swept = len(swept_params)
%>
from __future__ import print_function

import logging
import itertools
import argparse
import pickle
import os.path

import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
except ImportError:
    logging.warning('pycuda not available, CUDA driver not usable.')

import matplotlib.pyplot as plt
import tqdm

from tvb.simulator.lab import connectivity

here = os.path.dirname(os.path.abspath(__file__))


class DriverSetup:
    """Setup class for ${model_name} CUDA simulation."""

    def __init__(self):
        self.args = self.parse_args()
        self.logger = logging.getLogger('tvbo.rateml.${model_name_lower}')
        self.logger.setLevel(level='INFO' if self.args.verbose else 'WARNING')

        self.validate_args()

        self.dt = self.args.delta_time
        self.connectivity = self.load_connectivity(self.args.n_regions)
        self.weights = self.connectivity.weights
        self.lengths = self.connectivity.tract_lengths
        self.tavg_period = 1.0
        self.n_inner_steps = int(self.tavg_period / self.dt)

        self.params = self.setup_params()
        self.n_work_items, self.n_params = self.params.shape

        # Buffer length based on max delay
        self.buf_len_ = int((self.lengths / self.args.speeds_min / self.dt).max() + 1)
        self.buf_len = 2 ** int(np.ceil(np.log2(self.buf_len_)))

        self.states = ${n_states}
        self.exposures = ${n_exposures}

        self.log_config()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Run ${model_name} CUDA simulation.')

        # Swept parameter grid sizes
        % for i, sp in enumerate(swept_params):
        parser.add_argument('-s${i}', '--n_sweep_${sp}', default=4, type=int,
                            help='Grid points for ${sp}')
        % endfor

        parser.add_argument('-n', '--n_time', default=400, type=int, help='Number of time steps')
        parser.add_argument('-v', '--verbose', default=False, action='store_true')
        parser.add_argument('-m', '--model', default='${model_name_lower}', help='Model name')
        parser.add_argument('-s', '--states', default=${n_states}, type=int)
        parser.add_argument('-x', '--exposures', default=${n_exposures}, type=int)
        parser.add_argument('-bx', '--blockszx', default=8, type=int, help='GPU block size x')
        parser.add_argument('-by', '--blockszy', default=8, type=int, help='GPU block size y')
        parser.add_argument('-r', '--n_regions', default=68, type=int, help='Number of nodes')
        parser.add_argument('-dt', '--delta_time', default=0.1, type=float, help='Time step')
        parser.add_argument('-sm', '--speeds_min', default=1.0, type=float, help='Min speed for buffer')
        parser.add_argument('-p', '--plot_data', type=int, help='Plot state index')
        parser.add_argument('-w', '--write_data', default=False, action='store_true')

        args, _ = parser.parse_known_args()
        return args

    def validate_args(self):
        assert self.args.n_time > 0, "n_time must be > 0"
        assert self.args.n_regions > 0, "n_regions must be > 0"
        assert self.args.blockszx > 0 and self.args.blockszx <= 32
        assert self.args.blockszy > 0 and self.args.blockszy <= 32
        assert self.args.delta_time > 0

    def load_connectivity(self, n_nodes):
        conn = connectivity.Connectivity.from_file(
            source_file=f"connectivity_{n_nodes}.zip"
        )
        conn.configure()
        return conn

    def setup_params(self):
        """Setup parameter sweep grid."""
        % for i, sp in enumerate(swept_params):
        sweep_${sp} = np.linspace(self.args.${sp}_lo, self.args.${sp}_hi,
                                   getattr(self.args, 'n_sweep_${sp}', 4))
        % endfor

        params = itertools.product(
            % for i, sp in enumerate(swept_params):
            sweep_${sp}${',' if i < len(swept_params)-1 else ''}
            % endfor
        )
        return np.array([vals for vals in params], np.float32)

    def log_config(self):
        self.logger.info('dt: %f', self.dt)
        self.logger.info('n_nodes: %d', self.args.n_regions)
        self.logger.info('n_time: %d', self.args.n_time)
        self.logger.info('n_work_items: %d', self.n_work_items)
        self.logger.info('buf_len: %d', self.buf_len)
        self.logger.info('states: %d', self.states)


class DriverExecute(DriverSetup):
    """Execute ${model_name} CUDA simulation."""

    def __init__(self, setup=None):
        if setup is None:
            super().__init__()
        else:
            self.__dict__.update(setup.__dict__)

        self.cuda_file = os.path.join(here, '${model_name_lower}.c')

    def make_kernel(self):
        """Compile CUDA kernel."""
        with open(self.cuda_file, 'r') as f:
            source = f.read()
            source = source.replace('M_PI_F', f'{np.pi}f')

        opts = ['--ptxas-options=-v', '-maxrregcount=32']
        opts.append(f'-DWARP_SIZE=32')
        opts.append(f'-DNH={self.buf_len}')

        module = SourceModule(source, options=opts, no_extern_c=True)
        return module.get_function('${model_name}')

    def run(self):
        """Run the simulation."""
        kernel = self.make_kernel()

        n_nodes = self.args.n_regions
        n_work_items = self.n_work_items
        n_states = self.states
        n_exposures = self.exposures
        buf_len = self.buf_len

        # Allocate GPU memory
        weights_gpu = gpuarray.to_gpu(self.weights.astype(np.float32).flatten())
        lengths_gpu = gpuarray.to_gpu(self.lengths.astype(np.float32).flatten())
        params_gpu = gpuarray.to_gpu(self.params.T.astype(np.float32).flatten())
        state_gpu = gpuarray.zeros((buf_len * n_states * n_nodes * n_work_items,), np.float32)
        tavg_gpu = gpuarray.zeros((n_exposures * n_nodes * n_work_items,), np.float32)

        # Grid/block dimensions
        block = (self.args.blockszx, self.args.blockszy, 1)
        grid = (int(np.ceil(n_work_items / (block[0] * block[1]))), 1, 1)

        # Run simulation
        n_steps = self.args.n_time
        dt = np.float32(self.dt)

        for i_step in tqdm.trange(0, n_steps, self.n_inner_steps):
            kernel(
                np.uint32(i_step),
                np.uint32(n_nodes),
                np.uint32(buf_len),
                np.uint32(self.n_inner_steps),
                np.uint32(n_work_items),
                dt,
                weights_gpu,
                lengths_gpu,
                params_gpu,
                state_gpu,
                tavg_gpu,
                block=block,
                grid=grid
            )

        # Get results
        tavg = tavg_gpu.get().reshape((n_exposures, n_nodes, n_work_items))

        if self.args.write_data:
            np.save('tavg_${model_name_lower}.npy', tavg)

        if self.args.plot_data is not None:
            plt.figure()
            plt.imshow(tavg[self.args.plot_data, :, :], aspect='auto')
            plt.colorbar()
            plt.title('${model_name} - State ${exposures[0] if exposures else "0"}')
            plt.xlabel('Work item')
            plt.ylabel('Node')
            plt.show()

        return tavg


if __name__ == "__main__":
    import numpy as np
    from tvbo.data.types import SimulationResult, ExperimentResult
    import xarray as xr

    driver = DriverExecute()
    tavg = driver.run()
    print(f"Output shape: {tavg.shape}")

    # Wrap raw output in ExperimentResult
    da = xr.DataArray(data=np.asarray(tavg), dims=['time', 'node'][:tavg.ndim])
    sim = SimulationResult(data=da)
    results = ExperimentResult(integration=sim, name="${model_name}")
    print(results)
