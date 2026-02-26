import jax.numpy as jnp
import numpy as np
from tvbo.knowledge.study import SimulationStudy

s = SimulationStudy.from_file('database/studies/Jansen1995/Jansen1995_extracted.yaml')
exp3 = s.get_experiment(3)
res = exp3.run('tvboptim')
er = res.exploration.K_sweep_symmetric_fig5
n0, n1 = er.shape
grid = er.results.reshape(n0, n1, -1, 2)
K_vals = np.array(er.axes[0]['values'])
print('K values:', K_vals)
print('grid shape:', grid.shape)

# Check stability across entire grid
max_abs = np.array(jnp.max(jnp.abs(grid), axis=2))
print('\nMax abs per (K0, K1) for node 0:')
for i in range(n0):
    row = [f'{float(max_abs[i,j,0]):8.1f}' for j in range(n1)]
    print(f'K0={float(K_vals[i]):3.0f}: {" ".join(row)}')

std_map = np.array(jnp.std(grid, axis=2))
print('\nStd per (K0, K1) for node 0:')
for i in range(n0):
    row = [f'{float(std_map[i,j,0]):6.2f}' for j in range(n1)]
    print(f'K0={float(K_vals[i]):3.0f}: {" ".join(row)}')
