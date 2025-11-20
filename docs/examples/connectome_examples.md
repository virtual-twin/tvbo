# Connectome Examples

This document provides detailed examples for working with the `Connectome` class.

## Loading Normative Connectomes

Load a connectome from a brain atlas:

```python
from tvbo.data.tvbo_data.connectomes import Connectome

# Load connectome with DesikanKilliany atlas
sc = Connectome(
    parcellation={"atlas": {"name": "DesikanKilliany"}},
)

# Visualize the connectivity matrices
fig = sc.plot_matrix()

# Access connectivity data
weights = sc.weights_matrix  # Connection weights
lengths = sc.lengths_matrix  # Tract lengths
print(f"Number of regions: {sc.number_of_regions}")
```

## Creating Custom Connectomes

Create a connectome from numpy arrays:

```python
import numpy as np
from tvbo.data.tvbo_data.connectomes import Connectome

# Create custom connectivity matrices
n_regions = 10
weights = np.random.rand(n_regions, n_regions)
lengths = np.random.rand(n_regions, n_regions) * 100

sc = Connectome(
    weights=weights,
    lengths=lengths,
    number_of_regions=n_regions
)

# Compute delays based on conduction speed
delays = sc.calculate_delays(conduction_speed=3.0)
```

## Graph Visualization

Visualize the connectome as a network graph:

```python
import matplotlib.pyplot as plt
from tvbo.data.tvbo_data.connectomes import Connectome

# Load a connectome
sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

# Create a simple graph visualization
fig, ax = plt.subplots(figsize=(10, 10))
mappable = sc.plot_graph(
    ax=ax,
    node_size="in-strength",
    edge_color="weight",
    threshold_percentile=75
)
plt.colorbar(mappable, ax=ax, label="Connection Strength")
plt.show()
```

## Brain-Based Layout

Visualize with anatomical brain positions:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 12))
sc.plot_graph(
    ax=ax,
    plot_brain="horizontal",  # or "sagittal", "coronal"
    node_size="in-strength",
    edge_color="weight",
    threshold_percentile=90,
    node_labels=False
)
ax.set_title("Horizontal Brain View")
plt.show()
```

## Complete Overview

Create a comprehensive visualization with matrices and graph:

```python
from tvbo.data.tvbo_data.connectomes import Connectome

sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

# Complete overview with all visualizations
fig = sc.plot_overview(
    log_weights=True,
    graph_kwargs={"threshold_percentile": 80, "node_labels": False}
)
fig.savefig("connectome_overview.png", dpi=300, bbox_inches="tight")
```

## Working with TVB

Convert to TVB format for use with The Virtual Brain:

```python
from tvbo.data.tvbo_data.connectomes import Connectome

sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

# Convert to TVB connectivity object
tvb_conn = sc.execute(format="tvb")

# Now use with TVB simulator
# simulator = tvb.simulator.Simulator(connectivity=tvb_conn, ...)
```

## Normalization

Normalize connectivity weights:

```python
from tvbo.data.tvbo_data.connectomes import Connectome

sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

# Apply min-max normalization
sc.normalize_weights(equation_rhs="(W - W_min) / (W_max - W_min)")

# Access normalized weights
normalized_weights = sc.weights_matrix
```

## JAX Compatibility

Use with JAX transformations:

```python
import jax
import jax.numpy as jnp
from tvbo.data.tvbo_data.connectomes import Connectome

sc = Connectome(parcellation={"atlas": {"name": "DesikanKilliany"}})

# Connectome is a JAX pytree
def compute_mean_weight(conn):
    return jnp.mean(conn.weights_matrix)

# Use with JAX transformations
grad_fn = jax.grad(compute_mean_weight)
# result = grad_fn(sc)
```
