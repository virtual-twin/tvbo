

## Roessler
This three-dimensional continuous system is due to Rössler [^Rössler1976]. It is a
system that by design behaves similarly to the Lorenz system and displays a strange
attractor. However, it is easier to analyze qualitatively, as for example the attractor
is composed of a single manifold. Default values are the same as the original paper.

### State Equations
$$
\dot{x} = - y - z
$$
$$
\dot{y} = x + a*y
$$
$$
\dot{z} = b + z*\left(x - c\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 0.2 | N/A | Linear dissipation parameter in y equation. |
| $b$ | 0.2 | N/A | Constant drive term in z equation. |
| $c$ | 5.7 | N/A | Z-scaling parameter setting folding scale. |





