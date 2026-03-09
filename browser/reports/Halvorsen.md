

## Halvorsen
Smooth three-dimensional chaotic flow with rotational symmetry and robust strange
attractor arising from quadratic cross-couplings and constant drive. Exhibits
intertwined scroll-like structure and sensitive dependence useful for benchmarking
nonlinear state estimation and control strategies.

### State Equations
$$
\dot{x} = - y^{2} - 4*y - 4*z - a*x
$$
$$
\dot{y} = - z^{2} - 4*x - 4*z - a*y
$$
$$
\dot{z} = c - x^{2} - 4*x - 4*y - a*z
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 1.4 | N/A | Linear damping / coupling strength. |
| $b$ | 0.4 | N/A | Quadratic saturation coefficient. |
| $c$ | 1.0 | N/A | Constant drive parameter. |





