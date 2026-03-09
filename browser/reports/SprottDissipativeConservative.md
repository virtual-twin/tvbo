

## SprottDissipativeConservative
Mixed dissipative/conservative three-dimensional system where initial condition
choice yields either quasi-periodic torus motion or chaotic attractor behavior.

### State Equations
$$
\dot{x} = y + x*z + a*x*y
$$
$$
\dot{y} = 1 - 2*x^{2} + b*y*z
$$
$$
\dot{z} = - x^{2} - y^{2} + c*x
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 2.0 | N/A | Nonlinear mixed term coefficient a. |
| $b$ | 1.0 | N/A | yz coupling coefficient b. |
| $c$ | 1.0 | N/A | Linear term coefficient c in z equation. |





