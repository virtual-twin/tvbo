

## Lorenz84
Low-order atmospheric circulation model exhibiting multistability and fractal
basin boundaries with coexisting attractors under standard parameter set.

### State Equations
$$
\dot{x} = - y^{2} - z^{2} + F*a - a*x
$$
$$
\dot{y} = G - y + x*y - b*x*z
$$
$$
\dot{z} = - z + x*z + b*x*y
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $F$ | 6.846 | N/A | Baroclinic forcing F. |
| $G$ | 1.287 | N/A | Annual cycle modulation G. |
| $a$ | 0.25 | N/A | Linear damping parameter a. |
| $b$ | 4.0 | N/A | Coupling parameter b. |





