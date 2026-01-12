

## Lorenz84
Low-order atmospheric circulation model exhibiting multistability and fractal
basin boundaries with coexisting attractors under standard parameter set.


### State Equations
$$
\frac{d}{d t} x = - y^{2} - z^{2} + F*a - a*x
$$
$$
\frac{d}{d t} y = G - y + x*y - b*x*z
$$
$$
\frac{d}{d t} z = - z + x*z + b*x*y
$$


### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $F$ | 6.846 | N/A | Baroclinic forcing F. |
| $G$ | 1.287 | N/A | Annual cycle modulation G. |
| $a$ | 0.25 | N/A | Linear damping parameter a. |
| $b$ | 4.0 | N/A | Coupling parameter b. |



