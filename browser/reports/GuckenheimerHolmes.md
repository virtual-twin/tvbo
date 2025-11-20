

# GuckenheimerHolmes
GuckenheimerHolmes

## Equations


### State Equations
$$
\frac{d}{d t} x = a*x + x*\left(y^{2} + z^{2}\right) + d*y*z
$$
$$
\frac{d}{d t} y = b*y + y*\left(x^{2} + z^{2}\right) + e*x*z
$$
$$
\frac{d}{d t} z = c*z + z*\left(x^{2} + y^{2}\right) + f*x*y
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 0.5 | N/A | Linear growth/damping parameter in x-equation. |
| $b$ | 0.5 | N/A | Linear growth/damping parameter in y-equation. |
| $c$ | 0.5 | N/A | Linear growth/damping parameter in z-equation. |
| $d$ | 1.0 | N/A | Nonlinear coupling parameter for x. |
| $e$ | 1.0 | N/A | Nonlinear coupling parameter for y. |
| $f$ | 1.0 | N/A | Nonlinear coupling parameter for z. |



