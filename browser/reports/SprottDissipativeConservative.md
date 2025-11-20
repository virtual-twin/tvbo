

# SprottDissipativeConservative
SprottDissipativeConservative

## Equations


### State Equations
$$
\frac{d}{d t} x = y + x*z + a*x*y
$$
$$
\frac{d}{d t} y = 1 - 2*x^{2} + b*y*z
$$
$$
\frac{d}{d t} z = - x^{2} - y^{2} + c*x
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 2.0 | N/A | Nonlinear mixed term coefficient a. |
| $b$ | 1.0 | N/A | yz coupling coefficient b. |
| $c$ | 1.0 | N/A | Linear term coefficient c in z equation. |



