

## HyperRoessler
Four-dimensional hyperchaotic extension of the Rössler system introducing an
additional variable that yields a second positive Lyapunov exponent over wide
parameter ranges while preserving the spiral-type attractor structure.


### State Equations
$$
\frac{d}{d t} x = - y - z
$$
$$
\frac{d}{d t} y = w + x + a*y
$$
$$
\frac{d}{d t} z = b + z*\left(x - c\right)
$$
$$
\frac{d}{d t} w = w*y - d*z
$$


### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 0.25 | N/A | Parameter a. |
| $b$ | 3.0 | N/A | Parameter b. |
| $c$ | 0.5 | N/A | Parameter c. |
| $d$ | 0.05 | N/A | Parameter d for fourth dimension coupling. |



