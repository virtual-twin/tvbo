

# CoupledRoessler
CoupledRoessler

## Equations


### State Equations
$$
\frac{d}{d t} x_{1} = - z_{1} - \omega_{1}*y_{1}
$$
$$
\frac{d}{d t} y_{1} = a*y_{1} + k_{1}*\left(y_{2} - y_{1}\right) + \omega_{1}*x_{1}
$$
$$
\frac{d}{d t} z_{1} = b + z_{1}*\left(x_{1} - c\right)
$$
$$
\frac{d}{d t} x_{2} = - z_{2} - \omega_{2}*y_{2}
$$
$$
\frac{d}{d t} y_{2} = a*y_{2} + k_{2}*\left(y_{1} - y_{2}\right) + \omega_{2}*x_{2}
$$
$$
\frac{d}{d t} z_{2} = b + z_{2}*\left(x_{2} - c\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\omega_{1}$ | 0.18 | N/A | Frequency scaling of first oscillator. |
| $\omega_{2}$ | 0.22 | N/A | Frequency scaling of second oscillator. |
| $a$ | 0.2 | N/A | Linear term coefficient a. |
| $b$ | 0.2 | N/A | Drive parameter b. |
| $c$ | 5.7 | N/A | Folding parameter c. |
| $k_{1}$ | 0.115 | N/A | Diffusive coupling from oscillator 2 to 1. |
| $k_{2}$ | 0.0 | N/A | Diffusive coupling from oscillator 1 to 2. |



