

# VanDerPolForced
VanDerPolForced

## Equations


### State Equations
$$
\frac{d}{d t} x = y
$$
$$
\frac{d}{d t} y = - x + F*\sin{\left(\frac{2*\pi*t}{T} \right)} + \mu*y*\left(1 - x^{2}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\mu$ | 1.5 | N/A | Nonlinear damping parameter μ. |
| $F$ | 1.2 | N/A | Forcing amplitude F. |
| $T$ | 10.0 | N/A | Forcing period T. |



