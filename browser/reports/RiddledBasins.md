

# RiddledBasins
RiddledBasins

## Equations


### State Equations
$$
\frac{d}{d t} x = vx
$$
$$
\frac{d}{d t} y = vy
$$
$$
\frac{d}{d t} vx = - y^{2} - \gamma*vx + 4*x*\left(1 - x^{2}\right) + f_{0}*x_{0}*\sin{\left(\omega*t \right)}
$$
$$
\frac{d}{d t} vy = - \gamma*vy - 2*y*\left(x + \bar{x}\right) + f_{0}*y_{0}*\sin{\left(\omega*t \right)}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\gamma$ | 0.05 | N/A | Damping coefficient γ. |
| $\bar{x}$ | 1.9 | N/A | Shift parameter x̄. |
| $f_{0}$ | 2.3 | N/A | Forcing amplitude f₀. |
| $\omega$ | 3.5 | N/A | Forcing angular frequency ω. |
| $x_{0}$ | 1.0 | N/A | Forcing x-projection coefficient. |
| $y_{0}$ | 0.0 | N/A | Forcing y-projection coefficient. |



