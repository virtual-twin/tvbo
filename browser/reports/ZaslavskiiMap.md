

# ZaslavskiiMap
ZaslavskiiMap

## Equations

### Derived Variables
$$
\mu = \frac{1 - e^{- r}}{r}
$$

### State Equations
$$
x = \left(x + \nu*\left(1 + \mu*y\right) + eps*\mu*\nu*\cos{\left(2*\pi*x \right)}\right) \bmod 1
$$
$$
y = \left(y + eps*\cos{\left(2*\pi*x \right)}\right)*e^{- r}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $eps$ | 5.0 | N/A | Forcing amplitude ε. |
| $\nu$ | 0.2 | N/A | Coupling strength ν. |
| $r$ | 2.0 | N/A | Damping parameter r. |



