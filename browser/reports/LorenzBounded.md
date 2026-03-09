

## LorenzBounded
Lorenz system modified by smooth radial confinement factor f = 1 - (||X||^2)/r^2
producing bounded dynamics with altered basin geometry while retaining core
stretching and folding mechanisms.

### State Equations
$$
\dot{X} = f*\sigma*\left(Y - X\right)
$$
$$
\dot{Y} = f*\left(- Y + X*\left(\rho - Z\right)\right)
$$
$$
\dot{Z} = f*\left(X*Y - Z*\beta\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\beta$ | 2.667 | N/A | β parameter from original Lorenz with confinement. |
| $r$ | 64.0 | N/A | Radius of confining potential sphere. |
| $\rho$ | 28.0 | N/A | ρ parameter as in Lorenz. |
| $\sigma$ | 10.0 | N/A | σ parameter as in Lorenz. |

### Derived Quantities
#### Derived Variables
$$
f = 1 - \frac{X^{2} + Y^{2} + Z^{2}}{r^{2}}
$$




