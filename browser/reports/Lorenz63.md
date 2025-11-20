

# Lorenz63
Lorenz63

## Equations


### State Equations
$$
\frac{d}{d t} X = \sigma*\left(Y - X\right)
$$
$$
\frac{d}{d t} Y = - Y + X*\left(\rho - Z\right)
$$
$$
\frac{d}{d t} Z = X*Y - Z*\beta
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\sigma$ | 10.0 | N/A | Prandtl number (σ) controlling rate of convection. |
| $\rho$ | 28.0 | N/A | Rayleigh number (ρ) scaled parameter driving convection. |
| $\beta$ | 2.6666666666666665 | N/A | Geometric factor (β = 8/3) related to aspect ratio. |



