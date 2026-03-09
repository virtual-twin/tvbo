

## Lorenz63
The famous three dimensional system due to Lorenz [^Lorenz1963], shown to exhibit
so-called "deterministic nonperiodic flow". It was originally invented to study a
simplified form of atmospheric convection.

Currently, it is most famous for its strange attractor (occuring at the default
parameters), which resembles a butterfly. For the same reason it is
also associated with the term "butterfly effect" (a term which Lorenz himself disliked)
even though the effect applies generally to dynamical systems.
Default values are the ones used in the original paper.

### State Equations
$$
\dot{X} = \sigma*\left(Y - X\right)
$$
$$
\dot{Y} = - Y + X*\left(\rho - Z\right)
$$
$$
\dot{Z} = X*Y - Z*\beta
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\sigma$ | 10.0 | N/A | Prandtl number (σ) controlling rate of convection. |
| $\rho$ | 28.0 | N/A | Rayleigh number (ρ) scaled parameter driving convection. |
| $\beta$ | 2.6666666666666665 | N/A | Geometric factor (β = 8/3) related to aspect ratio. |





