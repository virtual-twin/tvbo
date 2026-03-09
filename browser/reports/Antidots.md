

## Antidots
Hamiltonian antidot lattice (periodic Sinai-like billiard) with smooth circular
repulsive potentials and perpendicular magnetic field. Models guiding center
dynamics and magnetotransport phenomena in mesoscopic systems. Smooth radial
repulsion implemented as a quartic polynomial potential inside radius (d0/2 + c)
matching the original implementation (Datseris 2019); outside region force is zero.

### State Equations
$$
\dot{x} = vx
$$
$$
\dot{y} = vy
$$
$$
\dot{vx} = - Ux + 2*B*vy*\sqrt{2}
$$
$$
\dot{vy} = - Uy - 2*B*vx*\sqrt{2}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $d_{0}$ | 0.5 | N/A | Effective antidot diameter. |
| $c$ | 0.2 | N/A | Smoothing factor c. |
| $B$ | 1.0 | N/A | Magnetic field strength (normalized). |

### Derived Quantities
#### Derived Variables
$$
xt = -0.5 + \left(1.0*x \bmod 1\right)
$$
$$
yt = -0.5 + \left(1.0*y \bmod 1\right)
$$
$$
\rho = \sqrt{xt^{2} + yt^{2}}
$$
$$
Ux = \begin{cases} \frac{- 4*\left(0.5*d_{0} + c - \rho\right)^{3}*xt}{c^{4}*\left(\rho + 1.0*10^{-12}\right)} & \text{for}\: \rho < 0.5*d_{0} + c \\0 & \text{otherwise} \end{cases}
$$
$$
Uy = \begin{cases} \frac{- 4*\left(0.5*d_{0} + c - \rho\right)^{3}*yt}{c^{4}*\left(\rho + 1.0*10^{-12}\right)} & \text{for}\: \rho < 0.5*d_{0} + c \\0 & \text{otherwise} \end{cases}
$$




