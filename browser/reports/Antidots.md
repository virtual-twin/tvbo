

# Antidots
Antidots

## Equations

### Derived Variables
$$
Uy = 0
$$
$$
Ux = 0
$$
$$
xt = -0.5 + \left(1.0*x \bmod 1\right)
$$
$$
yt = -0.5 + \left(1.0*y \bmod 1\right)
$$
$$
\rho = \sqrt{xt^{2} + yt^{2}}
$$

### State Equations
$$
\frac{d}{d t} x = vx
$$
$$
\frac{d}{d t} y = vy
$$
$$
\frac{d}{d t} vx = - Ux + 2*B*vy*\sqrt{2}
$$
$$
\frac{d}{d t} vy = - Uy - 2*B*vx*\sqrt{2}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $d_{0}$ | 0.5 | N/A | Effective antidot diameter. |
| $c$ | 0.2 | N/A | Smoothing factor c. |
| $B$ | 1.0 | N/A | Magnetic field strength (normalized). |



