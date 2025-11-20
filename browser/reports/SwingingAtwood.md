

# SwingingAtwood
SwingingAtwood

## Equations

### Derived Variables
$$
energy = 0.5*m_{1}*\left(v^{2} + l^{2}*\omega^{2} + 2*l*\omega*v*\sin{\left(\theta \right)}\right) + 0.5*m_{2}*v^{2} + g*y*\left(m_{1} + m_{2}\right) + g*l*m_{1}*\left(1 - \cos{\left(\theta \right)}\right)
$$

### State Equations
$$
\frac{d}{d t} \theta = \omega
$$
$$
\frac{d}{d t} \omega = \frac{- g*\left(m_{1} + m_{2}\right)*\sin{\left(\theta \right)} - m_{1}*\omega^{2}*\cos{\left(\theta \right)}*\sin{\left(\theta \right)}}{l*\left(m_{1} + m_{2}\right) - l*m_{1}*\cos^{2}{\left(\theta \right)}}
$$
$$
\frac{d}{d t} y = v
$$
$$
\frac{d}{d t} v = \frac{g*\left(m_{2} - m_{1}*\sin^{2}{\left(\theta \right)}\right)}{m_{1} + m_{2} - m_{1}*\cos^{2}{\left(\theta \right)}} - \frac{l*m_{1}*\omega^{2}*\cos{\left(\theta \right)}}{m_{1} + m_{2} - m_{1}*\cos^{2}{\left(\theta \right)}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $g$ | 9.81 | N/A | Gravitational acceleration. |
| $l$ | 1.0 | N/A | Length of the pendulum string segment (nondimensionalized scale). |
| $m_{1}$ | 1.0 | N/A | Mass of pendulum bob. |
| $m_{2}$ | 3.0 | N/A | Mass of counterweight. |



