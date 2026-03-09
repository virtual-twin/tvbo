

## SwingingAtwood
The swinging Atwood machine couples a pendulum to a vertically moving mass via a
single inextensible string over a pulley. Competing exchanges between vertical and
angular motion create a rich mixed phase space with coexisting regular islands and
chaotic seas, providing a canonical laboratory example of low-dimensional Hamiltonian
chaos and resonance overlap.

### State Equations
$$
\dot{\theta} = \omega
$$
$$
\dot{\omega} = \frac{- g*\left(m_{1} + m_{2}\right)*\sin{\left(\theta \right)} - m_{1}*\omega^{2}*\cos{\left(\theta \right)}*\sin{\left(\theta \right)}}{l*\left(m_{1} + m_{2}\right) - l*m_{1}*\cos^{2}{\left(\theta \right)}}
$$
$$
\dot{y} = v
$$
$$
\dot{v} = \frac{g*\left(m_{2} - m_{1}*\sin^{2}{\left(\theta \right)}\right)}{m_{1} + m_{2} - m_{1}*\cos^{2}{\left(\theta \right)}} - \frac{l*m_{1}*\omega^{2}*\cos{\left(\theta \right)}}{m_{1} + m_{2} - m_{1}*\cos^{2}{\left(\theta \right)}}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $g$ | 9.81 | N/A | Gravitational acceleration. |
| $l$ | 1.0 | N/A | Length of the pendulum string segment (nondimensionalized scale). |
| $m_{1}$ | 1.0 | N/A | Mass of pendulum bob. |
| $m_{2}$ | 3.0 | N/A | Mass of counterweight. |

### Derived Quantities
#### Derived Variables
$$
energy = 0.5*m_{1}*\left(v^{2} + l^{2}*\omega^{2} + 2*l*\omega*v*\sin{\left(\theta \right)}\right) + 0.5*m_{2}*v^{2} + g*y*\left(m_{1} + m_{2}\right) + g*l*m_{1}*\left(1 - \cos{\left(\theta \right)}\right)
$$




