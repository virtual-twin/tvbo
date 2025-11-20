

# MagneticPendulum
MagneticPendulum

## Equations

### Derived Variables
$$
sum_{\gamma terms x} = \gamma*\left(\frac{-1 + x}{\left(d^{2} + y^{2} + \left(-1 + x\right)^{2}\right)^{\frac{3}{2}}} + \frac{\frac{1}{2} + x}{\left(d^{2} + \left(\frac{1}{2} + x\right)^{2} + \left(y + \frac{\sqrt{3}}{2}\right)^{2}\right)^{\frac{3}{2}}} + \frac{\frac{1}{2} + x}{\left(d^{2} + \left(\frac{1}{2} + x\right)^{2} + \left(y - \frac{\sqrt{3}}{2}\right)^{2}\right)^{\frac{3}{2}}}\right)
$$
$$
sum_{\gamma terms y} = \gamma*\left(\frac{y}{\left(d^{2} + y^{2} + \left(-1 + x\right)^{2}\right)^{\frac{3}{2}}} + \frac{y + \frac{\sqrt{3}}{2}}{\left(d^{2} + \left(\frac{1}{2} + x\right)^{2} + \left(y + \frac{\sqrt{3}}{2}\right)^{2}\right)^{\frac{3}{2}}} + \frac{y - \frac{\sqrt{3}}{2}}{\left(d^{2} + \left(\frac{1}{2} + x\right)^{2} + \left(y - \frac{\sqrt{3}}{2}\right)^{2}\right)^{\frac{3}{2}}}\right)
$$

### State Equations
$$
\frac{d}{d t} x = vx
$$
$$
\frac{d}{d t} y = vy
$$
$$
\frac{d}{d t} vx = - sum_{\gamma terms x} - \alpha*vx - x*\omega^{2}
$$
$$
\frac{d}{d t} vy = - sum_{\gamma terms y} - \alpha*vy - y*\omega^{2}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\gamma$ | 1.0 | N/A | Base magnetic strength γ (if γs not explicitly varied). |
| $d$ | 0.3 | N/A | Vertical offset of pendulum from magnet plane. |
| $\alpha$ | 0.2 | N/A | Linear damping coefficient α. |
| $\omega$ | 0.5 | N/A | Natural frequency ω. |
| $N$ | 3.0 | N/A | Number of magnets equally spaced on unit circle. |



