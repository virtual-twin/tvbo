

## VanDerPolForced
The forced van der Pol oscillator features nonlinear (amplitude dependent) damping and
sinusoidal forcing. The unforced system has a unique attracting relaxation cycle; with
forcing it displays entrainment, quasi-periodicity or chaos depending on parameters.

### State Equations
$$
\dot{x} = y
$$
$$
\dot{y} = - x + F*\sin{\left(\frac{2*\pi*t}{T} \right)} + \mu*y*\left(1 - x^{2}\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\mu$ | 1.5 | N/A | Nonlinear damping parameter μ. |
| $F$ | 1.2 | N/A | Forcing amplitude F. |
| $T$ | 10.0 | N/A | Forcing period T. |





