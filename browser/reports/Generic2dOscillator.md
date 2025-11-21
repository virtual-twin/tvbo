

# Generic2dOscillator
The Generic 2-Dimensional Oscillator (G2D) is a phenomenological, coupled, nonlinear two-dimensional (i.e., two state-variables ('V', 'W')) oscillatory, neural mass model. The G2D is a generalization of the well-known FitzHugh-Nagumo model (FitzHugh, 1961; Nagumo et. al, 1962), adapted here for reproducing a wilder class of dynamical configurations of physiological phenomena as observed in neuronal population using phase-portrait method.

## Equations


### State Equations
$$
\frac{d}{d t} V = d*\tau*\left(I*\gamma + V*g + V*c_{local} + W*\alpha + c_{glob}*\gamma + e*V^{2} - f*V^{3}\right)
$$
$$
\frac{d}{d t} W = \frac{d*\left(a + V*b + c*V^{2} - W*\beta\right)}{\tau}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $I$ | 0.0 | N/A | Baseline shift of the cubic nullcline |
| $a$ | -2.0 | N/A | Vertical shift of the configurable nullcline |
| $\alpha$ | 1.0 | N/A | Constant parameter to scale the rate of feedback from the slow variable to the fast variable. |
| $b$ | -10.0 | N/A | Linear slope of the configurable nullcline |
| $\beta$ | 1.0 | N/A | Constant parameter to scale the rate of feedback from the             slow variable to itself |
| $c$ | 0.0 | N/A | Parabolic term of the configurable nullcline |
| $d$ | 0.02 | N/A | Temporal scale factor |
| $e$ | 3.0 | N/A | Coefficient of the quadratic term of the cubic nullcline |
| $f$ | 1.0 | N/A | Coefficient of the cubic term of the cubic nullcline |
| $g$ | 0.0 | N/A | Coefficient of the linear term of the cubic nullcline |
| $\gamma$ | 1.0 | N/A | Constant parameter to reproduce FHN dynamics where                excitatory input currents are negative |
| $\tau$ | 1.0 | N/A | A time-scale hierarchy can be introduced for the state         variables :math:`V` and :math:`W` |



## References
FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal*, 1(6), 445-466.

Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*, 50(10), 2061-2070.
