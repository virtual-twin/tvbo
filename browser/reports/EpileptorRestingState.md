

# EpileptorRestingState
Epileptor Resting-State (ERS) is an extension of the phenomenological neural mass model of Epileptor5D, tuned to express regionally specific physiological oscillations in addition to the epileptiform discharges (Courtiol et al., 2020). This extension was made using the Generic 2-dimensional Oscillator model (parametrized close to a supercritical Hopf Bifurcation) (Sanz-Leon et al., 2013, 2015) to reproduce the spontaneous local field potential-like signal.

This model, its motivation and derivation can be found in the published article (Courtiol et al., 2020).

## Equations

### Derived Variables
$$
lc_{1} = c_{local}*x_{rs}
$$
$$
output = p*\left(x_{2} - x_{1}\right) + x_{rs}*\left(1 - p\right)
$$
$$
x1cond = \begin{cases} - a*x_{1}^{2} + b*x_{1} & \text{for}\: x_{1} < 0 \\slope - x_{2} + 0.6*\left(z - 1*4.0\right)^{2} & \text{otherwise} \end{cases}
$$
$$
y2cond = \begin{cases} 0.0 & \text{for}\: x_{2} < -0.25 \\aa*\left(x_{2} + 0.25\right) & \text{otherwise} \end{cases}
$$
$$
zcond = \begin{cases} - 0.1*z^{7} & \text{for}\: z < 0 \\0.0 & \text{otherwise} \end{cases}
$$

### State Equations
$$
\frac{d}{d t} g = tt*\left(0.001*x_{1} - 0.01*g\right)
$$
$$
\frac{d}{d t} x_{1} = tt*\left(Iext + y_{1} - z + Kvf*c_{global} + c_{local}*x_{1} + x_{1}*x1cond\right)
$$
$$
\frac{d}{d t} x_{2} = tt*\left(1.05 + Iext_{2} + x_{2} - y_{2} - x_{2}^{3} - 0.3*z + Kf*c_{pop1} + bb*g\right)
$$
$$
\frac{d}{d t} x_{rs} = d_{rs}*\tau_{rs}*\left(lc_{1} + I_{rs}*\gamma_{rs} + \alpha_{rs}*y_{rs} + e_{rs}*x_{rs}^{2} - f_{rs}*x_{rs}^{3} + K_{rs}*c_{pop2}*\gamma_{rs}\right)
$$
$$
\frac{d}{d t} y_{1} = tt*\left(c - y_{1} - d*x_{1}^{2}\right)
$$
$$
\frac{d}{d t} y_{2} = \frac{tt*\left(y2cond - y_{2}\right)}{\tau}
$$
$$
\frac{d}{d t} y_{rs} = \frac{d_{rs}*\left(a_{rs} + b_{rs}*x_{rs} - \beta_{rs}*y_{rs}\right)}{\tau_{rs}}
$$
$$
\frac{d}{d t} z = r*tt*\left(zcond - z - 4*x_{0} + 4*x_{1} + Ks*c_{global}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $I_{rs}$ | 0.0 | N/A | External input current to the third population (x_rs, y_rs) |
| $Iext_{2}$ | 0.45 | N/A | External input current to the second population (x2, y2) |
| $Iext$ | 3.1 | N/A | External input current to the first population (x1, y1) |
| $K_{rs}$ | 1.0 | N/A | Coupling scaling on a fast time scale |
| $Kf$ | 0.0 | N/A | Coupling scaling on a fast time scale |
| $Ks$ | 0.0 | N/A | Permittivity coupling, that is from the very fast time scale         toward the slow time scale |
| $Kvf$ | 0.0 | N/A | Coupling scaling on a very fast time scale |
| $a$ | 1.0 | N/A | Coefficient of the cubic term in the first state-variable x1 |
| $a_{rs}$ | -2.0 | N/A | Vertical shift of the configurable nullcline         in the state-variable y_rs |
| $aa$ | 6.0 | N/A | Linear coefficient in the fifth state-variable y2 |
| $\alpha_{rs}$ | 1.0 | N/A | Constant parameter to scale the rate of feedback from the         slow variable y_rs to the fast variable x_rs |
| $b$ | 3.0 | N/A | Coefficient of the squared term in the first state-variable x1 |
| $b_{rs}$ | -10.0 | N/A | Linear coefficient of the state-variable y_rs |
| $bb$ | 2.0 | N/A | Linear coefficient of lowpass excitatory coupling in the fourth         state-variable x2 |
| $\beta_{rs}$ | 1.0 | N/A | Constant parameter to scale the rate of feedback from the         slow variable y_rs to itself |
| $c$ | 1.0 | N/A | Additive coefficient for the second state-variable y1,         called :math:'y_{0}' in Jirsa et al |
| $d$ | 5.0 | N/A | Coefficient of the squared term in the second state-variable y1 |
| $d_{rs}$ | 0.02 | N/A | Temporal scaling of the whole third system (x_rs, y_rs) |
| $e_{rs}$ | 3.0 | N/A | Coefficient of the squared term in the sixth state-variable x_rs |
| $f_{rs}$ | 1.0 | N/A | Coefficient of the cubic term in the sixth state-variable x_rs |
| $\gamma_{rs}$ | 1.0 | N/A | Constant parameter to reproduce FHN dynamics where         excitatory input currents are negative |
| $p$ | 0.0 | N/A | Linear coefficient |
| $r$ | 0.00035 | N/A | Temporal scaling in the third state-variable z,         called :math:'1/	au_{0}' in Jirsa et al |
| $slope$ | 0.0 | N/A | Linear coefficient in the first state-variable x1 |
| $\tau$ | 10.0 | N/A | Temporal scaling coefficient in the fifth state-variable y2 |
| $\tau_{rs}$ | 1.0 | N/A | Temporal scaling coefficient in the third population (x_rs, y_rs) |
| $tt$ | 1.0 | N/A | Time scaling of the Epileptor |
| $x_{0}$ | -1.6 | N/A | Epileptogenicity parameter |



## References
Citation key 'Courtiol2020' not found.

Jirsa, V., Stacey, W., Quilichini, P., Ivanov, A., & Bernard, C. (2014). On the nature of seizure dynamics. *Brain*, 137(8), 2210-2230.
