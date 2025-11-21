

# Epileptor2D
Epileptor2D (E2D) is a phenomenological neural mass model consisting in the two-dimensional reduction ('x', 'z') of the original Epileptor model (see Epileptor5D; Proix et al., 2014, 2017).

Note:
------
- Equations and default parameters are taken from (Proix et al.,2014),
- The slow permittivity state-variable (z_E2D) can be modified to account for the time difference between the interictal (between seizures) and ictal (during seizure) states (see Proix et al., 2014).

## Equations

### Derived Variables
$$
x1cond = \begin{cases} a*x_{1}^{2} + x_{1}*\left(- b + d\right) & \text{for}\: x_{1} < 0 \\d*x_{1} - slope - 0.6*\left(z - 1*4.0\right)^{2} & \text{otherwise} \end{cases}
$$
$$
zcond = \begin{cases} - 0.1*z^{7} & \text{for}\: z < 0 \\0 & \text{otherwise} \end{cases}
$$
$$
h = \begin{cases} x_{0} + \frac{3.0}{e^{\frac{- x_{1} - 0.5}{0.1}} + 1.0} & \text{for}\: modification > 0 \\zcond + 4*\left(- x_{0} + x_{1}\right) & \text{otherwise} \end{cases}
$$

### State Equations
$$
\frac{d}{d t} x_{1} = tt*\left(Iext + c - z + Kvf*c_{global} + c_{local}*x_{1} - x_{1}*x1cond\right)
$$
$$
\frac{d}{d t} z = r*tt*\left(h - z + Ks*c_{global}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $Iext$ | 3.1 | N/A | External input current to the first state-variable x_E2D, in Epileptor2D (Proix et al |
| $Ks$ | 0.0 | N/A | Permittivity coupling on the slow permittivity state-variable z_E2D in Epileptor2D (Proix et al |
| $Kvf$ | 0.0 | N/A | Coupling scaling on a very fast time scale |
| $a$ | 1.0 | N/A | Coefficient of the cubic term in the first state-variable x_E2D via the function f(x)_E2D, in Epileptor2D Proix et al |
| $b$ | 3.0 | N/A | Coefficient of the squared term in the first state-variable x_E2D via the function f_E2D, in Epileptor2D (Proix et al |
| $c$ | 1.0 | N/A | Additive coefficient for the second state-variable x_{2},         called :math:`y_{0}` in Jirsa paper |
| $d$ | 5.0 | N/A | Coefficient of the squared term in the first state-variable x_E2D via the function f in Epileptor2D (Proix et al |
| $modification$ | 0.0 | N/A | When modification is True, the function h_E2D uses a nonlinear influence on z_E2D |
| $r$ | 0.00035 | N/A | Temporal scaling in the slow state-variable, \         called :math:`1\tau_{0}` in Jirsa paper (see class Epileptor) |
| $slope$ | 0.0 | N/A | Linear coefficient in the first state-variable x_E2D via the function f_E2D, in Epileptor2D (Proix et al |
| $tt$ | 1.0 | N/A | Characteristic time scale of the whole-system Epileptor2D |
| $x_{0}$ | -1.6 | N/A | Degree of excitability or epileptogenicity in Epileptor2D (Proix et al |



## References
Citation key 'Proix2014' not found.

Citation key 'Proix2017' not found.
