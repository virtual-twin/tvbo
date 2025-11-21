

# Epileptor5D
Epilepor5D (E5D) is a phenomenological, coupled, nonlinear five-dimensional (i.e., five state-variables ('x1', 'y1', 'z', 'x2', 'y2')) neural mass model able to realistically reproduce the temporal dynamics of epileptic seizures and the alternating sequence of seizures (ictal and interictal state; Jirsa et al.,2014; El Houssaini et al., 2015, 2020).

Epileptor5D comprises three different time scales interacting together and accounting for various electrographic patterns:
- the fastest and intermediate time scales are two coupled oscillators ((x1, y1) and (x2, y2)), accounting respectively for the low-voltage fast discharges (i.e., very fast oscillations) and spike-and-wave discharges.
- the slowest time scale is responsible for leading the autonomous switch between interictal and ictal states and is driven by a slow-permittivity variable z. This switching is accompanied by a direct current (DC) shift that has been recorded in vitro and in vivo.

The main output of the model: -x1+ x2, bears analogy with the field potential, while the precise biophysical equivalent of the z variable is unknown and will be likely complex.

Note:
------
- Equations and default parameters are taken from (Jirsa et al.,2014 & El Houssaini et al., 2015),
- The integral coupling function g(x1) can be rewritten as an ordinary differential equation, which is technically introduced, here, as a sixth state-variable (see Jirsa et al.,2014),
- The slow permittivity state-variable (z_E5D) can be modified to account for the time difference between the interictal (between seizures) and ictal (during seizure) states (see Proix et al., 2014).

## Equations

### Derived Variables
$$
x1cond = \begin{cases} - a*x_{1}^{2} + b*x_{1} & \text{for}\: x_{1} < 0 \\slope - x_{2} + 0.6*\left(z - 1*4.0\right)^{2} & \text{otherwise} \end{cases}
$$
$$
y2cond = \begin{cases} 0.0 & \text{for}\: x_{2} < -0.25 \\aa*\left(x_{2} + 0.25\right) & \text{otherwise} \end{cases}
$$
$$
zcond = \begin{cases} - 0.1*z^{7} & \text{for}\: z < 0 \\0.0 & \text{otherwise} \end{cases}
$$
$$
h = \begin{cases} x_{0} + \frac{3}{e^{\frac{- x_{1} - 0.5}{0.1}} + 1} & \text{for}\: modification > 0 \\zcond + 4*\left(- x_{0} + x_{1}\right) & \text{otherwise} \end{cases}
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
\frac{d}{d t} y_{1} = tt*\left(c - y_{1} - d*x_{1}^{2}\right)
$$
$$
\frac{d}{d t} y_{2} = \frac{tt*\left(y2cond - y_{2}\right)}{\tau}
$$
$$
\frac{d}{d t} z = r*tt*\left(h - z + Ks*c_{global}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $Iext_{2}$ | 0.45 | N/A | External input current to the second population |
| $Iext$ | 3.1 | N/A | External input current to the first sub-population (x1_E5D, y1_E5D) via the state-variable x1_E5D, in Epileptor5D (Jirsa et al |
| $Kf$ | 0.0 | N/A | Correspond to the coupling scaling on a fast time scale |
| $Ks$ | 0.0 | N/A | Permittivity coupling on the slow permittivity state-variable z_E5D in Epileptor5D (Proix et al |
| $Kvf$ | 0.0 | N/A | Coupling scaling on a very fast time scale |
| $a$ | 1.0 | N/A | Coefficient of the cubic term in the first state variable |
| $aa$ | 6.0 | N/A | Linear coefficient in fifth state variable |
| $b$ | 3.0 | N/A | Coefficient of the squared term in the first state variabel |
| $bb$ | 2.0 | N/A | Linear coefficient of lowpass excitatory coupling in fourth state variable |
| $c$ | 1.0 | N/A | Additive coefficient for the second state variable,         called :math:`y_{0}` in Jirsa paper |
| $d$ | 5.0 | N/A | Coefficient of the squared term in the derivative of the second state-variable y1_E5D in Epileptor5D (Jirsa et al |
| $modification$ | 0.0 | N/A | When modification is True, the function h_E5D uses a nonlinear influence on z_E5D |
| $r$ | 0.00035 | N/A | Temporal scaling in the third state variable,         called :math:`1/\tau_{0}` in Jirsa paper |
| $s$ | 4.0 | N/A | Linear coefficient in the slow permittivity state-variable z_E5D in Epileptor5D (Jirsa et al |
| $slope$ | 0.0 | N/A | Linear coefficient in the first state variable |
| $\tau$ | 10.0 | N/A | Temporal scaling coefficient in fifth state variable |
| $tt$ | 1.0 | N/A | Characteristic time scale of the whole-system Epileptor5D |
| $x_{0}$ | -1.6 | N/A | Epileptogenicity Parameter |



## References
Citation key 'Proix2014' not found.

Jirsa, V., Stacey, W., Quilichini, P., Ivanov, A., & Bernard, C. (2014). On the nature of seizure dynamics. *Brain*, 137(8), 2210-2230.
