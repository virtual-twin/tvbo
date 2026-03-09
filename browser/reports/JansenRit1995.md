

## JansenRit1995
Lumped-parameter cortical column model with three neural populations: pyramidal cells, excitatory interneurons, and inhibitory interneurons. Each population is modeled by a PSP block (2nd-order linear filter) and a sigmoidal potential-to-firing-rate transformation. The model produces EEG-like output as the pyramidal PSP difference v_pyr = y1 - y2. Based on Lopes da Silva et al. (1974, 1976) and Jansen et al. (1993).

### State Equations
$$
\dot{y_{0}} = y_{3}
$$
$$
\dot{y_{3}} = - y_{0}*a^{2} - 2*a*y_{3} + A*a*\operatorname{Sigm}{\left(y_{1} - y_{2} \right)}
$$
$$
\dot{y_{1}} = y_{4}
$$
$$
\dot{y_{4}} = - y_{1}*a^{2} - 2*a*y_{4} + A*a*\left(c_{intercolumn} + p + C_{2}*\operatorname{Sigm}{\left(C_{1}*y_{0} \right)}\right)
$$
$$
\dot{y_{2}} = y_{5}
$$
$$
\dot{y_{5}} = - y_{2}*b^{2} - 2*b*y_{5} + B*C_{4}*b*\operatorname{Sigm}{\left(C_{3}*y_{0} \right)}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $A$ | 3.25 | mV | Maximum amplitude of excitatory PSP (EPSP) |
| $B$ | 22.0 | mV | Maximum amplitude of inhibitory PSP (IPSP) |
| $C$ | 135.0 | N/A | Lumped connectivity constant (C = C1) |
| $a$ | 100.0 | s^-1 | Reciprocal of EPSP time constant |
| $b$ | 50.0 | s^-1 | Reciprocal of IPSP time constant |
| $v_{0}$ | 6.0 | mV | PSP for 50% firing rate (sigmoid midpoint) |
| $e_{0}$ | 2.5 | s^-1 | Half of maximum firing rate (2*e0 = max rate) |
| $r$ | 0.56 | mV^-1 | Steepness of sigmoid transformation |
| $p$ | 220.0 | s^-1 | Mean external input (afferent pulse density) |

### Derived Quantities
#### Derived Parameters
$$
C_{1} = C
$$
$$
C_{2} = 0.8*C
$$
$$
C_{3} = 0.25*C
$$
$$
C_{4} = 0.25*C
$$
#### Derived Variables
$$
v_{pyr} = y_{1} - y_{2}
$$
#### Functions
$$
\operatorname{Sigm}{\left(v \right)} = \frac{2*e_{0}}{1 + e^{r*\left(v_{0} - v\right)}}
$$

### Output Transforms
$$
v_{pyr} = y_{1} - y_{2}
$$



