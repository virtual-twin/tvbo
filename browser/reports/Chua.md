

## Chua
This is a 3D continuous system that exhibits chaos.

Chua designed an electronic circuit with the expressed goal of exhibiting
chaotic motion, and this system is obtained by rescaling the circuit units
to simplify the form of the equation. [^Chua1992]

The parameters are a, b, m0, and m1. Setting a = 15.6, m0 = -8/7 and m1 = -5/7, and
varying the parameter b from b = 25 to b = 51, one observes a classic period-doubling
bifurcation route to chaos. [^Chua2007]

### State Equations
$$
\dot{x} = a*\left(y - h_{x} - x\right)
$$
$$
\dot{y} = x + z - y
$$
$$
\dot{z} = - b*y
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 15.6 | N/A | Parameter a scaling the active conductance term. |
| $b$ | 25.58 | N/A | Parameter b scaling the third state equation term. |
| $m_{0}$ | -1.1428571428571428 | N/A | Inner slope of Chua's piecewise-linear element. |
| $m_{1}$ | -0.7142857142857143 | N/A | Outer slope of Chua's piecewise-linear element. |

### Derived Quantities
#### Derived Variables
$$
h_{x} = m_{1}*x + \left(- \operatorname{abs}{\left(-1 + x \right)} + \operatorname{abs}{\left(1 + x \right)}\right)*\left(0.5*m_{0} - 0.5*m_{1}\right)
$$




