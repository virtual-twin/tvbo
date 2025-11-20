

# ModelJansen1995
ModelJansen1995

## Equations

### Derived Parameters
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
### Functions
$$
\operatorname{Sigm}{\left(v \right)} = \frac{2*e_{0}}{1 + e^{r*\left(v_{0} - v\right)}}
$$
### Derived Variables
$$
v_{pyr} = y_{1} - y_{2}
$$

### State Equations
$$
\frac{d}{d t} y_{0} = y_{3}
$$
$$
\frac{d}{d t} y_{3} = - y_{0}*a^{2} - 2*a*y_{3} + A*a*\operatorname{Sigm}{\left(y_{1} - y_{2} \right)}
$$
$$
\frac{d}{d t} y_{1} = y_{4}
$$
$$
\frac{d}{d t} y_{4} = - y_{1}*a^{2} - 2*a*y_{4} + A*a*\left(c_{glob} + p + C_{2}*\operatorname{Sigm}{\left(C_{1}*y_{0} \right)}\right)
$$
$$
\frac{d}{d t} y_{2} = y_{5}
$$
$$
\frac{d}{d t} y_{5} = - y_{2}*b^{2} - 2*b*y_{5} + B*C_{4}*b*\operatorname{Sigm}{\left(C_{3}*y_{0} \right)}
$$

### Output Transforms
$$
v_{pyr} = y_{1} - y_{2}
$$

## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $A$ | 3.25 | mV | None |
| $B$ | 22.0 | mV | None |
| $C$ | 135.0 | N/A | None |
| $a$ | 0.1 | ms^-1 | None |
| $b$ | 0.05 | ms^-1 | None |
| $v_{0}$ | 6.0 | mV | None |
| $e_{0}$ | 0.0025 | ms^-1 | None |
| $r$ | 0.56 | mV^-1 | None |
| $p$ | 0.24 | ms^-1 | None |



