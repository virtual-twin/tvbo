

## UlamRing4
Four unidirectionally coupled quadratic maps on a ring using the classic Ulam map
f(x)=2 - x^2 as local dynamics; higher N can be constructed analogously.

### State Equations
$$
x_{0} = 2 - \left(eps*x_{3} + x_{0}*\left(1 - eps\right)\right)^{2}
$$
$$
x_{1} = 2 - \left(eps*x_{0} + x_{1}*\left(1 - eps\right)\right)^{2}
$$
$$
x_{2} = 2 - \left(eps*x_{1} + x_{2}*\left(1 - eps\right)\right)^{2}
$$
$$
x_{3} = 2 - \left(eps*x_{2} + x_{3}*\left(1 - eps\right)\right)^{2}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $eps$ | 0.6 | N/A | Unidirectional coupling ε on a 4-node ring. |





