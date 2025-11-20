

# NLDCoupledLogisticMaps4
NLDCoupledLogisticMaps4

## Equations


### State Equations
$$
x_{0} = lam - x_{0}^{2} + k*\left(x_{1}^{2} + x_{2}^{2} + x_{3}^{2} - 3*x_{0}^{2}\right)
$$
$$
x_{1} = lam - x_{1}^{2} + k*\left(x_{0}^{2} + x_{2}^{2} + x_{3}^{2} - 3*x_{1}^{2}\right)
$$
$$
x_{2} = lam - x_{2}^{2} + k*\left(x_{0}^{2} + x_{1}^{2} + x_{3}^{2} - 3*x_{2}^{2}\right)
$$
$$
x_{3} = lam - x_{3}^{2} + k*\left(x_{0}^{2} + x_{1}^{2} + x_{2}^{2} - 3*x_{3}^{2}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $lam$ | 1.2 | N/A | Local quadratic map parameter λ. |
| $k$ | 0.08 | N/A | All-to-all coupling strength k. |



