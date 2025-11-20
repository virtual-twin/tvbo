

# CoupledStandardMaps3
CoupledStandardMaps3

## Equations


### State Equations
$$
\theta_{0} = \left(p_{0} + \theta_{0} + k_{0}*\sin{\left(\theta_{0} \right)} - \Gamma*\left(- \sin{\left(\theta_{0} - \theta_{1} \right)} - \sin{\left(\theta_{0} - \theta_{2} \right)}\right)\right) \bmod 2*\pi
$$
$$
p_{0} = \left(p_{0} + k_{0}*\sin{\left(\theta_{0} \right)} - \Gamma*\left(- \sin{\left(\theta_{0} - \theta_{1} \right)} - \sin{\left(\theta_{0} - \theta_{2} \right)}\right)\right) \bmod 2*\pi
$$
$$
\theta_{1} = \left(p_{1} + \theta_{1} + k_{1}*\sin{\left(\theta_{1} \right)} - \Gamma*\left(- \sin{\left(\theta_{1} - \theta_{2} \right)} + \sin{\left(\theta_{0} - \theta_{1} \right)}\right)\right) \bmod 2*\pi
$$
$$
p_{1} = \left(p_{1} + k_{1}*\sin{\left(\theta_{1} \right)} - \Gamma*\left(- \sin{\left(\theta_{1} - \theta_{2} \right)} + \sin{\left(\theta_{0} - \theta_{1} \right)}\right)\right) \bmod 2*\pi
$$
$$
\theta_{2} = \left(p_{2} + \theta_{2} + k_{2}*\sin{\left(\theta_{2} \right)} - \Gamma*\left(\sin{\left(\theta_{0} - \theta_{2} \right)} + \sin{\left(\theta_{1} - \theta_{2} \right)}\right)\right) \bmod 2*\pi
$$
$$
p_{2} = \left(p_{2} + k_{2}*\sin{\left(\theta_{2} \right)} - \Gamma*\left(\sin{\left(\theta_{0} - \theta_{2} \right)} + \sin{\left(\theta_{1} - \theta_{2} \right)}\right)\right) \bmod 2*\pi
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $k_{0}$ | 1.0 | N/A | Nonlinearity parameter for map 0. |
| $k_{1}$ | 1.0 | N/A | Nonlinearity parameter for map 1. |
| $k_{2}$ | 1.0 | N/A | Nonlinearity parameter for map 2. |
| $\Gamma$ | 1.0 | N/A | Coupling strength Γ among nearest neighbors on a ring. |



