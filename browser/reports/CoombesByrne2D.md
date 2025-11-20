

# CoombesByrne2D


## Equations


### State Equations
$$
\frac{d}{d t} V = c_{global} + \eta + V^{2} - \pi^{2}*r^{2} + \pi*k*r*\left(v_{syn} - V\right)
$$
$$
\frac{d}{d t} r = \frac{\Delta}{\pi} + 2*V*r - \pi*k*r^{2}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta$ | 1.0 | N/A | Half-width of heterogeneous noise distribution |
| $\eta$ | 2.0 | N/A | Constant parameter to scale the rate of feedback from the             firing rate variable to itself |
| $k$ | 1.0 | N/A | Local coupling strength |
| $v_{syn}$ | -4.0 | N/A | QIF membrane reversal potential |



## References
Citation key 'Byrne2020' not found.
