

# CoombesByrne


## Equations


### State Equations
$$
\frac{d}{d t} V = c_{global} + \eta + V^{2} + g*\left(v_{syn} - V\right) - \pi^{2}*r^{2}
$$
$$
\frac{d}{d t} g = \alpha*q
$$
$$
\frac{d}{d t} q = \alpha*\left(- g - 2*q + \pi*k*r\right)
$$
$$
\frac{d}{d t} r = \frac{\Delta}{\pi} - g*r + 2*V*r
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta$ | 0.5 | N/A | Half-width of heterogeneous noise distribution |
| $\alpha$ | 0.95 | N/A | Parameter of the alpha-function |
| $\eta$ | 20.0 | N/A | Constant parameter to scale the rate of feedback from the             firing rate variable to itself |
| $k$ | 1.0 | N/A | Local coupling strength |
| $v_{syn}$ | -10.0 | N/A | QIF membrane reversal potential |



## References
Citation key 'Byrne2020' not found.
