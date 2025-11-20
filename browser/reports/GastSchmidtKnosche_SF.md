

# GastSchmidtKnosche_SF


## Equations


### State Equations
$$
\frac{d}{d t} A = \frac{B}{\tau_{A}}
$$
$$
\frac{d}{d t} B = \frac{- A - 2*B + \alpha*r}{\tau_{A}}
$$
$$
\frac{d}{d t} V = \frac{I + \eta + V^{2} - A + c_{global}*cr + c_{pop1}*cv + J*r*\tau - \pi^{2}*r^{2}*\tau^{2}}{\tau}
$$
$$
\frac{d}{d t} r = \frac{2*V*r + \frac{\Delta}{\pi*\tau}}{\tau}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta$ | 2.0 | N/A | Half-width of heterogeneous noise distribution |
| $I$ | 0.0 | N/A | External homogeneous current |
| $J$ | 21.2132 | N/A | Synaptic weight |
| $\alpha$ | 10.0 | N/A | adaptation rate |
| $cr$ | 1.0 | N/A | It is the weight on Coupling through variable r |
| $cv$ | 0.0 | N/A | It is the weight on Coupling through variable V |
| $\eta$ | 1.0 | N/A | Mean of heterogeneous noise distribution |
| $\tau_{A}$ | 10.0 | N/A | Adaptation time scale |
| $\tau$ | 1.0 | N/A | Characteristic time |



## References
Citation key 'Gast2020' not found.
