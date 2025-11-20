

# MontbrioPazoRoxin


## Equations


### State Equations
$$
\frac{d}{d t} V = \frac{I + \eta + V^{2} + c_{pop1}*cv + J*r*\tau + c_{global}*cr*\tau - \pi^{2}*r^{2}*\tau^{2}}{\tau}
$$
$$
\frac{d}{d t} r = \frac{2*V*r + \frac{\Delta}{\pi*\tau}}{\tau}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta$ | 1.0 | N/A | Mean heterogeneous noise |
| $I$ | 0.0 | N/A | External Current |
| $J$ | 15.0 | N/A | Mean Synaptic weight |
| $cr$ | 1.0 | N/A | It is the weight on Coupling through variable r |
| $cv$ | 0.0 | N/A | It is the weight on Coupling through variable V |
| $\eta$ | -5.0 | N/A | Constant parameter to scale the rate of feedback from the firing rate variable to itself |
| $\tau$ | 1.0 | N/A | Membrane time constant |



## References
Citation key 'Montbrio2015' not found.
