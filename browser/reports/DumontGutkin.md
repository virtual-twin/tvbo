

# DumontGutkin


## Equations


### State Equations
$$
\frac{d}{d t} V_{e} = \frac{I_{e} + \eta_{e} + V_{e}^{2} + s_{ee}*\tau_{e} - s_{ei}*\tau_{e} - \pi^{2}*r_{e}^{2}*\tau_{e}^{2}}{\tau_{e}}
$$
$$
\frac{d}{d t} V_{i} = \frac{I_{i} + \eta_{i} + V_{i}^{2} + s_{ie}*\tau_{i} - s_{ii}*\tau_{i} - \pi^{2}*r_{i}^{2}*\tau_{i}^{2}}{\tau_{i}}
$$
$$
\frac{d}{d t} r_{e} = \frac{2*V_{e}*r_{e} + \frac{\Delta_{e}}{\pi*\tau_{e}}}{\tau_{e}}
$$
$$
\frac{d}{d t} r_{i} = \frac{2*V_{i}*r_{i} + \frac{\Delta_{i}}{\pi*\tau_{i}}}{\tau_{i}}
$$
$$
\frac{d}{d t} s_{ee} = \frac{c_{global} - s_{ee} + J_{ee}*r_{e}}{\tau_{s}}
$$
$$
\frac{d}{d t} s_{ei} = \frac{- s_{ei} + J_{ei}*r_{i}}{\tau_{s}}
$$
$$
\frac{d}{d t} s_{ie} = \frac{- s_{ie} + \Gamma*c_{global} + J_{ie}*r_{e}}{\tau_{s}}
$$
$$
\frac{d}{d t} s_{ii} = \frac{- s_{ii} + J_{ii}*r_{i}}{\tau_{s}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta_{e}$ | 1.0 | N/A | Half-width of heterogeneous noise distribution over excitatory population |
| $\Delta_{i}$ | 1.0 | N/A | Half-width of heterogeneous noise distribution over inhibitory population |
| $\Gamma$ | 5.0 | N/A | Ratio of excitatory VS inhibitory global couplings G_ie/G_ee  |
| $I_{e}$ | 0.0 | N/A | External homogeneous current on excitatory population |
| $I_{i}$ | 0.0 | N/A | External current on inhibitory population |
| $J_{ee}$ | 0.0 | N/A | Synaptic weight e-->e |
| $J_{ei}$ | 10.0 | N/A | Synaptic weight i-->e |
| $J_{ie}$ | 0.0 | N/A | Synaptic weight e-->i |
| $J_{ii}$ | 15.0 | N/A | Synaptic weight i-->i |
| $\eta_{e}$ | -5.0 | N/A | Mean heterogeneous current on excitatory population |
| $\eta_{i}$ | -5.0 | N/A | Mean heterogeneous current on inhibitory population |
| $\tau_{e}$ | 10.0 | N/A | Characteristic time of excitatory population |
| $\tau_{i}$ | 10.0 | N/A | Characteristic time of inhibitory population |
| $\tau_{s}$ | 1.0 | N/A | Synaptic time constant |



## References
Citation key 'Dumont2019' not found.
