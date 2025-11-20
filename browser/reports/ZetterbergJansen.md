

# ZetterbergJansen


## Equations

### Derived Variables
$$
coupled_{input} = \frac{2*e_{0}}{1 + e^{\rho_{1}*\left(\rho_{2} - c_{global} - c_{local}*v_{6}\right)}}
$$
$$
\sigma_{v1} = \begin{cases} 0 & \text{for}\: \rho_{1}*\left(\rho_{2} - v_{1}\right) > 709 \\\frac{2*e_{0}}{e^{\rho_{1}*\left(\rho_{2} - v_{1}\right)} + 1} & \text{otherwise} \end{cases}
$$
$$
\sigma_{v23} = \begin{cases} 0 & \text{for}\: \rho_{1}*\left(\rho_{2} - \left(v_{2} - v_{3}\right)\right) > 709 \\\frac{2*e_{0}}{e^{\rho_{1}*\left(\rho_{2} - \left(v_{2} - v_{3}\right)\right)} + 1} & \text{otherwise} \end{cases}
$$
$$
\sigma_{v45} = \begin{cases} 0 & \text{for}\: \rho_{1}*\left(\rho_{2} - \left(v_{4} - v_{5}\right)\right) > 709 \\\frac{2*e_{0}}{e^{\rho_{1}*\left(\rho_{2} - \left(v_{4} - v_{5}\right)\right)} + 1} & \text{otherwise} \end{cases}
$$

### State Equations
$$
\frac{d}{d t} v_{1} = y_{1}
$$
$$
\frac{d}{d t} v_{2} = y_{2}
$$
$$
\frac{d}{d t} v_{3} = y_{3}
$$
$$
\frac{d}{d t} v_{4} = y_{4}
$$
$$
\frac{d}{d t} v_{5} = y_{5}
$$
$$
\frac{d}{d t} v_{6} = y_{2} - y_{3}
$$
$$
\frac{d}{d t} v_{7} = y_{4} - y_{5}
$$
$$
\frac{d}{d t} y_{1} = - v_{1}*ke^{2} - 2*ke*y_{1} + He*ke*\left(\gamma_{1}*\sigma_{v23} + \gamma_{1T}*\left(U + coupled_{input}\right)\right)
$$
$$
\frac{d}{d t} y_{2} = - v_{2}*ke^{2} - 2*ke*y_{2} + He*ke*\left(\gamma_{2}*\sigma_{v1} + \gamma_{2T}*\left(P + coupled_{input}\right)\right)
$$
$$
\frac{d}{d t} y_{3} = - v_{3}*ki^{2} - 2*ki*y_{3} + Hi*\gamma_{4}*ki*\sigma_{v45}
$$
$$
\frac{d}{d t} y_{4} = - v_{4}*ke^{2} - 2*ke*y_{4} + He*ke*\left(\gamma_{3}*\sigma_{v23} + \gamma_{3T}*\left(Q + coupled_{input}\right)\right)
$$
$$
\frac{d}{d t} y_{5} = - v_{5}*ke^{2} - 2*ki*y_{5} + Hi*\gamma_{5}*ki*\sigma_{v45}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $He$ | 3.25 | N/A | Maximum amplitude of EPSP [mV] |
| $Hi$ | 22.0 | N/A | Maximum amplitude of IPSP [mV] |
| $P$ | 0.12 | N/A | Maximum firing rate to the pyramidal population [ms^-1] |
| $Q$ | 0.12 | N/A | Maximum firing rate to the interneurons population [ms^-1] |
| $U$ | 0.12 | N/A | Maximum firing rate to the stellate population [ms^-1] |
| $e_{0}$ | 0.0025 | N/A | Half of the maximum population mean firing rate [ms^-1] |
| $\gamma_{1T}$ | 1.0 | N/A | Coupling factor from the extrinisic input to the spiny stellate population |
| $\gamma_{1}$ | 135.0 | N/A | Average number of synapses between populations (pyramidal to stellate) |
| $\gamma_{2T}$ | 1.0 | N/A | Coupling factor from the extrinisic input to the pyramidal population |
| $\gamma_{2}$ | 108.0 | N/A | Average number of synapses between populations (stellate to pyramidal) |
| $\gamma_{3T}$ | 1.0 | N/A | Coupling factor from the extrinisic input to the inhibitory population |
| $\gamma_{3}$ | 33.75 | N/A | Connectivity constant (pyramidal to interneurons) |
| $\gamma_{4}$ | 33.75 | N/A | Connectivity constant (interneurons to pyramidal) |
| $\gamma_{5}$ | 15.0 | N/A | Connectivity constant (interneurons to interneurons) |
| $ke$ | 0.1 | N/A | Reciprocal of the time constant of passive membrane and all         other spatially distributed delays in the dendritic network [ms^-1] |
| $ki$ | 0.05 | N/A | Reciprocal of the time constant of passive membrane and all         other spatially distributed delays in the dendritic network [ms^-1] |
| $\rho_{1}$ | 0.56 | N/A | Steepness of the sigmoidal transformation [mV^-1] |
| $\rho_{2}$ | 6.0 | N/A | Firing threshold (PSP) for which a 50% firing rate is achieved |



## References
Citation key 'Zetterberg1978' not found.
