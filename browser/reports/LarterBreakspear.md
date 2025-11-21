

# LarterBreakspear
The Larter-Breakspear is an extension (Breakspear et al., 2003a, 2003b) of the biophysical-inspired neural mass model of a cortical column (or area) from Larter et al. (1999), initially developed to simulate firing rate activity from focal region involved in partial seizure. It is determined by voltage- and ligand-gated ions channels and feedback between intensively interconnected excitatory and inhibitory neurons.

The Larter-Breakspear is a 3D model describing the local average states of two interconnected neural populations: pyramidal cells (PCs) and inhibitory interneurons (IINs), with an additional variable representing the potassium channels in the population of PCs.

The membrane potential of the pyramidal cells is the focus of the model and is governed by sodium,
potassium, calcium and “leaky” ion channels, of which the voltage-gated potassium channels are modelled in more detail.

The excitatory to excitatory connections are modelled in more detail as glutamatergic connections with AMPA and NMDA receptors.

Note:
- Equations and default parameters are taken from (Breakspear et al., 2003b),
- All equations and parameters are non-dimensional and normalized to neural capacitance C = 1.

## Equations

### Derived Variables
$$
Q_{V} = 0.5*Q_{Vmax}*\left(1 + \tanh{\left(\frac{V - V_{T}}{\delta_{V}} \right)}\right)
$$
$$
Q_{Z} = 0.5*Q_{Zmax}*\left(1 + \tanh{\left(\frac{Z - Z_{T}}{\delta_{Z}} \right)}\right)
$$
$$
m_{Ca} = 0.5 + 0.5*\tanh{\left(\frac{V - T_{Ca}}{\delta_{Ca}} \right)}
$$
$$
m_{K} = 0.5 + 0.5*\tanh{\left(\frac{V - T_{K}}{\delta_{K}} \right)}
$$
$$
m_{Na} = 0.5 + 0.5*\tanh{\left(\frac{V - T_{Na}}{\delta_{Na}} \right)}
$$
$$
lc_{0} = Q_{V}*c_{local}
$$

### State Equations
$$
\frac{d}{d t} V = t_{scale}*\left(I_{ext}*a_{ne} - g_{L}*\left(V - V_{L}\right) - \left(V - V_{Na}\right)*\left(g_{Na}*m_{Na} + C*a_{ee}*c_{global} + a_{ee}*\left(1.0 - C\right)*\left(Q_{V} + lc_{0}\right)\right) + m_{Ca}*\left(V - V_{Ca}\right)*\left(- g_{Ca} - C*a_{ee}*c_{global}*r_{NMDA} - a_{ee}*r_{NMDA}*\left(1.0 - C\right)*\left(Q_{V} + lc_{0}\right)\right) - Q_{Z}*Z*a_{ie} - W*g_{K}*\left(V - V_{K}\right)\right)
$$
$$
\frac{d}{d t} W = \frac{\phi*t_{scale}*\left(m_{K} - W\right)}{\tau_{K}}
$$
$$
\frac{d}{d t} Z = b*t_{scale}*\left(I_{ext}*a_{ni} + Q_{V}*V*a_{ei}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $C$ | 0.1 | N/A | Coupling scaling factor |
| $I_{ext}$ | 0.3 | N/A | Subcortical input current |
| $Q_{Vmax}$ | 1.0 | Kilohertz | Maximal firing rate for excitatory populations |
| $Q_{Zmax}$ | 1.0 | Kilohertz | Maximal firing rate for inhibitory population |
| $T_{Ca}$ | -0.01 | N/A | Threshold value for Ca channels |
| $T_{K}$ | 0.0 | N/A | Threshold value for K channels |
| $T_{Na}$ | 0.3 | N/A | Threshold value for sodium channels |
| $V_{Ca}$ | 1.0 | N/A | Calcium Nernst potential |
| $V_{K}$ | -0.7 | N/A | K Nernst potential |
| $V_{L}$ | -0.5 | N/A | Nernst potential leak channels |
| $V_{Na}$ | 0.53 | N/A | Na Nernst potential |
| $V_{T}$ | 0.0 | N/A | Threshold potential for excitatory neurons |
| $Z_{T}$ | 0.0 | N/A | Threshold potential (mean) for inihibtory neurons |
| $a_{ee}$ | 0.4 | N/A | Excitatory-to-excitatory synaptic strength |
| $a_{ei}$ | 2.0 | N/A | Excitatory-to-inhibitory synaptic strength |
| $a_{ie}$ | 2.0 | N/A | Inhibitory-to-excitatory synaptic strength |
| $a_{ne}$ | 1.0 | N/A | Non-specific-to-excitatory synaptic strength |
| $a_{ni}$ | 0.4 | N/A | Non-specific-to-inhibitory synaptic strength |
| $b$ | 0.1 | N/A | Time constant scaling factor |
| $\delta_{Ca}$ | 0.15 | N/A | Variance of Calcium channel threshold |
| $\delta_{K}$ | 0.3 | N/A | Variance of Potassium channel threshold |
| $\delta_{Na}$ | 0.15 | N/A | Variance of sodium channel threshold |
| $\delta_{V}$ | 0.65 | N/A | Variance of excitatory threshold |
| $\delta_{Z}$ | 0.7 | N/A | Variance of inhibitory threshold |
| $g_{Ca}$ | 1.1 | N/A | Conductance of population of calcium (Ca++) channels |
| $g_{K}$ | 2.0 | N/A | Conductance of population of potassium (K) channels |
| $g_{L}$ | 0.5 | N/A | Conductance of population of leak channels |
| $g_{Na}$ | 6.7 | N/A | Conductance of population of Na channels |
| $\phi$ | 0.7 | N/A | Temperature scaling factor |
| $r_{NMDA}$ | 0.25 | N/A | Ratio of NMDA to AMPA receptors |
| $t_{scale}$ | 1.0 | N/A | Time scale factor |
| $\tau_{K}$ | 1.0 | N/A | Time constant for K relaxation time (ms) |



## References
Breakspear, M., Terry, J., & Friston, K. (2003). Modulation of excitatory synaptic coupling facilitates synchronization and complex dynamics in a biophysical model of neuronal dynamics.. *Network (Bristol, England)*, 14, 703-732.

Larter, R., Speelman, B., & Worth, R. (1999). A coupled ordinary differential equation lattice model for the simulation of epileptic seizures. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 9(3), 795-804.

Breakspear, M., R., J., & J., K. (2003). Modulation of excitatory synaptic coupling facilitates synchronization and complex dynamics in a nonlinear model of neuronal dynamics. *Neurocomputing*, 52–54, 151-158.
