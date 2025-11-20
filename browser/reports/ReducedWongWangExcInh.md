

# ReducedWongWangExcInh
Reduced WongWang Exc-Inh (RWW_EI) is a biologically-inspired two-dimensional (i.e., two state-variables ('S_E','S_I')) neural mass model describing the dynamics of a cortical area consisting of local networks of excitatory (AMPA) and inhibitory (GABA-A) populations of spiking neurons interconnected via NMDA synapses. These neurons are organized into an inhibitory population accounting for 20% of the neurons and an excitatory population accounting for 80% of the neurons.

## Equations

### Derived Variables
$$
J_{N S e} = J_{N}*S_{e}
$$
$$
coupling = G*J_{N}*\left(c_{glob} + S_{e}*c_{local}\right)
$$
$$
x_{e} = - b_{e} + a_{e}*\left(I_{ext} + coupling + I_{o}*W_{e} + J_{N S e}*w_{p} - J_{i}*S_{i}\right)
$$
$$
x_{i} = - b_{i} + a_{i}*\left(J_{N S e} - S_{i} + I_{o}*W_{i} + coupling*\lambda\right)
$$
$$
H_{e} = \frac{x_{e}}{1 - e^{- d_{e}*x_{e}}}
$$
$$
H_{i} = \frac{x_{i}}{1 - e^{- d_{i}*x_{i}}}
$$

### State Equations
$$
\frac{d}{d t} S_{e} = - \frac{S_{e}}{\tau_{e}} + H_{e}*\gamma_{e}*\left(1 - S_{e}\right)
$$
$$
\frac{d}{d t} S_{i} = H_{i}*\gamma_{i} - \frac{S_{i}}{\tau_{i}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $G$ | 2.0 | N/A | Global coupling scaling |
| $I_{ext}$ | 0.0 | N/A | External stimulation parameter for evoked potential simulated activity applied to the excitatory population firing rate IE (Deco et al |
| $I_{o}$ | 0.382 | nA | Effective external input parameter to the excitatory E and inhibitory I population (Deco et al |
| $J_{N}$ | 0.15 | N/A | Excitatory synaptic coupling parameter (Deco et al |
| $J_{i}$ | 1.0 | N/A | Local feedback inhibitory synaptic coupling to the excitatory population firing rate IE (Deco et al |
| $W_{e}$ | 1.0 | N/A | Excitatory population external input scaling weight |
| $W_{i}$ | 0.7 | N/A | Inhibitory population external input scaling weight |
| $a_{e}$ | 310.0 | nC^-1 | Slope (or gain) parameter of the sigmoid function HE_RWW_EI (Deco et al |
| $a_{i}$ | 615.0 | nC^-1 | Slope (or gain) parameter of the sigmoid function HI_RWW_EI of the inhibitory population (Deco et al |
| $b_{e}$ | 125.0 | Hz | Shift parameter of the sigmoid function HE_RWW_EI of the excitatory population (Deco et al |
| $b_{i}$ | 177.0 | Hz | Shift parameter of the sigmoid function HI_RWW_EI of the inhibitory population (Deco et al |
| $d_{e}$ | 0.16 | s | Scaling parameter of the sigmoid function HE_RWW_EI of the excitatory population (Deco et al |
| $d_{i}$ | 0.087 | s | Scaling parameter of the sigmoid function HI_RWW_EI of the inhibitory population (Deco et al |
| $\gamma_{e}$ | 0.000641 | N/A | Excitatory population kinetic parameter |
| $\gamma_{i}$ | 0.001 | N/A | Inhibitory population kinetic parameter |
| $\lambda$ | 0.0 | N/A | Inhibitory global coupling scaling |
| $\tau_{e}$ | 100.0 | ms | Kinetic parameter  that represents the decay times for NMDA synapses (Deco et al |
| $\tau_{i}$ | 10.0 | ms | Kinetic parameter that represents the decay times for inhibitory GABA synapses (Deco et al |
| $w_{p}$ | 1.4 | N/A | Excitatory population recurrence weight |



## References
Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G., Mantini, D., & Corbetta, M. (2014). How local excitation-inhibition ratio impacts the whole brain dynamics. *Journal of Neuroscience*, 34, 7886-7898.
