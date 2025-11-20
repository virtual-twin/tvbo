

# JansenRit
The Jansen-Rit is a neurophysiologically-inspired neural mass model of a cortical column (or area), developed to simulate the electrical brain activity, i.e., the electroencephalogram (EEG), and evoked-potentials (EPs; Jansen et al., 1993; Jansen & Rit, 1995). It is a 6-dimensional, non-linear, model describing the local average states of three interconnected neural populations: pyramidal cells (PCs), excitatory and inhibitory interneurons (EINs and IINs), interacting through positive and negative feedback loops. The main output of the model is the average membrane potential of the pyramidal cell population, as the sum of the potential of these cells is thought to be the source of the potential recorded in the EEG.

## Equations

### Derived Variables
$$
\sigma_{y0 1} = \frac{2.0*\nu_{max}}{1.0 + e^{r*\left(v_{0} - J*a_{1}*y_{0}\right)}}
$$
$$
\sigma_{y0 3} = \frac{2.0*\nu_{max}}{1.0 + e^{r*\left(v_{0} - J*a_{3}*y_{0}\right)}}
$$
$$
\sigma_{y1 y2} = \frac{2.0*\nu_{max}}{1.0 + e^{r*\left(v_{0} + y_{2} - y_{1}\right)}}
$$

### State Equations
$$
\frac{d}{d t} y_{0} = y_{3}
$$
$$
\frac{d}{d t} y_{1} = y_{4}
$$
$$
\frac{d}{d t} y_{2} = y_{5}
$$
$$
\frac{d}{d t} y_{3} = - y_{0}*a^{2} - 2.0*a*y_{3} + A*a*\sigma_{y1 y2}
$$
$$
\frac{d}{d t} y_{4} = - y_{1}*a^{2} - 2.0*a*y_{4} + A*a*\left(c_{global} + \mu + c_{local}*\left(y_{1} - y_{2}\right) + J*a_{2}*\sigma_{y0 1}\right)
$$
$$
\frac{d}{d t} y_{5} = - y_{2}*b^{2} - 2.0*b*y_{5} + B*J*a_{4}*b*\sigma_{y0 3}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $A$ | 3.25 | Millivolt | Maximum amplitude of EPSP [mV] |
| $B$ | 22.0 | Millivolt | Maximum amplitude of IPSP [mV] |
| $J$ | 135.0 | N/A | Average number of synapses between three neuronal populations of the model |
| $a_{1}$ | 1.0 | N/A | Average probability constant of the number of synapses made by the pyramidal cells to the dendrites of the excitatory interneurons  (feedback excitatory loop) |
| $a_{2}$ | 0.8 | N/A | Average probability constant of the number of synapses made by the EINs to the dendrites of the PCs |
| $a_{3}$ | 0.25 | N/A | Average probability constant of the number of synapses made by the PCs to the dendrites of the IINs |
| $a_{4}$ | 0.25 | N/A | Average probability constant of the number of synapses made by the IINs to the dendrites of the PCs |
| $a$ | 0.1 | ms^-1 | Reciprocal of the time constant of passive membrane and all other spatially distributed delays in the dendritic network. Also called average synaptic time constant. |
| $b$ | 0.05 | ms^-1 | Rate constant of the inhibitory post-synaptic potential (IPSP) |
| $\mu$ | 0.22 | ms^-1 | Mean excitatory external input to the derivative of the state-variable y4_JR (PCs) represented by a pulse density, that consists of activity originating from adjacent and more distant cortical columns, as well as from subcortical structures (e |
| $\nu_{max}$ | 0.0025 | ms^-1 | Asymptotic of the sigmoid function Sigm_JR corresponds to the maximum firing rate of the neural populations |
| $r$ | 0.56 | mV^-1 | Steepness (or gain) parameter of the sigmoid function Sigm_JR |
| $v_{0}$ | 5.52 | mV | Average firing threshold (PSP) for which half of the firing rate is achieved |



## References
Jansen, B., Zouridakis, G., & Brandt, M. (1993). A neurophysiologically-based mathematical model of flash visual evoked potentials. *Biological Cybernetics*, 68(3), 275-283.

Jansen, B. & Rit, V. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biological Cybernetics*, 73(4), 357-366.
