

# WilsonCowan
The Wilson and Cowan model consists of two populations or masses, one excitatory and one inhibitory, that are described by their mean firings rates E and I respectively (Wilson and Cowan, 1972, 1973).

This model is the minimal representation of a NMM with a coarse-grained description of the overall activity of a large-scale neuronal network, as opposed to being a detailed biophysical model. While employing just two differential equations, it has been used to build various biophysically realistic models (Liley et al., 1999; Daffertshofer and van Wijk, 2011).

Key parameters in the model are the strength of connectivity between each subtype of population (excitatory and inhibitory) and the strength of input to each subpopulation. The Input parameters P and Q also provide the entry point for local and long-range connectivity, that is, the activity coming from neighboring and distant populations respectively.

Varying Input and connectivity generates a diversity of dynamical behaviors that are representative of observed activity in the brain, like multistability, oscillations, traveling waves and spatial patterns.
We consider the transmission parameters of the excitatory population to be glutamatergic and therefore to be modified by glutamatergic receptors. The inhibitory population is considered as GABAergic.

Note:
- Equations and parameter names are taken from (Wilson and Cowan, 1972 and Sanz-Leon et al., 2015)
- Default parameters are taken from Fig. 4 p.10 (Wilson and Cowan, 1972)
- The model in Sanz-Leon et., 2015 includes more parameters than the original model, which can be traced in the description of the parameters.

## Equations

### Derived Variables
$$
lc_{0} = E*c_{local}
$$
$$
lc_{1} = I*c_{local}
$$
$$
x_{e} = \alpha_{e}*\left(P + c_{global} + lc_{0} + lc_{1} - \theta_{e} + E*c_{ee} - I*c_{ei}\right)
$$
$$
x_{i} = \alpha_{i}*\left(Q + lc_{0} + lc_{1} - \theta_{i} + E*c_{ie} - I*c_{ii}\right)
$$
$$
s_{e} = \frac{c_{e}}{1.0 + e^{- a_{e}*\left(x_{e} - b_{e}\right)}} - \frac{1.0*shift_{sigmoid}}{1.0 + e^{a_{e}*b_{e}}}
$$
$$
s_{i} = \frac{c_{i}}{1.0 + e^{- a_{i}*\left(x_{i} - b_{i}\right)}} - \frac{1.0*shift_{sigmoid}}{1.0 + e^{a_{i}*b_{i}}}
$$

### State Equations
$$
\frac{d}{d t} E = \frac{- E + s_{e}*\left(k_{e} - E*r_{e}\right)}{\tau_{e}}
$$
$$
\frac{d}{d t} I = \frac{- I + s_{i}*\left(k_{i} - I*r_{i}\right)}{\tau_{i}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $P$ | 0.0 | N/A | External stimulus to the excitatory population (Wilson and Cowan, 1972) |
| $Q$ | 0.0 | N/A | External stimulus to the inhibitory population (Wilson and Cowan, 1972) |
| $a_{e}$ | 1.2 | N/A | Steepness of the excitatory response function S_e_WC (Wilson and Cowan, 1972) |
| $a_{i}$ | 1.0 | N/A | Steepness of the excitatory response function S_i_WC (Wilson and Cowan, 1972) |
| $\alpha_{e}$ | 1.0 | N/A | Balance parameter between excitatory and inhibitory masses (Sanz-Leon et al |
| $\alpha_{i}$ | 1.0 | N/A | Balance parameter between excitatory and inhibitory masses (Sanz-Leon et al |
| $b_{e}$ | 2.8 | N/A | Position of the maximum slope of the excitatory response function S_e_WC (Sanz-Leon et al |
| $b_{i}$ | 4.0 | N/A | Position of the maximum slope of a sigmoid function [in         threshold units] |
| $c_{e}$ | 1.0 | N/A | The amplitude parameter for the excitatory response function |
| $c_{ee}$ | 12.0 | N/A | Excitatory to excitatory  coupling coefficient |
| $c_{ei}$ | 4.0 | N/A | Inhibitory to excitatory coupling coefficient |
| $c_{i}$ | 1.0 | N/A | The amplitude parameter for the excitatory response function S_i_WC (Sanz-Leon et al |
| $c_{ie}$ | 13.0 | N/A | Excitatory to inhibitory coupling coefficient (Sanz-Leon et al |
| $c_{ii}$ | 11.0 | N/A | Inhibitory to inhibitory coupling coefficient (Sanz-Leon et al |
| $k_{e}$ | 1.0 | N/A | Maximum value of the excitatory response function |
| $k_{i}$ | 1.0 | N/A | Maximum value of the inhibitory response function |
| $r_{e}$ | 1.0 | N/A | Excitatory refractory period |
| $r_{i}$ | 1.0 | N/A | Inhibitory refractory period |
| $shift_{sigmoid}$ | 1.0 | N/A | In order to have resting state (E=0 and I=0) in absence of external input,         the logistic curve are translated downward S(0)=0 |
| $\tau_{e}$ | 10.0 | ms | Excitatory population, membrane time-constant (Wilson and Cowan, 1972) |
| $\tau_{i}$ | 10.0 | ms | Inhibitory population, membrane time-constant (Wilson and Cowan, 1972) |
| $\theta_{e}$ | 0.0 | N/A | Excitation threshold of excitatory population (Sanz-Leon et al |
| $\theta_{i}$ | 0.0 | N/A | Excitation threshold of inhibitory population (Sanz-Leon et al |



## References
Citation key 'Wilson1972' not found.

Citation key 'Wilson1973' not found.
