

# ReducedWongWang
Reduced WongWang (RWW) is a biologically-inspired one-dimensional (i.e., only one state-variable 'S') neural mass model that approximates the realistic temporal dynamics of a detailed spiking and conductance-based synaptic large-scale network (Deco et al., 2013).

RWW is the dynamical mean-field (DMF) reduction of the Reduced WongWang Exc-Inh model, that consists in disentangling the contribution of the two neuronal populations (excitatory and inhibitory) in order to study the time evolution of just one pool of neurons for each network node (Wong & Wang, 2006). It results that the dynamics of each network node described the temporal evolution of the opening probability of the NMDA channels.

## Equations

### Derived Variables
$$
x = I_{o} + J_{N}*c_{global} + J_{N}*S*c_{local} + J_{N}*S*w
$$
$$
H = \frac{- b + a*x}{1 - e^{- d*\left(- b + a*x\right)}}
$$

### State Equations
$$
\frac{d}{d t} S = - \frac{S}{\tau_{s}} + H*\gamma*\left(1 - S\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $I_{o}$ | 0.33 | nA | External input current to the neurons population (Deco et al |
| $J_{N}$ | 0.2609 | nA | Excitatory recurrence |
| $a$ | 0.27 | (pC)^-1 | Slope (or gain) parameter of the sigmoid input-output function H_RWW (Deco et al |
| $b$ | 0.108 | kHz | Shift parameter of the sigmoid input-output function H_RWW (Deco et al |
| $d$ | 154.0 | ms | Scaling parameter of the sigmoid input-output function H_RWW (Deco et al |
| $\gamma$ | 0.641 | N/A | Kinetic parameter |
| $\tau_{s}$ | 100.0 | ms | Kinetic parameter |
| $w$ | 0.6 | dimensionless | Excitatory recurrence |



## References
Citation key 'Deco2013' not found.

Citation key 'WongWang2006' not found.
