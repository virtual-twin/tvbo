

## TsodyksMarkram
Tsodyks-Markram mean-field model of short-term synaptic plasticity. A three-variable neural mass model with firing rate E, synaptic depression variable x, and facilitation variable u. The transfer function uses a softplus nonlinearity α·log(1 + exp(s/α)). Reproduces the dynamics from BifurcationKit.jl/examples/TMModel.jl.

### State Equations
$$
\dot{E} = \frac{SS_{1} - E}{\tau}
$$
$$
\dot{x} = \frac{1 - x}{tauD} - E*u*x
$$
$$
\dot{u} = \frac{U_{0} - u}{tauF} + E*U_{0}*\left(1 - u\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $J$ | 3.07 | N/A | Synaptic coupling strength |
| $\alpha$ | 1.5 | N/A | Softplus nonlinearity steepness (smoothed threshold gain) |
| $E_{0}$ | -2.0 | N/A | External input (baseline excitability) |
| $\tau$ | 0.013 | N/A | Membrane time constant (firing rate relaxation) |
| $tauD$ | 0.2 | N/A | Synaptic depression recovery time constant |
| $tauF$ | 1.5 | N/A | Synaptic facilitation decay time constant |
| $U_{0}$ | 0.3 | N/A | Baseline release probability |

### Derived Quantities
#### Derived Variables
$$
SS_{0} = E_{0} + E*J*u*x
$$
$$
SS_{1} = \alpha*\log{\left(1 + e^{\frac{SS_{0}}{\alpha}} \right)}
$$




