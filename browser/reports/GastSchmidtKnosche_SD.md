

# GastSchmidtKnosche_SD
4D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) with Synaptic Depression adaptation
    mechanisms [Gastetal_2020]_.

    The two state variables :math:`r` and :math:`V` represent the average firing rate and
    the average membrane potential of our QIF neurons.
    :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

    The equations of the infinite QIF 2D population model read

    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r (1 - A) + I)\\
            \dot{A} &= 1/\tau_A (B)\\
            \dot{B} &= 1/\tau_A (-2 B - A + \alpha  r) \\

    .. [Gastetal_2020] Gast, R., Schmidt, H., & Knösche, T. R. (2020). A mean-field description of bursting dynamics in spiking neural networks with short-term adaptation. *Neural Computation*, 32(9), 1615-1634.

## Equations


### State Equations
$$
\frac{d}{d t} A = \frac{B}{\tau_{A}}
$$
$$
\frac{d}{d t} B = \frac{- A - 2*B + \alpha*r}{\tau_{A}}
$$
$$
\frac{d}{d t} V = \frac{I + \eta + V^{2} + c_{global}*cr + c_{pop1}*cv - \pi^{2}*r^{2}*\tau^{2} + J*r*\tau*\left(1 - A\right)}{\tau}
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
| $\alpha$ | 0.5 | N/A | adaptation rate |
| $cr$ | 1.0 | N/A | It is the weight on Coupling through variable r |
| $cv$ | 0.0 | N/A | It is the weight on Coupling through variable V |
| $\eta$ | -6.0 | N/A | Mean of heterogeneous noise distribution |
| $\tau_{A}$ | 10.0 | N/A | Adaptation time scale |
| $\tau$ | 1.0 | N/A | Characteristic time |



## References
Citation key 'Gast2020' not found.
