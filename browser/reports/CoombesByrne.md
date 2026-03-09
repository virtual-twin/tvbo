

## CoombesByrne
4D model describing the Ott-Antonsen reduction of infinite all-to-all
coupled QIF neurons (Theta-neurons).

Note: the original equations describe the dynamics of the Kuramoto parameter
$Z$. Using the conformal transformation $Z=(1-W^\star)/(1+W^\star)$ and $W= \pi r + i V$, we express the system dynamics in terms of two state variables $r$ and $V$ representing the average firing rate and the average membrane potential of our QIF neurons. The conductance variable and its derivative are $g$ and $q$.


### State Equations
$$
\dot{V} = c_{global} + \eta + V^{2} + g*\left(v_{syn} - V\right) - \pi^{2}*r^{2}
$$
$$
\dot{g} = \alpha*q
$$
$$
\dot{q} = \alpha*\left(- g - 2*q + \pi*k*r\right)
$$
$$
\dot{r} = \frac{\Delta}{\pi} - g*r + 2*V*r
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\Delta$ | 0.5 | N/A | Half-width of heterogeneous noise distribution |
| $\alpha$ | 0.95 | N/A | Parameter of the alpha-function |
| $\eta$ | 20.0 | N/A | Constant parameter to scale the rate of feedback from the             firing rate variable to itself |
| $k$ | 1.0 | N/A | Local coupling strength |
| $v_{syn}$ | -10.0 | N/A | QIF membrane reversal potential |





## References
Citation key 'Byrne2020' not found.
