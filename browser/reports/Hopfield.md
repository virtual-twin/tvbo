

# Hopfield
The Hopfield neural network is a discrete time dynamical system composed of multiple binary nodes, with a connectivity matrix built from a predetermined set of patterns. The update, inspired from the spin-glass model (used to describe magnetic properties of dilute alloys), is based on a random scanning of every node. The existence of a fixed point dynamics is guaranteed by a Lyapunov function. The Hopfield network is expected to have those multiple patterns as attractors (multistable dynamical system).
When the initial conditions are close to one of the 'learned' patterns, the dynamical system is expected to relax on the corresponding attractor. A possible output of the system is the final attractive state (interpreted as an associative memory).

Various extensions of the initial model have been proposed, among which a noiseless and continuous version [Hopfield 1984] having a slightly different Lyapunov function, but essentially the same dynamical properties, with more straightforward physiological Interpretation. A continuous Hopfield neural network (with a sigmoid transfer function) can indeed be interpreted as a network of neural masses with every node corresponding to the mean field activity of a local brain region, with many bridges with the Wilson Cowan model [WC_1972].

Note:
- This model uses the modifications implemented by Golos et al. (2015).

## Equations


### State Equations
$$
\frac{d}{d t} \theta = \frac{c_{pop1} - \theta}{tauT}
$$
$$
\frac{d}{d t} x = \frac{c_{global} - x}{taux}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $tauT$ | 5.0 | N/A | The slow time-scale for threshold calculus :math:`\\theta`, state-variable of the model |
| $taux$ | 1.0 | N/A | The fast time-scale for potential calculus :math:`x`, state-variable of the model |



## References
Citation key 'Hopfield1982' not found.

Citation key 'Hopfield1984' not found.
