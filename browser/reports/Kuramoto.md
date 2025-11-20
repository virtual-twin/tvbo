

# Kuramoto
The Kuramoto model is a model of synchronization phenomena derived by Yoshiki Kuramoto in 1975 which has since been applied to diverse domains including the study of neuronal oscillations and synchronization.

## Equations

### Derived Variables
$$
lc_{0} = \sin{\left(c_{local}*\theta \right)}
$$
$$
I = c_{global} + lc_{0}
$$

### State Equations
$$
\frac{d}{d t} \theta = I + \omega
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\omega$ | 1.0 | rad/ms | Sets the base line frequency for the Kuramoto oscillator |



## References
Cabral, J., Hugues, E., Sporns, O., & Deco, G. (2011). Role of local network oscillations in resting-state functional connectivity. *NeuroImage*, 57(1), 130-139.

Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics*, 420-422.

Strogatz, S. (2000). From kuramoto to crawford: exploring the onset of synchronization in populations of coupled oscillators. *Physica D: Nonlinear Phenomena*, 143(1–4), 1-20.
