

## LotkaVolterraPredPrey
The famous Lotka-Volterra model is a simple ecological model describing the interaction
between a predator and a prey species (or also parasite and host species). It has been
used independently in epidemics, ecology, and economics. The prey (x) grows
exponentially in absence of predators; predation converts prey into predator growth;
predator (y) declines without prey. Default parameters produce sustained oscillations.

### State Equations
$$
\dot{x} = \alpha*x - \beta*x*y
$$
$$
\dot{y} = - \gamma*y + \delta*x*y
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\alpha$ | 1.5 | N/A | Prey intrinsic growth rate α. |
| $\beta$ | 1.0 | N/A | Predation interaction coefficient β. |
| $\delta$ | 1.0 | N/A | Predator growth efficiency δ. |
| $\gamma$ | 3.0 | N/A | Predator mortality rate γ. |





