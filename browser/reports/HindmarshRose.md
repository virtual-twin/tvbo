

## HindmarshRose
The Hindmarsh-Rose model reproduces the bursting behavior of a neuron's membrane
potential, characterized by a fast sequence of spikes followed by a quiescent period.
The x variable is the membrane potential; y the fast recovery (ionic current) variable;
z a slow adaptation current. Parameter sets modulate transitions between quiescence,
tonic spiking and bursting.

### State Equations
$$
\dot{x} = I + y - z + b*x^{2} - a*x^{3}
$$
$$
\dot{y} = c - y - d*x^{2}
$$
$$
\dot{z} = r*\left(- z + s*\left(x - xr\right)\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 1.0 | N/A | Cubic nonlinearity scaling a. |
| $b$ | 3.0 | N/A | Quadratic term scaling b. |
| $c$ | 1.0 | N/A | Constant current term c in y equation. |
| $d$ | 5.0 | N/A | Scaling of x^2 in y equation. |
| $r$ | 0.001 | N/A | Slow adaptation rate r. |
| $s$ | 4.0 | N/A | Adaptation coupling strength s. |
| $xr$ | -1.6 | N/A | Reference potential x_r. |
| $I$ | 2.0 | N/A | External applied current I. |





