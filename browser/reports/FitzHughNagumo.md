

## FitzHughNagumo
Famous excitable system which emulates the firing of a neuron, reducing the biophysical
Hodgkin-Huxley description to two variables while preserving excitability and recovery
dynamics. Captures threshold behavior, refractoriness and sustained spiking under drive.

### State Equations
$$
\dot{v} = I - w + a*v*\left(1 - v\right)*\left(v - b\right)
$$
$$
\dot{w} = \epsilon*\left(v - w\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 3.0 | N/A | Excitability scaling parameter a. |
| $b$ | 0.2 | N/A | Recovery nullcline parameter b. |
| $\epsilon$ | 0.01 | N/A | Time-scale separation ε. |
| $I$ | 0.0 | N/A | External input current I. |





