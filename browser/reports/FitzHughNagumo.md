

# FitzHughNagumo
FitzHughNagumo

## Equations


### State Equations
$$
\frac{d}{d t} v = I - w + a*v*\left(1 - v\right)*\left(v - b\right)
$$
$$
\frac{d}{d t} w = \epsilon*\left(v - w\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 3.0 | N/A | Excitability scaling parameter a. |
| $b$ | 0.2 | N/A | Recovery nullcline parameter b. |
| $\epsilon$ | 0.01 | N/A | Time-scale separation ε. |
| $I$ | 0.0 | N/A | External input current I. |



