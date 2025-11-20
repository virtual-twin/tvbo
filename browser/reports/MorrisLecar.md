

# MorrisLecar
MorrisLecar

## Equations

### Derived Variables
$$
G = 0.5 + 0.5*\tanh{\left(\frac{V - V_{3}}{V_{4}} \right)}
$$
$$
M = 0.5 + 0.5*\tanh{\left(\frac{V - V_{1}}{V_{2}} \right)}
$$

### State Equations
$$
\frac{d}{d t} V = I - gL*\left(V - VL\right) - M*gCa*\left(V - VCa\right) - N*gK*\left(V - VK\right)
$$
$$
\frac{d}{d t} N = \frac{G - N}{\tau}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $I$ | 0.15 | N/A | Applied current I. |
| $V_{3}$ | 0.1 | N/A | Half-activation parameter V3. |
| $V_{1}$ | -0.0 | N/A | Half-activation parameter V1. |
| $V_{2}$ | 0.15 | N/A | Slope factor V2. |
| $V_{4}$ | 0.1 | N/A | Slope factor V4. |
| $VCa$ | 1.0 | N/A | Calcium reversal potential. |
| $VL$ | -0.5 | N/A | Leak reversal potential. |
| $VK$ | -0.7 | N/A | Potassium reversal potential. |
| $gCa$ | 1.2 | N/A | Calcium conductance. |
| $gK$ | 2.0 | N/A | Potassium conductance. |
| $gL$ | 0.5 | N/A | Leak conductance. |
| $\tau$ | 3.0 | N/A | Recovery time constant τ. |



