

## PomeauMannevilleMap
Intermittent map with laminar phases and chaotic bursts; here given in the symmetric
piecewise form commonly used in the literature.

### State Equations
$$
x = \begin{cases} 3 - 4*x & \text{for}\: x > 0.5 \\-3 - 4*x & \text{for}\: x < -0.5 \\x*\left(1 + \left(2*\left|{x}\right|\right)^{-1 + z}\right) & \text{for}\: \left|{x}\right| \leq 0.5 \\0 & \text{otherwise} \end{cases}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $z$ | 2.5 | N/A | Intermittency exponent z. |





