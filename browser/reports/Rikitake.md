

## Rikitake
Rikitake two-disk dynamo model for geomagnetic polarity reversals. Exhibits
chaotic switching between polarity states through nonlinear coupling of
mechanical and electromagnetic components.

### State Equations
$$
\dot{x} = y*z - \mu*x
$$
$$
\dot{y} = x*\left(z - \alpha\right) - \mu*y
$$
$$
\dot{z} = 1 - x*y
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\mu$ | 1.0 | N/A | Dissipation parameter μ. |
| $\alpha$ | 1.0 | N/A | Axial asymmetry parameter α. |





