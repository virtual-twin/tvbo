

## Gissinger
Low-order model for geomagnetic field reversals exhibiting chaotic polarity changes.
Captures interplay between dipole (Q), quadrupole (D) and velocity (V) modes.

### State Equations
$$
\dot{Q} = Q*\mu - D*V
$$
$$
\dot{D} = Q*V - D*\nu
$$
$$
\dot{V} = \Gamma - V + D*Q
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\mu$ | 0.119 | N/A | Growth parameter μ for dipole component Q. |
| $\nu$ | 0.1 | N/A | Damping parameter ν of D variable. |
| $\Gamma$ | 0.9 | N/A | Constant source term Γ in V equation. |





