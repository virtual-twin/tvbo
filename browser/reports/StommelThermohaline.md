

## StommelThermohaline
Two-box thermohaline circulation model capturing temperature (T) and salinity (S)
driven density contrasts leading to multiple equilibria in overturning strength.

### State Equations
$$
\dot{T} = \eta_{1} - T - T*\operatorname{abs}{\left(T - S \right)}
$$
$$
\dot{S} = \eta_{2} - S*\eta_{3} - S*\operatorname{abs}{\left(T - S \right)}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\eta_{1}$ | 3.0 | N/A | Thermal forcing parameter η₁. |
| $\eta_{2}$ | 1.0 | N/A | Freshwater/salinity source parameter η₂. |
| $\eta_{3}$ | 0.3 | N/A | Salinity damping coefficient η₃. |





