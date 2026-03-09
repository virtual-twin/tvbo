

## QuadrupoleBosonHamiltonian
Conservative Hamiltonian system modeling quadrupole surface vibrations in a nuclear
context. Adds a quartic term to Henon–Heiles-like structure producing energy-dependent
chaoticity. Default initial condition is chaotic.

### State Equations
$$
\dot{q_{0}} = A*p_{0}
$$
$$
\dot{p_{0}} = - A*q_{0} - D*q_{0}*\left(q_{0}^{2} + q_{2}^{2}\right) - \frac{3*B*\sqrt{2}*\left(q_{2}^{2} - q_{0}^{2}\right)}{2}
$$
$$
\dot{q_{2}} = A*p_{2}
$$
$$
\dot{p_{2}} = - q_{2}*\left(A + D*\left(q_{0}^{2} + q_{2}^{2}\right) + 3*B*q_{0}*\sqrt{2}\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $A$ | 1.0 | N/A | Harmonic scaling parameter A. |
| $B$ | 0.55 | N/A | Cubic interaction coefficient B. |
| $D$ | 0.4 | N/A | Quartic interaction coefficient D. |





