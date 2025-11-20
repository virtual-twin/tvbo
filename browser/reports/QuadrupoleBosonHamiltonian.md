

# QuadrupoleBosonHamiltonian
QuadrupoleBosonHamiltonian

## Equations


### State Equations
$$
\frac{d}{d t} q_{0} = A*p_{0}
$$
$$
\frac{d}{d t} p_{0} = - A*q_{0} - D*q_{0}*\left(q_{0}^{2} + q_{2}^{2}\right) - \frac{3*B*\sqrt{2}*\left(q_{2}^{2} - q_{0}^{2}\right)}{2}
$$
$$
\frac{d}{d t} q_{2} = A*p_{2}
$$
$$
\frac{d}{d t} p_{2} = - q_{2}*\left(A + D*\left(q_{0}^{2} + q_{2}^{2}\right) + 3*B*q_{0}*\sqrt{2}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $A$ | 1.0 | N/A | Harmonic scaling parameter A. |
| $B$ | 0.55 | N/A | Cubic interaction coefficient B. |
| $D$ | 0.4 | N/A | Quartic interaction coefficient D. |



