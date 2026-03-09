

## KuramotoModel2


### State Equations
$$
\dot{\theta} = I + \omega
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $\omega$ | 1.0 | N/A | None |

### Derived Quantities
#### Derived Parameters
$$
f = \frac{500*\omega}{\pi}
$$
#### Derived Variables
$$
lc_{0} = \sin{\left(c_{local}*\theta \right)}
$$
$$
phase = \frac{\theta}{2*\pi}
$$
$$
signal = \sin{\left(\theta \right)}
$$
$$
I = c_{glob} + lc_{0}
$$

### Output Transforms
$$
phase = \frac{\theta}{2*\pi}
$$
$$
signal = \sin{\left(\theta \right)}
$$



