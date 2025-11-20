

# IkedaMap
IkedaMap

## Equations

### Derived Variables
$$
\theta = c - \frac{d}{1 + x^{2} + y^{2}}
$$

### State Equations
$$
x = a + b*\left(x*\cos{\left(\theta \right)} - y*\sin{\left(\theta \right)}\right)
$$
$$
y = b*\left(x*\sin{\left(\theta \right)} + y*\cos{\left(\theta \right)}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 1.0 | N/A | Constant bias a. |
| $b$ | 1.0 | N/A | Linear scaling b. |
| $c$ | 0.4 | N/A | Phase offset c. |
| $d$ | 6.0 | N/A | Phase nonlinearity scale d. |



