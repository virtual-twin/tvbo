

# SupHopf
Supercritical Hopf (SUPH) is an oscillatory/excitable neural mass model that describe the normal form of a supercritical Hopf bifurcation in Cartesian coordinates (Kuznetsov, 2013; Deco et al., 2017).

This normal form has a supercritical bifurcation at 'a=0' with 'a' the bifurcation parameter in the model. So for 'a < 0', the local dynamics has a stable fixed point and the system corresponds to a damped oscillatory state, whereas for 'a > 0', the local dynamics enters in a stable limit cycle and the system switches to an oscillatory state.

## Equations

### Derived Variables
$$
lc_{0} = c_{local}*x
$$

### State Equations
$$
\frac{d}{d t} x = c_{global} + lc_{0} + x*\left(a - x^{2} - y^{2}\right) - \omega*y
$$
$$
\frac{d}{d t} y = c_{pop1} + \omega*x + y*\left(a - x^{2} - y^{2}\right)
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | -0.5 | N/A | Local bifurcation parameter |
| $\omega$ | 1.0 | N/A | Angular frequency |



## References
Citation key 'Deco2017' not found.
