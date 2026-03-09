

## HenonHeiles
The Hénon–Heiles system is a conservative 2 DoF Hamiltonian model introduced to
study stellar motion near a galactic center and the search for a third integral
of motion. The default initial condition is a typical chaotic orbit.

### State Equations
$$
\dot{x} = px
$$
$$
\dot{y} = py
$$
$$
\dot{px} = - x - 2*x*y
$$
$$
\dot{py} = y^{2} - y - x^{2}
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|


### Derived Quantities
#### Derived Variables
$$
H = - \frac{y^{3}}{3} + 0.5*px^{2} + 0.5*py^{2} + 0.5*x^{2} + 0.5*y^{2} + y*x^{2}
$$




