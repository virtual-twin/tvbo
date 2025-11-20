

# DoublePendulum
DoublePendulum

## Equations

### Derived Variables
$$
\phi = \theta_{2} - \theta_{1}
$$
$$
\Delta = M_{1} + M_{2} - M_{2}*\cos^{2}{\left(\phi \right)}
$$

### State Equations
$$
\frac{d}{d t} \theta_{1} = \omega_{1}
$$
$$
\frac{d}{d t} \omega_{1} = \frac{M_{2}*\left(\left(G*\sin{\left(\theta_{2} \right)} + L_{1}*\omega_{1}^{2}*\sin{\left(\phi \right)}\right)*\cos{\left(\phi \right)} + L_{2}*\omega_{2}^{2}*\sin{\left(\phi \right)}\right) - G*\left(M_{1} + M_{2}\right)*\sin{\left(\theta_{1} \right)}}{\Delta*L_{1}}
$$
$$
\frac{d}{d t} \theta_{2} = \omega_{2}
$$
$$
\frac{d}{d t} \omega_{2} = \frac{\left(M_{1} + M_{2}\right)*\left(G*\left(- \sin{\left(\theta_{2} \right)} + \cos{\left(\phi \right)}*\sin{\left(\theta_{1} \right)}\right) - L_{1}*\omega_{1}^{2}*\sin{\left(\phi \right)}\right) - L_{2}*M_{2}*\omega_{2}^{2}*\cos{\left(\phi \right)}*\sin{\left(\phi \right)}}{\Delta*L_{2}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $G$ | 10.0 | N/A | Gravitational acceleration. |
| $L_{1}$ | 1.0 | N/A | Length of first rod. |
| $L_{2}$ | 1.0 | N/A | Length of second rod. |
| $M_{1}$ | 1.0 | N/A | Mass of first bob. |
| $M_{2}$ | 1.0 | N/A | Mass of second bob. |



