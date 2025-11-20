

# Shinriki
Shinriki

## Equations

### Derived Variables
$$
Vnl = 2.295*10^{-5}*e^{3.0038*v_{1} - 3.0038*v_{2}} - 2.295*10^{-5}*e^{3.0038*v_{2} - 3.0038*v_{1}}
$$

### State Equations
$$
\frac{d}{d t} v_{1} = 6.89655172413793*v_{2} - 100.0*Vnl - 6.89655172413793*v_{1} + 100.0*v_{1}*\left(0.144927536231884 - \frac{1}{R_{1}}\right)
$$
$$
\frac{d}{d t} v_{2} = 10.0*Vnl + 0.689655172413793*v_{1} - 10.0*v_{3} - 0.689655172413793*v_{2}
$$
$$
\frac{d}{d t} v_{3} = 3.125*v_{2} - 0.3125*v_{3}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $R_{1}$ | 22.0 | N/A | Variable resistor R1 controlling nonlinearity. |



