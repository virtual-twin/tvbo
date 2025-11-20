

# Chua
Chua

## Equations

### Derived Variables
$$
h_{x} = m_{1}*x + \left(- \left|{-1 + x}\right| + \left|{1 + x}\right|\right)*\left(0.5*m_{0} - 0.5*m_{1}\right)
$$

### State Equations
$$
\frac{d}{d t} x = a*\left(y - h_{x} - x\right)
$$
$$
\frac{d}{d t} y = x + z - y
$$
$$
\frac{d}{d t} z = - b*y
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $a$ | 15.6 | N/A | Parameter a scaling the active conductance term. |
| $b$ | 25.58 | N/A | Parameter b scaling the third state equation term. |
| $m_{0}$ | -1.1428571428571428 | N/A | Inner slope of Chua's piecewise-linear element. |
| $m_{1}$ | -0.7142857142857143 | N/A | Outer slope of Chua's piecewise-linear element. |



