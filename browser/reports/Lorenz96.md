

## Lorenz96
Ten-dimensional explicit expansion of the Lorenz-96 system: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
with cyclic indices modulo N (here N=10 unrolled to avoid symbolic indexing that the
current parser cannot interpret). Initial conditions follow the Julia reference
default u0 = range(0; length=N, step=0.1) i.e. x_i(0) = 0.1*i.

### State Equations
$$
\dot{x_{0}} = F - x_{0} + x_{9}*\left(x_{1} - x_{8}\right)
$$
$$
\dot{x_{1}} = F - x_{1} + x_{0}*\left(x_{2} - x_{9}\right)
$$
$$
\dot{x_{2}} = F - x_{2} + x_{1}*\left(x_{3} - x_{0}\right)
$$
$$
\dot{x_{3}} = F - x_{3} + x_{2}*\left(x_{4} - x_{1}\right)
$$
$$
\dot{x_{4}} = F - x_{4} + x_{3}*\left(x_{5} - x_{2}\right)
$$
$$
\dot{x_{5}} = F - x_{5} + x_{4}*\left(x_{6} - x_{3}\right)
$$
$$
\dot{x_{6}} = F - x_{6} + x_{5}*\left(x_{7} - x_{4}\right)
$$
$$
\dot{x_{7}} = F - x_{7} + x_{6}*\left(x_{8} - x_{5}\right)
$$
$$
\dot{x_{8}} = F - x_{8} + x_{7}*\left(x_{9} - x_{6}\right)
$$
$$
\dot{x_{9}} = F - x_{9} + x_{8}*\left(x_{0} - x_{7}\right)
$$

### Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $F$ | 0.01 | N/A | Constant large-scale forcing. |
| $N$ | 10.0 | N/A | Number of state variables (dimension explicitly expanded below). |





