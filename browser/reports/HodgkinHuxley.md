

# HodgkinHuxley
HodgkinHuxley

## Equations

### Derived Variables
$$
IK = gK*n^{4}*\left(V - EK\right)
$$
$$
IL = gL*\left(V - EL\right)
$$
$$
INa = gNa*h*m^{3}*\left(V - ENa\right)
$$
$$
\alpha_{h} = 0.07*e^{- \frac{13}{4} - \frac{V}{20}}
$$
$$
\alpha_{m} = \frac{4.0 + 0.1*V}{1 - e^{-4 - \frac{V}{10}}}
$$
$$
\alpha_{n} = \frac{0.55 + 0.01*V}{1 - e^{- \frac{11}{2} - \frac{V}{10}}}
$$
$$
\beta_{h} = \frac{1}{1 + e^{- \frac{7}{2} - \frac{V}{10}}}
$$
$$
\beta_{m} = 4*e^{- \frac{65}{18} - \frac{V}{18}}
$$
$$
\beta_{n} = 0.125*e^{- \frac{13}{16} - \frac{V}{80}}
$$

### State Equations
$$
\frac{d}{d t} V = \frac{Iext - gL*\left(V - EL\right) - gK*n^{4}*\left(V - EK\right) - gNa*h*m^{3}*\left(V - ENa\right)}{Cm}
$$
$$
\frac{d}{d t} m = \alpha_{m}*\left(1 - m\right) - \beta_{m}*m
$$
$$
\frac{d}{d t} h = \alpha_{h}*\left(1 - h\right) - \beta_{h}*h
$$
$$
\frac{d}{d t} n = \alpha_{n}*\left(1 - n\right) - \beta_{n}*n
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $Cm$ | 1.0 | N/A | Membrane capacitance (uF/cm^2). |
| $gNa$ | 120.0 | N/A | Maximum sodium conductance. |
| $gK$ | 36.0 | N/A | Maximum potassium conductance. |
| $gL$ | 0.3 | N/A | Leak conductance. |
| $ENa$ | 50.0 | N/A | Sodium reversal potential (mV). |
| $EK$ | -77.0 | N/A | Potassium reversal potential (mV). |
| $EL$ | -54.387 | N/A | Leak reversal potential (mV). |
| $Iext$ | 10.0 | N/A | External applied current (uA/cm^2). |



