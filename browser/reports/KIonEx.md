

# KIonEx
It describes the mean-field activity of a population of Hodgkin-Huxley-type neurons (Depannemaker et al 2022) linking the slow fluctuations of intra- and extra-cellular potassium ion concentrations to the mean membrane potential, and the synaptic input to the population firing rate.
The model is derived as the mathematical limit of an infinite number of all-to-all coupled neurons, resulting in 5 state variables:
    :math:`x` represents a phenomenological variable connected to the firing rate,
    :math:`V` represent the average membrane potential,
    :math:`n` represents the gating variable for potassium K,
    :math:`\Delta K_{int}` represent the intracellular potassium concentration,
    :math:`K_g` represents the extracellular potassium buffering by the external bath
    """

## Equations

### Derived Variables
$$
DNa_{i} = - DKi
$$
$$
I_{Cl} = g_{Cl}*\left(V + 26.64*\log{\left(\frac{Cl_{o0}}{Cl_{i0}} \right)}\right)
$$
$$
K_{i} = DKi + K_{i0}
$$
$$
\beta = \frac{w_{i}}{w_{o}}
$$
$$
h = 1.1 - \frac{1.0}{1.0 + 24.5325301971094*e^{- 8.0*n}}
$$
$$
minf = \frac{1.0}{1.0 + e^{\frac{Cmna - V}{DCmna}}}
$$
$$
ninf = \frac{1.0}{1.0 + e^{\frac{Cnk - V}{DCnk}}}
$$
$$
r = \frac{R_{minus}*x}{\pi}
$$
$$
DK_{o} = - DKi*\beta
$$
$$
DNa_{o} = - DNa_{i}*\beta
$$
$$
Na_{i} = DNa_{i} + Na_{i0}
$$
$$
xcond = \begin{cases} \Delta - J*r*x + 2*R_{minus}*x*\left(V - c_{minus}\right) & \text{for}\: V \leq Vstar \\\Delta - J*r*x + 2*R_{plus}*x*\left(V - c_{plus}\right) & \text{otherwise} \end{cases}
$$
$$
K_{o} = DK_{o} + K_{o0} + Kg
$$
$$
Na_{o} = DNa_{o} + Na_{o0}
$$
$$
I_{K} = \left(V - 26.64*\log{\left(\frac{K_{o}}{K_{i}} \right)}\right)*\left(g_{Kl} + g_{K}*n\right)
$$
$$
I_{Na} = \left(V - 26.64*\log{\left(\frac{Na_{o}}{Na_{i}} \right)}\right)*\left(g_{Nal} + g_{Na}*h*minf\right)
$$
$$
I_{pump} = \frac{1.0*\rho}{\left(1.0 + e^{\frac{Ckp - K_{o}}{DCkp}}\right)*\left(1.0 + e^{\frac{Cnap - Na_{i}}{DCnap}}\right)}
$$
$$
V_{temp} = \frac{- 1.0*I_{Cl} - 1.0*I_{K} - 1.0*I_{Na} - 1.0*I_{pump}}{Cm}
$$
$$
Vcond = \begin{cases} \frac{R_{minus}*c_{global}*\left(- V + E\right)}{\pi} - R_{minus}*x^{2} + V_{temp} + \eta & \text{for}\: V \leq Vstar \\\frac{R_{minus}*c_{global}*\left(- V + E\right)}{\pi} - R_{plus}*x^{2} + V_{temp} + \eta & \text{otherwise} \end{cases}
$$

### State Equations
$$
\frac{d}{d t} DKi = - \frac{\gamma*\left(I_{K} - 2.0*I_{pump}\right)}{w_{i}}
$$
$$
\frac{d}{d t} Kg = \epsilon*\left(K_{bath} - K_{o}\right)
$$
$$
\frac{d}{d t} V = Vcond
$$
$$
\frac{d}{d t} n = \frac{ninf - n}{\tau_{n}}
$$
$$
\frac{d}{d t} x = xcond
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $Cm$ | 1.0 | N/A | membrane capacitance |
| $\Delta$ | 1.0 | N/A | HWHM heterogeneous noise |
| $E$ | 0.0 | N/A | Reversal Potential |
| $J$ | 0.1 | N/A | Mean Synaptic weight |
| $K_{bath}$ | 5.5 | N/A | Potassium concentration in bath |
| $R_{minus}$ | 0.5 | N/A | curvature left parabola |
| $R_{plus}$ | -0.5 | N/A | curvature right parabola |
| $Vstar$ | -31.0 | N/A | x-coordinate meeting point of parabolas |
| $c_{minus}$ | -40.0 | N/A | x-coordinate left parabola |
| $c_{plus}$ | -20.0 | N/A | x-coordinate right parabola |
| $\epsilon$ | 0.001 | N/A | diffusion rate |
| $\eta$ | 0.0 | N/A | Mean heterogeneous noise |
| $\gamma$ | 0.04 | N/A | conversion factor |
| $\tau_{n}$ | 4.0 | N/A | time constant of gating variable |
| $Chn$ | 0.4 | N/A |  |
| $Ckp$ | 5.5 | mol.m**-3 |  |
| $Cl_{i0}$ | 4.8 | mMol/m**3 | initial concentration of intracellular Cl |
| $Cl_{o0}$ | 112.0 | mMol/m**3 | initial concentration of extracellular Cl |
| $Cmna$ | -24.0 | mV |  |
| $Cnap$ | 21.0 | mol.m**-3 |  |
| $Cnk$ | -19.0 | mV |  |
| $DChn$ | -8.0 | N/A |  |
| $DCkp$ | 1.0 | mol.m**-3 |  |
| $DCmna$ | 12.0 | mV |  |
| $DCnap$ | 21.0 | mol.m**-3 |  |
| $DCnk$ | 18.0 | mV |  |
| $K_{i0}$ | 130.0 | mMol/m**3 | initial concentration of intracellular K |
| $K_{o0}$ | 4.8 | mMol/m**3 | initial concentration of extracellular K |
| $Na_{i0}$ | 16.0 | mMol/m**3 | initial concentration of intracellular Na |
| $Na_{o0}$ | 138.0 | mMol/m**3 | initial concentration of extracellular Na |
| $g_{Cl}$ | 7.5 | nS | chloride conductance |
| $g_{K}$ | 22.0 | nS | maximal potassium conductance |
| $g_{Kl}$ | 0.12 | nS | potassium leak conductance |
| $g_{Na}$ | 40.0 | nS | maximal sodiumconductance |
| $g_{Nal}$ | 0.02 | nS | sodium leak conductance |
| $\rho$ | 250.0 | pA | maximal Na/K pump current |
| $w_{i}$ | 2160.0 | umeter**3 | intracellular volume |
| $w_{o}$ | 720.0 | umeter**3 | extracellular volume |



## References
Citation key 'Rabuffo2024' not found.

Citation key 'Depannemaecker2023' not found.
