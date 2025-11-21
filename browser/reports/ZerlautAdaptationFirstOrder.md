

# ZerlautAdaptationFirstOrder


## Equations

### Derived Variables
$$
lc_{E} = E*c_{local}
$$
$$
lc_{I} = I*c_{local}
$$
$$
Fe_{ext} = \begin{cases} 0 & \text{for}\: K_{ext e}*\left(c_{global} + lc_{E} + ou_{drift}*weight_{noise}\right) < 0 \\c_{global} + lc_{E} + ou_{drift}*weight_{noise} & \text{otherwise} \end{cases}
$$
$$
Fi_{ext} = lc_{I}
$$
$$
fe_{e} = K_{ext e}*\left(Fe_{ext} + external_{input ex ex}\right) + N_{tot}*p_{connect e}*\left(\frac{1}{1000000} + E\right)*\left(1.0 - g\right)
$$
$$
fe_{i} = K_{ext e}*\left(Fe_{ext} + external_{input in ex}\right) + N_{tot}*p_{connect e}*\left(\frac{1}{1000000} + E\right)*\left(1.0 - g\right)
$$
$$
fi_{e} = K_{ext i}*\left(Fi_{ext} + external_{input ex in}\right) + N_{tot}*g*p_{connect i}*\left(\frac{1}{1000000} + I\right)
$$
$$
fi_{i} = K_{ext i}*\left(Fi_{ext} + external_{input in in}\right) + N_{tot}*g*p_{connect i}*\left(\frac{1}{1000000} + I\right)
$$
$$
\mu_{Ge e} = Q_{e}*fe_{e}*\tau_{e}
$$
$$
\mu_{Ge i} = Q_{e}*fe_{i}*\tau_{e}
$$
$$
\mu_{Gi e} = Q_{i}*fi_{e}*\tau_{i}
$$
$$
\mu_{Gi i} = Q_{i}*fi_{i}*\tau_{i}
$$
$$
\mu_{G e} = g_{L} + \mu_{Ge e} + \mu_{Gi e}
$$
$$
\mu_{G i} = g_{L} + \mu_{Ge i} + \mu_{Gi i}
$$
$$
T_{m e} = \frac{C_{m}}{\mu_{G e}}
$$
$$
T_{m i} = \frac{C_{m}}{\mu_{G i}}
$$
$$
\mu_{V e} = \frac{- W_{e} + E_{L e}*g_{L} + E_{e}*\mu_{Ge e} + E_{i}*\mu_{Gi e}}{\mu_{G e}}
$$
$$
\mu_{V i} = \frac{- W_{i} + E_{L i}*g_{L} + E_{e}*\mu_{Ge i} + E_{i}*\mu_{Gi i}}{\mu_{G i}}
$$
$$
U_{e e} = \frac{Q_{e}*\left(E_{e} - \mu_{V e}\right)}{\mu_{G e}}
$$
$$
U_{e i} = \frac{Q_{e}*\left(E_{e} - \mu_{V i}\right)}{\mu_{G i}}
$$
$$
U_{i e} = \frac{Q_{i}*\left(E_{i} - \mu_{V e}\right)}{\mu_{G e}}
$$
$$
U_{i i} = \frac{Q_{i}*\left(E_{i} - \mu_{V i}\right)}{\mu_{G i}}
$$
$$
V_{e} = \frac{\mu_{V e} - muV_{0}}{DmuV_{0}}
$$
$$
V_{i} = \frac{\mu_{V i} - muV_{0}}{DmuV_{0}}
$$
$$
T_{V e} = \frac{fe_{e}*U_{e e}^{2}*\tau_{e}^{2} + fi_{e}*U_{i e}^{2}*\tau_{i}^{2}}{\frac{fe_{e}*U_{e e}^{2}*\tau_{e}^{2}}{T_{m e} + \tau_{e}} + \frac{fi_{e}*U_{i e}^{2}*\tau_{i}^{2}}{T_{m e} + \tau_{i}}}
$$
$$
T_{V i} = \frac{fe_{i}*U_{e i}^{2}*\tau_{e}^{2} + fi_{i}*U_{i i}^{2}*\tau_{i}^{2}}{\frac{fe_{i}*U_{e i}^{2}*\tau_{e}^{2}}{T_{m i} + \tau_{e}} + \frac{fi_{i}*U_{i i}^{2}*\tau_{i}^{2}}{T_{m i} + \tau_{i}}}
$$
$$
\sigma_{V e} = \sqrt{\frac{fe_{e}*U_{e e}^{2}*\tau_{e}^{2}}{2.0*T_{m e} + 2.0*\tau_{e}} + \frac{fi_{e}*U_{i e}^{2}*\tau_{i}^{2}}{2.0*T_{m e} + 2.0*\tau_{i}}}
$$
$$
\sigma_{V i} = \sqrt{\frac{fe_{i}*U_{e i}^{2}*\tau_{e}^{2}}{2.0*T_{m i} + 2.0*\tau_{e}} + \frac{fi_{i}*U_{i i}^{2}*\tau_{i}^{2}}{2.0*T_{m i} + 2.0*\tau_{i}}}
$$
$$
S_{e} = \frac{\sigma_{V e} - sV_{0}}{DsV_{0}}
$$
$$
S_{i} = \frac{\sigma_{V i} - sV_{0}}{DsV_{0}}
$$
$$
T_{e} = \frac{- TvN_{0} + \frac{T_{V e}*g_{L}}{C_{m}}}{DTvN_{0}}
$$
$$
T_{i} = \frac{- TvN_{0} + \frac{T_{V i}*g_{L}}{C_{m}}}{DTvN_{0}}
$$
$$
V_{thre e} = 1000.0*P_{0 e} + 1000.0*P_{1 e}*V_{e} + 1000.0*P_{2 e}*S_{e} + 1000.0*P_{3 e}*T_{e} + 1000.0*P_{4 e}*V_{e}^{2} + 1000.0*P_{5 e}*S_{e}^{2} + 1000.0*P_{6 e}*T_{e}^{2} + 1000.0*P_{7 e}*S_{e}*V_{e} + 1000.0*P_{8 e}*T_{e}*V_{e} + 1000.0*P_{9 e}*S_{e}*T_{e}
$$
$$
V_{thre i} = 1000.0*P_{0 i} + 1000.0*P_{1 i}*V_{i} + 1000.0*P_{2 i}*S_{i} + 1000.0*P_{3 i}*T_{i} + 1000.0*P_{4 i}*V_{i}^{2} + 1000.0*P_{5 i}*S_{i}^{2} + 1000.0*P_{6 i}*T_{i}^{2} + 1000.0*P_{7 i}*S_{i}*V_{i} + 1000.0*P_{8 i}*T_{i}*V_{i} + 1000.0*P_{9 i}*S_{i}*T_{i}
$$
$$
f_{out e} = \frac{\operatorname{erfc}{\left(\frac{\sqrt{2}*\left(V_{thre e} - \mu_{V e}\right)}{2*\sigma_{V e}} \right)}}{2*T_{V e}}
$$
$$
f_{out i} = \frac{\operatorname{erfc}{\left(\frac{\sqrt{2}*\left(V_{thre i} - \mu_{V i}\right)}{2*\sigma_{V i}} \right)}}{2*T_{V i}}
$$

### State Equations
$$
\frac{d}{d t} E = \frac{f_{out e} - E}{T}
$$
$$
\frac{d}{d t} I = \frac{f_{out i} - I}{T}
$$
$$
\frac{d}{d t} W_{e} = E*b_{e} - \frac{W_{e}}{\tau_{w e}} + \frac{a_{e}*\left(\mu_{V e} - E_{L e}\right)}{\tau_{w e}}
$$
$$
\frac{d}{d t} W_{i} = I*b_{i} - \frac{W_{i}}{\tau_{w i}} + \frac{a_{i}*\left(\mu_{V i} - E_{L i}\right)}{\tau_{w i}}
$$
$$
\frac{d}{d t} ou_{drift} = - \frac{ou_{drift}}{\tau_{OU}}
$$


## Parameters

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
| $C_{m}$ | 200.0 | N/A | membrane capacitance [pF] |
| $DTvN_{0}$ | 1.0 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $DmuV_{0}$ | 10.0 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $DsV_{0}$ | 6.0 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $E_{L e}$ | -65.0 | N/A | leak reversal potential for excitatory [mV] |
| $E_{L i}$ | -65.0 | N/A | leak reversal potential for inhibitory [mV] |
| $E_{e}$ | 0.0 | N/A | excitatory reversal potential [mV] |
| $E_{i}$ | -80.0 | N/A | inhibitory reversal potential [mV] |
| $K_{ext e}$ | 400.0 | N/A | Number of excitatory connexions from external population |
| $K_{ext i}$ | 0.0 | N/A | Number of inhibitory connexions from external population |
| $N_{tot}$ | 10000.0 | N/A | cell number |
| $P_{0 e}$ | -0.04983106 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{0 i}$ | -0.05149122024209484 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{1 e}$ | 0.005063550882777035 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{1 i}$ | 0.004003689190271077 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{2 e}$ | -0.023470121807314552 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{2 i}$ | -0.008352013668528155 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{3 e}$ | 0.0022951513725067503 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{3 i}$ | 0.0002414237992765705 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{4 e}$ | -0.0004105302652029825 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{4 i}$ | -0.0005070645080016026 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{5 e}$ | 0.010547051343547399 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{5 i}$ | 0.0014345394104282397 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{6 e}$ | -0.03659252821136933 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{6 i}$ | -0.014686689498949967 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{7 e}$ | 0.007437487505797858 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{7 i}$ | 0.004502706285435741 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{8 e}$ | 0.001265064721846073 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{8 i}$ | 0.0028472190352532454 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{9 e}$ | -0.04072161294490446 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{9 i}$ | -0.015357804594594548 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $P_{e}$ | -0.04983106 | N/A | Polynome of excitatory phenomenological threshold (order 9) |
| $P_{i}$ | -0.05149122024209484 | N/A | Polynome of inhibitory phenomenological threshold (order 9) |
| $Q_{e}$ | 1.5 | N/A | excitatory quantal conductance [nS] |
| $Q_{i}$ | 5.0 | N/A | inhibitory quantal conductance [nS] |
| $S_{i}$ | 1.0 | N/A | Scaling of the remote input for the inhibitory population with         respect to the excitatory population |
| $T$ | 20.0 | N/A | Time scale of describing network activity |
| $TvN_{0}$ | 0.5 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $a_{e}$ | 4.0 | N/A | Excitatory adaptation conductance [nS] |
| $a_{i}$ | 0.0 | N/A | Inhibitory adaptation conductance [nS] |
| $b_{e}$ | 60.0 | N/A | Excitatory adaptation current increment [pA] |
| $b_{i}$ | 0.0 | N/A | Inhibitory adaptation current increment [pA] |
| $external_{input ex ex}$ | 0.0 | N/A | external drive |
| $external_{input ex in}$ | 0.0 | N/A | external drive |
| $external_{input in ex}$ | 0.0 | N/A | external drive |
| $external_{input in in}$ | 0.0 | N/A | external drive |
| $g_{L}$ | 10.0 | N/A | leak conductance [nS] |
| $g$ | 0.2 | N/A | fraction of inhibitory cells |
| $muV_{0}$ | -60.0 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $p_{connect e}$ | 0.05 | N/A | connectivity probability |
| $p_{connect i}$ | 0.05 | N/A | connectivity probability |
| $sV_{0}$ | 4.0 | N/A | Normalization factors page 48 after the equation 4 from [ZD_2018] |
| $\tau_{OU}$ | 5.0 | N/A | time constant noise |
| $\tau_{e}$ | 5.0 | N/A | excitatory decay [ms] |
| $\tau_{i}$ | 5.0 | N/A | inhibitory decay [ms] |
| $\tau_{w e}$ | 500.0 | N/A | Adaptation time constant of excitatory neurons [ms] |
| $\tau_{w i}$ | 1.0 | N/A | Adaptation time constant of inhibitory neurons [ms] |
| $weight_{noise}$ | 10.5 | N/A | weight noise |



## References
Citation key 'diVolo2019' not found.

Citation key 'Zerlaut2018' not found.
