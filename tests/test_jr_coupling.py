#
# Module: test_jr_coupling.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
# test_jr_coupling
import tvb
from tvb.simulator.simulator import Simulator
from tvb.simulator import models
from tvbo.export import templater
from tvb.datatypes.connectivity import Connectivity
import numpy as np
import matplotlib.pyplot as plt

coupling = templater.coupling2class("SigmoidalJansenRit", print_source=True)
coupling.configure()

sim = Simulator(
    model=templater.model2class("JansenRit"),
    connectivity=Connectivity.from_file(),
    coupling=coupling,
    # coupling=tvb.simulator.coupling.SigmoidalJansenRit(),
)
sim.configure()
init = np.zeros((1, sim.model._nvar, sim.number_of_nodes, 1))

sim2 = Simulator(
    model=models.JansenRit(),
    connectivity=Connectivity.from_file(),
    coupling=tvb.simulator.coupling.SigmoidalJansenRit(),
    initial_conditions=init,
)

sim.initial_conditions = init
sim.model.variables_of_interest = sim2.model.variables_of_interest
sim.configure()

tv, xv = sim.run()[0]
print(xv.shape)
plt.plot(tv, xv[:, :, :, 0].mean(axis=2), color="black", alpha=0.5)


sim2.configure()
tv, xv = sim2.run()[0]
print(xv.shape)
plt.plot(tv, xv[:, :, :, 0].mean(axis=2), color="red", alpha=0.5, linestyle="dashed")

# plt.show()
