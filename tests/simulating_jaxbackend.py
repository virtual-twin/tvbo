#
# Module: simulating_jaxbackend.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator import models
from tvb_autodiff.jax import JaxBackend
from tvb.simulator.simulator import Simulator, connectivity

from tvbo.export import templater
from tvbo.knowledge import ontology

warnings.filterwarnings("ignore", category=FutureWarning, module="jax._src.ops.scatter")

np.random.seed(1312)
supported_models = sorted(
    (
        "LarterBreakspear",
        "ReducedWongWang",
        "WilsonCowan",
        "SupHopf",
        "Kuramoto",
        "Generic2dOscillator",
        "ReducedWongWangExcInh",
        "Linear",
        "JansenRit",
        "MontbrioPazoRoxin",
    )
)
supported_models = ontology.get_models().keys()

# Calculate the number of rows and columns for the subplots
n_models = len(supported_models)
n_cols = 5  # Max number of columns
n_rows = n_models // n_cols + (1 if n_models % n_cols > 0 else 0)

# Create the subplots
fig, axs = plt.subplots(
    nrows=n_rows,
    ncols=min(n_cols, n_models),
    figsize=(n_cols * 3, n_rows * 2),
    layout="constrained",
    subplot_kw={"aspect": "auto"},
)
fig.suptitle("TVB-O-Jax Simulations", fontweight="bold")

# Flatten axs array if there are multiple rows
if n_rows > 1:
    axs = axs.flatten()
# sc = connectivity.sinlge_node()
sc = connectivity.Connectivity.from_file()

for i, model in enumerate(supported_models):
    axs[i].set_title(model)

    print(model)
    tvbo_model = templater.model2class(
        model,
        print_source=False,
        split_nonintegrated_variables=True,
    )
    tvbo_model.configure()

    # try:
    sim = Simulator(model=tvbo_model, connectivity=sc, simulation_length=200)
    sim.configure()
    sim.coupling.a = np.array([0])
    sim.coupling.b = np.array([0])
    sim.initial_conditions = np.zeros((1, sim.model._nvar, sim.number_of_nodes, 1))
    sim.configure()

    jb = JaxBackend()
    try:
        kf, params = jb.build_sim(sim, print_source=False)
        tv, xv = kf(*params)
    except:
        kf, params = jb.build_sim(sim, print_source=False)
        pass

    # for j in range(xv.shape[1]):
    axs[i].plot(
        tv,
        xv[:, :, :, 0].mean(axis=2),
        color="#042A2B",
        label="TVB-O",
        alpha=0.8,
        linewidth=2,
    )
    # except:
    #     print(f"TVB-O sim Failed for {model}")

    # TVB comparison
    tvb_model = getattr(
        models,
        model.replace("Epileptor5D", "Epileptor").replace("GenericLinear", "Linear"),
    )()
    sim2 = Simulator(model=tvb_model, connectivity=sc, simulation_length=200)
    sim2.configure()
    sim2.coupling.a = np.array([0])
    sim2.coupling.b = np.array([0])
    sim2.initial_conditions = np.zeros((1, sim2.model._nvar, sim2.number_of_nodes, 1))
    sim2.configure()

    jb2 = JaxBackend()

    try:
        kf2, params2 = jb2.build_sim(sim2)
        tv2, xv2 = kf2(*params2)
    except:
        print(
            f"TVB model {model} not supported by TVB-O-Jax. Running generic TVB instead"
        )

        tv2, xv2 = sim2.run()[0]

    for j in range(xv.shape[1]):
        axs[i].plot(
            tv2,
            xv2[:, :, :, 0].mean(axis=2),
            color="#F4E04D",
            label="TVB",
            alpha=0.8,
            linewidth=2,
            linestyle="dashed",
            dashes=(3, 8),
        )
axs[i].legend()

# fig.savefig(f"jax_validation/{model}.png", bbox_inches="tight")
plt.show()
