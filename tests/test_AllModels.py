#
# Module: test_AllModels.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
# %%
"""
NMM Test
========
Test all models in TVB-O and compare with TVB for different backends.
"""
import os
import pickle
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import integrators, models
from tvb.simulator.simulator import Simulator
from tvb_autodiff.jax import JaxBackend

from drafts import rateml
from tvbo.export import lemsgenerator, templater
from tvbo.knowledge import ontology
from tvbo.knowledge.simulation.networkmodel import BrainNetworkModel
from tvbo.parse import ontology_loader

get_logger("tvb.datatypes").setLevel("ERROR")
get_logger("tvb.datatypes.Connectivity").setLevel("ERROR")
get_logger("tvb.simulator").setLevel("ERROR")
get_logger("tvb.simulator.simulator").setLevel("ERROR")
get_logger("tvb.simulator.history").setLevel("ERROR")
get_logger("tvb.simulator.noise").setLevel("ERROR")
get_logger("tvb.simulator.integrators").setLevel("ERROR")
get_logger("tvb.simulator.models").setLevel("ERROR")

ROOT = abspath(dirname(__file__))
odir = join(ROOT, "tvb_validation")
os.makedirs(odir, exist_ok=True)
np.random.seed(1312)


def simulator(model, nsig=1e-7, dt=0.001):
    sc = Connectivity.from_file()
    sc.configure()
    coupling = templater.coupling2class("Linear", print_source=False, a=0.0, b=0.0)
    sim = Simulator(
        model=model,
        connectivity=sc,
        integrator=integrators.HeunDeterministic(dt=dt),
        coupling=coupling,
        simulation_length=100,
        initial_conditions=np.zeros((1, model._nvar, sc.weights.shape[0], 1)),
    )
    sim.configure()
    return sim


tvbo_models = ontology.get_models()
data = dict()
for j, model in enumerate(sorted(tvbo_models)):
    plt.cla()
    if model not in ontology.functional_models:
        continue
    # data[model] = dict()
    print(model)
    bnm = BrainNetworkModel(model=model)

    bnm.simulator.configure()
    ((time, data),) = bnm.simulator.run()

    methods = ["TVB-O", "TVB-O-Jax", "TVB"]# , "TVB-O numba", "TVB-O-Jax", "TVB", "TVB-Jax"]

    # Create the subplots
    fig, ax = plt.subplots(
        nrows=len(methods),
        layout="constrained",
        subplot_kw={"aspect": "auto"},
        sharex="all",
        sharey="all",
        figsize=(16, 16),
    )

    fig.suptitle(f"{model}", fontweight="bold")

    for k, method in enumerate(methods):
        print(f"{method}")
        ax[k].set_title(f"{method}")
        ax[k].set_xlabel("Time (ms)")
        ax[k].set_ylabel("Activity")

        # Define Neural Mass Model Class
        if method == "TVB-O":
            nmm = templater.model2class(
                model,
                split_nonintegrated_variables=False,
            )
            vois = nmm.variables_of_interest
        elif method == "TVB-O-LEMS-RateML":
            if model in [
                "Epileptor2D",
                "Epileptor5D",
                "LarterBreakspear",
                "ZerlautAdaptationFirstOrder",
                "KIonEx",
            ]:
                continue
            lemsgenerator.export_lems_model(model=model)
            rateml.lems2python(model=model)
            nmm = ontology_loader.load_tvb_model(model)
        elif method == "TVB-O numba":
            nmm = templater.model2class(
                model,
                return_instance=False,
            )
            nmm.use_numba = True
            nmm = nmm()
            nmm.configure()
        elif method == "TVB-O-Jax":
            if model in [
                "KIonEx",
            ]:
                continue
            nmm = templater.model2class(
                model,
                split_nonintegrated_variables=True,
            )
        elif method in ["TVB", "TVB-Jax"]:
            nmm = getattr(
                models,
                model.replace("GenericLinear", "Linear").replace(
                    "Epileptor5D", "Epileptor"
                ),
            )()
            nmm.variables_of_interest = vois

        sim = simulator(nmm)
        if "Jax" in method:
            jb = JaxBackend()
            try:
                kf, params = jb.build(sim, print_source=False)
            except:
                continue
            tv, xv = kf(*params)
            xv = np.expand_dims(xv, axis=3)
        else:
            tv, xv = sim.run()[0]

        ax[k].plot(tv, xv[:, :, 0, 0], linewidth=2)
        ax[k].legend(
            labels=sim.model.variables_of_interest,
            ncols=8,
            fontsize=8,
            loc="lower left",
        )
        print(f"successful for {model}")
        print()

    fig.savefig(join(odir, f"{model}.pdf"))
    plt.close()
    print(f"Saved {model} - TVB")
    print()
    print()

    # with open(join(odir, "tvb_validation_data.pkl"), "wb") as f:
    #     pickle.dump(data, f)
    # wait = input("Press Enter to continue.")

# %%
