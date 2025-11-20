#  tvbo_examples_allModels.py
#
# Created on Mon Jan 15 2024
# Author: Leon K. Martin
#
# Copyright (c) 2024 Charité Universitätsmedizin Berlin
#

"""
All Models of TVB-O
====================

This script showcases a selection of neural models from the TVB-O (The Virtual Brain - Oscillations) library. TVB-O is an extension of The Virtual Brain project, focusing specifically on oscillatory dynamics in neural systems. This gallery aims to provide a visual and interactive overview of various models available in the library, demonstrating their behavior and characteristics.

Each model is simulated and its dynamics are plotted to give an insight into its temporal evolution. This script is particularly useful for neuroscientists, computational biologists, and anyone interested in neural modeling and simulation.

The script follows these steps:

1. Import necessary modules: `matplotlib.pyplot` for plotting, and relevant TVB-O modules like `lemsgenerator`, `rateml`, `bnm`, and `ontology`.

2. Define a list of models to be showcased. This includes models like "CoombesByrne", "DumontGutkin", "Epileptor2D", etc. Some models are commented out and can be included as per requirement.

3. Set up the plotting environment: Calculate the number of rows and columns needed for the subplots based on the number of models. Then, create a figure with subplots arranged accordingly.

4. Iterate over each model: For each model, perform the necessary pre-processing like exporting LEMS models and converting them to Python where applicable. Then, initialize and simulate each model using `bnm.ODESystemTVB`, and plot their dynamics.

5. Display the plots: Show the resulting figure with subplots representing the dynamics of each model.

This gallery serves as an educational tool for understanding the diverse range of neural models and their behaviors in a concise and visual manner. It provides an easy way to compare and contrast the dynamics of different models in the TVB-O library.

"""

import matplotlib.pyplot as plt
import numpy as np

from tvbo.export import templater
from tvbo.knowledge import ontology
from tvbo.knowledge.simulation import networkmodel

np.random.seed(1312)

models = ontology.get_models()
# models.pop("LarterBreakspear")

# Calculate the number of rows and columns for the subplots
n_models = len(models)
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
fig.suptitle("TVB-O Simulations", fontweight="bold")

noise = 1e-4

# Flatten axs array if there are multiple rows
if n_rows > 1:
    axs = axs.flatten()

for i, model in enumerate(sorted(models)):
    print(model)
    axs[i].set_title(model)
    model = templater.model2class(
        model,
        print_source=False,
        split_nonintegrated_variables=False,
    )

    sim = networkmodel.ODESystemTVB(model)
    sim.integrate(simulation_length=500, integrator="Dopri5Stochastic", nsig=noise)
    axs[i].plot(sim.xv[100:])

plt.show()
