#
# Module: functions.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import matplotlib.pyplot as plt
import numpy as np
import owlready2 as owl
from sympy import lambdify, latex, symbols

from tvbo.knowledge import ontology, config  # Assuming this module handles your equations
from tvbo.knowledge.simulation import equations


def plot_coupling_function(CF, ax=None):
    # Extract acronym and set parameter configuration
    acr = CF.acronym  # TODO: acr not used, remove?
    param_config = config.get_coupling_parameters(CF)

    # Define symbolic variables
    x_i, x_j, y1_j, y2_j = symbols("x_i x_j y1_j y2_j")  # TODO: variables not used, remove?

    # Determine the range of values based on the CF name
    if "Kuramoto" in CF.name:
        lo, hi = -np.pi, np.pi
    elif CF.name == "Sigmoidal":
        lo, hi = -1000, 1000
    else:
        lo, hi = -10, 10

    # Generate the values for plotting
    x_vals = y_vals = np.linspace(lo, hi, 100)

    # Convert the CF equation and substitute parameters
    eq = equations.sympify_value(CF)
    latex_eq = latex(eq, mul_symbol="dot")
    eq = eq.subs(param_config)

    if isinstance(ax, type(None)):
        fig = plt.figure()
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection="3d" if len(list(eq.free_symbols)) > 1 else None,
        )
        return_fig = True
    else:
        return_fig = False

    # Determine if a 2D or 3D plot is required
    if len(list(eq.free_symbols)) > 1:
        # 3D Plot
        X, Y = np.meshgrid(x_vals, y_vals)
        eq_sympy = lambdify(
            (list(eq.free_symbols)[0], list(eq.free_symbols)[1]), eq, "numpy"
        )
        Z = eq_sympy(X, Y)
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("$x_j$")
        ax.set_ylabel("$x_i$")
        ax.set_zlabel("$c_{glob0}$")

        # Additional 3D plot configurations
        ax.grid(False)
        ax.set_facecolor("white")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(azim=45)

        ax.text(
            ax.get_xlim()[1],
            ax.get_ylim()[0],
            ax.get_zlim()[1]
            + 0.5
            * (
                ax.get_zlim()[1] - ax.get_zlim()[0]
            ),  # if CF.name == "KuramotoCouplingFunction" else ax.get_zlim()[1],
            f"${latex_eq}$",
            # fontsize=12,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor=(1, 1, 1, 0.8), edgecolor="black"
            ),
        )

    else:
        # 2D Plot
        eq_sympy = lambdify(list(eq.free_symbols)[0], eq, "numpy")
        Z = eq_sympy(x_vals)
        ax.plot(x_vals, Z)
        ax.set_xlabel("$x_j$")
        ax.set_ylabel("$c_{glob0}$")

        # Common configurations for both 2D and 3D plots
        ax.text(
            ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"${latex_eq}$",
            # fontsize=12,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor=(1, 1, 1, 0.8), edgecolor="black"
            ),
        )
    ax.set_title(CF.name)

    if return_fig:
        return fig


def plot_temporal_equation(
    EQ, t_ms, title="Stimulation pulse sequence", plot_kwargs=None, ax=None, **kwargs
):
    if plot_kwargs is None:
        plot_kwargs = {}
    if isinstance(EQ, owl.ThingClass):
        eq = equations.sympify_value(EQ)
        parameters = ontology.get_default_values(EQ)
        for k, v in kwargs.items():
            if k in parameters:
                parameters[k] = v
        eq = eq.subs(parameters)
    else:
        eq = EQ
        for k, v in kwargs.items():
            eq = eq.subs({k: v})
    # Adjust the range for milliseconds (0 to 200 ms)
    expr_values_ms = [eq.subs({"t": v}) for v in t_ms]

    if isinstance(ax, type(None)):
        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(title)
        return_fig = True
    else:
        return_fig = False
    ax.plot(t_ms, expr_values_ms, label=latex(eq, mul_symbol="dot"), **plot_kwargs)

    ax.set_xlabel("t[ms]")
    ax.set_ylabel("Stimulus")
    if return_fig:
        plt.close()
        return fig
