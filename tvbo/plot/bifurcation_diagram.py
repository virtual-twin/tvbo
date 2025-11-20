import matplotlib.pyplot as plt
from sympy import parse_expr, pycode, symbols
import pandas as pd
from tvbo.knowledge.simulation import equations


def compute_voi(df, VOI, prefix=""):
    exp = parse_expr(VOI, equations._clash1)
    variables = list(exp.free_symbols)
    exp = exp.subs({v: symbols(f"{prefix}{v}") for v in variables})
    return df.eval(pycode(exp, fully_qualified_modules=False))


def plot_equilibrium_branch(df, ax, ICS=None, VOI=None, **kwargs):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stable_color = color_cycle[0]  # First color for stable
    unstable_color = color_cycle[0]  # Second color for unstable
    points = df.specialpoint.dropna().unique()
    # Julia BifurcationKit color mapping replicated (subset)
    julia_colorbif = {
        'fold': 'black',
        'hopf': 'red',
        'bp': 'blue',
        'nd': 'magenta',
        'none': 'yellow',
        'ns': 'orange',
        'pd': 'green',
        'bt': 'red',
        'cusp': 'sienna',
        'gh': 'brown',
        'zh': 'burlywood',
        'hh': 'green',
        'R': 'chartreuse',
        'R1': 'chartreuse',
        'R2': 'chartreuse',
        'R3': 'chartreuse',
        'R4': 'blue',
        'foldFlip': 'navy',
        'ch': 'darkred',
        'foldNS': 'cyan',
        'flipNS': 'goldenrod',
        'pdNS': 'maroon',
        'nsns': 'purple',
        'gpd': 'salmon',
        'user': 'goldenrod'
    }
    colormap = julia_colorbif

    if VOI is None:
        VOI = df.columns[0]
    # Create a new column for segment labeling based on stability
    df["segment"] = (df.stable != df.stable.shift()).cumsum()

    # Iterate over the unique segments and plot each with its corresponding style and label
    for segment_id, segment_data in df.groupby("segment"):
        # Determine the stability of the segment
        is_stable = segment_data.iloc[0].stable
        label = "Stable" if is_stable else "Unstable"

        # Add label only if it hasn't been added before
        current_labels = ax.get_legend_handles_labels()[1]
        ax.plot(
            segment_data["param"],
            compute_voi(segment_data, VOI),
            "-" if is_stable else "--",
            zorder=1,
            label=label if label not in current_labels else None,
            color=stable_color if is_stable else unstable_color,
        )

    # Scatter special points, excluding 'endpoint'
    for i, r in df[df["specialpoint"].notna()].iterrows():
        if r.specialpoint != "endpoint":
            # Add label only if it hasn't been added before
            current_labels = ax.get_legend_handles_labels()[1]
            color = colormap.get(r.specialpoint, 'black')
            ax.scatter(
                r.param,
                compute_voi(df, VOI).loc[i],
                zorder=2,
                label=r.specialpoint if r.specialpoint not in current_labels else None,
                color=color,
            )




def plot_periodic_orbit(df_po, VOI, ax, color_cycle_index=1, **kwargs):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    periodic_color = color_cycle[color_cycle_index]  # Third color for periodic orbit

    if VOI is None:
        VOI = df_po.columns[0].replace("min_", "").replace("max_", "")

    # Add periodic orbit label only once
    current_labels = ax.get_legend_handles_labels()[1]
    label = "Periodic orbit" if "Periodic orbit" not in current_labels else None

    ax.plot(
        df_po["param"],
        compute_voi(df_po, VOI, prefix="min_"),
        # df_po[f"min_{VOI}"],
        zorder=1,
        label=label,
        color=periodic_color,
    )
    ax.plot(
        df_po["param"],
        compute_voi(df_po, VOI, prefix="max_"),
        # df_po[f"max_{VOI}"],
        zorder=1,
        color=periodic_color,
    )
