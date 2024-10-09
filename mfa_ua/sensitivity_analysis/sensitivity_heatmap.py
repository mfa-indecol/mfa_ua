import sympy as sy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_sensitivity_heatmap(
    sensitvities: dict[str, dict[sy.Symbol, dict[str, float]]],
    figsize: tuple = None,
) -> tuple[plt.figure, plt.Axes]:
    """
    Create a heatmap of the relative sensitivities of the model to the parameters.

    Args:
        sensitivities: the sensitivities - ordered by the flow, then by the parameter.
        figsize (tuple, optional): the size of the figure. Defaults to (16, 9).

    Returns:
        tuple[plt.figure, plt.Axes]: the figure and the axes of the plot.
    """
    if figsize is None:
        figsize = (6, 4.5)

    flow_names = list(sensitvities.keys())
    parameters = list(sensitvities[flow_names[0]].keys())
    parameter_names = [p.__str__() for p in parameters]
    relative_sensitivity_matrix = np.zeros((len(flow_names), len(parameters)))
    for i, flow in enumerate(flow_names):
        for j, parameter in enumerate(parameters):
            relative_sensitivity_matrix[i, j] = sensitvities[flow][parameter][
                "relative_sensitivity"
            ]

    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(relative_sensitivity_matrix, cmap="seismic", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameter_names, rotation=75)
    ax.set_yticks(np.arange(len(flow_names)))
    ax.set_yticklabels(flow_names)
    plt.colorbar()
    plt.title("Relative Sensitivities")
    norm = Normalize(vmin=-1, vmax=1)
    for i, _ in enumerate(flow_names):
        for j, _ in enumerate(parameters):
            background_color = plt.cm.seismic(norm(relative_sensitivity_matrix[i, j]))
            if sum(background_color[:3]) / 3 > 0.5:
                text_color = "black"
                bbox_color = "white"
            else:
                text_color = "white"
                bbox_color = "black"
            ax.text(
                j,
                i,
                f"{relative_sensitivity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                bbox=dict(facecolor=bbox_color, alpha=0.5),
            )
    plt.show()

    return fig, ax
