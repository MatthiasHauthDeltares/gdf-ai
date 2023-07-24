import numpy as np
import matplotlib.pyplot as plt

from src.random_field_generator import Field2D


def plot_field(field: np.array, x_values, y_values):
    cmap = "Spectral_r"

    plt.imshow(field, aspect=0.25, cmap=cmap, vmin=2, vmax=22)
    plt.colorbar()

    # Choose 5 x-ticks and y-ticks evenly distributed over the data
    x_ticks = np.linspace(0, len(x_values), num=5, dtype=int)
    y_ticks = np.linspace(0, len(y_values), num=5, dtype=int)

    x_ticklabels = np.linspace(x_values[0], x_values[-1], num=5)
    y_ticklabels = - np.round(np.linspace(y_values[0], y_values[-1], num=5), 3)

    plt.xticks(x_ticks, x_ticklabels)
    plt.yticks(y_ticks, y_ticklabels)

    plt.show()


def plot_field_comparison(field_list: list[Field2D], x_values, y_values):
    fig, axs = plt.subplots(len(field_list))
    for ax_idx, field in enumerate(field_list):
        axs[ax_idx].imshow(field.field_data, aspect=0.25, cmap="Spectral_r", vmin=2, vmax=22)
        # set title on the left side of the plot:
        axs[ax_idx].set_title(field.name, y=1, pad=-14)

        #ticks
        x_ticks = np.linspace(0, len(x_values), num=5, dtype=int)
        y_ticks = np.linspace(0, len(y_values), num=5, dtype=int)

        x_ticklabels = np.linspace(x_values[0], x_values[-1], num=5)
        y_ticklabels = - np.round(np.linspace(y_values[0], y_values[-1], num=5), 3)

        axs[ax_idx].set_xticks(x_ticks, x_ticklabels)
        axs[ax_idx].set_yticks(y_ticks, y_ticklabels)

    fig.colorbar(axs[0].imshow(field_list[0].field_data, aspect=0.25, cmap="Spectral_r", vmin=2, vmax=22), ax=axs, location='right')

    plt.savefig("results/field_comparison.png")
    plt.show()
