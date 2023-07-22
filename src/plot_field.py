import numpy as np
import matplotlib.pyplot as plt


def plot_field(field: np.array, x_values, y_values):
    plt.imshow(field, aspect=0.25, cmap="Spectral", vmin=2, vmax=22)
    plt.colorbar()

    # Choose 5 x-ticks and y-ticks evenly distributed over the data
    x_ticks = np.linspace(0, len(x_values), num=5, dtype=int)
    y_ticks = np.linspace(0, len(y_values), num=5, dtype=int)

    x_ticklabels = np.linspace(x_values[0], x_values[-1], num=5)
    y_ticklabels = - np.round(np.linspace(y_values[0], y_values[-1], num=5), 3)

    plt.xticks(x_ticks, x_ticklabels)
    plt.yticks(y_ticks, y_ticklabels)

    plt.show()

def plot_field_comparison(field_list, x_values, y_values):

    fig, axs = plt.subplots(2)
    for ax_idx, field in enumerate(field_list):
        axs[ax_idx].imshow(field, aspect=0.25, cmap="Spectral", vmin=2, vmax=22)


    # plt.colorbar()
    plt.show()
