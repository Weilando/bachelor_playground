import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch import nn

from data import plotter_evaluation


def plot_linear_layer_on_ax(ax, layer, norm):
    """ Plots the weights of linear 'layer' in a rectangle and translates values into colors.
    'norm' is used to avoid clipping and to center all values at zero. """
    assert isinstance(layer, nn.Linear)
    assert isinstance(norm, TwoSlopeNorm)

    ax.imshow(layer.weight.data, norm=norm, cmap='seismic')
    ax.label_outer()


def plot_fc(sequential):
    """ Plot all linear layers in 'sequential'. """
    assert isinstance(sequential, nn.Sequential)

    linear_layer_count = sum([isinstance(layer, nn.Linear) for layer in sequential])
    fig, ax = plt.subplots(linear_layer_count, 1, figsize=(10, 5))
    plot_counter = 0

    norm = plotter_evaluation.get_norm_for_sequential(sequential)

    for layer in sequential:
        if isinstance(layer, nn.Linear):
            plot_linear_layer_on_ax(ax[plot_counter], layer, norm)
            plot_counter += 1

    fig.colorbar(ax[0].images[0], ax=ax, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    plt.show()
