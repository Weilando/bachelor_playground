import matplotlib.pyplot as plt
from torch import nn

from data import plotter_evaluation


def plot_fc(sequential):
    """ Plots the weights of all linear layers from 'sequential' as rectangles and translates values into colors.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(sequential, nn.Sequential)

    linear_layers = [layer for layer in sequential if isinstance(layer, nn.Linear)]
    weight_norm = plotter_evaluation.get_norm_for_sequential(sequential)

    fig, ax = plt.subplots(len(linear_layers), 1, figsize=(10, 5))
    for plot_counter, layer in enumerate(linear_layers):
        ax[plot_counter].imshow(layer.weight.data, norm=weight_norm, cmap='seismic')
        ax[plot_counter].label_outer()

    fig.colorbar(ax[0].images[0], ax=ax, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    plt.show()


def plot_kernels(conv_2d, num_cols=8):
    """ Plots the weights of all kernels from 'conv_2d' as rectangles and translates values into colors.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(conv_2d, nn.Conv2d)

    weights = conv_2d.weight.data.permute(0, 2, 3, 1).numpy()  # adapt dimensions to [image_num, height, width, color]
    weight_norm = plotter_evaluation.get_norm_for_sequential(nn.Sequential(conv_2d))

    num_kernels = weights.shape[0]
    num_cols = min(num_cols, num_kernels)
    num_rows = 1 + num_kernels // num_cols

    fig = plt.figure(figsize=(num_cols, num_rows))
    for plot_counter, kernel in enumerate(weights[:], 1):
        ax = fig.add_subplot(num_rows, num_cols, plot_counter)
        ax.imshow(weight_norm(kernel), cmap='seismic', vmin=weight_norm.vmin, vmax=weight_norm.vmax)
        ax.axis('off')

    fig.colorbar(fig.axes[0].images[0], ax=fig.axes, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.75)
    plt.show()
