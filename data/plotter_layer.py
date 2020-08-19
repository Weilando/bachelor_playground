import matplotlib.pyplot as plt
import numpy as np
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

    linear_layers = [layer for layer in sequential if isinstance(layer, nn.Linear)]
    fig, ax = plt.subplots(len(linear_layers), 1, figsize=(10, 5))

    norm = plotter_evaluation.get_norm_for_sequential(sequential)

    for plot_counter, layer in enumerate(linear_layers):
        plot_linear_layer_on_ax(ax[plot_counter], layer, norm)

    fig.colorbar(ax[0].images[0], ax=ax, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    plt.show()


def plot_kernel_on_ax(ax, kernel, norm):
    """ Plots the weights of convolutional 'kernel' in a rectangle and translates values into colors for three channels.
    'norm' is used to avoid clipping and to center all values at zero. """
    assert isinstance(kernel, np.ndarray)

    if kernel.shape[2] == 3:
        ax.imshow(norm(kernel), cmap='seismic', vmin=norm.vmin, vmax=norm.vmax)
        ax.label_outer()


def plot_kernels(conv_2d, num_cols=8):
    """ Plots all kernels in 'conv_2d'. """
    assert isinstance(conv_2d, nn.Conv2d)

    weights = conv_2d.weight.data.permute(0, 2, 3, 1).numpy()  # adapt dimensions to [image_num, height, width, color]
    weight_norm = plotter_evaluation.get_norm_for_sequential(nn.Sequential(conv_2d))

    num_kernels = weights.shape[0]
    num_cols = min(num_cols, num_kernels)
    num_rows = num_kernels // num_cols

    fig = plt.figure(figsize=(num_cols, num_rows))

    if num_kernels == 1:
        ax = fig.add_subplot(num_rows, num_cols, 1)
        plot_kernel_on_ax(ax, weights[0], weight_norm)
    else:
        for plot_counter, kernel in enumerate(weights[:], 1):
            ax = fig.add_subplot(num_rows, num_cols, plot_counter)
            plot_kernel_on_ax(ax, kernel, weight_norm)

    fig.colorbar(fig.axes[0].images[0], ax=fig.axes, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.75)
    plt.show()
