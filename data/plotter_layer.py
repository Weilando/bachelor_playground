import warnings

import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.nn.utils import prune

from data import plotter_evaluation


def get_cmap():
    """ Generate a diverging colormap which shows NANs in black. """
    cmap = plt.get_cmap('bwr')
    cmap.set_bad(color='black')
    return cmap


def plot_kernels(conv_2d, num_cols=8):
    """ Plot the weights of all kernels from 'conv_2d' as rectangles on a new figure and map values to colors.
    Create one normalized image per channel and kernel to avoid clipping and to center all values at zero.
    Show a colorbar with integrated histogram on the right. """
    assert isinstance(conv_2d, nn.Conv2d), f"'conv_2d' has invalid type {type(conv_2d)}"

    weights = conv_2d.weight.data.clone().numpy()  # shape is [kernels, channels, height, width]
    if prune.is_pruned(conv_2d):  # mark masked weights with NAN to highlight them later
        weights[np.where(conv_2d.weight_mask.numpy() == 0)] = np.nan
    if (weights.shape[0] * weights.shape[1]) > 512:  # restrict number of images to 512, do not plot partial kernels
        last_kernel = ceil(512 / weights.shape[1])
        weights = weights[:last_kernel]
        warnings.warn(f"Too many kernels to plot, only plot the first {last_kernel} kernels.")

    weight_norm = plotter_evaluation.get_norm_for_sequential(nn.Sequential(conv_2d))
    num_cols, num_rows = plotter_evaluation.get_row_and_col_num(weights.shape, num_cols)

    fig = plt.figure(figsize=(num_cols + 1, num_rows), constrained_layout=False)
    gs = fig.add_gridspec(num_rows, num_cols + 1, wspace=0.1, hspace=0.4)  # extra column for colorbar and histogram
    for kernel_counter, kernel in enumerate(weights[:]):
        for channel_counter, channel in enumerate(kernel[:]):
            ax_counter = kernel_counter * kernel.shape[0] + channel_counter
            ax = fig.add_subplot(gs[ax_counter // num_cols, ax_counter % num_cols])
            ax.imshow(channel, cmap=get_cmap(), norm=weight_norm)
            ax.set_title(f"K{kernel_counter + 1}.{channel_counter + 1}").set_position([.5, 0.95])
            ax.axis('off')

    # append two columns for colorbar and histogram
    tmp_ax = fig.add_subplot(gs[:, -1])
    tmp_ax.axis('off')
    div = make_axes_locatable(tmp_ax)
    cax = div.append_axes("right", size="25%", pad=0)
    hax = div.append_axes("right", size="75%", pad=0)

    fig.colorbar(fig.axes[0].images[0], cax=cax, aspect=10)
    cax.yaxis.set_ticks_position('left')

    hax.hist(weights.flatten(), orientation='horizontal', density=False, bins=30, color='gray')
    hax.yaxis.set_visible(False)
    for loc in ['top', 'right', 'left']:
        hax.spines[loc].set_visible(False)

    return fig


def plot_conv(sequential, num_cols=8):
    """ Plot the kernel-weights of each Conv2D layer from 'sequential' as single figure.
    Create one normalized image per channel and kernel to avoid clipping and to center all values at zero. """
    assert isinstance(sequential, nn.Sequential), f"'sequential' has invalid type {type(sequential)}"
    fig_list = []

    for layer in sequential:
        if isinstance(layer, nn.Conv2d):
            fig_list.append(plot_kernels(layer, num_cols))
    return fig_list


def plot_fc(sequential):
    """ Plot the weights of all linear layers from 'sequential' as rectangles on a figure and maps values to colors.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(sequential, nn.Sequential), f"'sequential' has invalid type {type(sequential)}"

    linear_layers = [layer for layer in sequential if isinstance(layer, nn.Linear)]
    weight_norm = plotter_evaluation.get_norm_for_sequential(sequential)

    fig, ax = plt.subplots(len(linear_layers), 1, figsize=(10, 5))
    for plot_counter, layer in enumerate(linear_layers):
        weights = layer.weight.data.clone().numpy()
        if prune.is_pruned(layer):  # mark masked weights with NAN to highlight them later
            pruning_mask = layer.weight_mask.numpy()
            weights[np.where(pruning_mask == 0)] = np.nan
        ax[plot_counter].imshow(weights, norm=weight_norm, cmap=get_cmap())
        ax[plot_counter].label_outer()

    fig.colorbar(ax[0].images[0], ax=ax, fraction=0.1)
    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    return fig
