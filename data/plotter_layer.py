import warnings

import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from torch import nn
from torch.nn.utils import prune

from data import plotter_evaluation
from data.plotter_evaluation import get_row_and_col_num


def get_cmap():
    """ Generates a diverging colormap which shows NANs in black. """
    cmap = plt.get_cmap('seismic')
    cmap.set_bad(color='black')
    return cmap


def plot_kernels_rgb_on_fig(fig, weights, weight_norm, num_cols, num_rows, cmap):
    """ Plots each kernel from 'weights' as normalized RGB-images on 'fig'. """
    for plot_counter, kernel in enumerate(weights[:], 1):
        ax = fig.add_subplot(num_rows, num_cols, plot_counter)
        ax.imshow(weight_norm(kernel), cmap=cmap, vmin=weight_norm.vmin, vmax=weight_norm.vmax)
        ax.set_title(f"K{plot_counter}").set_position([.5, 0.95])
        ax.axis('off')


def plot_kernels_single_channel_on_fig(fig, weights, weight_norm, num_cols, num_rows, cmap):
    """ Plots each kernel from 'weights' as normalized image per channel on 'fig'. """
    for kernel_counter, kernel in enumerate(weights[:]):
        kernel = kernel.transpose(2, 0, 1)  # adapt dimensions to [color, height, width]
        for channel_counter, channel in enumerate(kernel[:], 1):
            ax = fig.add_subplot(num_rows, num_cols, kernel_counter * kernel.shape[0] + channel_counter)
            ax.imshow(weight_norm(channel), cmap=cmap, vmin=weight_norm.vmin, vmax=weight_norm.vmax)
            ax.set_title(f"K{kernel_counter + 1}.{channel_counter}").set_position([.5, 0.95])
            ax.axis('off')


def plot_kernels(conv_2d, num_cols=8):
    """ Plots the weights of all kernels from 'conv_2d' as rectangles and translates values into colors.
    Plot each kernel as RGB image if the layer has three input channels or one plot per channel and kernel otherwise.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(conv_2d, nn.Conv2d)

    weights = conv_2d.weight.data.clone().permute(0, 2, 3, 1).numpy()  # adapt dims to [kernel, height, width, color]
    if prune.is_pruned(conv_2d):  # mark masked weights with NAN to highlight them later
        pruning_mask = conv_2d.weight_mask.permute(0, 2, 3, 1).numpy()
        weights[np.where(pruning_mask == 0)] = np.nan
    if (weights.shape[0] * weights.shape[3]) > 512:  # restrict number of images to 512, do not plot partial kernels
        last_kernel = ceil(512 / weights.shape[3])
        weights = weights[:last_kernel]
        warnings.warn(f"Too many kernels to plot, only plot the first {last_kernel} kernels.")

    weight_norm = plotter_evaluation.get_norm_for_sequential(nn.Sequential(conv_2d))
    num_cols, num_rows = get_row_and_col_num(weights.shape, num_cols)

    fig = plt.figure(figsize=(num_cols, num_rows))
    if weights.shape[3] == 3:
        plot_kernels_rgb_on_fig(fig, weights, weight_norm, num_cols, num_rows, get_cmap())
    else:
        plot_kernels_single_channel_on_fig(fig, weights, weight_norm, num_cols, num_rows, get_cmap())

    fig.colorbar(fig.axes[0].images[0], ax=fig.axes, fraction=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.75)
    plt.show()


def plot_conv(sequential, num_cols=8):
    """ Plots the weights of all Conv2D layers from 'sequential' as rows of square representing the kernels.
    Plot them in RGB if the layer has three input channels, or each channel alone otherwise.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(sequential, nn.Sequential)

    for layer in sequential:
        if isinstance(layer, nn.Conv2d):
            plot_kernels(layer, num_cols)


def plot_fc(sequential):
    """ Plots the weights of all linear layers from 'sequential' as rectangles and translates values into colors.
    Use normalization to avoid clipping and to center all values at zero. """
    assert isinstance(sequential, nn.Sequential)

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
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    plt.show()
