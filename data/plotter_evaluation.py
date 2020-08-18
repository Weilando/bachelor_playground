import numpy as np
from matplotlib import colors
from torch import nn


def find_early_stop_indices(loss_hists):
    """ Find the early-stop indices in 'loss_hists', i.e. the smallest indices with minimum loss along the last axis.
    Usually the criterion is performed on the validation-loss.
    Suppose 'loss_hists' has shape (net_count, prune_count+1, iteration_count).
    The result has shape (net_count, prune_count+1). """
    return np.argmin(loss_hists, axis=2)


def find_early_stop_iterations(loss_hists, plot_step):
    """ Apply early-stopping criterion on 'loss_hists' to find corresponding iterations.
    Suppose 'loss_hists' has shape (net_count, prune_count[+1], data_length). """
    early_stop_indices = find_early_stop_indices(loss_hists)
    return scale_early_stop_indices_to_iterations(early_stop_indices, plot_step)


def find_acc_at_early_stop_indices(loss_hists, acc_hists):
    """ Apply early-stopping criterion on 'loss_hists' to find corresponding accuracies from 'acc_hists'.
    Suppose 'acc_hists' and 'loss_hists' have shape (net_count, prune_count[+1], data_length). """
    early_stop_indices = find_early_stop_indices(loss_hists)
    return get_values_at_stop_iteration(early_stop_indices, acc_hists)


def format_time(time):
    """ Format a given integer (UNIX time) into a string.
    Convert times shorter than a minute into seconds with four decimal places.
    Convert longer times into rounded minutes and seconds, separated by a colon. """
    if time >= 60:
        minutes, seconds = divmod(time, 60)
        return f"{round(minutes)}:{round(seconds):02d}min"
    return f"{time:.4f}sec"


def get_means_and_y_errors(arr):
    """ Calculate means and error bars for given array of arrays.
    The array is supposed to be of shape (net_count, prune_count+1, ...).
    Then this function squashes the net-dimension, so mean, max and min have shape (prune_count+1, ...). """
    arr_mean = np.mean(arr, axis=0)
    arr_neg_y_err = arr_mean - np.min(arr, axis=0)
    arr_pos_y_err = np.max(arr, axis=0) - arr_mean
    return arr_mean, arr_neg_y_err, arr_pos_y_err


def get_values_at_stop_iteration(stop_indices, hists):
    """ Find the actual values from 'hists' at given 'stop_indices' as calculated by find_early_stop_indices(...).
    Usually 'stop_indices' are found on the validation-loss and one needs accuracies at these times.
    Suppose 'stop_indices' has shape (net_count, prune_count+1) and 'hists' has shape
    (net_count, prune_count+1, iteration_count).
    The result has shape (net_count, prune_count+1, 1) and holds corresponding values from 'hists' in the last dim. """
    return np.take_along_axis(hists, np.expand_dims(stop_indices, axis=2), axis=2)


def scale_early_stop_indices_to_iterations(stop_indices, plot_step):
    """ Scale 'stop_indices', as calculated by find_early_stop_indices(...)), to match early-stopping iterations. """
    return (stop_indices + 1) * plot_step


def get_norm_for_sequential(sequential):
    """ Generates a Normalize-object with the minimum and maximum of all weights from all layers 'sequential'. """
    assert isinstance(sequential, nn.Sequential)
    weight_list = [lay.weight.data for lay in sequential if (isinstance(lay, nn.Linear) or isinstance(lay, nn.Conv2d))]
    min_weight = min(weights.min().item() for weights in weight_list)
    max_weight = max(weights.max().item() for weights in weight_list)
    return colors.TwoSlopeNorm(vcenter=0.0, vmin=min_weight, vmax=max_weight)
