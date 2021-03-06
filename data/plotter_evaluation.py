import numpy as np
from math import ceil
from matplotlib import colors


def find_early_stop_indices(loss_hists):
    """ Find the early-stop indices in 'loss_hists', i.e. the smallest indices with minimum loss along the last axis.
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


def get_norm_for_tensor(tensor):
    """ Generate a zero-centered TwoSlopeNorm (or Normalize-object) to normalize np array 'tensor'. """
    assert isinstance(tensor, np.ndarray), f"'tensor' has invalid type {type(tensor)}."
    min_val = np.nanmin(tensor)
    max_val = np.nanmax(tensor)
    if min_val >= 0.0 or max_val <= 0.0:
        return colors.Normalize(vmin=min_val, vmax=max_val)
    return colors.TwoSlopeNorm(vcenter=0.0, vmin=min_val, vmax=max_val)


def get_row_and_col_num(weight_shape, num_cols):
    """ Calculate the correct row and column numbers from convolutional 'weight_shape' and preferred number of columns.
    Suppose 'weight_shape' is a tuple with entries (kernels, channels, height, width). """
    num_cols = min(num_cols, weight_shape[0] * weight_shape[1])
    num_rows = ceil((weight_shape[0] * weight_shape[1]) / num_cols)
    return num_cols, num_rows


def get_values_at_stop_iteration(stop_indices, hists):
    """ Find the actual values from 'hists' at given 'stop_indices' as calculated by find_early_stop_indices(...).
    Suppose 'stop_indices' has shape (net_count, prune_count+1) and 'hists' (net_count, prune_count+1, iteration_count).
    The result has shape (net_count, prune_count+1, 1) and holds corresponding values from 'hists' in the last dim. """
    return np.take_along_axis(hists, np.expand_dims(stop_indices, axis=2), axis=2)


def scale_early_stop_indices_to_iterations(stop_indices, plot_step):
    """ Scale 'stop_indices', as calculated by find_early_stop_indices(...)), to match early-stopping iterations. """
    return (stop_indices + 1) * plot_step
