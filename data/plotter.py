from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class PlotType(str, Enum):
    """ Defines available types of plots. """
    TRAIN_LOSS = "Training-Loss"
    VAL_LOSS = "Validation-Loss"
    VAL_ACC = "Validation-Accuracy"
    TEST_ACC = "Test-Accuracy"
    EARLY_STOP_ITER = "Early-Stopping Iteration"


# evaluation
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


# generators
def gen_iteration_space(arr, plot_step):
    """ Generate a linear space from 'plot_step' with the same length as 'arr' and step size 'plot_step'. """
    len_arr = len(arr)
    return np.linspace(start=plot_step, stop=len_arr * plot_step, num=len_arr)


def gen_labels_on_ax(ax, plot_type: PlotType, iteration=True):
    """ Generates labels for the x- and y-axis on given ax.
    'iteration' defines if the x-axis shows iterations or sparsity. """
    ax.set_ylabel(f"{plot_type.value}")
    ax.set_xlabel(f"{'Iteration' if iteration else 'Sparsity'}")


def gen_new_single_ax():
    """ Generates a new axes from new figure. """
    fig = plt.figure(None, figsize=(7, 6))
    return fig.subplots(1, 1, sharex=False)


def gen_title_on_ax(ax, net_count, plot_type: PlotType, early_stop=False):
    """ Generates plot-title for 'net_count' nets on given ax.
    'early_stop' defines if early-stopping should be mentioned. """
    if not early_stop:
        ax.set_title(f"Average {plot_type.value} for {net_count} pruned Networks")
    else:
        ax.set_title(f"Average {plot_type.value} for {net_count} pruned Networks at early-stop")


def setup_grids_on_ax(ax, force_zero=False):
    """ Setup grids on given ax.
    'force_zero' sets the minimum y-value to zero, e.g. for loss plots. """
    ax.grid()
    if force_zero:
        ax.set_ylim(bottom=0)


def setup_labeling_on_ax(ax, net_count, plot_type: PlotType, iteration=True):
    """ Setup complete labeling on ax, i.e. generate title, labels and legend. """
    gen_title_on_ax(ax, net_count, plot_type)
    gen_labels_on_ax(ax, plot_type, iteration)
    ax.legend()


# subplots
def plot_average_at_early_stop_on_ax(ax, hists, sparsity_hist):
    """ Plot means and error-bars for given early-stopping iterations or accuracies on ax.
    Suppose 'hists' has shape (net_count, prune_count+1, 1) for accuracies or (net_count, prune_count+1) for iterations.
    Suppose 'sparsity_hist' has shape (prune_count+1).
    'hists' is a solid line with error bars. """
    # calculate means
    mean, neg_y_err, pos_y_err = get_means_and_y_errors(hists)

    # for accuracies each mean and y_err has shape (prune_count+1, 1), so squeeze them to get shape (prune_count+1)
    mean = np.squeeze(mean)
    neg_y_err = np.squeeze(neg_y_err)
    pos_y_err = np.squeeze(pos_y_err)

    # plot
    ax.errorbar(x=sparsity_hist, y=mean, elinewidth=1, yerr=[neg_y_err, pos_y_err], marker='x', ls='-', color='C0')


def plot_random_average_at_early_stop_on_ax(ax, random_hists, sparsity_hist):
    """ Plot means and error-bars for given random early-stopping iterations or accuracies on ax.
    Suppose 'hists' has shape (net_count, prune_count, 1) for accuracies or (net_count, prune_count) for iterations.
    Suppose 'sparsity_hist' has shape (prune_count+1).
    'hists' is a solid line with error bars. """
    # calculate means
    mean, neg_y_err, pos_y_err = get_means_and_y_errors(random_hists)

    # for accuracies each mean and y_err has shape (prune_count+1, 1), so squeeze them to get shape (prune_count+1)
    mean = np.squeeze(mean)
    neg_y_err = np.squeeze(neg_y_err)
    pos_y_err = np.squeeze(pos_y_err)

    # plot
    ax.errorbar(x=sparsity_hist[1:], y=mean, elinewidth=1, yerr=[neg_y_err, pos_y_err], marker='x', ls=':', color='C0')


def plot_averages_on_ax(ax, hists, sparsity_hist, plot_step):
    """ Plot means and error-bars for given hists on ax.
    Suppose hists has shape (net_count, prune_count+1) and 'sparsity_hist' has shape (prune_count+1).
    Plots dashed baseline (unpruned) and a solid line for each pruning step. """
    _, prune_count, _ = hists.shape
    prune_count -= 1  # baseline at index 0, thus first pruned round at index 1

    hists_mean, hists_neg_y_err, hists_pos_y_err = get_means_and_y_errors(hists)
    xs = gen_iteration_space(hists_mean[0], plot_step)

    plot_baseline_mean_on_ax(ax, xs, hists_mean[0], hists_neg_y_err[0], hists_pos_y_err[0])
    plot_pruned_means_on_ax(ax, xs, hists_mean[1:], hists_neg_y_err[1:], hists_pos_y_err[1:], sparsity_hist[1:],
                            prune_count)


def plot_random_averages_on_ax(ax, random_hists, sparsity_hist, plot_step):
    """ Plot means and error-bars for given random hists on ax.
    Suppose hists has shape (net_count, prune_count) and 'sparsity_hist' has shape (prune_count+1).
    Plots dashed baseline (unpruned) and a solid line for each pruning step. """
    _, prune_count, _ = random_hists.shape

    hists_mean, hists_neg_y_err, hists_pos_y_err = get_means_and_y_errors(random_hists)
    xs = gen_iteration_space(hists_mean[0], plot_step)

    plot_pruned_means_on_ax(ax, xs, hists_mean, hists_neg_y_err, hists_pos_y_err, sparsity_hist, prune_count, ':')


def plot_baseline_mean_on_ax(ax, xs, ys, y_err_neg, y_err_pos):
    """ Plots the baseline as dashed line wit error bars on given ax. """
    ax.errorbar(x=xs, y=ys, yerr=[y_err_neg, y_err_pos], elinewidth=1.2, ls='--', color="C0", label="Sparsity 1.0000")


def plot_pruned_means_on_ax(ax, xs, ys, y_err_neg, y_err_pos, sparsity_hist, prune_count, ls='-'):
    """ Plots means per pruning level as line with error bars on given ax.
    Labels contain the sparsity at given level of pruning.
    'prune_min' and 'prune_max' specify the prune-levels to plot.
    'ls' specifies the style for all plotted lines (e.g. '-'=solid and ':'=dotted).
    Colors start with color-spec C1. """
    for p in range(prune_count):
        ax.errorbar(x=xs, y=ys[p], yerr=[y_err_neg[p], y_err_pos[p]], color=f"C{p + 1}", elinewidth=1.2, ls=ls,
                    label=f"Sparsity {sparsity_hist[p]:.4f}")


# plots
def plot_acc_at_early_stop(loss_hists, acc_hists, sparsity_hist, plot_type: PlotType, random_loss_hists=None,
                           random_acc_hists=None):
    """ Plot means and error bars for the given accuracies at the time an early stopping criterion would end training.
    Use 'loss_hists' to find accuracies from 'acc_hists', analog for random histories, if given.
    Suppose 'acc_hists' and 'loss_hists' have shape (net_count, prune_count+1, data_length), 'random_acc_hists' and
    'random_loss_hists' have shape (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1)
    with prune_count > 1.
    The x-axis shows the sparsity at each time.
    Plot accuracies as solid line with error bars and random accuracies as dotted line with error bars. """
    assert (sparsity_hist.shape[0] > 1) and (sparsity_hist.shape[0] == loss_hists.shape[1]), \
        f"'prune_count' (dimension of 'sparsity_hist') needs to be greater than one, but is {sparsity_hist.shape}."

    # setup and plot
    ax = gen_new_single_ax()
    early_stop_acc = find_acc_at_early_stop_indices(loss_hists, acc_hists)
    plot_average_at_early_stop_on_ax(ax, early_stop_acc, sparsity_hist)
    if random_loss_hists is not None and random_acc_hists is not None:
        random_early_stop_acc = find_acc_at_early_stop_indices(random_loss_hists, random_acc_hists)
        plot_random_average_at_early_stop_on_ax(ax, random_early_stop_acc, sparsity_hist)
    ax.set_xticks(sparsity_hist)
    ax.invert_xaxis()  # also inverts plot!
    setup_grids_on_ax(ax)  # for correct scaling the grids need to be set after plotting

    # labeling
    net_count, _, _ = acc_hists.shape
    gen_title_on_ax(ax, net_count, plot_type, early_stop=True)
    gen_labels_on_ax(ax, plot_type, iteration=False)


def plot_average_hists(hists, sparsity_hist, plot_step, plot_type: PlotType, random_hists=None):
    """ Plot means and error bars for the given histories in 'hists' and 'random_hists' (if given).
    Suppose 'hists' has shape (net_count, prune_count+1, data_length), 'random_hists' has shape
    (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1).
    The x-axis is labeled with iterations, which are reconstructed from plot_step.
    The baseline (i.e. the lowest sparsity) is a dashed line, all further pruning-levels from 'hists' are solid lines,
    all levels of pruning from 'random_hists' are dotted lines. """
    # setup and plot
    ax = gen_new_single_ax()
    plot_averages_on_ax(ax, hists, sparsity_hist, plot_step)
    if random_hists is not None:
        plot_random_averages_on_ax(ax, random_hists, sparsity_hist[1:], plot_step)
    setup_grids_on_ax(ax)  # for correct scaling the grids need to be set after plotting

    # labeling
    net_count, _, _ = hists.shape
    setup_labeling_on_ax(ax, net_count, plot_type)


def plot_early_stop_iterations(loss_hists, sparsity_hist, plot_step, random_loss_hists=None):
    """ Plot means and error bars for early-stopping iterations based on 'loss_hists' and 'random_loss_hists', if given.
    Suppose 'loss_hists' has shape (net_count, prune_count+1, data_length), 'loss_hists' has shape
    (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1) with prune_count > 1.
    The x-axis shows the sparsity at each time.
    Plot iterations as solid line with error bars and random iterations as dotted line with error bars. """
    assert (sparsity_hist.shape[0] > 1) and (sparsity_hist.shape[0] == loss_hists.shape[1]), \
        f"'prune_count' (dimension of 'sparsity_hist') needs to be greater than one, but is {sparsity_hist.shape}."
    # setup and plot
    ax = gen_new_single_ax()
    early_stop_iterations = find_early_stop_iterations(loss_hists, plot_step)
    plot_average_at_early_stop_on_ax(ax, early_stop_iterations, sparsity_hist)
    if random_loss_hists is not None:
        random_early_stop_iterations = find_early_stop_iterations(random_loss_hists, plot_step)
        plot_random_average_at_early_stop_on_ax(ax, random_early_stop_iterations, sparsity_hist)
    ax.set_xticks(sparsity_hist)
    ax.invert_xaxis()  # also inverts plot!
    setup_grids_on_ax(ax)  # for correct scaling the grids need to be set after plotting

    # labeling
    net_count, _, _ = loss_hists.shape
    gen_title_on_ax(ax, net_count, PlotType.EARLY_STOP_ITER, early_stop=True)
    gen_labels_on_ax(ax, PlotType.EARLY_STOP_ITER, iteration=False)


def plot_two_average_hists(hists_left, hists_right, sparsity_hist, plot_step, type_left: PlotType, type_right: PlotType,
                           force_zero_left=False, force_zero_right=False, random_hists_left=None,
                           random_hists_right=None):
    """ Plot means and error bars for two hists side by side with random-histories (if given).
    Suppose 'hists_left' and 'hists_right' have shape (net_count, prune_count+1, data_length), 'random_hists_left' and
    and 'random_hists_right' have shape (net_count, prune_count, data_length) and 'sparsity_hist' has shape
    (prune_count+1).
    The x-axis is labeled with iterations, which are reconstructed from plot_step.
    The baseline (i.e. the lowest sparsity) is a dashed line, all further pruning-levels are solid lines and
    all levels of pruning from random histories are dotted lines. """
    # setup and plot
    fig = plt.figure(None, figsize=(14, 6))
    ax_left, ax_right = fig.subplots(1, 2, sharex=False)
    plot_averages_on_ax(ax_left, hists_left, sparsity_hist, plot_step)
    plot_averages_on_ax(ax_right, hists_right, sparsity_hist, plot_step)
    if random_hists_left is not None:
        plot_random_averages_on_ax(ax_left, random_hists_left, sparsity_hist[1:], plot_step)
    if random_hists_right is not None:
        plot_random_averages_on_ax(ax_right, random_hists_right, sparsity_hist[1:], plot_step)
    setup_grids_on_ax(ax_left, force_zero_left)  # for correct scaling the grids need to be set after plotting
    setup_grids_on_ax(ax_right, force_zero_right)

    # labeling
    net_count, _, _ = hists_left.shape
    setup_labeling_on_ax(ax_left, net_count, type_left)
    setup_labeling_on_ax(ax_right, net_count, type_right)
