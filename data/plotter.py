from enum import Enum

import numpy as np

from data.plotter_evaluation import find_acc_at_early_stop_indices, find_early_stop_iterations, get_means_and_y_errors


class PlotType(str, Enum):
    """ Defines available types of plots. """
    TRAIN_LOSS = "Training-Loss"
    VAL_LOSS = "Validation-Loss"
    VAL_ACC = "Validation-Accuracy"
    TEST_ACC = "Test-Accuracy"
    EARLY_STOP_ITER = "Early-Stopping Iteration"


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


def gen_title_on_ax(ax, plot_type: PlotType, early_stop=False):
    """ Generates plot-title on given ax.
    'early_stop' defines if early-stopping should be mentioned. """
    ax.set_title(f"Average {plot_type.value}{' at early-stop' if early_stop else ''}")


def setup_early_stop_ax(ax, force_zero, log_step=7):
    """ Inverts x-axis and activates log-scale with 'log_step' steps for x-axis. """
    ax.set_xscale('log', basex=2)
    ax.set_xticks([2 ** (-p) for p in range(log_step)])
    ax.set_xticklabels([2 ** (-p) for p in range(log_step)])

    ax.invert_xaxis()  # also inverts plot!
    setup_grids_on_ax(ax, force_zero)  # for correct scaling the grids need to be set after plotting


def setup_grids_on_ax(ax, force_zero=False):
    """ Setup grids on given ax.
    'force_zero' sets the minimum y-value to zero, e.g. for loss plots. """
    ax.grid()
    if force_zero:
        ax.set_ylim(bottom=0)


def setup_labeling_on_ax(ax, plot_type: PlotType, iteration=True, early_stop=False):
    """ Setup complete labeling on ax, i.e. generate title, labels and legend. """
    gen_title_on_ax(ax, plot_type, early_stop)
    gen_labels_on_ax(ax, plot_type, iteration)
    ax.legend()


# subplots
def plot_average_at_early_stop_on_ax(ax, hists, sparsity_hist, net_name, random=False, color=None):
    """ Plot means and error-bars for given early-stopping iterations or accuracies on ax.
    Suppose 'hists' has shape (net_count, prune_count+1, 1) for accuracies or (net_count, prune_count+1) for iterations.
    Suppose 'sparsity_hist' has shape (prune_count+1).
    'hists' is a solid line with error bars if 'random' is False, and a dotted line otherwise.
    If 'color' is not specified, choose the next color from the color cycle. """
    # calculate means
    mean, neg_y_err, pos_y_err = get_means_and_y_errors(hists)

    # for accuracies each mean and y_err has shape (prune_count+1, 1), so squeeze them to get shape (prune_count+1)
    mean = np.squeeze(mean)
    neg_y_err = np.squeeze(neg_y_err)
    pos_y_err = np.squeeze(pos_y_err)

    # plot and return instance of `ErrorbarContainer` to read its color
    if random:
        return ax.errorbar(x=sparsity_hist[1:], y=mean, elinewidth=1, yerr=[neg_y_err, pos_y_err], marker='x', ls=':',
                           color=color, label=f"{net_name} reinit")
    return ax.errorbar(x=sparsity_hist, y=mean, elinewidth=1, yerr=[neg_y_err, pos_y_err], marker='x', ls='-',
                       color=color, label=net_name)


def plot_averages_on_ax(ax, hists, sparsity_hist, plot_step, random=False):
    """ Plot means and error-bars for 'hists' on ax.
    Suppose hists has shape (net_count, prune_count+1) and 'sparsity_hist' has shape (prune_count+1).
    Plots dashed baseline (unpruned) and a solid line for each pruning step, if 'random' is False.
    Plots dotted lines for each pruning step, if 'random' is True. """
    _, prune_count, _ = hists.shape
    prune_count -= 1  # baseline at index 0, thus first pruned round at index 1

    h_mean, h_neg_y_err, h_pos_y_err = get_means_and_y_errors(hists)
    xs = gen_iteration_space(h_mean[0], plot_step)

    if random:
        plot_pruned_means_on_ax(ax, xs, h_mean, h_neg_y_err, h_pos_y_err, sparsity_hist, prune_count, ':')
    else:
        plot_baseline_mean_on_ax(ax, xs, h_mean[0], h_neg_y_err[0], h_pos_y_err[0])
        plot_pruned_means_on_ax(ax, xs, h_mean[1:], h_neg_y_err[1:], h_pos_y_err[1:], sparsity_hist[1:], prune_count)


def plot_baseline_mean_on_ax(ax, xs, ys, y_err_neg, y_err_pos):
    """ Plots the baseline as dashed line wit error bars on given ax. """
    ax.errorbar(x=xs, y=ys, yerr=[y_err_neg, y_err_pos], elinewidth=1.2, ls='--', color="C0", label="Sparsity 1.0000",
                errorevery=5, capsize=2)


def plot_pruned_means_on_ax(ax, xs, ys, y_err_neg, y_err_pos, sparsity_hist, prune_count, ls='-'):
    """ Plots means per pruning level as line with error bars on given ax.
    Labels contain the sparsity at given level of pruning.
    'prune_min' and 'prune_max' specify the prune-levels to plot.
    'ls' specifies the style for all plotted lines (e.g. '-'=solid and ':'=dotted).
    Colors start with color-spec C1. """
    for p in range(prune_count):
        ax.errorbar(x=xs, y=ys[p], yerr=[y_err_neg[p], y_err_pos[p]], color=f"C{p + 1}", elinewidth=1.2, ls=ls,
                    label=f"Sparsity {sparsity_hist[p]:.4f}", errorevery=5, capsize=2)


# plots
def plot_acc_at_early_stop_on_ax(ax, loss_hists, acc_hists, sparsity_hist, net_name, plot_type: PlotType,
                                 rnd_loss_hists=None, rnd_acc_hists=None, force_zero=False, setup_ax=True, log_step=7):
    """ Plot means and error bars for the given accuracies at the time an early stopping criterion would end training.
    Use 'loss_hists' to find accuracies from 'acc_hists', analog for random histories, if given.
    Suppose 'acc_hists' and 'loss_hists' have shape (net_count, prune_count+1, data_length), 'rnd_acc_hists' and
    'rnd_loss_hists' have shape (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1)
    with prune_count > 1.
    If 'setup_ax' is True, add grids and labels, invert x-axis and apply log-scale to x-axis.
    Use 'net_name' to generate labels for the legend.
    Plot accuracies as solid line and random accuracies as dotted line in the same color. """
    assert (sparsity_hist.shape[0] > 1) and (sparsity_hist.shape[0] == loss_hists.shape[1]), \
        f"'prune_count' (dimension of 'sparsity_hist') needs to be greater than one, but is {sparsity_hist.shape}."

    early_stop_acc = find_acc_at_early_stop_indices(loss_hists, acc_hists)
    original_plot = plot_average_at_early_stop_on_ax(ax, early_stop_acc, sparsity_hist, net_name, random=False)
    if rnd_loss_hists is not None and rnd_acc_hists is not None:
        random_early_stop_acc = find_acc_at_early_stop_indices(rnd_loss_hists, rnd_acc_hists)
        plot_average_at_early_stop_on_ax(ax, random_early_stop_acc, sparsity_hist, net_name, random=True,
                                         color=original_plot.lines[0].get_color())
    if setup_ax:
        setup_early_stop_ax(ax, force_zero, log_step)
        setup_labeling_on_ax(ax, plot_type, iteration=False, early_stop=True)


def plot_average_hists_on_ax(ax, hists, sparsity_hist, plot_step, plot_type: PlotType, rnd_hists=None,
                             force_zero=False):
    """ Plot means and error bars for the given histories in 'hists' and 'rnd_hists' (if given).
    Suppose 'hists' has shape (net_count, prune_count+1, data_length), 'rnd_hists' has shape
    (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1).
    The x-axis is labeled with iterations, which are reconstructed from plot_step.
    The baseline (i.e. the lowest sparsity) is a dashed line, all further pruning-levels from 'hists' are solid lines,
    all levels of pruning from 'rnd_hists' are dotted lines. """
    plot_averages_on_ax(ax, hists, sparsity_hist, plot_step, random=False)
    if rnd_hists is not None:
        plot_averages_on_ax(ax, rnd_hists, sparsity_hist[1:], plot_step, random=True)

    setup_grids_on_ax(ax, force_zero)  # for correct scaling the grids need to be set after plotting
    setup_labeling_on_ax(ax, plot_type, iteration=True, early_stop=False)


def plot_early_stop_iterations_on_ax(ax, loss_hists, sparsity_hist, plot_step, net_name, rnd_loss_hists=None,
                                     force_zero=False, setup_ax=True, log_step=7):
    """ Plot means and error bars for early-stopping iterations based on 'loss_hists' and 'rnd_loss_hists', if given.
    Suppose 'loss_hists' has shape (net_count, prune_count+1, data_length), 'loss_hists' has shape
    (net_count, prune_count, data_length) and 'sparsity_hist' has shape (prune_count+1) with prune_count > 1.
    If 'setup_ax' is True, add grids and labels, invert x-axis and apply log-scale to x-axis.
    Use 'net_name' to generate labels for the legend.
    Plot iterations as solid line and random iterations as dotted line in the same color. """
    assert (sparsity_hist.shape[0] > 1) and (sparsity_hist.shape[0] == loss_hists.shape[1]), \
        f"'prune_count' (dimension of 'sparsity_hist') needs to be greater than one, but is {sparsity_hist.shape}."

    early_stop_iterations = find_early_stop_iterations(loss_hists, plot_step)
    original_plot = plot_average_at_early_stop_on_ax(ax, early_stop_iterations, sparsity_hist, net_name, random=False)
    if rnd_loss_hists is not None:
        random_early_stop_iterations = find_early_stop_iterations(rnd_loss_hists, plot_step)
        plot_average_at_early_stop_on_ax(ax, random_early_stop_iterations, sparsity_hist, net_name, random=True,
                                         color=original_plot.lines[0].get_color())
    if setup_ax:
        setup_early_stop_ax(ax, force_zero, log_step)
        setup_labeling_on_ax(ax, PlotType.EARLY_STOP_ITER, iteration=False, early_stop=True)
