import matplotlib.pyplot as plt
import numpy as np


# evaluation functions
def get_means_and_y_errors(arr):
    """ Calculate means and error bars for given array of arrays.
    The array is supposed to be of shape (net_count, prune_count+1, data_length).
    Then this function squashes the net-dimension, so mean, max and min have shape (prune_count+1, data_length). """
    arr_mean = np.mean(arr, axis=0)
    arr_neg_y_err = arr_mean - np.min(arr, axis=0)
    arr_pos_y_err = np.max(arr, axis=0) - arr_mean
    return arr_mean, arr_neg_y_err, arr_pos_y_err


def find_stop_iteration(loss_hists):
    """ Find the minimum for each loss history in 'loss_hists' and return its indices.
    'loss_hists' is supposed be of shape (net_count, prune_count, iteration_count).
    The result has shape (net_count, prune_count) and contains the earliest index with minimum loss. """
    return np.argmin(loss_hists, axis=2)


def format_time(time):
    """ Format a given integer (UIX time) into a string.
    Convert times shorter than a minute into seconds with four decimal places.
    Convert longer times into rounded minutes and seconds, separated by a colon. """
    if time >= 60:
        minutes, seconds = divmod(time, 60)
        return f"{round(minutes)}:{round(seconds):02d}min"
    return f"{time:.4f}sec"


# generator functions for plots
def gen_baseline_mean_plot_on_ax(ax, xs, ys, y_err_neg, y_err_pos):
    """ Plots the baseline as dashed line wit error bars on given ax. """
    ax.errorbar(x=xs, y=ys, yerr=[y_err_neg, y_err_pos], label="Mean after 0 prunes", ls='--')


def gen_pruned_mean_plots_on_ax(ax, xs, ys, y_err_neg, y_err_pos, sparsity_hist, prune_min, prune_max):
    """ Plots means per pruning level as line with error bars on given ax.
     Labels contain the sparsity at given level of pruning.
     prune_min and prune_max specify the prune-levels to plot. """
    for p in range(prune_min, prune_max + 1):
        ax.errorbar(x=xs, y=ys[p], yerr=[y_err_neg[p], y_err_pos[p]],
                    label=f"Mean after {p} prunes (sparsity  {sparsity_hist[p]:.4})")


def setup_loss_grids_on_ax(ax):
    """ Setup grids for loss plots on given ax. """
    ax.grid('major')
    ax.set_ylim(bottom=0)


def setup_acc_grids_on_ax(ax, force_one=False):
    """ Setup grids for accuracy plots on given ax. """
    ax.grid('major')
    if force_one:
        ax.set_ylim(top=1)


def gen_plot_average_acc_on_ax(ax, acc_hists, sparsity_hist, force_one=False):
    """ Generate plots of means and error-bars for given accuracies (per epoch) on ax. """
    net_count, prune_count, epoch_count = acc_hists.shape
    prune_count -= 1
    acc_mean, acc_neg_y_err, acc_pos_y_err = get_means_and_y_errors(acc_hists)
    acc_xs = np.linspace(start=1, stop=epoch_count, num=epoch_count)

    gen_baseline_mean_plot_on_ax(ax, acc_xs, acc_mean[0], acc_neg_y_err[0], acc_pos_y_err[0])
    gen_pruned_mean_plots_on_ax(ax, acc_xs, acc_mean, acc_neg_y_err, acc_pos_y_err, sparsity_hist, 1, prune_count)
    setup_acc_grids_on_ax(ax, force_one)


def gen_plot_average_loss_on_ax(ax, loss_hists, sparsity_hist, plot_step):
    """ Generate plots of means and error-bars for given losses on ax. """
    net_count, prune_count, _ = loss_hists.shape
    prune_count -= 1
    loss_mean, loss_neg_y_err, loss_pos_y_err = get_means_and_y_errors(loss_hists)
    loss_xs = np.linspace(start=plot_step, stop=len(loss_mean[0]) * plot_step, num=len(loss_mean[0]))

    gen_baseline_mean_plot_on_ax(ax, loss_xs, loss_mean[0], loss_neg_y_err[0], loss_pos_y_err[0])
    gen_pruned_mean_plots_on_ax(ax, loss_xs, loss_mean, loss_neg_y_err, loss_pos_y_err, sparsity_hist, 1, prune_count)
    setup_loss_grids_on_ax(ax)


def gen_plot_average_acc_at_early_stop_on_ax(ax, acc_hists, sparsity_hist, force_one=False):
    """ Generate plots of means and error-bars for given accuracies (per sparsity) on ax. """
    acc_mean, acc_neg_y_err, acc_pos_y_err = get_means_and_y_errors(acc_hists)  # each result has shape (prune_count)

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=sparsity_hist, y=acc_mean, yerr=[acc_neg_y_err, acc_pos_y_err], label="Mean of accs", marker='x')
    ax.invert_xaxis()
    setup_acc_grids_on_ax(ax, force_one)


def gen_labels_for_acc_on_ax(ax, test=True, epoch=True):
    """ Generate standard labels for validation-plots on given ax.
    'test' defines if y-labels should be generated for test- or validation-accuracies.
    'epoch' defines if x-labels should be generated for epochs or sparsity. """
    ax.set_ylabel(f"{'Test' if test else 'Validation'}-Accuracy")
    ax.set_xlabel(f"{'Epoch' if epoch else 'Sparsity'}")
    ax.legend()


def gen_labels_for_loss_on_ax(ax):
    """ Generate standard labels for loss-plots on given ax. """
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()


# plotting functions
def plot_average_acc(acc_hists, sparsity_hist, test=True):
    """ Plot means and error bars for the given accuracies.
    'test' defines if labels should be generated for test- or validation-accuracies. """
    # setup and plot
    fig = plt.figure(None, (7, 6))
    ax = fig.subplots(1, 1, sharex=False)
    gen_plot_average_acc_on_ax(ax, acc_hists, sparsity_hist)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax.set_title(f"Average {'Test' if test else 'Validation'}-Accuracy for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax, test)


def plot_average_val_acc_and_loss(acc_hists, loss_hists, sparsity_hist, plot_step):
    """ Plot means and error bars for validation-accuracies and loss. """
    # setup and plot
    fig = plt.figure(None, (14, 6))
    ax_acc, ax_loss = fig.subplots(1, 2, sharex=False)
    gen_plot_average_acc_on_ax(ax_acc, acc_hists, sparsity_hist)
    gen_plot_average_loss_on_ax(ax_loss, loss_hists, sparsity_hist, plot_step)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax_acc.set_title(f"Average Validation-Accuracy for {net_count} pruned Networks")
    ax_loss.set_title(f"Average Loss for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax_acc, test=False)
    gen_labels_for_loss_on_ax(ax_loss)


def plot_acc_at_early_stopping(acc_hists, loss_hists, sparsity_hist, test=True):
    """ Plot means and error bars for the given accuracies at the time an early stopping criterion would end training.
    The x-axis shows the sparsity at each time..
    'test' defines if labels should be generated for test- or validation-accuracies. """
    fig = plt.figure(None, (7, 6))
    ax = fig.subplots(1, 1, sharex=False)
    early_stop_indices = find_stop_iteration(loss_hists)
    gen_plot_average_acc_at_early_stop_on_ax(ax, acc_hists[-1, -1, early_stop_indices], sparsity_hist, force_one=False)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax.set_title(
        f"Average {'Test' if test else 'Validation'}-Accuracy for {net_count} pruned Networks at early stopping")
    gen_labels_for_acc_on_ax(ax, test, epoch=False)
