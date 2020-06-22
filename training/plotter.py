import numpy as np
import matplotlib.pyplot as plt

# evaluation functions
def get_means_and_y_errors(arr):
    """ Calculate means and error bars for given array of arrays.
    The array is supposed to be of shape (net_count, prune_count+1, data_length).
    In this case this function squashes the net-dimension, so one gets mean, max and min with shape (prune_count+1, data_length). """
    arr_mean = np.mean(arr, axis=0)
    arr_neg_yerr = arr_mean - np.min(arr, axis=0)
    arr_pos_yerr = np.max(arr, axis=0) - arr_mean
    return arr_mean, arr_neg_yerr, arr_pos_yerr

def find_stop_iteration(loss_hists):
    """ Find the minimum for each loss history in 'loss_hists' and return its indices.
    'loss_hists' is supposed be of shape (net_count, prune_count, iteration_count).
    The result has shape (net_count, prune_count) and contains the earliest index with minimum loss. """
    return np.argmin(loss_hists, axis=2)

def format_time(time):
    """ Format a given integer (UIX time) into a string.
    Convert times shorter than a minute into seconds with four decimal places, and times longer than a minute into minutes and seconds. """
    if time >= 60:
        minutes, seconds = divmod((time), 60)
        return f"{round(minutes)}:{round(seconds):02d}min"
    return f"{time:.4f}sec"


# generator functions for plots
def gen_plot_average_acc_on_ax(ax, acc_hists, sparsity_hist, force_one=False):
    """ Generate plots of means and error-bars for given accuracies (per epoch) on ax. """
    net_count, prune_count, epoch_count = acc_hists.shape
    prune_count -= 1
    acc_mean, acc_neg_yerr, acc_pos_yerr = get_means_and_y_errors(acc_hists)
    acc_xs = np.linspace(start=1, stop=epoch_count, num=epoch_count)

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=acc_xs, y=acc_mean[0], yerr=[acc_neg_yerr[0], acc_pos_yerr[0]], label="Mean after 0 prunes", ls='--')

    # plot pruned results
    for p in range(1, prune_count+1):
        ax.errorbar(x=acc_xs, y=acc_mean[p], yerr=[acc_neg_yerr[p], acc_pos_yerr[p]], label=f"Mean after {p} prunes (sparsity  {sparsity_hist[p]:.4})")

    # setup grids
    ax.grid('major')
    if force_one:
        ax.set_ylim(top=1)

def gen_plot_average_acc_at_early_stop_on_ax(ax, acc_hists, sparsity_hist, force_one=False):
    """ Generate plots of means and error-bars for given accuracies (per sparsity) on ax. """
    net_count, prune_count = acc_hists.shape
    acc_mean, acc_neg_yerr, acc_pos_yerr = get_means_and_y_errors(acc_hists) # each result has shape (prune_count)

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=sparsity_hist, y=acc_mean, yerr=[acc_neg_yerr, acc_pos_yerr], label="Mean of accs", marker='x')

    # setup grids
    ax.grid('major')
    if force_one:
        ax.set_ylim(top=1)

def gen_plot_average_loss_on_ax(ax, loss_hists, sparsity_hist, plot_step):
    """ Generate plots of means and error-bars for given losses on ax. """
    net_count, prune_count, _ = loss_hists.shape
    prune_count -= 1
    loss_mean, loss_neg_yerr, loss_pos_yerr = get_means_and_y_errors(loss_hists)
    loss_xs = np.linspace(start=plot_step, stop=len(loss_mean[0])*plot_step, num=len(loss_mean[0]))

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=loss_xs, y=loss_mean[0], yerr=[loss_neg_yerr[0], loss_pos_yerr[0]], label="Mean after 0 prunes", ls='--')

    # plot pruned results
    for p in range(1, prune_count+1):
        ax.errorbar(x=loss_xs, y=loss_mean[p], yerr=[loss_neg_yerr[p], loss_pos_yerr[p]], label=f"Mean after {p} prunes (sparsity {sparsity_hist[p]:.4})")

    ax.grid('major')
    ax.set_ylim(bottom=0)

def gen_labels_for_acc_on_ax(ax, test=True, epoch=True):
    """ Generate standard labels for validation-plots on given ax.
    'test' defines if y-labels should be generated for test- or validation-accuracies.
    'epoch' defines if x-labels should be generated for epochs or sparsities. """
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
    ax = fig.subplots(1,1, sharex=False)
    gen_plot_average_acc_on_ax(ax, acc_hists, sparsity_hist)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax.set_title(f"Average {'Test' if test else 'Validation'}-Accuracy for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax, test)

def plot_average_val_acc_and_loss(acc_hists, loss_hists, sparsity_hist, plot_step):
    """ Plot means and error bars for validation-accuracies and loss. """
    # setup and plot
    fig = plt.figure(None, (14, 6))
    ax_acc, ax_loss = fig.subplots(1,2, sharex=False)
    gen_plot_average_acc_on_ax(ax_acc, acc_hists, sparsity_hist)
    gen_plot_average_loss_on_ax(ax_loss, loss_hists, sparsity_hist, plot_step)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax_acc.set_title(f"Average Validation-Accuracy for {net_count} pruned Networks")
    ax_loss.set_title(f"Average Loss for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax_acc, test=False)
    gen_labels_for_loss_on_ax(ax_loss)

def plot_acc_at_early_stopping(acc_hists, loss_hists, sparsity_hist, plot_step, test=True):
    """ Plot means and error bars for the given accuracies at the time an early stopping criterion would end training.
    The x-axis shows the sparsities at each time..
    'test' defines if labels should be generated for test- or validation-accuracies. """
    fig = plt.figure(None, (7, 6))
    ax = fig.subplots(1,1, sharex=False)
    early_stop_indices = find_stop_iteration(loss_hists)
    gen_plot_average_acc_at_early_stop_on_ax(ax, acc_hists[-1,-1,early_stop_indices], sparsity_hist, force_one=False)

    # labeling
    net_count, prune_count, _ = acc_hists.shape
    ax.set_title(f"Average {'Test' if test else 'Validation'}-Accuracy for {net_count} pruned Networks at early stopping")
    gen_labels_for_acc_on_ax(ax, test, epoch=False)
