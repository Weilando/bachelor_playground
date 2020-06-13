import numpy as np
import matplotlib.pyplot as plt

def get_means_and_y_errors(arr):
    """ Calculate means and error bars for given array of arrays. """
    arr_mean = np.mean(arr, axis=0)
    arr_neg_yerr = arr_mean - np.min(arr, axis=0)
    arr_pos_yerr = np.max(arr, axis=0) - arr_mean
    return arr_mean, arr_neg_yerr, arr_pos_yerr

def format_time(time):
    if time >= 60:
        minutes, seconds = divmod((time), 60)
        return f"{round(minutes)}:{round(seconds):02d}min"
    return f"{time:.4f}sec"

def gen_plot_average_acc_on_ax(ax, acc_histories, sparsity_history):
    """ Generate plots of means and error-bars for given accuracies (per epoch) on ax. """
    net_count, prune_count, epoch_count = acc_histories.shape
    prune_count -= 1
    acc_mean, acc_neg_yerr, acc_pos_yerr = get_means_and_y_errors(acc_histories)
    acc_xs = np.linspace(start=1, stop=epoch_count, num=epoch_count)

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=acc_xs, y=acc_mean[0], yerr=[acc_neg_yerr[0], acc_pos_yerr[0]], label="Mean after 0 prunes", ls='--')

    # plot pruned results
    for p in range(1, prune_count+1):
        ax.errorbar(x=acc_xs, y=acc_mean[p], yerr=[acc_neg_yerr[p], acc_pos_yerr[p]], label=f"Mean after {p} prunes (sparsity  {sparsity_history[p]:.4})")

    # setup grids
    ax.grid('major')
    ax.set_ylim(top=1)

def gen_plot_average_loss_on_ax(ax, loss_histories, sparsity_history, loss_plot_step):
    """ Generate plots of means and error-bars for given losses on ax. """
    net_count, prune_count, _ = loss_histories.shape
    prune_count -= 1
    loss_mean, loss_neg_yerr, loss_pos_yerr = get_means_and_y_errors(loss_histories)
    loss_xs = np.linspace(start=loss_plot_step, stop=len(loss_mean[0])*loss_plot_step, num=len(loss_mean[0]))

    # plot baseline, i.e. unpruned results
    ax.errorbar(x=loss_xs, y=loss_mean[0], yerr=[loss_neg_yerr[0], loss_pos_yerr[0]], label="Mean after 0 prunes", ls='--')

    # plot pruned results
    for p in range(1, prune_count+1):
        ax.errorbar(x=loss_xs, y=loss_mean[p], yerr=[loss_neg_yerr[p], loss_pos_yerr[p]], label=f"Mean after {p} prunes (sparsity {sparsity_history[p]:.4})")

    ax.grid('major')
    ax.set_ylim(bottom=0)

def gen_labels_for_acc_on_ax(ax, test=True):
    """ Generate standard labels for validation-plots on given ax.
    'test' defines if labels should be generated for test- or validation-accuracies. """
    ax.set_ylabel(f"{'Test' if test else 'Validation'}-Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

def gen_labels_for_loss_on_ax(ax):
    """ Generate standard labels for loss-plots on given ax. """
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()


def plot_average_acc(acc_histories, sparsity_history, test=True):
    """ Plot means and error bars for the given accuracies.
    'test' defines if labels should be generated for test- or validation-accuracies. """
    # setup and plot
    fig = plt.figure(None, (7, 6))
    ax = fig.subplots(1,1, sharex=False)
    gen_plot_average_acc_on_ax(ax, acc_histories, sparsity_history)

    # labeling
    net_count, prune_count, _ = acc_histories.shape
    ax.set_title(f"Average {'Test' if test else 'Validation'}-Accuracy for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax, test)

def plot_average_val_acc_and_loss(acc_histories, loss_histories, sparsity_history, loss_plot_step):
    """ Plot means and error bars for validation-accuracies and loss. """
    # setup and plot
    fig = plt.figure(None, (14, 6))
    ax_acc, ax_loss = fig.subplots(1,2, sharex=False)
    gen_plot_average_acc_on_ax(ax_acc, acc_histories, sparsity_history)
    gen_plot_average_loss_on_ax(ax_loss, loss_histories, sparsity_history, loss_plot_step)

    # labeling
    net_count, prune_count, _ = acc_histories.shape
    ax_acc.set_title(f"Average Validation-Accuracy for {net_count} pruned Networks")
    ax_loss.set_title(f"Average Loss for {net_count} pruned Networks")
    gen_labels_for_acc_on_ax(ax_acc, test=False)
    gen_labels_for_loss_on_ax(ax_loss)
