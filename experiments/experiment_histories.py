import math
from dataclasses import dataclass, astuple

import numpy as np


@dataclass
class ExperimentHistories:
    """ Stores np.arrays with histories from one experiment. """
    train_loss: np.array = None
    val_loss: np.array = None
    val_acc: np.array = None
    test_acc: np.array = None
    sparsity: np.array = None

    def setup(self, net_count, prune_count, epoch_length, epoch_count, plot_step):
        """ Setup histories with correct shapes.
        Loss and accuracies are saved at each plot_step iteration.
        Sparsity is saved once for all models. """
        history_length = calc_hist_length_per_net(epoch_length, epoch_count, plot_step)
        self.train_loss = np.zeros((net_count, prune_count + 1, history_length), dtype=float)
        self.val_loss = np.zeros_like(self.train_loss, dtype=float)
        self.val_acc = np.zeros_like(self.train_loss, dtype=float)
        self.test_acc = np.zeros_like(self.train_loss, dtype=float)

        self.sparsity = np.ones((prune_count + 1), dtype=float)

    def __eq__(self, other):
        """ Checks if all fields (each is np.array) are equal. """
        assert isinstance(other, ExperimentHistories), f"other must have type ExperimentSettings, but is {type(other)}."

        self_tuple, other_tuple = astuple(self), astuple(other)
        return all(np.array_equal(self_arr, other_arr) for self_arr, other_arr in zip(self_tuple, other_tuple))


@dataclass
class EarlyStopHistories:
    """ Stores checkpoints (i.e. state_dicts) and indices at early-stopping for one model. """
    checkpoints: np.array = None
    indices: np.array = None

    def setup(self, net_count, prune_count):
        """ Setup histories with correct shapes.
        Checkpoints and indices are saved once per net and level of pruning. """
        self.checkpoints = np.empty((net_count, prune_count + 1), dtype=dict)
        self.indices = np.full((net_count, prune_count + 1), fill_value=-1, dtype=int)

    def __eq__(self, other):
        """ Checks if all fields (each is np.array) are equal. """
        assert isinstance(other, EarlyStopHistories), f"other must have type ExperimentSettings, but is {type(other)}."

        self_tuple, other_tuple = astuple(self), astuple(other)
        return all(np.array_equal(self_arr, other_arr) for self_arr, other_arr in zip(self_tuple, other_tuple))


def calc_hist_length_per_net(batch_count, epoch_count, plot_step):
    """ Calculate length of history arrays based on batch_count, epoch_count and plot_step. """
    return math.floor((batch_count * epoch_count) / plot_step)
