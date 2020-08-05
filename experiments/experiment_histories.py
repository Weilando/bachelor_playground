from copy import deepcopy
from dataclasses import dataclass, astuple

import math
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
        """ Check if all fields (each is np.array) are equal. """
        assert isinstance(other,
                          ExperimentHistories), f"other must have type ExperimentHistories, but is {type(other)}."

        self_tuple, other_tuple = astuple(self), astuple(other)
        return all(np.array_equal(self_arr, other_arr) for self_arr, other_arr in zip(self_tuple, other_tuple))

    def stack_histories(self, other):
        """ Generate a new ExperimentHistories object, which contains joined arrays for losses and accuracies.
        Add arrays from 'other' to corresponding arrays from 'self' (basically on nets-dimension).
        Levels of sparsity and all dimensions need to match. """
        assert isinstance(other,
                          ExperimentHistories), f"other must have type ExperimentHistories, but is {type(other)}."
        np.testing.assert_array_almost_equal(self.sparsity, other.sparsity)

        new_history = ExperimentHistories()
        new_history.sparsity = deepcopy(self.sparsity)
        new_history.train_loss = np.vstack([self.train_loss, other.train_loss])
        new_history.val_loss = np.vstack([self.val_loss, other.val_loss])
        new_history.val_acc = np.vstack([self.val_acc, other.val_acc])
        new_history.test_acc = np.vstack([self.test_acc, other.test_acc])

        return new_history


def calc_hist_length_per_net(batch_count, epoch_count, plot_step):
    """ Calculate length of history arrays based on batch_count, epoch_count and plot_step. """
    return math.floor((batch_count * epoch_count) / plot_step)
