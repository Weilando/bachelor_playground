from dataclasses import dataclass, astuple

import numpy as np


@dataclass
class ExperimentHistories:
    train_loss: np.array = None
    val_loss: np.array = None
    val_acc: np.array = None
    test_acc: np.array = None
    sparsity: np.array = None

    def __eq__(self, other):
        """ Checks if all fields (each is np.array) are equal. """
        assert isinstance(other, ExperimentHistories), f"other must have type ExperimentSettings, but is {type(other)}."

        self_tuple, other_tuple = astuple(self), astuple(other)
        return all(np.array_equal(self_arr, other_arr) for self_arr, other_arr in zip(self_tuple, other_tuple))
