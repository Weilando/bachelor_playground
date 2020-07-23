from dataclasses import dataclass

import numpy as np


@dataclass
class EarlyStopHistoryList:
    """ Stores EarlyStopHistory-objects for several models and levels of pruning. """
    histories: np.array = None

    def setup(self, net_count, prune_count):
        """ Setup histories with correct shapes.
        An EarlyStopHistory is saved once per net and level of pruning. """
        self.histories = np.empty(net_count, dtype=EarlyStopHistory)
        for n in range(len(self.histories)):
            self.histories[n] = EarlyStopHistory()
            self.histories[n].setup(prune_count)

    def __eq__(self, other):
        """ Check if all histories are equal (under omission of state_dicts). """
        assert isinstance(other, EarlyStopHistoryList), \
            f"'other' must have type EarlyStopHistoryList, but is {type(other)}."
        return np.array_equal(self.histories, other.histories)


@dataclass
class EarlyStopHistory:
    """ Stores state_dicts and indices at early-stop for one model and several levels of pruning. """
    state_dicts: np.array = None
    indices: np.array = None

    def setup(self, prune_count):
        """ Setup histories with correct shapes.
        state_dicts and indices are saved once per level of pruning. """
        self.state_dicts = np.empty(prune_count + 1, dtype=dict)
        self.indices = np.full(prune_count + 1, fill_value=-1, dtype=int)

    def __eq__(self, other):
        """ Check if all fields (each is np.array) are equal.
        The equality of the saved state_dicts is not part of the check! """
        assert isinstance(other, EarlyStopHistory), f"'other' must have type EarlyStopHistory, but is {type(other)}."
        return np.array_equal(self.indices, other.indices)
