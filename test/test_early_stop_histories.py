from unittest import TestCase, main as unittest_main

import numpy as np
from torch.nn import Linear

from experiments.early_stop_histories import EarlyStopHistory, EarlyStopHistoryList


class TestEarlyStopHistories(TestCase):
    """ Tests for the early_stop_histories module.
    Call with 'python -m test.test_early_stop_histories' from project root '~'.
    Call with 'python -m test_early_stop_histories' from inside '~/test'. """

    def test_early_stop_histories_are_equal(self):
        """ Should return True, because both EarlyStopHistories are the same. """
        history = EarlyStopHistory()
        history.setup(0)
        self.assertIs(EarlyStopHistory.__eq__(history, history), True)

    def test_early_stop_histories_are_unequal(self):
        """ Should return False, because both EarlyStopHistories contain unequal indices. """
        net = Linear(1, 1)
        history1, history2 = EarlyStopHistory(), EarlyStopHistory()
        history1.setup(1)
        history2.setup(1)
        history1.state_dicts[0], history2.state_dicts[0] = net.state_dict(), net.state_dict()
        history1.indices[0], history2.indices[0] = 3, 42

        self.assertIs(EarlyStopHistory.__eq__(history1, history2), False)

    def test_early_stop_history_error_on_invalid_type(self):
        """ Should return False, because a dict is no EarlyStopHistory. """
        with self.assertRaises(AssertionError):
            EarlyStopHistory.__eq__(EarlyStopHistory(), dict())

    # noinspection PyMethodMayBeStatic
    def test_setup_early_stop_history(self):
        """ Should setup all np.arrays correctly. """
        history = EarlyStopHistory()
        history.setup(1)

        np.testing.assert_array_equal(np.empty(2, dtype=dict), history.state_dicts)
        np.testing.assert_array_equal(np.full(2, fill_value=-1, dtype=int), history.indices)

    def test_early_stop_history_lists_are_equal(self):
        """ Should return True, because both EarlyStopHistoryLists are the same. """
        history_list = EarlyStopHistoryList()
        history_list.setup(2, 1)
        self.assertIs(EarlyStopHistoryList.__eq__(history_list, history_list), True)

    def test_early_stop_history_lists_are_unequal(self):
        history_list0, history_list1 = EarlyStopHistoryList(), EarlyStopHistoryList()
        history_list0.setup(2, 1)
        history_list1.setup(2, 1)
        history_list1.histories[0].indices[0] = 5

        self.assertIs(EarlyStopHistoryList.__eq__(history_list0, history_list1), False)

    def test_setup_early_stop_history_list(self):
        """ Should setup all np.arrays correctly. """
        histories = EarlyStopHistoryList()
        histories.setup(1, 0)

        self.assertIsInstance(histories.histories[0], EarlyStopHistory)
        self.assertIsInstance(histories.histories, np.ndarray)
        self.assertEqual(1, len(histories.histories))


if __name__ == '__main__':
    unittest_main()
