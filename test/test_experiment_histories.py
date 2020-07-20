from unittest import TestCase
from unittest import main as unittest_main

import numpy as np

from experiments.experiment_histories import calc_hist_length_per_net, ExperimentHistories, EarlyStopHistories


class MyTestCase(TestCase):
    """ Tests for the experiment_histories module.
    Call with 'python -m test.test_experiment_histories' from project root '~'.
    Call with 'python -m test_experiment_histories' from inside '~/test'. """

    def test_experiment_histories_are_equal(self):
        """ Should return True, because both ExperimentHistories contain equal arrays. """
        histories1 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))
        histories2 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), True)

    def test_experiment_histories_are_unequal(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        histories1 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2))
        histories2 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), False)

    def test_experiment_histories_error_on_invalid_type(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        with self.assertRaises(AssertionError):
            ExperimentHistories.__eq__(ExperimentHistories(), dict())

    def test_early_stop_histories_are_equal(self):
        """ Should return True, because both EarlyStopHistories contain equal arrays. """
        histories1 = EarlyStopHistories(np.zeros(3), np.zeros(3))
        histories2 = EarlyStopHistories(np.zeros(3), np.zeros(3))

        self.assertIs(EarlyStopHistories.__eq__(histories1, histories2), True)

    def test_early_stop_histories_are_unequal(self):
        """ Should return False, because both EarlyStopHistories contain unequal arrays. """
        histories1 = EarlyStopHistories(np.zeros(3), np.zeros(3))
        histories2 = EarlyStopHistories(np.zeros(3), np.ones(3))

        self.assertIs(EarlyStopHistories.__eq__(histories1, histories2), False)

    def test_early_stop_histories_error_on_invalid_type(self):
        """ Should return False, because both EarlyStopHistories contain unequal arrays. """
        with self.assertRaises(AssertionError):
            EarlyStopHistories.__eq__(EarlyStopHistories(), dict())

    def test_calc_hist_length_per_net(self):
        """ Should calculate the correct length of a history for one net.
        As one entry is generated per epoch, the history should have a length of two. """
        self.assertEqual(calc_hist_length_per_net(3, 2, 3), 2)

    # noinspection PyMethodMayBeStatic
    def test_setup_experiment_histories(self):
        """ Should setup all np.arrays correctly. """
        histories = ExperimentHistories()
        histories.setup(3, 1, 42, 4, 7)

        expected_history = np.zeros((3, 2, 24), dtype=float)  # expected history for accuracies and losses
        np.testing.assert_array_equal(histories.train_loss, expected_history)
        np.testing.assert_array_equal(histories.val_loss, expected_history)
        np.testing.assert_array_equal(histories.val_acc, expected_history)
        np.testing.assert_array_equal(histories.test_acc, expected_history)
        np.testing.assert_array_equal(histories.sparsity, np.ones(2, dtype=float))

    # noinspection PyMethodMayBeStatic
    def test_setup_early_stop_histories(self):
        """ Should setup all np.arrays correctly. """
        histories = EarlyStopHistories()
        histories.setup(3, 1)

        np.testing.assert_array_equal(histories.checkpoints, np.empty((3, 2), dtype=dict))
        np.testing.assert_array_equal(histories.indices, np.full((3, 2), fill_value=-1, dtype=int))

    def test_calc_hist_length_per_net_rounding(self):
        """ Should calculate the correct length of a history for one net.
        As one entry is generated in the last epoch, the history should have a length of one. """
        self.assertEqual(calc_hist_length_per_net(3, 2, 4), 1)


if __name__ == '__main__':
    unittest_main()
