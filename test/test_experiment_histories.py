from unittest import TestCase
from unittest import main as unittest_main

import numpy as np

from experiments.experiment_histories import calc_hist_length_per_net, ExperimentHistories


class TestExperimentHistories(TestCase):
    """ Tests for the experiment_histories module.
    Call with 'python -m test.test_experiment_histories' from project root '~'.
    Call with 'python -m test_experiment_histories' from inside '~/test'. """

    def test_experiment_histories_are_equal(self):
        """ Should return True, because both ExperimentHistories contain equal arrays. """
        histories1 = ExperimentHistories()
        histories2 = ExperimentHistories()
        histories1.setup(1, 1, 1, 1, 1)
        histories2.setup(1, 1, 1, 1, 1)

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), True)

    def test_experiment_histories_are_unequal(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        histories1 = ExperimentHistories()
        histories2 = ExperimentHistories()
        histories1.setup(1, 1, 1, 1, 1)
        histories2.setup(1, 1, 2, 1, 1)

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), False)

    def test_experiment_histories_error_on_invalid_type(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        with self.assertRaises(AssertionError):
            ExperimentHistories.__eq__(ExperimentHistories(), dict())

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

    def test_stack_histories(self):
        """ Should stack two histories, i.e. append their arrays at the net-dimension.
        The sparsity histories should not be appended, as they need to be equal. """
        histories0 = ExperimentHistories()
        histories1 = ExperimentHistories()
        histories0.setup(2, 1, 2, 3, 2)  # arrays for accuracies and losses have shape (2, 2, 3), all filled with zeroes
        histories1.setup(2, 1, 2, 3, 2)

        # fill arrays for accuracies and losses with ones
        histories1.train_loss = np.ones_like(histories1.train_loss)
        histories1.val_loss = np.ones_like(histories1.val_loss)
        histories1.val_acc = np.ones_like(histories1.val_acc)
        histories1.test_acc = np.ones_like(histories1.test_acc)

        result_history = histories0.stack_histories(histories1)

        # expected history for accuracies and losses
        expected_history = np.array([[[0., 0., 0.], [0., 0., 0.]],
                                     [[0., 0., 0.], [0., 0., 0.]],
                                     [[1., 1., 1.], [1., 1., 1.]],
                                     [[1., 1., 1.], [1., 1., 1.]]], dtype=float)

        np.testing.assert_array_equal(result_history.train_loss, expected_history)
        np.testing.assert_array_equal(result_history.val_loss, expected_history)
        np.testing.assert_array_equal(result_history.val_acc, expected_history)
        np.testing.assert_array_equal(result_history.test_acc, expected_history)
        np.testing.assert_array_equal(result_history.sparsity, histories0.sparsity)

    def test_stack_histories_fail_unequal_sparsity(self):
        """ Should raise an error, as the sparsity-histories do not match. """
        histories0 = ExperimentHistories()
        histories1 = ExperimentHistories()
        histories0.setup(1, 1, 2, 3, 2)
        histories1.setup(1, 1, 2, 3, 2)
        histories0.sparsity = np.array([1., 0.5])
        histories1.sparsity = np.array([1., 0.7])

        with self.assertRaises(AssertionError):
            histories0.stack_histories(histories1)

    def test_stack_histories_fail_wrong_type(self):
        """ Should raise an error, as other is no ExperimentHistory. """
        histories0 = ExperimentHistories()
        histories0.setup(1, 1, 1, 1, 1)

        with self.assertRaises(AssertionError):
            histories0.stack_histories(list)

    def test_calc_hist_length_per_net_rounding(self):
        """ Should calculate the correct length of a history for one net.
        As one entry is generated in the last epoch, the history should have a length of one. """
        self.assertEqual(calc_hist_length_per_net(3, 2, 4), 1)


if __name__ == '__main__':
    unittest_main()
