from unittest import TestCase, main as unittest_main

import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from torch import nn

from data import plotter_evaluation


class TestPlotterEvaluation(TestCase):
    """ Tests for the plotter_evaluation module.
    Call with 'python -m test.test_plotter_evaluation' from project root '~'.
    Call with 'python -m test_plotter_evaluation' from inside '~/test'. """

    def test_find_early_stop_indices(self):
        """ Should find the correct indices for the early-stopping criterion.
        The input has shape (1,2,5), thus the result needs to have shape (1,2). """
        loss = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        result_indices = plotter_evaluation.find_early_stop_indices(loss)
        np.testing.assert_array_equal(np.array([[4, 2]]), result_indices)

    def test_find_early_stop_iterations(self):
        """ Should find the correct iterations for the early-stopping iterations.
        The input has shape (1,2,5), thus the result needs to have shape (1,2). """
        loss = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        result_iterations = plotter_evaluation.find_early_stop_iterations(loss, 10)
        np.testing.assert_array_equal(np.array([[50, 30]]), result_iterations)  # indices are 4 and 2

    def test_find_acc_at_early_stop_indices(self):
        """ Should find the correct accuracies for the early-stopping iterations.
        The inputs have shape (1,2,5), thus the result needs to have shape (1,2,1). """
        loss = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        accuracies = np.array([[[0.5, 0.4, 0.3, 0.2, 0.1], [0.9, 0.8, 0.7, 0.8, 0.9]]], dtype=float)
        result_accuracies = plotter_evaluation.find_acc_at_early_stop_indices(loss, accuracies)
        np.testing.assert_array_equal(np.array([[[0.1], [0.7]]]), result_accuracies)  # indices are 4 and 2

    def test_time_smaller_than_minute_should_be_seconds(self):
        """ If the given time is smaller than one minute, it should be returned as seconds with four digits. """
        self.assertEqual("1.2222sec", plotter_evaluation.format_time(1.22222))

    def test_time_smaller_than_minute_should_be_seconds2(self):
        """ If the given time is smaller than one minute, it should be returned as seconds with four digits. """
        self.assertEqual("59.1234sec", plotter_evaluation.format_time(59.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds(self):
        """ If the given time is bigger than a minute, it should be returned as minutes and seconds without digits. """
        self.assertEqual("1:00min", plotter_evaluation.format_time(60.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds2(self):
        """ If the given time is bigger than a minute, it should be returned as minutes and seconds without digits. """
        self.assertEqual("2:18min", plotter_evaluation.format_time(138.44444))

    def test_get_mean_and_y_errors(self):
        """ Should calculate the correct averages, maxima and minima.
        The input has shape (3,2,4), thus all returned arrays need to have shape (2,4).
        Negative and positive error-bars are equal in this case, as it holds min=[[3, 4, 5, 7], [2, 21, 9, 25]] and
        max=[[9, 12, 15, 21], [6, 63, 15, 75]]. """
        arr = np.array(
            [[[3, 12, 10, 7], [2, 63, 9, 50]], [[6, 4, 15, 14], [6, 42, 27, 75]], [[9, 8, 5, 21], [4, 21, 18, 25]]],
            dtype=int)
        expected_y_error = np.array([[3, 4, 5, 7], [2, 21, 9, 25]])

        result_mean, result_neg_y_error, result_pos_y_error = plotter_evaluation.get_means_and_y_errors(arr)

        np.testing.assert_array_equal(np.array([[6, 8, 10, 14], [4, 42, 18, 50]]), result_mean)
        np.testing.assert_array_equal(expected_y_error, result_neg_y_error)
        np.testing.assert_array_equal(expected_y_error, result_pos_y_error)

    def test_get_norm_for_sequence_zero_center(self):
        """ Should find a TwoSlopeNorm-object with correct min and max of all weights from all layers in 'seq'. """
        torch.manual_seed(123)
        seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Conv2d(2, 2, 2))

        norm = plotter_evaluation.get_norm_for_sequential(seq)

        self.assertIsInstance(norm, TwoSlopeNorm)
        self.assertAlmostEqual(0.0, norm.vcenter)  # colormap becomes zero centered
        self.assertAlmostEqual(-0.35119062662124634, norm.vmin)
        self.assertAlmostEqual(0.26665955781936646, norm.vmax)

    def test_get_norm_for_sequence(self):
        """ Should find a Normalize-object with correct min and max of all weights from all layers in 'seq'. """
        seq = nn.Sequential(nn.Linear(2, 2))
        seq[0].weight.data = torch.eye(2)  # plot of identity matrix is not zero centered

        norm = plotter_evaluation.get_norm_for_sequential(seq)

        self.assertIsInstance(norm, Normalize)
        self.assertAlmostEqual(0.0, norm.vmin)
        self.assertAlmostEqual(1.0, norm.vmax)

    def test_get_row_and_col_num(self):
        """ Should return 4 columns and 8 rows as each row holds all channels for one kernel. """
        num_cols, num_rows = plotter_evaluation.get_row_and_col_num(weight_shape=(8, 4, 5, 5), num_cols=4)
        self.assertEqual(4, num_cols)
        self.assertEqual(8, num_rows)

    def test_get_row_and_col_num_clip(self):
        """ Should return 5 columns and 7 rows as all channels for one kernel do not fit into one row. """
        num_cols, num_rows = plotter_evaluation.get_row_and_col_num(weight_shape=(8, 4, 5, 5), num_cols=5)
        self.assertEqual(5, num_cols)
        self.assertEqual(7, num_rows)

    def test_get_values_at_stop_iteration(self):
        """ Should find the corresponding values for the early-stopping indices.
        The original array has shape (1,2,5) and indices have shape (1,2).
        Thus the result needs to have shape (1,2,1). """
        hists = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        stop_indices = np.array([[4, 2]])
        result_values = plotter_evaluation.get_values_at_stop_iteration(stop_indices, hists)
        np.testing.assert_array_equal(np.array([[[1], [7]]]), result_values)

    def test_scale_early_stop_indices_to_iterations(self):
        """ Should scale the given indices correctly, i.e. start counting by plot_step and count up with plot_step. """
        result_iterations = plotter_evaluation.scale_early_stop_indices_to_iterations(np.array([[4, 0]]), plot_step=10)
        np.testing.assert_array_equal(np.array([[50, 10]]), result_iterations)


if __name__ == '__main__':
    unittest_main()
