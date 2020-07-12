from unittest import TestCase
from unittest import main as unittest_main

import matplotlib.pyplot as plt
import numpy as np

from data import plotter


class TestPlotter(TestCase):
    """ Tests for the plotter module.
    Call with 'python -m test.test_plotter' from project root '~'.
    Call with 'python -m test_plotter' from inside '~/test'. """

    # evaluation
    def test_find_early_stop_indices(self):
        """ Should find the correct indices for the early-stopping criterion.
        The input has shape (1,2,5), thus the result needs to have shape (1,2). """
        arr = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        expected_iterations = np.array([[4, 2]])

        result_iterations = plotter.find_early_stop_indices(arr)
        np.testing.assert_array_equal(expected_iterations, result_iterations)

    def test_get_values_at_stop_iteration(self):
        """ Should find the corresponding values for the early-stopping indices.
        The original array has shape (1,2,5) and indices have shape (1,2).
        Thus the result needs to have shape (1,2,1). """
        hists = np.array([[[5, 4, 3, 2, 1], [9, 8, 7, 8, 9]]], dtype=float)
        stop_indices = np.array([[4, 2]])

        expected_values = np.array([[[1], [7]]])

        result_values = plotter.get_values_at_stop_iteration(stop_indices, hists)
        np.testing.assert_array_equal(expected_values, result_values)

    def test_time_smaller_than_minute_should_be_seconds(self):
        """ If the given time is smaller than one minute, it should be returned as seconds with four digits. """
        self.assertEqual("1.2222sec", plotter.format_time(1.22222))

    def test_time_smaller_than_minute_should_be_seconds2(self):
        """ If the given time is smaller than one minute, it should be returned as seconds with four digits. """
        self.assertEqual("59.1234sec", plotter.format_time(59.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds(self):
        """ If the given time is bigger than a minute, it should be returned as minutes and seconds without digits. """
        self.assertEqual("1:00min", plotter.format_time(60.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds2(self):
        """ If the given time is bigger than a minute, it should be returned as minutes and seconds without digits. """
        self.assertEqual("2:18min", plotter.format_time(138.44444))

    def test_calculate_correct_mean_and_y_error(self):
        """ Should calculate the correct averages, maxima and minima.
        The input has shape (3,2,4), thus all returned arrays need to have shape (2,4).
        Negative and positive error-bars are equal in this case, as it holds min=[[3, 4, 5, 7], [2, 21, 9, 25]] and
        max=[[9, 12, 15, 21], [6, 63, 15, 75]]. """
        arr = np.array(
            [[[3, 12, 10, 7], [2, 63, 9, 50]], [[6, 4, 15, 14], [6, 42, 27, 75]], [[9, 8, 5, 21], [4, 21, 18, 25]]],
            dtype=int)
        expected_y_error = np.array([[3, 4, 5, 7], [2, 21, 9, 25]])
        expected_mean = np.array([[6, 8, 10, 14], [4, 42, 18, 50]])

        result_mean, result_neg_y_error, result_pos_y_error = plotter.get_means_and_y_errors(arr)

        np.testing.assert_array_equal(expected_mean, result_mean)
        np.testing.assert_array_equal(expected_y_error, result_neg_y_error)
        np.testing.assert_array_equal(expected_y_error, result_pos_y_error)

    def test_scale_early_stop_indices_to_iterations(self):
        """ Should scale the given indices correctly, i.e. start counting by plot_step and count up with plot_step. """
        indices = np.array([[4, 0]])
        expected_iterations = np.array([[50, 10]])
        result_iterations = plotter.scale_early_stop_indices_to_iterations(indices, 10)
        np.testing.assert_array_equal(expected_iterations, result_iterations)

    # generators
    def test_gen_iteration_space(self):
        """ Should generate the correct iteration space. """
        arr = np.zeros(3)
        plot_step = 2

        expected_space = np.array([2, 4, 6])
        result_space = plotter.gen_iteration_space(arr, plot_step)
        np.testing.assert_array_equal(expected_space, result_space)

    def test_gen_labels_for_train_loss_and_iterations(self):
        """ Should generate the correct labels. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_labels_on_ax(ax, plotter.PlotType.TRAIN_LOSS, iteration=True)
        self.assertEqual(ax.get_ylabel(), plotter.PlotType.TRAIN_LOSS)
        self.assertEqual(ax.get_xlabel(), "Iteration")

    def test_gen_labels_for_val_acc_and_sparsity(self):
        """ Should generate the correct labels. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_labels_on_ax(ax, plotter.PlotType.VAL_ACC, iteration=False)
        self.assertEqual(ax.get_ylabel(), plotter.PlotType.VAL_ACC)
        self.assertEqual(ax.get_xlabel(), "Sparsity")

    def test_gen_new_single_ax(self):
        """ Should return a single axes object. """
        ax = plotter.gen_new_single_ax()
        self.assertIsInstance(ax, plt.Axes)

    def test_gen_title_for_test_acc_on_ax(self):
        """ Should generate the correct title. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_title_on_ax(ax, 3, plotter.PlotType.TEST_ACC, early_stop=False)
        self.assertEqual(ax.get_title(), f"Average {plotter.PlotType.TEST_ACC} for 3 pruned Networks")

    def test_gen_title_for_test_acc_with_early_stop_on_ax(self):
        """ Should generate the correct title with early-stop suffix. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_title_on_ax(ax, 3, plotter.PlotType.TEST_ACC, early_stop=True)
        self.assertEqual(ax.get_title(), f"Average {plotter.PlotType.TEST_ACC} for 3 pruned Networks at early-stop")

    def test_setup_grids_and_y_lim_on_ax(self):
        """ Should setup grids and set the minimum of y-limit to zero. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.set_ylim(bottom=1, top=2)
        plotter.setup_grids_on_ax(ax, force_zero=True)
        self.assertEqual(ax.get_ylim()[0], 0)

    def test_setup_grids_on_ax(self):
        """ Should setup grids and do not change y-limits. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.set_ylim(bottom=1, top=2)
        plotter.setup_grids_on_ax(ax, force_zero=False)
        self.assertEqual(ax.get_ylim()[0], 1)

    def test_setup_labeling_on_ax(self):
        """ Should generate the right title and labels, and generate a legend. """
        plot_type = plotter.PlotType.TEST_ACC
        net_count = 3
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.plot([0], label='f(x)')  # plot with label to avoid empty legend

        plotter.setup_labeling_on_ax(ax, net_count, plot_type, iteration=False)

        self.assertEqual(ax.get_title(), f"Average {plot_type.value} for {net_count} pruned Networks")
        self.assertEqual(ax.get_ylabel(), f"{plot_type.value}")
        self.assertEqual(ax.get_xlabel(), "Sparsity")
        self.assertIsNotNone(ax.get_legend())

    # plots
    def test_plot_average_hists(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        plotter.plot_average_hists(hists, sparsity, 10, plotter.PlotType.TRAIN_LOSS)

    def test_plot_two_average_hists(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        plotter.plot_two_average_hists(hists, hists, sparsity, 10, plotter.PlotType.VAL_ACC, plotter.PlotType.VAL_LOSS)

    def test_plot_acc_at_early_stopping(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        plotter.plot_acc_at_early_stopping(hists, hists, sparsity, plotter.PlotType.TEST_ACC)

    def test_plot_early_stop_iterations(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        plotter.plot_early_stop_iterations(hists, sparsity, 10)


if __name__ == '__main__':
    unittest_main()
