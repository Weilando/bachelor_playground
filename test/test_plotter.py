from unittest import TestCase, main as unittest_main

import matplotlib.pyplot as plt
import numpy as np

from data import plotter


class TestPlotter(TestCase):
    """ Tests for the plotter module.
    Call with 'python -m test.test_plotter' from project root '~'.
    Call with 'python -m test_plotter' from inside '~/test'. """

    # generators
    def test_gen_iteration_space(self):
        """ Should generate the correct iteration space. """
        result_space = plotter.gen_iteration_space(arr=np.zeros(3), plot_step=2)
        np.testing.assert_array_equal(np.array([2, 4, 6]), result_space)

    def test_gen_labels_for_train_loss_and_iterations(self):
        """ Should generate the correct labels. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_labels_on_ax(ax, plotter.PlotType.TRAIN_LOSS, iteration=True)
        self.assertEqual(plotter.PlotType.TRAIN_LOSS, ax.get_ylabel())
        self.assertEqual("Iteration", ax.get_xlabel())

    def test_gen_labels_for_val_acc_and_sparsity(self):
        """ Should generate the correct labels. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_labels_on_ax(ax, plotter.PlotType.VAL_ACC, iteration=False)
        self.assertEqual(plotter.PlotType.VAL_ACC, ax.get_ylabel())
        self.assertEqual("Sparsity", ax.get_xlabel())

    def test_gen_title_for_test_acc_on_ax(self):
        """ Should generate the correct title. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_title_on_ax(ax, plotter.PlotType.TEST_ACC, early_stop=False)
        self.assertEqual(f"Average {plotter.PlotType.TEST_ACC}", ax.get_title())

    def test_gen_title_for_test_acc_with_early_stop_on_ax(self):
        """ Should generate the correct title with early-stop suffix. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.gen_title_on_ax(ax, plotter.PlotType.TEST_ACC, early_stop=True)
        self.assertEqual(f"Average {plotter.PlotType.TEST_ACC} at early-stop", ax.get_title())

    def test_setup_grids_and_y_lim_on_ax(self):
        """ Should setup grids and set the minimum of y-limit to zero. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.set_ylim(bottom=1, top=2)
        plotter.setup_grids_on_ax(ax, force_zero=True)
        self.assertEqual(0, ax.get_ylim()[0])

    def test_setup_grids_on_ax(self):
        """ Should setup grids and do not change y-limits. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.set_ylim(bottom=1, top=2)
        plotter.setup_grids_on_ax(ax, force_zero=False)
        self.assertEqual(1, ax.get_ylim()[0])

    def test_setup_labeling_on_ax(self):
        """ Should generate the right title and labels, and generate a legend. """
        plot_type = plotter.PlotType.TEST_ACC
        ax = plt.figure().subplots(1, 1, sharex=False)
        ax.plot([0], label='f(x)')  # plot with label to avoid empty legend

        plotter.setup_labeling_on_ax(ax, plot_type, iteration=False)

        self.assertEqual(f"Average {plot_type.value}", ax.get_title())
        self.assertEqual(f"{plot_type.value}", ax.get_ylabel())
        self.assertEqual("Sparsity", ax.get_xlabel())
        self.assertIsNotNone(ax.get_legend())

    def test_setup_early_stop_ax(self):
        """ Should setup log-scale correctly and invert x-axis. """
        ax = plt.figure().subplots(1, 1, sharex=False)
        plotter.setup_early_stop_ax(ax, False, log_step=4)
        self.assertEqual([1.0, 0.5, 0.25, 0.125], ax.get_xticks().tolist())
        self.assertIs(ax.xaxis_inverted().item(), True)

    # plots
    def test_plot_acc_at_early_stop_on_ax(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_acc_at_early_stop_on_ax(ax, hists, hists, sparsity, 'Name', plotter.PlotType.TEST_ACC)

    def test_plot_acc_at_early_stop_on_ax_with_random_hists(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 3, 2))
        hists_random = np.ones((4, 2, 2))
        sparsity = np.ones(3)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_acc_at_early_stop_on_ax(ax, hists, hists, sparsity, 'Name', plotter.PlotType.TEST_ACC,
                                             hists_random, hists_random, log_step=3)
        self.assertEqual(3, len(ax.get_xticks()))

    def test_plot_average_hists_on_ax(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_average_hists_on_ax(ax, hists, sparsity, 10, plotter.PlotType.TRAIN_LOSS)

    def test_plot_average_hists_on_ax_with_random_hists(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 3, 2))
        hists_random = np.ones((4, 2, 2))
        sparsity = np.ones(3)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_average_hists_on_ax(ax, hists, sparsity, 10, plotter.PlotType.TRAIN_LOSS, hists_random)

    def test_plot_early_stop_iterations_on_ax(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 2, 2))
        sparsity = np.ones(2)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_early_stop_iterations_on_ax(ax, hists, sparsity, 10, 'Name')

    def test_plot_early_stop_iterations_on_ax_with_random_hists(self):
        """ Should run plot routine without errors. """
        hists = np.ones((2, 3, 2))
        hists_random = np.ones((4, 2, 2))
        sparsity = np.ones(3)
        ax = plt.figure().subplots(1, 1)
        plotter.plot_early_stop_iterations_on_ax(ax, hists, sparsity, 10, 'Name', hists_random, log_step=2)
        self.assertEqual(2, len(ax.get_xticks()))


if __name__ == '__main__':
    unittest_main()
