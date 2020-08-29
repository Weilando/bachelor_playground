from unittest import TestCase, main as unittest_main

from matplotlib.figure import Figure
from torch import nn

from data import plotter_layer


class TestPlotterLayer(TestCase):
    """ Tests for the plotter_layer module.
    Call with 'python -m test.test_plotter_layer' from project root '~'.
    Call with 'python -m test_plotter_layer' from inside '~/test'. """

    def test_plot_kernels(self):
        """ Should run plot routine without errors and generate one figure. """
        conv_2d = nn.Conv2d(2, 2, kernel_size=3)
        result_figure = plotter_layer.plot_kernels(conv_2d)
        self.assertIsInstance(result_figure, Figure)

    def test_plot_conv(self):
        """ Should run plot routine without errors and generate one figure per Conv2d. """
        sequential = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Tanh(), nn.MaxPool2d(3, 2), nn.Conv2d(4, 6, 3), nn.ReLU())

        result_figure_list = plotter_layer.plot_conv(sequential)

        self.assertEqual(2, len(result_figure_list))
        self.assertIsInstance(result_figure_list[0], Figure)
        self.assertIsInstance(result_figure_list[1], Figure)

    def test_plot_fc(self):
        """ Should run plot routine without errors and generate one figure. """
        sequential = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2), nn.ReLU())
        result_figure = plotter_layer.plot_fc(sequential)
        self.assertIsInstance(result_figure, Figure)


if __name__ == '__main__':
    unittest_main()
