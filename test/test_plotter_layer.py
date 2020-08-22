from unittest import TestCase, mock
from unittest import main as unittest_main

from torch import nn

from data import plotter_layer


class TestPlotterLayer(TestCase):
    """ Tests for the plotter_layer module.
    Call with 'python -m test.test_plotter_layer' from project root '~'.
    Call with 'python -m test_plotter_layer' from inside '~/test'. """

    def test_plot_kernels(self):
        """ Should run plot routine without errors. """
        conv_2d = nn.Conv2d(2, 2, kernel_size=3)
        with mock.patch('matplotlib.pyplot.show') as plot_mock:
            plotter_layer.plot_kernels(conv_2d)
            plot_mock.assert_called_once()

    def test_plot_conv(self):
        """ Should run plot routine without errors. """
        sequential = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.Tanh(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(4, 6, 3),
            nn.ReLU()
        )
        with mock.patch('matplotlib.pyplot.show') as plot_mock:
            plotter_layer.plot_conv(sequential)
            self.assertEqual(2, plot_mock.call_count)

    def test_plot_fc(self):
        """ Should run plot routine without errors. """
        sequential = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 2),
            nn.ReLU()
        )
        with mock.patch('matplotlib.pyplot.show') as plot_mock:
            plotter_layer.plot_fc(sequential)
            plot_mock.assert_called_once()


if __name__ == '__main__':
    unittest_main()
