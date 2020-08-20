from unittest import TestCase
from unittest import main as unittest_main

from torch import nn

from data import plotter_layer


class TestPlotterLayer(TestCase):
    """ Tests for the plotter_layer module.
    Call with 'python -m test.test_plotter_layer' from project root '~'.
    Call with 'python -m test_plotter_layer' from inside '~/test'. """

    def test_get_row_and_col_num_color(self):
        """ Should return 4 columns and 2 rows as they can hold 8 kernels. """
        weight_shape = (8, 5, 5, 3)

        num_cols, num_rows = plotter_layer.get_row_and_col_num(weight_shape, 4)
        self.assertEqual(4, num_cols)
        self.assertEqual(2, num_rows)

    def test_get_row_and_col_num_color_clip(self):
        """ Should return 5 columns and 4 rows as the last row is not completely filled. """
        weight_shape = (18, 5, 5, 3)

        num_cols, num_rows = plotter_layer.get_row_and_col_num(weight_shape, 5)
        self.assertEqual(5, num_cols)
        self.assertEqual(4, num_rows)

    def test_get_row_and_col_num_single(self):
        """ Should return 4 columns and 8 rows as each row holds all channels for one kernel. """
        weight_shape = (8, 5, 5, 4)

        num_cols, num_rows = plotter_layer.get_row_and_col_num(weight_shape, 4)
        self.assertEqual(4, num_cols)
        self.assertEqual(8, num_rows)

    def test_get_row_and_col_num_single_clip(self):
        """ Should return 5 columns and 7 rows as all channels for one kernel do not fit into one row. """
        weight_shape = (8, 5, 5, 4)

        num_cols, num_rows = plotter_layer.get_row_and_col_num(weight_shape, 5)
        self.assertEqual(5, num_cols)
        self.assertEqual(7, num_rows)

    def test_plot_kernels_color(self):
        """ Should run plot routine without errors. """
        conv_2d = nn.Conv2d(3, 2, kernel_size=3)
        plotter_layer.plot_kernels(conv_2d)

    def test_plot_kernels_single(self):
        """ Should run plot routine without errors. """
        conv_2d = nn.Conv2d(2, 2, kernel_size=3)
        plotter_layer.plot_kernels(conv_2d)

    def test_plot_conv(self):
        """ Should run plot routine without errors. """
        sequential = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.Tanh(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(4, 6, 3),
            nn.ReLU()
        )
        plotter_layer.plot_conv(sequential)

    def test_plot_fc(self):
        """ Should run plot routine without errors. """
        sequential = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 2),
            nn.ReLU()
        )
        plotter_layer.plot_fc(sequential)


if __name__ == '__main__':
    unittest_main()
