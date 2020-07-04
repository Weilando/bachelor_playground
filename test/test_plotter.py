import unittest

import numpy as np

from training import plotter


class Test_plotter(unittest.TestCase):
    """ Tests for the plotter module.
    Call with 'python -m test.test_plotter' from project root '~'.
    Call with 'python -m test_plotter' from inside '~/test'. """

    def test_calculate_correct_mean_and_y_error(self):
        """ The correct averages, maxima and minima should be calculated.
        The input has shape (3,2,4), thus all returned arrays need to have shape (2,4). """
        arr = np.array(
            [[[3, 12, 10, 7], [2, 63, 9, 50]], [[6, 4, 15, 14], [6, 42, 27, 75]], [[9, 8, 5, 21], [4, 21, 18, 25]]],
            dtype=int)
        expected_min = np.array([[3, 4, 5, 7], [2, 21, 9, 25]])
        expected_max = np.array([[9, 12, 15, 21], [6, 63, 15, 75]])
        expected_mean = np.array([[6, 8, 10, 14], [4, 42, 12, 50]])
        expected_shape = (2, 4)

        result_mean, result_min, result_max = plotter.get_means_and_y_errors(arr)

        self.assertTrue(expected_shape == result_mean.shape)
        self.assertTrue(expected_shape == result_min.shape)
        self.assertTrue(expected_shape == result_max.shape)
        self.assertTrue((expected_mean == result_mean).all)
        self.assertTrue((expected_min == result_min).all)
        self.assertTrue((expected_max == result_max).all)

    def test_calculate_correct_early_stop_iteration(self):
        """ The correct iteration for the early stopping criterion should be calculated.
        The input has shape (1,2,5), thus the result needs to have shape (1,2). """
        arr = np.array([[[5, 4, 3, 2, 1], [3, 2, 3, 2, 3]]], dtype=float)
        expected_iterations = np.array([[4, 1]])
        expected_shape = (1, 2)

        result_iterations = plotter.find_stop_iteration(arr)

        self.assertTrue(expected_shape == result_iterations.shape)
        self.assertTrue((expected_iterations == result_iterations).all)

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


if __name__ == '__main__':
    unittest.main()
