import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training import plotter

class Test_Plotter(unittest.TestCase):
    """ Tests for the plotter module.
    Call with 'python -m test.test_plotter' from project root '~'.
    Call with 'python -m test_plotter' from inside '~/test'. """
    def test_time_smaller_than_minute_should_be_seconds(self):
        ''' If the given time is smaller than one minute, it should be returned as seconds with four digits. '''
        self.assertEqual("1.2222sec", plotter.format_time(1.22222))

    def test_time_smaller_than_minute_should_be_seconds2(self):
        ''' If the given time is smaller than one minute, it should be returned as seconds with four digits. '''
        self.assertEqual("59.1234sec", plotter.format_time(59.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds(self):
        ''' If the given time is bigger than one minute, it should be returned as minutes and seconds without digits. '''
        self.assertEqual("1:00min", plotter.format_time(60.12345))

    def test_time_bigger_than_minute_should_be_minutes_and_seconds2(self):
        ''' If the given time is bigger than one minute, it should be returned as minutes and seconds without digits. '''
        self.assertEqual("2:18min", plotter.format_time(138.44444))

if __name__ == '__main__':
    unittest.main()
