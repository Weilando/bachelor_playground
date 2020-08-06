from io import StringIO
from unittest import TestCase
from unittest import main as unittest_main

import sys

from experiments.experiment_specs import VerbosityLevel
from training import logger


class TestLogger(TestCase):
    """ Tests for the logger module.
    Call with 'python -m test.test_logger' from project root '~'.
    Call with 'python -m test_logger' from inside '~/test'. """

    def test_no_medium_print_for_level_silent(self):
        """ Should not print anything, because the verbosity_level is silent. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_from_medium(VerbosityLevel.SILENT, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "")

    def test_medium_print_for_level_medium(self):
        """ Should print the message, because the verbosity_level is medium. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_from_medium(VerbosityLevel.MEDIUM, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "Some message\n")

    def test_medium_print_for_level_detailed(self):
        """ Should print the message, because the verbosity_level is detailed. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_from_medium(VerbosityLevel.DETAILED, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "Some message\n")

    def test_no_detailed_print_for_level_silent(self):
        """ Should not print anything, because the verbosity_level is silent. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_detailed_only(VerbosityLevel.SILENT, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "")

    def test_no_detailed_print_for_level_medium(self):
        """ Should not print anything, because the verbosity_level is medium. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_detailed_only(VerbosityLevel.MEDIUM, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "")

    def test_no_detailed_print_for_level_detailed(self):
        """ Should print message, because the verbosity_level is detailed. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.log_detailed_only(VerbosityLevel.DETAILED, "Some message")
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "Some message\n")

    def test_print_message_and_append_new_line(self):
        """ Should print message and append a new line. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.print_message("Some message", True)
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "Some message\n")

    def test_print_message_and_append_no_new_line(self):
        """ Should print message and do not append a new line. """
        with StringIO() as interception:
            old_stdout = sys.stdout
            sys.stdout = interception
            logger.print_message("Some message", False)
            sys.stdout = old_stdout

            self.assertEqual(interception.getvalue(), "Some message")


if __name__ == '__main__':
    unittest_main()
