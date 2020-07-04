import unittest

from playground import should_override_epoch_count, should_override_prune_count, should_override_net_count


class Test_playground(unittest.TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_playground' from project root '~'.
    Call with 'python -m test_playground' from inside '~/test'. """
    def test_should_not_override_epoch_count_as_epochs_flag_not_set(self):
        """ Result should be false, because the 'epochs'-flag was not set. """
        self.assertIs(should_override_epoch_count(None), False)

    def test_should_override_epoch_count_as_epochs_is_valid(self):
        """ Result should be true, because the 'epochs'-flag was set with a valid value. """
        self.assertIs(should_override_epoch_count(1), True)

    def test_should_raise_exception_as_epochs_is_invalid(self):
        """ An assertion error should be thrown, because the 'epochs'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_epoch_count(0)

    def test_should_not_override_net_count_as_nets_flag_not_set(self):
        """ Result should be false, because the 'nets'-flag was not set. """
        self.assertIs(should_override_net_count(None), False)

    def test_should_override_net_count_as_nets_is_valid(self):
        """ Result should be true, because the 'nets'-flag was set with a valid value. """
        self.assertIs(should_override_net_count(1), True)

    def test_should_raise_exception_as_nets_is_invalid(self):
        """ An assertion error should be thrown, because the 'nets'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_net_count(0)

    def test_should_not_override_prune_count_as_prunes_flag_not_set(self):
        """ Result should be false, because the 'prunes'-flag was not set. """
        self.assertIs(should_override_prune_count(None), False)

    def test_should_override_prune_count_as_prunes_is_valid(self):
        """ Result should be true, because the 'prunes'-flag was set with a valid value. """
        self.assertIs(should_override_prune_count(0), True)

    def test_should_raise_exception_as_prunes_is_invalid(self):
        """ An assertion error should be thrown, because the 'prunes'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_prune_count(-1)


if __name__ == '__main__':
    unittest.main()
