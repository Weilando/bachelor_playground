from unittest import TestCase
from unittest import main as unittest_main

from nets import plan_check


class TestPlanCheck(TestCase):
    """ Tests for the plan_check module.
    Call with 'python -m test.test_plan_check' from project root '~'.
    Call with 'python -m test_plan_check' from inside '~/test'. """

    def test_one_is_numerical_spec(self):
        """ Should classify '1' as numerical spec. """
        input_string = '1'
        self.assertTrue(plan_check.is_numerical_spec(input_string))

    def test_42_is_numerical_spec(self):
        """ Should classify '42' as numerical spec. """
        input_string = '42'
        self.assertTrue(plan_check.is_numerical_spec(input_string))

    def test_zero_is_no_numerical_spec(self):
        """ Should not classify '0' as numerical spec. """
        input_string = '0'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_negative_one_is_no_numerical_spec(self):
        """ Should not classify '-1' as numerical spec. """
        input_string = '-1'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_007_is_no_numerical_spec(self):
        """ Should not classify '007' as numerical spec, because it has leading zeros. """
        input_string = '007'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_2B_is_batch_norm_spec(self):
        """ Should classify '2B' as numerical spec. """
        input_string = '2B'
        self.assertTrue(plan_check.is_batch_norm_spec(input_string))

    def test_42B_is_batch_norm_spec(self):
        """ Should classify '42B' as numerical spec. """
        input_string = '42B'
        self.assertTrue(plan_check.is_batch_norm_spec(input_string))

    def test_2BB_is_no_batch_norm_spec(self):
        """ Should not classify '2BB' as numerical spec. """
        input_string = '2BB'
        self.assertFalse(plan_check.is_batch_norm_spec(input_string))

    def test_2_is_no_batch_norm_spec(self):
        """ Should not classify '2' as numerical spec. """
        input_string = '2'
        self.assertFalse(plan_check.is_batch_norm_spec(input_string))

    def test_0B_is_no_batch_norm_spec(self):
        """ Should not classify '0B' as numerical spec. """
        input_string = '0B'
        self.assertFalse(plan_check.is_batch_norm_spec(input_string))

    def test_neg_1B_is_no_batch_norm_spec(self):
        """ Should not classify '-1B' as numerical spec. """
        input_string = '-1B'
        self.assertFalse(plan_check.is_batch_norm_spec(input_string))


if __name__ == '__main__':
    unittest_main()
