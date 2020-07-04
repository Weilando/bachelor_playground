import unittest

from nets import plan_check


class Test_plan_check(unittest.TestCase):
    """ Tests for the plan_check module.
    Call with 'python -m test.test_plan_check' from project root '~'.
    Call with 'python -m test_plan_check' from inside '~/test'. """
    def test_one_is_numerical_spec(self):
        """ '1' should be classified as numerical spec. """
        input_string = '1'
        self.assertTrue(plan_check.is_numerical_spec(input_string))

    def test_42_is_numerical_spec(self):
        """ '42' should be classified as numerical spec. """
        input_string = '42'
        self.assertTrue(plan_check.is_numerical_spec(input_string))

    def test_zero_is_no_numerical_spec(self):
        """ '0' should not be classified as numerical spec. """
        input_string = '0'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_negative_one_is_no_numerical_spec(self):
        """ '-1' should not be classified as numerical spec. """
        input_string = '-1'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_007_is_no_numerical_spec(self):
        """ '007' should not be classified as numerical spec, because it has leading zeros. """
        input_string = '007'
        self.assertFalse(plan_check.is_numerical_spec(input_string))

    def test_2B_is_batchnorm_spec(self):
        """ '2B' should be classified as numerical spec. """
        input_string = '2B'
        self.assertTrue(plan_check.is_batchnorm_spec(input_string))

    def test_42B_is_batchnorm_spec(self):
        """ '42B' should be classified as numerical spec. """
        input_string = '42B'
        self.assertTrue(plan_check.is_batchnorm_spec(input_string))

    def test_2BB_is_no_batchnorm_spec(self):
        """ '2BB' should not be classified as numerical spec. """
        input_string = '2BB'
        self.assertFalse(plan_check.is_batchnorm_spec(input_string))

    def test_2_is_no_batchnorm_spec(self):
        """ '2' should not be classified as numerical spec. """
        input_string = '2'
        self.assertFalse(plan_check.is_batchnorm_spec(input_string))

    def test_0B_is_no_batchnorm_spec(self):
        """ '0B' should not be classified as numerical spec. """
        input_string = '0B'
        self.assertFalse(plan_check.is_batchnorm_spec(input_string))

    def test_neg_1B_is_no_batchnorm_spec(self):
        """ '-1B' should not be classified as numerical spec. """
        input_string = '-1B'
        self.assertFalse(plan_check.is_batchnorm_spec(input_string))


if __name__ == '__main__':
    unittest.main()
