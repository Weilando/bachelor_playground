from unittest import TestCase
from unittest import main as unittest_main

from torch import nn

from experiments.experiment_specs import NetNames
from nets import plan_check
from nets.plan_check import get_activation


class TestPlanCheck(TestCase):
    """ Tests for the plan_check module.
    Call with 'python -m test.test_plan_check' from project root '~'.
    Call with 'python -m test_plan_check' from inside '~/test'. """

    def test_one_is_numerical_spec(self):
        """ Should classify '1' as numerical spec. """
        self.assertIs(plan_check.is_numerical_spec('1'), True)

    def test_42_is_numerical_spec(self):
        """ Should classify '42' as numerical spec. """
        self.assertIs(plan_check.is_numerical_spec('42'), True)

    def test_zero_is_no_numerical_spec(self):
        """ Should not classify '0' as numerical spec. """
        self.assertIs(plan_check.is_numerical_spec('0'), False)

    def test_negative_one_is_no_numerical_spec(self):
        """ Should not classify '-1' as numerical spec. """
        self.assertIs(plan_check.is_numerical_spec('-1'), False)

    def test_007_is_no_numerical_spec(self):
        """ Should not classify '007' as numerical spec, because it has leading zeros. """
        self.assertIs(plan_check.is_numerical_spec('007'), False)

    def test_2B_is_batch_norm_spec(self):
        """ Should classify '2B' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('2B'), True)

    def test_42B_is_batch_norm_spec(self):
        """ Should classify '42B' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('42B'), True)

    def test_2BB_is_no_batch_norm_spec(self):
        """ Should not classify '2BB' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('2BB'), False)

    def test_2_is_no_batch_norm_spec(self):
        """ Should not classify '2' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('2'), False)

    def test_0B_is_no_batch_norm_spec(self):
        """ Should not classify '0B' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('0B'), False)

    def test_neg_1B_is_no_batch_norm_spec(self):
        """ Should not classify '-1B' as numerical spec. """
        self.assertIs(plan_check.is_batch_norm_spec('-1B'), False)

    def test_get_number_from_numerical_spec_int(self):
        """ Should return number from numerical spec, i.e. return it, as it is an int. """
        self.assertEqual(42, plan_check.get_number_from_numerical_spec(42))

    def test_get_number_from_numerical_spec_string(self):
        """ Should return number from numerical spec, i.e. parse it, as it is a str. """
        self.assertEqual(42, plan_check.get_number_from_numerical_spec('42'))

    def test_get_number_from_batch_norm_spec(self):
        """ Should return number from batch-norm spec. """
        self.assertEqual(42, plan_check.get_number_from_batch_norm_spec('42B'))

    def test_should_raise_error_on_failed_numerical_parse(self):
        """ Should raise a ValueError, as the string cannot be parsed into an int. """
        with self.assertRaises(ValueError):
            plan_check.get_number_from_numerical_spec('not a number')

    def test_should_raise_error_on_failed_batch_norm_parse(self):
        """ Should raise a ValueError, as the string cannot be parsed into an int. """
        with self.assertRaises(ValueError):
            plan_check.get_number_from_batch_norm_spec('not a numberB')

    def test_get_activation_lenet(self):
        """ Should return tanh for Lenet. """
        self.assertIsInstance(get_activation(NetNames.LENET), nn.Tanh)

    def test_get_activation_conv(self):
        """ Should return ReLU for Conv. """
        self.assertIsInstance(get_activation(NetNames.CONV), nn.ReLU)


if __name__ == '__main__':
    unittest_main()
