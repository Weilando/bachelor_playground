from unittest import TestCase
from unittest import main as unittest_main

from experiments.experiment_specs import get_specs, DatasetNames, ExperimentPresetNames, ExperimentSpecs, NetNames, \
    get_specs_lenet_toy, get_specs_conv_toy


class TestExperimentSpecs(TestCase):
    """ Tests for the experiment_specs module.
    Call with 'python -m test.test_experiment_specs' from project root '~'.
    Call with 'python -m test_experiment_specs' from inside '~/test'. """

    def test_get_value_list_from_experiment_names(self):
        """ Should generate a list of all values, i.e. strings of experiment names. """
        expected_list = ['lenet-mnist', 'conv2-cifar10', 'conv4-cifar10', 'conv6-cifar10']
        self.assertEqual(expected_list, ExperimentPresetNames.get_value_list())

    def test_get_specs_for_lenet_mnist(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs(ExperimentPresetNames.LENET_MNIST)

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.LENET)
        self.assertIs(experiment_specs.dataset, DatasetNames.MNIST)

    def test_get_specs_for_conv2_cifar10(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs(ExperimentPresetNames.CONV2_CIFAR10)

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.CONV)
        self.assertIs(experiment_specs.dataset, DatasetNames.CIFAR10)

    def test_get_specs_for_conv4_cifar10(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs(ExperimentPresetNames.CONV4_CIFAR10)

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.CONV)
        self.assertIs(experiment_specs.dataset, DatasetNames.CIFAR10)

    def test_get_specs_for_conv6_cifar10(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs(ExperimentPresetNames.CONV6_CIFAR10)

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.CONV)
        self.assertIs(experiment_specs.dataset, DatasetNames.CIFAR10)

    def test_get_specs_for_lenet_toy(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs_lenet_toy()

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.LENET)
        self.assertIs(experiment_specs.dataset, DatasetNames.TOY_MNIST)

    def test_get_specs_for_conv_toy(self):
        """ Should get specs without errors and verify the most important attributes. """
        experiment_specs = get_specs_conv_toy()

        self.assertIs(type(experiment_specs), ExperimentSpecs)
        self.assertIs(experiment_specs.net, NetNames.CONV)
        self.assertIs(experiment_specs.dataset, DatasetNames.TOY_CIFAR10)

    def test_get_specs_should_raise_assertion_error_on_invalid_name(self):
        """ Should raise an assertion error, because the given name is invalid. """
        with self.assertRaises(AssertionError):
            get_specs("Invalid experiment name")


if __name__ == '__main__':
    unittest_main()
