from unittest import TestCase
from unittest import main as unittest_main

from experiments.experiment_settings import get_settings, DatasetNames, ExperimentNames, ExperimentSettings, NetNames, \
    get_settings_lenet_toy


class TestExperimentSettings(TestCase):
    """ Tests for the experiment_settings module.
    Call with 'python -m test.test_experiment_settings' from project root '~'.
    Call with 'python -m test_experiment_settings' from inside '~/test'. """

    def test_get_value_list_from_experiment_names(self):
        """ ExperimentNames should generate a list of all values, i.e. strings of experiment names. """
        expected_list = ['lenet-mnist', 'conv2-cifar10', 'conv4-cifar10', 'conv6-cifar10']
        self.assertEqual(expected_list, ExperimentNames.get_value_list())

    def test_get_settings_for_lenet_mnist(self):
        """ Get results without errors and verify the most important attributes. """
        experiment_settings = get_settings(ExperimentNames.LENET_MNIST)

        self.assertIs(type(experiment_settings), ExperimentSettings)
        self.assertIs(experiment_settings.net, NetNames.LENET)
        self.assertIs(experiment_settings.dataset, DatasetNames.MNIST)

    def test_get_settings_for_conv2_cifar10(self):
        """ Get results without errors and verify the most important attributes. """
        experiment_settings = get_settings(ExperimentNames.CONV2_CIFAR10)

        self.assertIs(type(experiment_settings), ExperimentSettings)
        self.assertIs(experiment_settings.net, NetNames.CONV)
        self.assertIs(experiment_settings.dataset, DatasetNames.CIFAR10)

    def test_get_settings_for_conv4_cifar10(self):
        """ Get results without errors and verify the most important attributes. """
        experiment_settings = get_settings(ExperimentNames.CONV4_CIFAR10)

        self.assertIs(type(experiment_settings), ExperimentSettings)
        self.assertIs(experiment_settings.net, NetNames.CONV)
        self.assertIs(experiment_settings.dataset, DatasetNames.CIFAR10)

    def test_get_settings_for_conv6_cifar10(self):
        """ Get results without errors and verify the most important attributes. """
        experiment_settings = get_settings(ExperimentNames.CONV6_CIFAR10)

        self.assertIs(type(experiment_settings), ExperimentSettings)
        self.assertIs(experiment_settings.net, NetNames.CONV)
        self.assertIs(experiment_settings.dataset, DatasetNames.CIFAR10)

    def test_get_settings_for_lenet_toy(self):
        """ Get results without errors and verify the most important attributes. """
        experiment_settings = get_settings_lenet_toy()

        self.assertIs(type(experiment_settings), ExperimentSettings)
        self.assertIs(experiment_settings.net, NetNames.LENET)
        self.assertIs(experiment_settings.dataset, DatasetNames.TOY)

    def test_get_settings_should_raise_assertion_error_on_invalid_name(self):
        """ An assertion error should be thrown, because the given name is invalid. """
        with self.assertRaises(AssertionError):
            get_settings("This is an invalid experiment name for sure!")


if __name__ == '__main__':
    unittest_main()
