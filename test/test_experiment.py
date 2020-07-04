import unittest

import experiments.experiment_settings as experiment_settings
from experiments.experiment_conv_cifar10 import ExperimentConvCIFAR10
from experiments.experiment_lenet_mnist import ExperimentLenetMNIST


class Test_experiment(unittest.TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_experiment' from project root '~'.
    Call with 'python -m test_experiment' from inside '~/test'. """
    def test_experiment_lenet_mnist_should_init(self):
        settings = experiment_settings.get_settings_lenet_mnist()
        settings.net_count = 1
        experiment = ExperimentLenetMNIST(settings)
        experiment.setup_experiment()

    def test_experiment_conv_cifar10_should_init(self):
        settings = experiment_settings.get_settings_conv2_cifar10()
        settings.net_count = 1
        experiment = ExperimentConvCIFAR10(settings)
        experiment.setup_experiment()


if __name__ == '__main__':
    unittest.main()
