from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import experiments.experiment_settings as experiment_settings
from experiments.experiment_imp import ExperimentIMP


class TestExperiment(TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_experiment' from project root '~'.
    Call with 'python -m test_experiment' from inside '~/test'. """

    def test_experiment_lenet_mnist_should_init(self):
        """ Should setup IMP-Experiment with Lenet and MNIST without errors. """
        settings = experiment_settings.get_settings_lenet_mnist()
        settings.net_count = 1
        experiment = ExperimentIMP(settings)
        experiment.setup_experiment()

    def test_experiment_conv6_cifar10_should_init(self):
        """ Should setup IMP-Experiment with Conv-6 and CIFAR-10 without errors. """
        settings = experiment_settings.get_settings_conv6_cifar10()
        settings.net_count = 1
        experiment = ExperimentIMP(settings)
        experiment.setup_experiment()

    def test_perform_toy_experiment(self):
        """ Should run IMP-Experiment with small Lenet and toy-dataset without errors. """
        settings = experiment_settings.get_settings_lenet_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()


if __name__ == '__main__':
    unittest_main()
