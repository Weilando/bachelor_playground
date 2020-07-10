from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import numpy as np

import experiments.experiment_settings as experiment_settings
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_imp import ExperimentIMP


class TestExperiment(TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_experiment' from project root '~'.
    Call with 'python -m test_experiment' from inside '~/test'. """

    def test_experiment_lenet_mnist_should_init(self):
        """ Should setup IMP-Experiment with Lenet and MNIST without errors. """
        settings = experiment_settings.get_settings_lenet_mnist()
        settings.dataset = experiment_settings.DatasetNames.TOY_MNIST  # use toy set for speedup
        settings.net_count = 1  # use one net for speed up
        experiment = ExperimentIMP(settings)
        experiment.setup_experiment()

    def test_experiment_conv6_cifar10_should_init(self):
        """ Should setup IMP-Experiment with Conv-6 and CIFAR-10 without errors. """
        settings = experiment_settings.get_settings_conv6_cifar10()
        settings.dataset = experiment_settings.DatasetNames.TOY_CIFAR10  # use toy set for speedup
        settings.net_count = 1  # use one net for speed up
        experiment = ExperimentIMP(settings)
        experiment.setup_experiment()

    def test_perform_toy_lenet_experiment(self):
        """ Should run IMP-Experiment with small Lenet and toy-dataset without errors. """
        settings = experiment_settings.get_settings_lenet_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()

    def test_perform_toy_conv_experiment(self):
        """ Should run IMP-Experiment with small Conv and toy-dataset without errors. """
        settings = experiment_settings.get_settings_conv_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()

    def test_experiment_histories_are_equal(self):
        """ Should return True, because both ExperimentHistories contain equal arrays. """
        histories1 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))
        histories2 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), True)

    def test_experiment_histories_are_unequal(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        histories1 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2))
        histories2 = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        self.assertIs(ExperimentHistories.__eq__(histories1, histories2), False)

    def test_experiment_histories_error_on_invalid_type(self):
        """ Should return False, because both ExperimentHistories contain unequal arrays. """
        with self.assertRaises(AssertionError):
            ExperimentHistories.__eq__(ExperimentHistories(), dict())


if __name__ == '__main__':
    unittest_main()
