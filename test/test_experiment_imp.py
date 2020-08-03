import glob
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import experiments.experiment_settings as experiment_settings
from experiments.experiment_imp import ExperimentIMP


class TestExperimentIMP(TestCase):
    """ Tests for the experiment_imp module.
    Call with 'python -m test.test_experiment_imp' from project root '~'.
    Call with 'python -m test_experiment_imp' from inside '~/test'. """

    def test_perform_toy_lenet_experiment(self):
        """ Should run IMP-Experiment with small Lenet and toy-dataset without errors. """
        settings = experiment_settings.get_settings_lenet_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))), 1)

    def test_perform_toy_conv_experiment(self):
        """ Should run IMP-Experiment with small Conv and toy-dataset without errors. """
        settings = experiment_settings.get_settings_conv_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))), 1)

    def test_raise_error_on_invalid_dataset(self):
        """ Should raise an assertion error, because the given dataset-name is invalid. """
        settings = experiment_settings.get_settings_conv_toy()
        with self.assertRaises(AssertionError):
            settings.dataset = 'Invalid dataset name'
            ExperimentIMP(settings).load_data_and_setup_trainer()

    def test_save_early_stop_checkpoints(self):
        """ Should save checkpoints from early-stopping. """
        settings = experiment_settings.get_settings_lenet_toy()
        settings.save_early_stop = True
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, '*-early-stop[0,1].pth'))), 2)

    def test_do_not_save_early_stop_checkpoints(self):
        """ Should not save checkpoints from early-stopping. """
        settings = experiment_settings.get_settings_lenet_toy()
        settings.save_early_stop = False
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(settings, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(glob.glob(os.path.join(tmp_dir_name, '*-early-stop*.pth')), [])


if __name__ == '__main__':
    unittest_main()
