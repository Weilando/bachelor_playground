import glob
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import experiments.experiment_specs as experiment_specs
from experiments.experiment_imp import ExperimentIMP


class TestExperimentIMP(TestCase):
    """ Tests for the experiment_imp module.
    Call with 'python -m test.test_experiment_imp' from project root '~'.
    Call with 'python -m test_experiment_imp' from inside '~/test'. """

    def test_perform_toy_lenet_experiment(self):
        """ Should run IMP-Experiment with small Lenet and toy-dataset without errors. """
        specs = experiment_specs.get_specs_lenet_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(1, len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))), )

    def test_perform_toy_conv_experiment(self):
        """ Should run IMP-Experiment with small Conv and toy-dataset without errors. """
        specs = experiment_specs.get_specs_conv_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(1, len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))))

    def test_raise_error_on_invalid_dataset(self):
        """ Should raise an assertion error, because the given dataset-name is invalid. """
        specs = experiment_specs.get_specs_conv_toy()
        with self.assertRaises(AssertionError):
            specs.dataset = 'Invalid dataset name'
            ExperimentIMP(specs).load_data_and_setup_trainer()

    def test_save_early_stop_checkpoints(self):
        """ Should save checkpoints from early-stopping. """
        specs = experiment_specs.get_specs_lenet_toy()
        specs.save_early_stop = True
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(2, len(glob.glob(os.path.join(tmp_dir_name, '*-early-stop[0,1].pth'))))

    def test_do_not_save_early_stop_checkpoints(self):
        """ Should not save checkpoints from early-stopping. """
        specs = experiment_specs.get_specs_lenet_toy()
        specs.save_early_stop = False
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentIMP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual([], glob.glob(os.path.join(tmp_dir_name, '*-early-stop*.pth')))


if __name__ == '__main__':
    unittest_main()
