import glob
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import experiments.experiment_specs as experiment_specs
from experiments.experiment_osp import ExperimentOSP


class TestExperimentOSP(TestCase):
    """ Tests for the experiment_osp module.
    Call with 'python -m test.test_experiment_osp' from project root '~'.
    Call with 'python -m test_experiment_osp' from inside '~/test'. """

    def test_perform_toy_lenet_osp_experiment(self):
        """ Should run OSP-Experiment with small Lenet and toy-dataset without errors. """
        specs = experiment_specs.get_specs_lenet_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentOSP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))), 1)

    def test_perform_toy_conv_osp_experiment(self):
        """ Should run OSP-Experiment with small Conv and toy-dataset without errors. """
        specs = experiment_specs.get_specs_conv_toy()
        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            experiment = ExperimentOSP(specs, tmp_dir_name)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))), 1)


if __name__ == '__main__':
    unittest_main()
