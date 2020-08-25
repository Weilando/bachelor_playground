import glob
import os
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from unittest import main as unittest_main

import numpy as np
import torch

import experiments.experiment_specs as experiment_specs
from experiments.experiment_osp import ExperimentOSP
from test.fake_data_loaders import generate_fake_mnist_data_loaders, generate_fake_cifar10_data_loaders
from test.fake_experiment_specs import get_specs_lenet_toy, get_specs_conv_toy


class TestExperimentOSP(TestCase):
    """ Tests for the experiment_osp module.
    Call with 'python -m test.test_experiment_osp' from project root '~'.
    Call with 'python -m test_experiment_osp' from inside '~/test'. """

    @classmethod
    def setUpClass(cls):
        cls.fake_mnist_data_loaders = generate_fake_mnist_data_loaders()
        cls.fake_cifar10_data_loaders = generate_fake_cifar10_data_loaders()

    def test_perform_toy_lenet_osp_experiment(self):
        """ Should run OSP-Experiment with small Lenet and toy-dataset without errors. """
        specs = get_specs_lenet_toy()
        specs.experiment_name = experiment_specs.ExperimentNames.OSP
        with mock.patch('experiments.experiment.get_mnist_data_loaders', return_value=self.fake_mnist_data_loaders):
            with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
                experiment = ExperimentOSP(specs, tmp_dir_name)
                experiment.run_experiment()
                self.assertEqual(1, len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))))

    def test_perform_toy_conv_osp_experiment(self):
        """ Should run OSP-Experiment with small Conv and toy-dataset without errors. """
        specs = get_specs_conv_toy()
        specs.experiment_name = experiment_specs.ExperimentNames.OSP
        with mock.patch('experiments.experiment.get_cifar10_data_loaders', return_value=self.fake_cifar10_data_loaders):
            with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
                experiment = ExperimentOSP(specs, tmp_dir_name)
                experiment.run_experiment()
                self.assertEqual(1, len(glob.glob(os.path.join(tmp_dir_name, '*-specs.json'))))

    def test_subnetworks_from_toy_lenet_osp_experiment_have_equal_init_weight(self):
        """ All subnetworks should have the same 'weight_init' buffer as the original network. """
        specs = get_specs_lenet_toy()
        specs.experiment_name = experiment_specs.ExperimentNames.OSP
        specs.net_count = 1
        specs.epoch_count = 1
        specs.prune_count = 2
        specs.prune_rate_fc = 0.2
        specs.plan_fc = [5]

        with mock.patch('experiments.experiment.get_mnist_data_loaders', return_value=self.fake_mnist_data_loaders):
            with TemporaryDirectory() as tmp_dir_name:  # temporary folder to check if experiment generates files
                experiment = ExperimentOSP(specs, tmp_dir_name)
                experiment.setup_experiment()
                initial_net = experiment.nets[0].get_new_instance(reset_weight=False)

                experiment.execute_experiment()

                # check if subnetworks have correct sparsity
                np.testing.assert_allclose(experiment.hists.sparsity, [1., 0.801, 0.642], atol=0.001)
                # check if subnetworks have been generated from original net
                subnet = experiment.nets[0].get_new_instance(reset_weight=False)
                self.assertIs(torch.equal(subnet.fc[0].weight_init, initial_net.fc[0].weight_init), True)
                self.assertIs(torch.equal(subnet.out.weight_init, initial_net.out.weight_init), True)
                # should not generate files
                self.assertListEqual([], os.listdir(tmp_dir_name))


if __name__ == '__main__':
    unittest_main()
