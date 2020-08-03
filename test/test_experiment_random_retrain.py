import glob
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import main as unittest_main

import torch

import experiments.experiment_settings as experiment_settings
from data import result_saver
from experiments.early_stop_histories import EarlyStopHistory, EarlyStopHistoryList
from experiments.experiment_random_retrain import ExperimentRandomRetrain
from nets.lenet import Lenet


class TestExperimentRandomRetrain(TestCase):
    """ Tests for the experiment_random_retrain module.
    Call with 'python -m test.test_experiment_random_retrain' from project root '~'.
    Call with 'python -m test_experiment_random_retrain' from inside '~/test'. """

    def test_generate_randomly_reinitialized_net(self):
        """ Should generate a network with equal masks but different weights. """
        specs = experiment_settings.get_settings_lenet_mnist()
        specs.plan_fc = [5]
        specs.save_early_stop = True
        torch.manual_seed(0)
        net = Lenet(specs.plan_fc)

        torch.manual_seed(1)
        new_net = ExperimentRandomRetrain.generate_randomly_reinitialized_net(specs, net.state_dict())

        self.assertIs(net.fc[0].weight.eq(new_net.fc[0].weight).all().item(), False)
        self.assertIs(net.fc[0].weight_mask.eq(new_net.fc[0].weight_mask).all().item(), True)
        self.assertIs(net.out.weight.eq(new_net.out.weight).all().item(), False)
        self.assertIs(net.out.weight_mask.eq(new_net.out.weight_mask).all().item(), True)

    def test_perform_toy_lenet_experiment(self):
        """ Should run IMP-Experiment with small Lenet and toy-dataset without errors. """
        specs = experiment_settings.get_settings_lenet_toy()
        specs.prune_count = 1
        specs.save_early_stop = True

        early_stop_history = EarlyStopHistory()
        early_stop_history.setup(specs.prune_count)

        net = Lenet(specs.plan_fc)
        early_stop_history.state_dicts[0] = net.state_dict()
        early_stop_history.state_dicts[1] = net.state_dict()
        early_stop_history_list = EarlyStopHistoryList()
        early_stop_history_list.setup(1, 0)
        early_stop_history_list.histories[0] = early_stop_history

        with TemporaryDirectory() as tmp_dir_name:  # save results into a temporary folder
            result_saver.save_specs(tmp_dir_name, 'prefix', specs)
            result_saver.save_early_stop_history_list(tmp_dir_name, 'prefix', early_stop_history_list)
            path_to_specs = os.path.join(tmp_dir_name, 'prefix-specs.json')
            experiment = ExperimentRandomRetrain(path_to_specs, 0, 1)
            experiment.run_experiment()
            self.assertEqual(len(glob.glob(os.path.join(tmp_dir_name, 'prefix-random-histories0.npz'))), 1)


if __name__ == '__main__':
    unittest_main()
