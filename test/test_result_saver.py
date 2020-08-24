import os
from copy import deepcopy
from dataclasses import asdict
from json import load as j_load
from tempfile import TemporaryDirectory
from unittest import main as unittest_main
from unittest import mock, TestCase

import numpy as np
from torch import load as t_load

import data.result_saver as result_saver
from experiments.early_stop_histories import EarlyStopHistoryList
from experiments.experiment_histories import ExperimentHistories
from fake_experiment_specs import get_specs_lenet_toy
from nets.lenet import Lenet


class TestResultSaver(TestCase):
    """ Tests for the result_saver module.
    Call with 'python -m test.test_result_saver' from project root '~'.
    Call with 'python -m test_result_saver' from inside '~/test'. """

    def test_setup_and_get_result_path(self):
        """ Should generate the correct absolute result path. """
        with mock.patch('data.result_saver.os') as mocked_os:
            expected_path = "//root/results"

            mocked_os.getcwd.return_value = "//root"
            mocked_os.path.exists.return_value = True
            mocked_os.path.join.return_value = expected_path

            result_path = result_saver.setup_and_get_result_path("results")

            self.assertEqual(expected_path, result_path)
            mocked_os.getcwd.assert_called_once()
            mocked_os.path.exists.assert_called_once_with(expected_path)
            mocked_os.path.join.called_once_with("//root", "results")
            mocked_os.mkdir.assert_not_called()

    def test_setup_and_get_result_path_with_make(self):
        """ Should generate the correct absolute result path and make the directory. """
        with mock.patch('data.result_saver.os') as mocked_os:
            expected_path = "//root/results"

            mocked_os.getcwd.return_value = "//root"
            mocked_os.path.exists.return_value = False
            mocked_os.path.join.return_value = expected_path

            result_path = result_saver.setup_and_get_result_path("results")

            self.assertEqual(expected_path, result_path)
            mocked_os.getcwd.assert_called_once()
            mocked_os.path.exists.assert_called_once_with(expected_path)
            mocked_os.path.join.called_once_with("//root", "results")
            mocked_os.mkdir.assert_called_once_with(expected_path)

    def test_generate_file_prefix_for_toy_experiment(self):
        """ Should generate the correct file_prefix for toy specs and a fake time-string. """
        specs = get_specs_lenet_toy()
        time_string = 'Time'

        expected_file_prefix = 'Time-Lenet-MNIST'

        self.assertEqual(expected_file_prefix, result_saver.generate_file_prefix(specs, time_string))

    def test_generate_specs_file_name(self):
        """ Should append '-specs.json' to given prefix. """
        prefix = 'prefix'
        self.assertEqual('prefix-specs.json', result_saver.generate_specs_file_name(prefix))

    def test_generate_histories_file_name(self):
        """ Should append '-histories.npz' to given prefix. """
        prefix = 'prefix'
        self.assertEqual('prefix-histories.npz', result_saver.generate_experiment_histories_file_name(prefix))

    def test_generate_random_histories_file_name(self):
        """ Should append '-random-histories.npz' to given prefix and append net_number. """
        prefix = 'prefix'
        expected_file_name = 'prefix-random-histories42.npz'
        self.assertEqual(expected_file_name, result_saver.generate_random_experiment_histories_file_name(prefix, 42))

    def test_generate_early_stop_file_name(self):
        """ Should append '-early-stop42.pth' to given prefix. """
        prefix = 'prefix'
        self.assertEqual('prefix-early-stop42.pth', result_saver.generate_early_stop_file_name(prefix, 42))

    def test_generate_net_file_name(self):
        """ Should append '-net42.pth' to given prefix if called with 42. """
        prefix = 'prefix'
        self.assertEqual('prefix-net42.pth', result_saver.generate_net_file_name(prefix, 42))

    def test_save_specs(self):
        """ Should save toy_specs into json file. """
        specs = get_specs_lenet_toy()

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_specs(tmp_dir_name, 'prefix', specs)

            result_file_path = os.path.join(tmp_dir_name, 'prefix-specs.json')
            with open(result_file_path, 'r') as result_file:
                self.assertEqual(j_load(result_file), asdict(specs))

    def test_save_experiment_histories(self):
        """ Should save fake histories into npz file. """
        histories = ExperimentHistories()
        histories.setup(2, 1, 3, 2, 3)

        with TemporaryDirectory() as tmp_dir_name:
            # save histories
            result_saver.save_experiment_histories(tmp_dir_name, 'prefix', histories)

            # load and validate histories from file
            result_file_path = os.path.join(tmp_dir_name, 'prefix-histories.npz')
            with np.load(result_file_path) as result_file:
                reconstructed_histories = ExperimentHistories(**result_file)
                self.assertEqual(histories, reconstructed_histories)

    def test_save_experiment_histories_random_retrain(self):
        """ Should save fake histories into npz file. """
        histories = ExperimentHistories()
        histories.setup(2, 1, 3, 2, 3)

        with TemporaryDirectory() as tmp_dir_name:
            # save histories
            result_saver.save_experiment_histories_random_retrain(tmp_dir_name, 'prefix', 42, histories)

            # load and validate histories from file
            result_file_path = os.path.join(tmp_dir_name, 'prefix-random-histories42.npz')
            with np.load(result_file_path) as result_file:
                reconstructed_histories = ExperimentHistories(**result_file)
                self.assertEqual(histories, reconstructed_histories)

    def test_save_early_stop_history_list(self):
        """ Should save two fake EarlyStopHistories into two pth files. """
        plan_fc = [2]
        net0 = Lenet(plan_fc)
        net1 = Lenet(plan_fc)
        history_list = EarlyStopHistoryList()
        history_list.setup(2, 0)
        history_list.histories[0].state_dicts[0] = deepcopy(net0.state_dict())
        history_list.histories[1].state_dicts[0] = deepcopy(net1.state_dict())
        history_list.histories[0].indices[0] = 3
        history_list.histories[1].indices[0] = 42

        with TemporaryDirectory() as tmp_dir_name:
            # save checkpoints
            result_saver.save_early_stop_history_list(tmp_dir_name, 'prefix', history_list)

            # load and validate histories from file
            result_file_path0 = os.path.join(tmp_dir_name, 'prefix-early-stop0.pth')
            result_file_path1 = os.path.join(tmp_dir_name, 'prefix-early-stop1.pth')
            for net_num, result_file_path in enumerate([result_file_path0, result_file_path1]):
                with open(result_file_path, 'rb') as result_file:
                    reconstructed_hist = t_load(result_file)
                    net = Lenet(plan_fc)
                    np.testing.assert_array_equal(reconstructed_hist.indices, history_list.histories[net_num].indices)
                    net.load_state_dict(reconstructed_hist.state_dicts[0])

    def test_save_nets(self):
        """ Should save two small Lenet instances into pth files. """
        plan_fc = [5]
        net_list = [Lenet(plan_fc), Lenet(plan_fc)]

        with TemporaryDirectory() as tmp_dir_name:
            # save nets
            result_saver.save_nets(tmp_dir_name, 'prefix', net_list)

            # load and reconstruct nets from their files
            result_file_path0 = os.path.join(tmp_dir_name, 'prefix-net0.pth')
            result_file_path1 = os.path.join(tmp_dir_name, 'prefix-net1.pth')
            for result_file_path in [result_file_path0, result_file_path1]:
                with open(result_file_path, 'rb') as result_file:
                    checkpoint = t_load(result_file)
                    net = Lenet(plan_fc)
                    net.load_state_dict(checkpoint)


if __name__ == '__main__':
    unittest_main()
