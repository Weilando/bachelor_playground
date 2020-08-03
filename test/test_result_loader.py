import os
from copy import deepcopy
from dataclasses import asdict
from tempfile import TemporaryDirectory
from unittest import main as unittest_main
from unittest import mock, TestCase

import numpy as np

import data.result_loader as result_loader
from data import result_saver
from experiments.early_stop_histories import EarlyStopHistoryList, EarlyStopHistory
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_settings import get_settings_lenet_toy, get_settings_conv_toy
from nets.conv import Conv
from nets.lenet import Lenet


class TestResultLoader(TestCase):
    """ Tests for the result_loader module.
    Call with 'python -m test.test_result_loader' from project root '~'.
    Call with 'python -m test_result_loader' from inside '~/test'. """

    # helper functions
    def test_is_valid_specs_path(self):
        """ Should return True, because the given path is valid and the file exists. """
        with mock.patch('data.result_loader.os') as mocked_os:
            specs_path = "//root/results/prefix-specs.json"
            mocked_os.path.exists.return_value = True

            self.assertIs(result_loader.is_valid_specs_path(specs_path), True)
            mocked_os.path.exists.assert_called_once_with(specs_path)

    def test_is_invalid_specs_path_wrong_suffix(self):
        """ Should return False, because the given path is invalid. """
        with mock.patch('data.result_loader.os'):
            specs_path = "//root/results/prefix-specs"
            self.assertIs(result_loader.is_valid_specs_path(specs_path), False)

    def test_is_invalid_specs_path_no_file(self):
        """ Should return False, because the referenced file does not exist. """
        with mock.patch('data.result_loader.os') as mocked_os:
            specs_path = "//root/results/prefix-specs"
            mocked_os.path.exists.return_value = False

            self.assertIs(result_loader.is_valid_specs_path(specs_path), False)
            mocked_os.path.exists.assert_called_once_with(specs_path)

    def test_generate_absolute_specs_path(self):
        """ Should extract absolute specs path from given relative path. """
        with mock.patch('data.result_loader.os') as mocked_os:
            expected_path = "//root/results/prefix-specs.json"
            relative_path = "results/prefix-specs.json"

            mocked_os.getcwd.return_value = "//root"
            mocked_os.path.join.return_value = expected_path

            result_path = result_loader.generate_absolute_specs_path(relative_path)

            self.assertEqual(expected_path, result_path)
            mocked_os.getcwd.assert_called_once()
            mocked_os.path.join.called_once_with("//root", relative_path)

    def test_generate_experiment_path_prefix(self):
        """ Should generate the experiment prefix from given specs-file path, i.e. remove suffix '-specs.json'. """
        expected_prefix = "//root/results/prefix"
        absolute_path = "//root/results/prefix-specs.json"
        result_path = result_loader.generate_experiment_path_prefix(absolute_path)
        self.assertEqual(expected_prefix, result_path)

    def test_generate_experiment_histories_file_path(self):
        """ Should generate histories file path, i.e. append '-histories.npz'. """
        experiment_path_prefix = "results/prefix"
        expected_path = "results/prefix-histories.npz"
        result_path = result_loader.generate_experiment_histories_file_path(experiment_path_prefix)
        self.assertEqual(expected_path, result_path)

    def test_generate_random_experiment_histories_file_path(self):
        """ Should generate histories file path, i.e. append '-random-histories#.npz' with #=net_number. """
        experiment_path_prefix = "results/prefix"
        expected_path = "results/prefix-random-histories42.npz"
        result_path = result_loader.generate_random_experiment_histories_file_path(experiment_path_prefix, 42)
        self.assertEqual(expected_path, result_path)

    def test_generate_early_stop_file_path(self):
        """ Should an early-stop file path, i.e. append '-early-stop#.pth' with # number. """
        experiment_path_prefix = "results/prefix"
        expected_path = "results/prefix-early-stop42.pth"
        result_path = result_loader.generate_early_stop_file_path(experiment_path_prefix, 42)
        self.assertEqual(expected_path, result_path)

    def test_generate_net_file_paths(self):
        """ Should generate a list of net file paths, i.e. append '-net#.pth' with # number. """
        experiment_path_prefix = "results/prefix"
        expected_paths = ["results/prefix-net0.pth", "results/prefix-net1.pth"]
        result_paths = result_loader.generate_net_file_paths(experiment_path_prefix, 2)
        self.assertEqual(expected_paths, result_paths)

    def test_generate_net_file_paths_invalid_net_count(self):
        """ Should not generate a list of net file paths, because 'net_count' is not positive. """
        experiment_path_prefix = "results/prefix"
        with self.assertRaises(AssertionError):
            result_loader.generate_net_file_paths(experiment_path_prefix, 0)

    # higher level functions
    def test_get_relative_spec_file_paths(self):
        """ Should return all relative paths to spec-file in ascending order. """
        with TemporaryDirectory() as tmp_dir:
            # create sub-directory 'results', two specs-files and another file
            tmp_results = os.path.join(tmp_dir, 'results')
            os.mkdir(tmp_results)
            relative_path_specs_file1 = os.path.join(tmp_results, 'prefix1-specs.json')
            relative_path_specs_file2 = os.path.join(tmp_results, 'prefix2-specs.json')
            relative_path_no_specs_file = os.path.join(tmp_results, 'prefix.json')
            open(relative_path_specs_file1, 'a').close()
            open(relative_path_specs_file2, 'a').close()
            open(relative_path_no_specs_file, 'a').close()

            result_paths = result_loader.get_relative_spec_file_paths(tmp_results)

            self.assertEqual([relative_path_specs_file1, relative_path_specs_file2], result_paths)

    def test_extract_experiment_path_prefix(self):
        """ Should extract experiment path prefix without error. """
        with mock.patch('data.result_loader.os') as mocked_os:
            relative_specs_path = "results/prefix-specs.json"
            absolute_specs_path = "//root/results/prefix-specs.json"
            expected_prefix = "//root/results/prefix"

            mocked_os.getcwd.return_value = "//root"
            mocked_os.path.join.return_value = absolute_specs_path
            mocked_os.path.exists.return_value = True

            result_prefix = result_loader.extract_experiment_path_prefix(relative_specs_path)

            self.assertEqual(expected_prefix, result_prefix)
            mocked_os.path.join.called_once_with("//root", relative_specs_path)
            mocked_os.path.exists.assert_called_once_with(absolute_specs_path)

    def test_extract_experiment_path_prefix_fail(self):
        """ Should raise assertion error, because the specs file does not exist. """
        with mock.patch('data.result_loader.os') as mocked_os:
            relative_specs_path = "results/prefix-specs.json"
            absolute_specs_path = "//root/results/prefix-specs.json"

            mocked_os.getcwd.return_value = "//root"
            mocked_os.path.join.return_value = absolute_specs_path
            mocked_os.path.exists.return_value = False

            with self.assertRaises(AssertionError):
                result_loader.extract_experiment_path_prefix(relative_specs_path)

    def test_get_specs_from_file_as_dict(self):
        """ Should load toy_specs from json file as dict. """
        experiment_settings = get_settings_lenet_toy()

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_specs(tmp_dir_name, 'prefix', experiment_settings)

            result_file_path = f"{tmp_dir_name}/prefix-specs.json"
            loaded_dict = result_loader.get_specs_from_file(result_file_path, as_dict=True)
            self.assertEqual(loaded_dict, asdict(experiment_settings))

    def test_get_specs_from_file_as_experiment_settings(self):
        """ Should load toy_specs from json file as ExperimentSettings. """
        experiment_settings = get_settings_lenet_toy()

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_specs(tmp_dir_name, 'prefix', experiment_settings)

            result_file_path = f"{tmp_dir_name}/prefix-specs.json"
            loaded_experiment_settings = result_loader.get_specs_from_file(result_file_path, as_dict=False)
            self.assertEqual(loaded_experiment_settings, experiment_settings)

    def test_get_experiment_histories_from_file(self):
        """ Should load fake histories from npz file. """
        histories = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_experiment_histories(tmp_dir_name, 'prefix', histories)

            # load and validate histories from file
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_histories = result_loader.get_experiment_histories_from_file(experiment_path_prefix)
            self.assertEqual(loaded_histories, histories)

    def test_get_early_stop_history_from_file(self):
        """ Should load fake EarlyStopHistory from pth file. """
        plan_fc = [2]
        net0 = Lenet(plan_fc)
        history = EarlyStopHistory()
        history.setup(0)
        history.state_dicts[0] = deepcopy(net0.state_dict())
        history.indices[0] = 3

        specs = get_settings_lenet_toy()
        specs.save_early_stop = True
        specs.net_count = 2
        specs.prune_count = 1

        with TemporaryDirectory() as tmp_dir_name:
            # save checkpoints
            history_list = EarlyStopHistoryList()
            history_list.setup(1, 0)
            history_list.histories[0] = history
            result_saver.save_early_stop_history_list(tmp_dir_name, 'prefix', history_list)

            # load and validate histories from file
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_history = result_loader.get_early_stop_history_from_file(experiment_path_prefix, specs, 0)
            self.assertEqual(loaded_history, history)
            net0.load_state_dict(history.state_dicts[0])

    def test_get_early_stop_history_from_file_invalid_specs(self):
        """ Should raise assertion error as specs do not have type ExperimentSettings. """
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_from_file("some_path", dict(), 42)

    def test_get_early_stop_history_from_file_no_save_early_stop(self):
        """ Should raise assertion error as 'save_early_stop' flag is not set in specs. """
        experiment_settings = get_settings_lenet_toy()
        experiment_settings.save_early_stop = False
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_from_file("some_path", experiment_settings, 42)

    def test_get_early_stop_history_from_file_invalid_number_small(self):
        """ Should raise assertion error as 'number' is too small. """
        experiment_settings = get_settings_lenet_toy()
        experiment_settings.save_early_stop = True
        experiment_settings.net_count = 3
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_from_file("some_path", experiment_settings, -1)

    def test_get_early_stop_history_from_file_invalid_number_tall(self):
        """ Should raise assertion error as 'number' is too tall. """
        experiment_settings = get_settings_lenet_toy()
        experiment_settings.save_early_stop = True
        experiment_settings.net_count = 3
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_from_file("some_path", experiment_settings, 3)

    def test_get_early_stop_history_list_from_files(self):
        """ Should load fake EarlyStopHistoryList from pth files. """
        plan_fc = [2]
        net0 = Lenet(plan_fc)
        net1 = Lenet(plan_fc)
        history_list = EarlyStopHistoryList()
        history_list.setup(2, 0)
        history_list.histories[0].state_dicts[0] = deepcopy(net0.state_dict())
        history_list.histories[1].state_dicts[0] = deepcopy(net1.state_dict())
        history_list.histories[0].indices[0] = 3
        history_list.histories[1].indices[0] = 42

        specs = get_settings_lenet_toy()
        specs.save_early_stop = True
        specs.net_count = 2
        specs.prune_count = 0

        with TemporaryDirectory() as tmp_dir_name:
            # save checkpoints
            result_saver.save_early_stop_history_list(tmp_dir_name, 'prefix', history_list)

            # load and validate histories from file
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_history_list = result_loader.get_early_stop_history_list_from_files(experiment_path_prefix, specs)
            self.assertEqual(loaded_history_list, history_list)
            net0.load_state_dict(history_list.histories[0].state_dicts[0])
            net1.load_state_dict(history_list.histories[1].state_dicts[0])

    def test_get_early_stop_history_list_from_files_invalid_specs(self):
        """ Should raise assertion error as specs do not have type ExperimentSettings. """
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_list_from_files("some_path", dict())

    def test_get_early_stop_history_list_from_files_no_save_early_stop(self):
        """ Should raise assertion error as 'save_early_stop' flag is not set in specs. """
        experiment_settings = get_settings_lenet_toy()
        experiment_settings.save_early_stop = False
        with self.assertRaises(AssertionError):
            result_loader.get_early_stop_history_list_from_files("some_path", experiment_settings)

    def test_get_lenet_from_file(self):
        """ Should load two small Lenet instances from pth files. """
        experiment_settings = get_settings_lenet_toy()
        net_list = [Lenet(experiment_settings.plan_fc), Lenet(experiment_settings.plan_fc)]

        with TemporaryDirectory() as tmp_dir_name:
            # save nets
            result_saver.save_nets(tmp_dir_name, 'prefix', net_list)

            # load and reconstruct nets from their files
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_nets = result_loader.get_models_from_files(experiment_path_prefix, experiment_settings)
            self.assertIsInstance(loaded_nets[0], Lenet)
            self.assertIsInstance(loaded_nets[1], Lenet)

    def test_get_conv_from_file(self):
        """ Should load two small Conv instances from pth files. """
        experiment_settings = get_settings_conv_toy()
        net_list = [Conv(experiment_settings.plan_conv, experiment_settings.plan_fc),
                    Conv(experiment_settings.plan_conv, experiment_settings.plan_fc)]

        with TemporaryDirectory() as tmp_dir_name:
            # save nets
            result_saver.save_nets(tmp_dir_name, 'prefix', net_list)

            # load and reconstruct nets from their files
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_nets = result_loader.get_models_from_files(experiment_path_prefix, experiment_settings)
            self.assertIsInstance(loaded_nets[0], Conv)
            self.assertIsInstance(loaded_nets[1], Conv)

    def test_get_models_from_file_invalid_model_name(self):
        """ Should raise assertion error if specs contain an invalid entry for 'net'. """
        experiment_settings = get_settings_lenet_toy()
        experiment_settings.net = "Some invalid name"
        with self.assertRaises(AssertionError):
            result_loader.get_models_from_files("some_path", experiment_settings)

    def test_get_models_from_file_invalid_specs(self):
        """ Should raise assertion error if specs do not have type ExperimentSettings. """
        with self.assertRaises(AssertionError):
            result_loader.get_models_from_files("some_path", dict())

    def test_random_histories_file_exists(self):
        """ Should return true, because a random-histories-file exists for 42. """
        histories = ExperimentHistories()
        histories.setup(2, 2, 2, 2, 2)
        with TemporaryDirectory() as tmp_dir_name:
            # save nets
            result_saver.save_experiment_histories_random_retrain(tmp_dir_name, 'prefix', 42, histories)

            # load and reconstruct nets from their files
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            self.assertIs(result_loader.random_histories_file_exists(experiment_path_prefix, 42), True)

    def test_random_histories_file_does_not_exist(self):
        """ Should return true, because no random-histories-file exists for 42. """
        histories = ExperimentHistories()
        histories.setup(2, 2, 2, 2, 2)
        with TemporaryDirectory() as tmp_dir_name:
            # save nets
            result_saver.save_experiment_histories_random_retrain(tmp_dir_name, 'prefix', 7, histories)

            # load and reconstruct nets from their files
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            self.assertIs(result_loader.random_histories_file_exists(experiment_path_prefix, 42), False)


if __name__ == '__main__':
    unittest_main()
