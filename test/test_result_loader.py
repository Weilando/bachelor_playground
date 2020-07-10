from dataclasses import asdict
from tempfile import TemporaryDirectory
from unittest import main as unittest_main
from unittest import mock, TestCase

import numpy as np

import data.result_loader as result_loader
from data import result_saver
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

    def test_generate_histories_file_path(self):
        """ Should generate histories file path, i.e. append '-histories.npz'. """
        experiment_path_prefix = "results/prefix"
        expected_path = "results/prefix-histories.npz"
        result_path = result_loader.generate_histories_file_path(experiment_path_prefix)
        self.assertEqual(expected_path, result_path)

    def test_generate_net_file_paths(self):
        """ Should generate a list of net file paths, i.e. append '-net#.pth' with # number. """
        experiment_path_prefix = "results/prefix"
        expected_paths = ["results/prefix-net0.pth", "results/prefix-net1.pth"]
        result_paths = result_loader.generate_net_file_paths(experiment_path_prefix, 2)
        self.assertEqual(expected_paths, result_paths)

    def test_generate_net_file_paths_invalid_net_count(self):
        """ Should generate a list of net file paths, i.e. append '-net#.pth' with # number. """
        experiment_path_prefix = "results/prefix"
        with self.assertRaises(AssertionError):
            result_loader.generate_net_file_paths(experiment_path_prefix, 0)

    # higher level functions
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

    def test_get_histories_from_file(self):
        """ Should load fake histories from npz file. """
        histories = ExperimentHistories(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(2))

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_histories(tmp_dir_name, 'prefix', histories)

            # load and validate histories from file
            experiment_path_prefix = f"{tmp_dir_name}/prefix"
            loaded_histories = result_loader.get_histories_from_file(experiment_path_prefix)
            self.assertEqual(loaded_histories, histories)

    def test_get_lenet_from_file(self):
        """ Should load two small Lenets from pth files. """
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
        """ Should load two small Convs from pth files. """
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

    def test_get_models_from_file_invalid_specs(self):
        """ Should raise assertion error if specs do not have type ExperimentSettings. """
        with self.assertRaises(AssertionError):
            result_loader.get_models_from_files("some_path", dict())


if __name__ == '__main__':
    unittest_main()
