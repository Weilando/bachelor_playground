import os
import unittest
from dataclasses import asdict
from json import load as j_load
from tempfile import TemporaryDirectory

import numpy as np
from torch import load as t_load

import data.result_saver as result_saver
from experiments.experiment_settings import get_settings_lenet_toy
from nets.lenet import Lenet


class Test_result_saver(unittest.TestCase):
    """ Tests for the result_saver module.
    Call with 'python -m test.test_result_saver' from project root '~'.
    Call with 'python -m test_result_saver' from inside '~/test'. """

    def test_generate_file_prefix_for_toy_experiment(self):
        """ Should generate the correct file_prefix for toy specs and a fake time-string. """
        specs = get_settings_lenet_toy()
        time_string = 'Time'

        expected_file_prefix = 'Time-Lenet-Toy'

        self.assertEqual(expected_file_prefix, result_saver.generate_file_prefix(specs, time_string))

    def test_generate_specs_file_name(self):
        """ Should append '-specs.json' to given prefix. """
        prefix = 'prefix'
        self.assertEqual('prefix-specs.json', result_saver.generate_specs_file_name(prefix))

    def test_generate_histories_file_name(self):
        """ Should append '-histories.npz' to given prefix. """
        prefix = 'prefix'
        self.assertEqual('prefix-histories.npz', result_saver.generate_histories_file_name(prefix))

    def test_generate_net_file_name(self):
        """ Should append '-net42.pth' to given prefix if called with 42. """
        prefix = 'prefix'
        self.assertEqual('prefix-net42.pth', result_saver.generate_net_file_name(prefix, 42))

    def test_save_specs(self):
        """ Should save toy_specs into json file. """
        specs = get_settings_lenet_toy()

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_specs(tmp_dir_name, 'prefix', specs)

            result_file_path = os.path.join(tmp_dir_name, 'prefix-specs.json')
            with open(result_file_path, 'r') as result_file:
                self.assertEqual(j_load(result_file), asdict(specs))

    def test_save_histories(self):
        """ Should save fake histories into npz file. """
        h1 = np.zeros(3)
        h2 = np.zeros(1)
        h3 = np.zeros(3)
        h4 = np.ones(1)
        h5 = np.ones(2)
        h6 = np.ones(3)

        with TemporaryDirectory() as tmp_dir_name:
            result_saver.save_histories(tmp_dir_name, 'prefix', h1, h2, h3, h4, h5, h6)

            result_file_path = os.path.join(tmp_dir_name, 'prefix-histories.npz')
            with open(result_file_path, 'rb') as result_file:
                # save histories
                result_file = np.load(result_file)

                # load and validate histories from file
                np.testing.assert_array_equal(h1, result_file['loss_h'])
                np.testing.assert_array_equal(h2, result_file['val_acc_h'])
                np.testing.assert_array_equal(h3, result_file['test_acc_h'])
                np.testing.assert_array_equal(h4, result_file['val_acc_ep_h'])
                np.testing.assert_array_equal(h5, result_file['test_acc_ep_h'])
                np.testing.assert_array_equal(h6, result_file['sparsity_h'])

    def test_save_nets(self):
        """ Should save two small Lenets into pth files. """
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
    unittest.main()
