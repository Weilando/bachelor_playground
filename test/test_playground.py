import sys
from io import StringIO
from unittest import TestCase, mock
from unittest import main as unittest_main

from experiments.experiment_settings import ExperimentNames, VerbosityLevel, get_settings_lenet_mnist
from playground import main as playground_main
from playground import should_override_epoch_count, should_override_prune_count, should_override_net_count, setup_cuda


class TestPlayground(TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_playground' from project root '~'.
    Call with 'python -m test_playground' from inside '~/test'. """

    def test_should_not_override_epoch_count_as_epochs_flag_not_set(self):
        """ Should return False, because the 'epochs'-flag was not set. """
        self.assertIs(should_override_epoch_count(None), False)

    def test_should_override_epoch_count_as_epochs_is_valid(self):
        """ Should return True, because the 'epochs'-flag was set with a valid value. """
        self.assertIs(should_override_epoch_count(1), True)

    def test_should_raise_exception_as_epochs_is_invalid(self):
        """ Should throw an assertion error, because the 'epochs'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_epoch_count(0)

    def test_should_not_override_net_count_as_nets_flag_not_set(self):
        """ Should return False, because the 'nets'-flag was not set. """
        self.assertIs(should_override_net_count(None), False)

    def test_should_override_net_count_as_nets_is_valid(self):
        """ Should return True, because the 'nets'-flag was set with a valid value. """
        self.assertIs(should_override_net_count(1), True)

    def test_should_raise_exception_as_nets_is_invalid(self):
        """ Should throw an assertion error, because the 'nets'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_net_count(0)

    def test_should_not_override_prune_count_as_prunes_flag_not_set(self):
        """ Should return False, because the 'prunes'-flag was not set. """
        self.assertIs(should_override_prune_count(None), False)

    def test_should_override_prune_count_as_prunes_is_valid(self):
        """ Should return True, because the 'prunes'-flag was set with a valid value. """
        self.assertIs(should_override_prune_count(0), True)

    def test_should_raise_exception_as_prunes_is_invalid(self):
        """ Should throw an assertion error, because the 'prunes'-flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_prune_count(-1)

    def test_should_setup_gpu_as_wanted_and_available(self):
        """ Should return 'cuda:0' as cuda device and its name as device_name, because cuda is wanted and available. """
        with mock.patch('playground.torch.cuda') as mocked_cuda:
            mocked_cuda.is_available.return_value = True
            mocked_cuda.get_device_name.return_value = 'GPU-name'

            device, device_name = setup_cuda(True)

            self.assertEqual(device, 'cuda:0')
            self.assertEqual(device_name, 'GPU-name')
            mocked_cuda.is_available.assert_called_once()
            mocked_cuda.get_device_name.assert_called_once_with(0)

    def test_should_not_setup_gpu_as_not_available(self):
        """ Should return 'cpu' as cuda device and device_name, because no cuda is available. """
        with mock.patch('playground.torch.cuda') as mocked_cuda:
            mocked_cuda.is_available.return_value = False

            device, device_name = setup_cuda(True)

            self.assertEqual(device, 'cpu')
            self.assertEqual(device_name, 'cpu')
            mocked_cuda.is_available.assert_called_once()
            mocked_cuda.get_device_name.assert_not_called()

    def test_should_setup_cpu_as_wanted(self):
        """ Should return 'cpu' as cuda device and device_name, because no cuda is wanted. """
        device, device_name = setup_cuda(False)
        self.assertEqual(device, 'cpu')
        self.assertEqual(device_name, 'cpu')

    def test_should_start_experiment(self):
        """ Playground should start the experiment with correct standard settings. """
        expected_settings = get_settings_lenet_mnist()
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, None, False, VerbosityLevel.SILENT, False)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_print_experiment_settings(self):
        """ Playground should not start the experiment and print the settings. """
        expected_settings = get_settings_lenet_mnist()
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            with StringIO() as interception:
                old_stdout = sys.stdout
                sys.stdout = interception

                playground_main(ExperimentNames.LENET_MNIST, None, None, None, False, VerbosityLevel.SILENT, True)

                sys.stdout = old_stdout

                self.assertEqual(interception.getvalue(), f"{expected_settings}\n")
                mocked_experiment.assert_not_called()

    def test_should_start_experiment_with_modified_epochs_parameter(self):
        """ Playground should start the experiment with modified epoch_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.epoch_count = 42
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, 42, None, None, False, VerbosityLevel.SILENT, False)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_nets_parameter(self):
        """ Playground should start the experiment with modified net_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.net_count = 1
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, 1, None, False, VerbosityLevel.SILENT, False)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_prunes_parameter(self):
        """ Playground should start the experiment with modified prune_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.prune_count = 4
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, 4, False, VerbosityLevel.SILENT, False)
            mocked_experiment.assert_called_once_with(expected_settings)


if __name__ == '__main__':
    unittest_main()
