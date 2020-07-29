import sys
from io import StringIO
from unittest import TestCase, mock
from unittest import main as unittest_main

from experiments.experiment_settings import ExperimentNames, VerbosityLevel, get_settings_lenet_mnist, \
    get_settings_conv2_cifar10
from playground import main as playground_main, setup_cuda, should_override_arg_plan, should_override_arg_rate, \
    should_override_arg_positive_int


class TestPlayground(TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_playground' from project root '~'.
    Call with 'python -m test_playground' from inside '~/test'. """

    def test_should_not_override_positive_int_as_flag_not_set(self):
        """ Should return False, because the flag was not set. """
        self.assertIs(should_override_arg_positive_int(None, 'Some argument'), False)

    def test_should_override_positive_int_as_value_is_valid(self):
        """ Should return True, because the flag was set with valid values. """
        self.assertIs(should_override_arg_positive_int(1, 'Some argument'), True)
        self.assertIs(should_override_arg_positive_int(42, 'Some argument'), True)

    def test_should_raise_exception_as_positive_int_is_invalid(self):
        """ Should throw an assertion error, because the flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_arg_positive_int(0, 'Some Argument')
            should_override_arg_positive_int(1.0, 'Some Argument')

    def test_should_not_override_rate_as_flag_not_set(self):
        """ Should return False, because the flag was not set. """
        self.assertIs(should_override_arg_rate(None, 'Some argument'), False)

    def test_should_override_rate_as_rate_is_valid(self):
        """ Should return True, because the flag was set with valid values. """
        self.assertIs(should_override_arg_rate(0.0, 'Some argument'), True)
        self.assertIs(should_override_arg_rate(0.8, 'Some argument'), True)
        self.assertIs(should_override_arg_rate(1.0, 'Some argument'), True)

    def test_should_raise_exception_as_rate_is_invalid(self):
        """ Should throw an assertion error, because the flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_arg_positive_int(1.1, 'Some Argument')
            should_override_arg_positive_int(0, 'Some Argument')

    def test_should_not_override_arg_plan_as_flag_not_set(self):
        """ Should return False, because the flag was not set. """
        self.assertIs(should_override_arg_plan(None, 'Some plan'), False)

    def test_should_override_arg_plan_as_it_is_valid(self):
        """ Should return True, because the flag was set with a valid value. """
        self.assertIs(should_override_arg_plan([1, 2], 'Some plan'), True)

    def test_should_raise_exception_as_plan_is_invalid(self):
        """ Should throw an assertion error, because the flag was set with an invalid value. """
        with self.assertRaises(AssertionError):
            should_override_arg_plan(3, 'Some plan')

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
            playground_main(ExperimentNames.LENET_MNIST, None, None, None, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_print_experiment_settings(self):
        """ Playground should not start the experiment and print the settings. """
        expected_settings = get_settings_lenet_mnist()
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            with StringIO() as interception:
                old_stdout = sys.stdout
                sys.stdout = interception

                playground_main(ExperimentNames.LENET_MNIST, None, None, None, None, None, None, None, None, False,
                                VerbosityLevel.SILENT, True, False, None)

                sys.stdout = old_stdout

                self.assertEqual(interception.getvalue(), f"{expected_settings}\n")
                mocked_experiment.assert_not_called()

    def test_should_start_experiment_with_modified_epochs_parameter(self):
        """ Playground should start the experiment with modified epoch_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.epoch_count = 42
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, 42, None, None, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_nets_parameter(self):
        """ Playground should start the experiment with modified net_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.net_count = 1
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, 1, None, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_prunes_parameter(self):
        """ Playground should start the experiment with modified prune_count. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.prune_count = 4
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, 4, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_learning_rate_parameter(self):
        """ Playground should start the experiment with modified learning_rate. """
        expected_settings = get_settings_conv2_cifar10()
        expected_settings.learning_rate = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.CONV2_CIFAR10, None, None, None, 0.5, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_prune_rate_conv_parameter(self):
        """ Playground should start the experiment with modified prune_rate_conv. """
        expected_settings = get_settings_conv2_cifar10()
        expected_settings.prune_rate_conv = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.CONV2_CIFAR10, None, None, None, None, 0.5, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_prune_rate_fc_parameter(self):
        """ Playground should start the experiment with modified prune_rate_fc. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.prune_rate_fc = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, None, None, None, 0.5, None, None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_plan_conv(self):
        """ Playground should start the experiment with modified plan_conv. """
        expected_settings = get_settings_conv2_cifar10()
        expected_settings.plan_conv = [1]
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.CONV2_CIFAR10, None, None, None, None, None, None, [1], None, False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_plan_fc(self):
        """ Playground should start the experiment with modified plan_conv. """
        expected_settings = get_settings_conv2_cifar10()
        expected_settings.plan_fc = [1]
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.CONV2_CIFAR10, None, None, None, None, None, None, None, [1], False,
                            VerbosityLevel.SILENT, False, False, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_early_stop(self):
        """ Playground should start the experiment with flag for early-stopping-checkpoints during training. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.save_early_stop = True
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, None, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, True, None)
            mocked_experiment.assert_called_once_with(expected_settings)

    def test_should_start_experiment_with_modified_plot_step_parameter(self):
        """ Playground should start the experiment with modified plot_step. """
        expected_settings = get_settings_lenet_mnist()
        expected_settings.plot_step = 42
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main(ExperimentNames.LENET_MNIST, None, None, None, None, None, None, None, None, False,
                            VerbosityLevel.SILENT, False, False, 42)
            mocked_experiment.assert_called_once_with(expected_settings)


if __name__ == '__main__':
    unittest_main()
