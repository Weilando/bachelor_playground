from io import StringIO
from unittest import TestCase, mock
from unittest import main as unittest_main

import sys

from experiments.experiment_specs import ExperimentPresetNames, VerbosityLevel, get_specs_lenet_mnist, \
    get_specs_conv2_cifar10, ExperimentNames
from playground import main as playground_main, setup_cuda, should_override_arg_plan, should_override_arg_rate, \
    should_override_arg_positive_int, parse_arguments


class TestPlayground(TestCase):
    """ Tests for the playground module.
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

    def test_should_show_help_on_missing_args(self):
        """ Should print help message to stderr, if args is empty. """
        with StringIO() as interception:
            old_stderr = sys.stderr
            sys.stderr = interception

            with self.assertRaises(SystemExit):
                parse_arguments([])

            sys.stderr = old_stderr

            self.assertIn("usage", interception.getvalue())

    def test_should_parse_arguments(self):
        """ Should parse all given arguments correctly. """
        parsed_args = parse_arguments([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-c', '-es', '-l', '-ps',
                                       '42', '-v', '-e', '1', '-n', '2', '-p', '3', '-lr', '0.01', '-prc', '0.2',
                                       '-prf', '0.3', '--plan_conv', '16', 'M', '--plan_fc', '300', '200'])

        self.assertEqual(parsed_args.cuda, True)
        self.assertEqual(parsed_args.early_stop, True)
        self.assertEqual(parsed_args.listing, True)
        self.assertEqual(parsed_args.plot_step, 42)
        self.assertEqual(parsed_args.verbose, VerbosityLevel.MEDIUM)
        self.assertEqual(parsed_args.epochs, 1)
        self.assertEqual(parsed_args.nets, 2)
        self.assertEqual(parsed_args.prunes, 3)
        self.assertEqual(parsed_args.learn_rate, 0.01)
        self.assertEqual(parsed_args.prune_rate_conv, 0.2)
        self.assertEqual(parsed_args.prune_rate_fc, 0.3)
        self.assertEqual(parsed_args.plan_conv, ['16', 'M'])
        self.assertEqual(parsed_args.plan_fc, ['300', '200'])

    def test_should_start_experiment_imp(self):
        """ Playground should start the IMP-experiment with correct standard specs. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.experiment_name = ExperimentNames.IMP
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_osp(self):
        """ Playground should start the OSP-experiment with correct standard specs. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.experiment_name = ExperimentNames.OSP
        with mock.patch('playground.ExperimentOSP') as mocked_experiment:
            playground_main([ExperimentNames.OSP, ExperimentPresetNames.LENET_MNIST])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_random_retrain(self):
        """ Playground should start the random-retraining-experiment with correct arguments. """
        with mock.patch('playground.ExperimentRandomRetrain') as mocked_experiment:
            playground_main([ExperimentNames.RR, 'some/path/pre-specs.json', '0', '3'])
            mocked_experiment.assert_called_once_with('../some/path/pre-specs.json', 0, 3)

    def test_should_print_experiment_specs_imp(self):
        """ Playground should not start the IMP-experiment and print the specs. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.experiment_name = ExperimentNames.IMP
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            with StringIO() as interception:
                old_stdout = sys.stdout
                sys.stdout = interception

                playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-l'])

                sys.stdout = old_stdout

                self.assertEqual(interception.getvalue(), f"{expected_specs}\n")
                mocked_experiment.assert_not_called()

    def test_should_print_experiment_specs_osp(self):
        """ Playground should not start the OSP-experiment and print the specs. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.experiment_name = ExperimentNames.OSP
        with mock.patch('playground.ExperimentOSP') as mocked_experiment:
            with StringIO() as interception:
                old_stdout = sys.stdout
                sys.stdout = interception

                playground_main([ExperimentNames.OSP, ExperimentPresetNames.LENET_MNIST, '-l'])

                sys.stdout = old_stdout

                self.assertEqual(interception.getvalue(), f"{expected_specs}\n")
                mocked_experiment.assert_not_called()

    def test_should_start_experiment_with_modified_epochs_parameter(self):
        """ Playground should start the experiment with modified epoch_count. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.epoch_count = 42
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-e', '42'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_nets_parameter(self):
        """ Playground should start the experiment with modified net_count. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.net_count = 1
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-n', '1'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_prunes_parameter(self):
        """ Playground should start the experiment with modified prune_count. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.prune_count = 4
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-p', '4'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_learning_rate_parameter(self):
        """ Playground should start the experiment with modified learning_rate. """
        expected_specs = get_specs_conv2_cifar10()
        expected_specs.learning_rate = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.CONV2_CIFAR10, '-lr', '0.5'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_prune_rate_conv_parameter(self):
        """ Playground should start the experiment with modified prune_rate_conv. """
        expected_specs = get_specs_conv2_cifar10()
        expected_specs.prune_rate_conv = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.CONV2_CIFAR10, '-prc', '0.5'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_prune_rate_fc_parameter(self):
        """ Playground should start the experiment with modified prune_rate_fc. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.prune_rate_fc = 0.5
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-prf', '0.5'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_plan_conv(self):
        """ Playground should start the experiment with modified plan_conv. """
        expected_specs = get_specs_conv2_cifar10()
        expected_specs.plan_conv = ['1']
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.CONV2_CIFAR10, '--plan_conv', '1'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_plan_fc(self):
        """ Playground should start the experiment with modified plan_conv. """
        expected_specs = get_specs_conv2_cifar10()
        expected_specs.plan_fc = ['1']
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.CONV2_CIFAR10, '--plan_fc', '1'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_early_stop(self):
        """ Playground should start the experiment with flag for early-stopping-checkpoints during training. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.save_early_stop = True
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-es'])
            mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_detailed_logging(self):
        """ Playground should start the experiment with detailed logging. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.verbosity = VerbosityLevel.DETAILED
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            with StringIO() as interception:
                old_stdout = sys.stdout
                sys.stdout = interception

                playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-vv'])

                sys.stdout = old_stdout

                self.assertEqual(interception.getvalue(), "cpu\n")
                mocked_experiment.assert_called_once_with(expected_specs)

    def test_should_start_experiment_with_modified_plot_step_parameter(self):
        """ Playground should start the experiment with modified plot_step. """
        expected_specs = get_specs_lenet_mnist()
        expected_specs.plot_step = 42
        with mock.patch('playground.ExperimentIMP') as mocked_experiment:
            playground_main([ExperimentNames.IMP, ExperimentPresetNames.LENET_MNIST, '-ps', '42'])
            mocked_experiment.assert_called_once_with(expected_specs)


if __name__ == '__main__':
    unittest_main()
