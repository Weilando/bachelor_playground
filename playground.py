from argparse import ArgumentParser
from torch.cuda import is_available, get_device_name

import os
import sys

current_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(current_path, '.')))
from experiments.experiment_settings import get_settings, ExperimentNames, VerbosityLevel
from experiments.experiment_lenet_mnist import ExperimentLenetMNIST
from experiments.experiment_conv_cifar10 import ExperimentConvCIFAR10

os.chdir(os.path.join(current_path, 'experiments'))


def should_override_epoch_count(epochs):
    """ Check if the 'epochs'-flag was set and if the given value is valid. """
    if epochs is not None:
        assert epochs > 0, f"Epoch count needs to be a positive number, but was {epochs}."
        return True
    return False


def should_override_prune_count(prunes):
    """ Check if the 'prunes'-flag was set and if the given value is valid. """
    if prunes is not None:
        assert prunes >= 0, f"Prune count needs to be a number greater or equal to zero, but was {prunes}."
        return True
    return False


def should_override_net_count(nets):
    """ Check if the 'nets'-flag was set and if the given value is valid. """
    if nets is not None:
        assert nets > 0, f"Net count needs to be a positive number, but was {nets}."
        return True
    return False


def main(experiment, epochs, nets, prunes, cuda, verbose):
    assert verbose in VerbosityLevel.__members__.values()
    if verbose != VerbosityLevel.SILENT:
        print("Welcome to bachelor_playground.")

    settings = get_settings(experiment)

    # enable CUDA
    use_cuda = cuda and is_available()
    settings.device = "cuda:0" if use_cuda else "cpu"
    settings.device_name = get_device_name(0) if use_cuda else "cpu"
    if verbose != VerbosityLevel.SILENT:
        print(settings.device_name)

    if should_override_epoch_count(epochs):
        settings.epoch_count = epochs
    if should_override_net_count(nets):
        settings.net_count = nets
    if should_override_prune_count(prunes):
        settings.prune_count = prunes
    settings.verbosity = VerbosityLevel(verbose)

    if experiment == ExperimentNames.LENET_MNIST:
        experiment = ExperimentLenetMNIST(settings)
    else:
        experiment = ExperimentConvCIFAR10(settings)
    experiment.run_experiment()


if __name__ == '__main__':
    p = ArgumentParser(description='bachelor_playground - Framework for pruning-experiments.')

    p.add_argument('experiment', choices=[n.name for n in ExperimentNames],
                   help='choose experiment')
    p.add_argument('-c', '--cuda', action='store_true', default=False, help='use cuda, if available')
    p.add_argument('-v', '--verbose', action='count', default=0,
                   help='activate output, use twice for more detailed output (i.e. -vv)')
    p.add_argument('-e', '--epochs', type=int, default=None, metavar='E', help='specify number of epochs')
    p.add_argument('-n', '--nets', type=int, default=None, metavar='N', help='specify number of trained networks')
    p.add_argument('-p', '--prunes', type=int, default=None, metavar='P', help='specify number of pruning steps')
    # p.add_argument('--plan_conv', type=str, nargs='+', help='specify convolutional layers as list of output-sizes')
    # p.add_argument('--plan_fc', nargs='+', help='specify fully-connected layers as list of output-sizes')

    args = p.parse_args()
    main(**vars(args))
