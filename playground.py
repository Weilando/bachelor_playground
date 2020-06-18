import argparse
import torch

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(current_path, '.')))
from experiments import experiment_settings
from experiments.experiment_lenet_mnist import Experiment_Lenet_MNIST
from experiments.experiment_conv_cifar10 import Experiment_Conv_CIFAR10
os.chdir(os.path.join(current_path, 'experiments'))

def main(experiment, epochs, nets, prunes, cuda, fast):
    print("Welcome to bachelor_playground.")

    # load settings
    if experiment=='lenet-mnist':
        if fast:
            settings = experiment_settings.get_settings_lenet_mnist_f()
        else:
            settings = experiment_settings.get_settings_lenet_mnist()
    elif experiment=='conv2-cifar10':
        if fast:
            settings = experiment_settings.get_settings_conv2_cifar10_f()
        else:
            settings = experiment_settings.get_settings_conv2_cifar10()
    elif experiment=='conv4-cifar10':
        if fast:
            settings = experiment_settings.get_settings_conv4_cifar10_f()
        else:
            settings = experiment_settings.get_settings_conv4_cifar10()
    else:
        if fast:
            settings = experiment_settings.get_settings_conv6_cifar10_f()
        else:
            settings = experiment_settings.get_settings_conv6_cifar10()

    # enable CUDA
    use_cuda = cuda and torch.cuda.is_available()
    settings['device'] = "cuda:0" if use_cuda else "cpu"
    settings['device_name'] = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(settings['device_name'])

    if epochs != None:
        settings['epoch_count'] = epochs
    if nets != None:
        settings['net_count'] = nets
    if prunes != None:
        settings['prune_count'] = prunes

    if experiment=='lenet-mnist':
        experiment = Experiment_Lenet_MNIST(settings)
    else:
        experiment = Experiment_Conv_CIFAR10(settings)
    experiment.run_experiment()

if __name__=='__main__':
    p = argparse.ArgumentParser(description='bachelor_playground - Framework for pruning-experiments.')

    p.add_argument('experiment', choices=['lenet-mnist', 'conv2-cifar10', 'conv4-cifar10', 'conv6-cifar10'], help='choose experiment')
    p.add_argument('-c', '--cuda', action='store_true', default=False, help='use cuda, if available')
    p.add_argument('-f', '--fast', action='store_true', default=False, help='use fast version, i.e. less epochs')
    p.add_argument('-e', '--epochs', type=int, default=None, metavar='E', help='specify number of epochs')
    p.add_argument('-n', '--nets', type=int, default=None, metavar='N', help='specify number of trained networks')
    p.add_argument('-p', '--prunes', type=int, default=None, metavar='P', help='specify number of pruning steps')
    # p.add_argument('--plan_conv', type=str, nargs='+', help='specify convolutional layers as list of output-sizes')
    # p.add_argument('--plan_fc', nargs='+', help='specify fully-connected layers as list of output-sizes')

    args = p.parse_args()
    main(**vars(args))
