import argparse

p = argparse.ArgumentParser()
p.add_argument('experiment', choices=['lenet-mnist', 'conv2-cifar10', 'conv4-cifar10', 'conv6-cifar10'], help='Choose experiment.')
p.add_argument('-f', '--fast', action='store_true', help='Use fast version, i.e. less epochs.')
#p.add_argument('--plan_fc', nargs='+', help='Specify fully-connected layers as list of output-sizes.')
#p.add_argument('--plan_conv', nargs='+', help='Specify convolutional layers as list of output-sizes.')
p.add_argument('-e', '--epochs', nargs=1, type=int, help='Specify number of epochs.')
p.add_argument('-n', '--nets', nargs=1, type=int, help='Specify number of trained networks.')

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(current_path, '.')))
from experiments import experiment_settings
from experiments.experiment_lenet_mnist import Experiment_Lenet_MNIST
from experiments.experiment_conv_cifar10 import Experiment_Conv_CIFAR10
os.chdir(os.path.join(current_path, 'experiments'))

def main(experiment, nets, epochs, fast):
    print("Welcome to bachelor_playground.")
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

    if not nets==None:
        settings['net_count'] = nets[0]
    if not epochs==None:
        settings['epoch_count'] = epochs[0]

    if experiment=='lenet-mnist':
        experiment = Experiment_Lenet_MNIST(settings)
    else:
        experiment = Experiment_Conv_CIFAR10(settings)
    experiment.run_experiment()


if __name__=='__main__':
    args = p.parse_args()
    main(**vars(args))
