import unittest

import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
test_dir = os.path.join(os.getcwd(), 'test')
if os.path.exists(test_dir):
    os.chdir(test_dir)
    
import experiments.experiment_settings as experiment_settings
from experiments.experiment_lenet_mnist import Experiment_Lenet_MNIST
from experiments.experiment_conv_cifar10 import Experiment_Conv_CIFAR10

class Test_experiment(unittest.TestCase):
    """ Tests for the experiment package.
    Call with 'python -m test.test_experiment' from project root '~'.
    Call with 'python -m test_experiment' from inside '~/test'. """
    def test_experiment_lenet_mnist_should_init(self):
        settings = experiment_settings.get_settings_lenet_mnist()
        settings['net_count'] = 1
        settings['device'] = 'cpu'
        experiment = Experiment_Lenet_MNIST(settings)
        experiment.setup_experiment()

    def test_experiment_conv_cifar10_should_init(self):
        settings = experiment_settings.get_settings_conv2_cifar10()
        settings['net_count'] = 1
        settings['device'] = 'cpu'
        experiment = Experiment_Conv_CIFAR10(settings)
        experiment.setup_experiment()

if __name__ == '__main__':
    unittest.main()