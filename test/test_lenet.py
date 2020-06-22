import torch
import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.lenet import Lenet

class Test_lenet(unittest.TestCase):
    """ Tests for the Lenet class.
    Call with 'python -m test.test_lenet' from project root '~'.
    Call with 'python -m test_lenet' from inside '~/test'. """
    def test_forward_pass(self):
        ''' The neural network should perform a forward pass without exceptions. '''
        net = Lenet()
        input_sample = torch.rand(28, 28)
        net(input_sample)
        self.assertEqual(266200, net.init_weight_count_net)

    def test_forward_pass_simple_architecture(self):
        ''' The neural network should perform a forward pass for  without exceptions. '''
        net = Lenet(plan_fc=[5, 10])
        input_sample = torch.rand(28, 28)
        net(input_sample)
        self.assertEqual(4070, net.init_weight_count_net)

    def test_sparsity_report_initial_weights(self):
        ''' The neural network should be fully connected right after initialization. '''
        net = Lenet()
        sparsity_report = net.sparsity_report()
        self.assertTrue(([1.0, 1.0, 1.0, 1.0] == sparsity_report).all())

    def test_sparsity_report_after_single_prune(self):
        ''' Each layer should be pruned with the given pruning rate, except for the last layer.
        The last layer needs to be pruned using half of the pruning rate.
        For the whole net's sparsity we get:
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9 + (300*100)*0.9 + (100*10)*0.95) / 266200 ~ 0.9002 '''
        net = Lenet()
        net.prune_net(prune_rate=0.1)
        sparsity_report = net.sparsity_report()
        self.assertTrue(([0.9002, 0.9, 0.9, 0.95] == sparsity_report).all())

    def test_sparsity_report_after_double_prune(self):
        ''' Each layer should be pruned with the given pruning rate, except for the last layer.
        The last layer needs to be pruned using half of the pruning rate.
        For the whole net's sparsity we get:
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9^2 + (300*100)*0.9^2 + (100*10)*0.95^2) / 266200 ~ 0.8103 '''
        net = Lenet()
        net.prune_net(prune_rate=0.1)
        net.prune_net(prune_rate=0.1)
        sparsity_report = net.sparsity_report()
        self.assertTrue(([0.8103, 0.81, 0.81, 0.902] == sparsity_report).all())

if __name__ == '__main__':
    unittest.main()
