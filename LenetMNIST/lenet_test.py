import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import unittest

from lenet import Lenet

class Lenet_test(unittest.TestCase):
    """ Tests for the Lenet class.
    Call with 'python3 Lenet_test.py'. """
    def test_forward_pass(self):
        ''' The neural network should perform a forward pass without exceptions. '''
        net = Lenet()
        input_sample = torch.rand(28, 28)
        net(input_sample)

    def test_sparsity_report_initial_weights(self):
        ''' The neural network should be fully connected right after initialization. '''
        net = Lenet()
        sparsity_report = net.sparsity_report()
        self.assertTrue(([1.0, 1.0, 1.0, 1.0] == sparsity_report).all())

    def test_correct_initial_weight_counts(self):
        ''' Initial weight counts should store correct dimensions. '''
        net = Lenet()
        self.assertEqual(28*28*300, int(net.layer1.weight.nonzero().numel()/2))
        self.assertEqual(300*100, int(net.layer2.weight.nonzero().numel()/2))
        self.assertEqual(100*10, int(net.layer3.weight.nonzero().numel()/2))
        self.assertEqual(28*28*300, net.init_weight_count1)
        self.assertEqual(300*100, net.init_weight_count2)
        self.assertEqual(100*10, net.init_weight_count3)

    def test_prune_mask_for_toy_layer_correcly_once(self):
        """ Prune the mask for an unpruned linear layer in one step.
        The two weights with the lowest magnitude should be zeroed out. """
        # Initialize linear layer with 10 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        test_net = Lenet() # Internal layers are not used
        test_mask_pruned = test_net.prune_mask(layer=test_layer, initial_weight_count=10, prune_rate=0.2)

        expected_mask = torch.tensor([[0.,1.,1.,1.,1.], [1.,1.,1.,0.,1.]])
        self.assertTrue(test_mask_pruned.equal(expected_mask))

    def test_prune_mask_for_toy_layer_correcly_twice(self):
        """ Prune the mask for a pruned linear layer in one step.
        The two weights (as ceil(8*0.2)) with the lowest magnitude should be zeroed out. """
        # Initialize linear layer with 10 given weights and pruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        initial_mask = torch.tensor([[0.,1.,1.,1.,1.], [1.,1.,1.,0.,1.]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=initial_mask)

        test_net = Lenet() # Internal layers are not used
        test_mask_pruned = test_net.prune_mask(layer=test_layer, initial_weight_count=10, prune_rate=0.2)

        expected_mask = torch.tensor([[0.,1.,1.,0.,1.], [0.,1.,1.,0.,1.]])
        self.assertTrue(test_mask_pruned.equal(expected_mask))

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
