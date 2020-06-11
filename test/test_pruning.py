import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pruning.magnitude_pruning as mp

class Pruning_test(unittest.TestCase):
    """ Tests for the pruning logic.
    Call with 'python -m test.pruning_test' from project root '~'.
    Call with 'python -m pruning_test' from inside '~/test'. """
    def test_setup_mask_correctly(self):
        """ Create weight-masks for all sublayers recursively. """
        test_net = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=2, padding=1),
            nn.Linear(2,3))

        test_net.apply(mp.setup_masks)

        expected_con = torch.ones_like(test_net[0].weight)
        expected_lin = torch.ones_like(test_net[1].weight)
        self.assertTrue(test_net[0].weight_mask.equal(expected_con))
        self.assertTrue(test_net[1].weight_mask.equal(expected_lin))

    def test_prune_mask_for_linear_layer_correcly(self):
        """ Prune the mask for an unpruned linear layer in one step.
        The two weights with the lowest magnitude should be zeroed out. """
        # Initialize linear layer with 10 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        test_mask_pruned = mp.prune_mask(layer=test_layer, prune_rate=0.2)

        expected_mask = torch.tensor([[0.,1.,1.,1.,1.], [1.,1.,1.,0.,1.]])
        self.assertTrue(test_mask_pruned.equal(expected_mask))

    def test_prune_linear_layer_correcly_once(self):
        """ Prune an unpruned linear layer in one step.
        The two weights with the lowest magnitude should be zeroed out. """
        # Initialize linear layer with 10 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        mp.prune_layer(layer=test_layer, prune_rate=0.2, init_weights=initial_weights)

        expected_weights = torch.tensor([[0., -2., 3., -1.5, -3.], [-1., 2., -4., 0., 1.5]])
        self.assertTrue(test_layer.weight.equal(expected_weights))

    def test_prune_linear_layer_correcly_twice(self):
        """ Prune the mask for a pruned linear layer in one step.
        The two weights (as ceil(8*0.2)=2) with the lowest magnitude should be zeroed out. """
        # Initialize linear layer with 10 given weights and pruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        initial_mask = torch.tensor([[0.,1.,1.,1.,1.], [1.,1.,1.,0.,1.]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=initial_mask)

        mp.prune_layer(layer=test_layer, prune_rate=0.2, init_weights=initial_weights)

        expected_weights = torch.tensor([[0., -2., 3., -0., -3.], [-0., 2., -4., 0., 1.5]])
        self.assertTrue(test_layer.weight.equal(expected_weights))

    def test_prune_mask_for_conv_layer_correcly(self):
        """ Prune the mask for an unpruned convolutional layer in one step.
        The two weights with the lowest magnitude should be zeroed out. """
        # Initialize conv layer with 16 given weights and unpruned mask
        initial_weights = torch.tensor([1.2, -0.1, 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0.2, 0.4, 1.4, 2.2, -0.8, 0.4, 0.9]).view(2,2,2,2)
        test_layer = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        test_mask_pruned = mp.prune_mask(layer=test_layer, prune_rate=0.2)

        expected_mask = torch.tensor([1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1.]).view(2,2,2,2)
        self.assertTrue(test_mask_pruned.equal(expected_mask))

    def test_prune_conv_layer_correcly_once(self):
        """ Prune an unpruned convolutional layer in one step.
        The four weights with the lowest magnitude should be zeroed out. """
        # Initialize conv layer with 16 given weights and unpruned mask
        initial_weights = torch.tensor([1.2, -0.1, 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0.2, 0.4, 1.4, 2.2, -0.8, 0.4, 0.9]).view(2,2,2,2)
        test_layer = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        mp.prune_layer(layer=test_layer, prune_rate=0.2, init_weights=initial_weights)

        expected_weights = torch.tensor([1.2, -0., 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0., 0., 1.4, 2.2, -0.8, 0., 0.9]).view(2,2,2,2)
        self.assertTrue(test_layer.weight.equal(expected_weights))

    def test_prune_linear_layer_correcly_twice(self):
        """ Prune the mask for a pruned linear layer in one step.
        The three weights (as ceil(12*0.2)=3) with the lowest magnitude should be zeroed out. """
        # Initialize conv layer with 16 given weights and unpruned mask
        initial_weights = torch.tensor([1.2, -0., 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0., 0., 1.4, 2.2, -0.8, 0., 0.9]).view(2,2,2,2)
        initial_mask = torch.tensor([1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1.]).view(2,2,2,2)
        test_layer = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=initial_mask)

        mp.prune_layer(layer=test_layer, prune_rate=0.2, init_weights=initial_weights)

        expected_weights = torch.tensor([1.2, -0., 1.2, 4.3, -2.1, -1.1, -0., 1.2, 0., 0., 0., 1.4, 2.2, -0., 0., 0.9]).view(2,2,2,2)
        self.assertTrue((test_layer.weight==expected_weights).all())


if __name__ == '__main__':
    unittest.main()
