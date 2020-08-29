from unittest import TestCase, main as unittest_main

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import pruning.magnitude_pruning as mp


class TestPruning(TestCase):
    """ Tests for the pruning logic.
    Call with 'python -m test.test_pruning' from project root '~'.
    Call with 'python -m test_pruning' from inside '~/test'. """

    def test_setup_mask_correctly(self):
        """ Should create weight-masks for all sub-layers recursively. """
        test_net = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=2, padding=1),
            nn.Linear(2, 3))

        test_net.apply(mp.setup_masks)

        self.assertIs(test_net[0].weight_mask.equal(torch.ones_like(test_net[0].weight)), True)
        self.assertIs(test_net[1].weight_mask.equal(torch.ones_like(test_net[1].weight)), True)

    def test_prune_mask_for_linear_layer_correctly(self):
        """ Prune the mask for an unpruned linear layer in one step.
        Should zero out the two weights with the lowest magnitude. """
        # initialize linear layer with 10 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        test_mask_pruned = mp.prune_mask(layer=test_layer, prune_rate=0.2)
        self.assertIs(test_mask_pruned.equal(torch.tensor([[0., 1., 1., 1., 1.], [1., 1., 1., 0., 1.]])), True)

    def test_prune_linear_layer_correctly(self):
        """ Prune the mask for a pruned linear layer in one step.
        Should zero out the two weights (as ceil(8*0.2)=2) with the lowest magnitude. """
        # initialize linear layer with 10 given weights and pruned mask
        initial_weights = torch.tensor([[1., -2., 3., -1.5, -3.], [-1., 2., -4., 0.5, 1.5]])
        initial_mask = torch.tensor([[0., 1., 1., 1., 1.], [1., 1., 1., 0., 1.]])
        test_layer = nn.Linear(2, 5)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer.register_buffer('weight_init', initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=initial_mask)

        mp.prune_layer(layer=test_layer, prune_rate=0.2)

        expected_weights = torch.tensor([[0., -2., 3., -0., -3.], [-0., 2., -4., 0., 1.5]])
        self.assertIs(test_layer.weight.equal(expected_weights), True)

    def test_prune_mask_for_conv_layer_correctly(self):
        """ Prune the mask for an unpruned convolutional layer in one step.
        Should zero out the two weights with the lowest magnitude. """
        # Initialize conv layer with 16 given weights and unpruned mask
        initial_weights = torch.tensor(
            [1.2, -0.1, 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0.2, 0.4, 1.4, 2.2, -0.8, 0.4, 0.9]).view(2, 2, 2, 2)
        test_layer = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        test_mask_pruned = mp.prune_mask(layer=test_layer, prune_rate=0.2)

        expected_mask = torch.tensor([1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1.]).view(2, 2, 2, 2)
        self.assertIs(test_mask_pruned.equal(expected_mask), True)

    def test_prune_conv_layer_correctly(self):
        """ Prune the mask for a pruned convolutional layer in one step.
        Should zero out the three weights (as ceil(12*0.2)=3) with the lowest magnitude. """
        # initialize conv layer with 16 given weights and pruned mask
        initial_weights = torch.tensor(
            [1.2, -0.1, 1.2, 4.3, -2.1, -1.1, -0.8, 1.2, 0.5, 0.2, 0.4, 1.4, 2.2, -0.8, 0.4, 0.9]).view(2, 2, 2, 2)
        initial_mask = torch.tensor([1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1.]).view(2, 2, 2, 2)
        test_layer = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        test_layer.weight = nn.Parameter(initial_weights.clone())
        test_layer.register_buffer('weight_init', initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=initial_mask)

        mp.prune_layer(layer=test_layer, prune_rate=0.2)

        expected_weights = torch.tensor(
            [1.2, -0., 1.2, 4.3, -2.1, -1.1, -0., 1.2, 0., 0., 0., 1.4, 2.2, -0., 0., 0.9]).view(2, 2, 2, 2)
        self.assertIs((test_layer.weight == expected_weights).all().item(), True)

    def test_apply_init_weight_after_pruning_linear_layer(self):
        """ Generate, modify and prune an unpruned linear layer.
        Its weights should be reset to the initial values. """
        # initialize linear layer with 6 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3.], [-4., 5., -6.]])
        test_layer = nn.Linear(2, 3)
        test_layer.weight = nn.Parameter(2 * initial_weights.clone())  # fake training, i.e. save modify weights
        test_layer.register_buffer('weight_init', initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        mp.prune_layer(layer=test_layer, prune_rate=0.2)

        expected_weights = torch.tensor([[0., -0., 3.], [-4., 5., -6.]])
        self.assertIs(test_layer.weight.equal(expected_weights), True)

    def test_do_not_apply_init_weight_after_pruning_linear_layer(self):
        """ Generate, modify and prune an unpruned linear layer.
        Its weights should not be reset. """
        # initialize linear layer with 6 given weights and unpruned mask
        initial_weights = torch.tensor([[1., -2., 3.], [-4., 5., -6.]])
        test_layer = nn.Linear(2, 3)
        test_layer.weight = nn.Parameter(2 * initial_weights.clone())  # fake training, i.e. save modify weights
        test_layer.register_buffer('weight_init', initial_weights.clone())
        test_layer = prune.custom_from_mask(test_layer, name='weight', mask=torch.ones_like(test_layer.weight))

        mp.prune_layer(layer=test_layer, prune_rate=0.2, reset=False)

        expected_weights = torch.tensor([[0., -0., 6.], [-8., 10., -12.]])
        self.assertIs(test_layer.weight.equal(expected_weights), True)

    def test_prune_mask_raise_error_on_invalid_layer_type(self):
        """ Should raise an assertion error, because no pruning procedure is defined. """
        with self.assertRaises(AssertionError):
            mp.prune_mask(nn.MaxPool2d(2), 0.2)


if __name__ == '__main__':
    unittest_main()
