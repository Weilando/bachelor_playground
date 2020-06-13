import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.weight_init import gaussian_glorot
from pruning.magnitude_pruning import prune_layer, setup_masks

class Lenet(nn.Module):
    """
    Lenet with FC layers for the MNIST dataset with input 28*28.
    Layer sizes can be set via argument fc_plan.
    Create Lenet 300-100 if no plan is specified.
    The neural network is prunable using iterative magnitude pruning (IMP).
    Initial weights for each layer are stored in the dict "init_weights" after applying the weight initialization with Gaussian Glorot.
    """
    def __init__(self, fc_plan=[300, 100]):
        super(Lenet, self).__init__()
        # statistics
        self.init_weight_count_net = 0
        self.init_weights = dict()

        # create and initialize layers with Gaussian Glorot
        fc_layers = []
        input = 784 # 28*28=784, dimension of samples in MNIST

        for spec in fc_plan:
            fc_layers.append(nn.Linear(input, spec))
            fc_layers.append(nn.Tanh())
            self.init_weight_count_net += input * spec
            input = spec

        self.fc = nn.Sequential(*fc_layers)
        self.out = nn.Linear(input, 10)
        self.init_weight_count_net += input * 10
        self.crit = nn.CrossEntropyLoss()

        self.apply(gaussian_glorot)
        self.store_initial_weights()
        self.apply(setup_masks)

    def store_initial_weights(self):
        for layer in self.fc:
            if isinstance(layer, torch.nn.Linear):
                self.init_weights[layer] = layer.weight.clone()
        self.init_weights[self.out] = self.out.weight.clone()

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        x = x.view(-1, 784) # 28*28=784, dimension of samples in MNIST
        x = self.fc(x)
        x = self.out(x)
        return x

    def prune_net(self, prune_rate):
        """ Prune all layers with the given prune rate (use half of it for the output layer).
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        for layer in self.fc:
            prune_layer(layer, prune_rate, self.init_weights.get(layer))
        # prune output-layer with half of the pruning rate
        prune_layer(self.out, prune_rate/2, self.init_weights.get(self.out))

    def sparsity_layer(self, layer):
        """ Calculates sparsity and counts unpruned weights for given layer. """
        unpr_weight_count = int(layer.weight.nonzero().numel()/2)
        init_weight_count = layer.in_features * layer.out_features

        sparsity = unpr_weight_count / init_weight_count
        return (sparsity, unpr_weight_count)

    def sparsity_report(self):
        """ Generate a list with sparsities for the whole network and per layer. """
        unpr_weight_counts = 0
        sparsities = []
        for layer in self.fc:
             if isinstance(layer, torch.nn.Linear):
                 curr_sparsity, curr_unpr_weight_count = self.sparsity_layer(layer)

                 sparsities.append(curr_sparsity)
                 unpr_weight_counts += curr_unpr_weight_count

        out_sparsity, out_unpr_weight_count = self.sparsity_layer(self.out)
        sparsities.append(out_sparsity)
        unpr_weight_counts += out_unpr_weight_count

        sparsity_net = unpr_weight_counts / self.init_weight_count_net
        sparsities.insert(0, sparsity_net)
        return np.round(sparsities, decimals=4)
