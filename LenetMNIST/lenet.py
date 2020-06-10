import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import math

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

        self.apply(gaussian_glorot)
        self.store_initial_weights()

        # setup masks for pruning (all ones in the beginning)
        for layer in self.fc:
            if isinstance(layer, torch.nn.Linear):
                layer = prune.custom_from_mask(layer, name='weight', mask=torch.ones_like(layer.weight))
        self.out = prune.custom_from_mask(self.out, name='weight', mask=torch.ones_like(self.out.weight))

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

    def prune_mask(self, layer, prune_rate):
        """ Prune mask for given layer, i.e. set its entries to zero for weights with smallest magnitudes.
        The amount of pruning is the product of the pruning rate and the number of the layer's unpruned weights. """
        initial_weight_count = layer.in_features * layer.out_features
        unpruned_weight_count = int(layer.weight_mask.flatten().sum())
        pruned_weight_count = int(initial_weight_count - unpruned_weight_count)
        prune_amount = math.ceil(unpruned_weight_count * prune_rate)

        sorted_weight_indices = layer.weight.flatten().abs().argsort()

        new_mask = layer.weight_mask.clone()
        new_mask.flatten()[sorted_weight_indices[pruned_weight_count:(pruned_weight_count+prune_amount)]] = 0
        return new_mask

    def prune_net(self, prune_rate):
        """ Prune all layers with the given prune rate (use half of it for the output layer).
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        layer_count = 0
        for layer in self.fc:
            if isinstance(layer, torch.nn.Linear):
                pruned_mask = self.prune_mask(layer, prune_rate)

                prune.remove(layer, name='weight') # temporarily remove pruning
                layer.weight = nn.Parameter(self.init_weights.get(layer)) # set weights to initial weights

                layer = prune.custom_from_mask(layer, name='weight',  mask=pruned_mask) # apply pruned mask

                layer_count += 1

        # prune output-layer with half of the pruning rate
        pruned_mask = self.prune_mask(self.out, prune_rate/2)
        prune.remove(self.out, name='weight')
        self.out.weight = nn.Parameter(self.init_weights.get(self.out))
        self.out = prune.custom_from_mask(self.out, name='weight',  mask=pruned_mask)

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
        return np.round(sparsities, 4)

def gaussian_glorot(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(layer.weight)
