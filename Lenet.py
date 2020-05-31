import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import math

class Lenet(nn.Module):
    """
    Lenet with FC layers 300, 100, 10 for the MNIST dataset with input 28*28.
    The neural network is prunable using iterative magnitude pruning (IMP).
    Each layer stores its initial weight as parameter "weight_initial" after applying the weight initialization with Gaussian Glorot.
    """
    def __init__(self):
        super(Lenet, self).__init__()
        self.init_weight_count1 = 28*28*300
        self.init_weight_count2 = 300*100
        self.init_weight_count3 = 100*10
        self.init_weight_count_all_layers = self.init_weight_count1 + self.init_weight_count2 + self.init_weight_count3
        self.init_weights = dict()

        # Initialize the layers with Gaussian Glorot
        self.layer1 = nn.Linear(28*28, 300) # input layer
        self.layer2 = nn.Linear(300, 100) # hidden layer
        self.layer3 = nn.Linear(100, 10) # output layer
        torch.nn.init.xavier_normal_(self.layer1.weight)
        torch.nn.init.xavier_normal_(self.layer2.weight)
        torch.nn.init.xavier_normal_(self.layer3.weight)
        self.store_initial_weights()

        # Setup masks for pruning (all ones in the beginning)
        self.layer1 = prune.custom_from_mask(self.layer1, name='weight', mask=torch.ones_like(self.layer1.weight))
        self.layer2 = prune.custom_from_mask(self.layer2, name='weight', mask=torch.ones_like(self.layer2.weight))
        self.layer3 = prune.custom_from_mask(self.layer3, name='weight', mask=torch.ones_like(self.layer3.weight))

    def store_initial_weights(self):
        self.init_weights[self.layer1] = self.layer1.weight.clone()
        self.init_weights[self.layer2] = self.layer2.weight.clone()
        self.init_weights[self.layer3] = self.layer3.weight.clone()

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        x = x.view(-1, 28*28)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

    def prune_mask(self, layer, initial_weight_count, prune_rate):
        """ Prune mask for given layer, i.e. set mask to zero for weights with smallest magnitudes.
        The amount of pruning is the product of the pruning rate and the number of the layer's unpruned weights. """
        unpruned_weight_count = int(layer.weight_mask.flatten().sum())
        pruned_weight_count = int(initial_weight_count - unpruned_weight_count)
        prune_amount = math.ceil(unpruned_weight_count * prune_rate)

        # sorted_weight_indices = layer.weight[layer.weight_mask==1].abs().argsort() #  tensor[mask==1] is flattened
        sorted_weight_indices = layer.weight.flatten().abs().argsort()

        new_mask = layer.weight_mask.clone()
        new_mask.flatten()[sorted_weight_indices[pruned_weight_count:(pruned_weight_count+prune_amount)]] = 0
        #print(len(layer.weight.flatten()), len(layer.weight[layer.weight_mask==1].abs().argsort()), len(new_mask.flatten()[sorted_weight_indices[:prune_amount]]))
        return new_mask

    def prune_net(self, prune_rate):
        """ Prune all layers with the given prune rate (use half of it for the output layer).
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        pruned_mask_layer1 = self.prune_mask(self.layer1, self.init_weight_count1, prune_rate)
        pruned_mask_layer2 = self.prune_mask(self.layer2, self.init_weight_count2, prune_rate)
        # output-layer is pruned with half of the pruning rate
        pruned_mask_layer3 = self.prune_mask(self.layer3, self.init_weight_count3, prune_rate/2)

        # temporarily remove pruning
        prune.remove(self.layer1, name='weight')
        prune.remove(self.layer2, name='weight')
        prune.remove(self.layer3, name='weight')

        # set weights to initial weights
        self.layer1.weight = nn.Parameter(self.init_weights.get(self.layer1))
        self.layer2.weight = nn.Parameter(self.init_weights.get(self.layer2))
        self.layer3.weight = nn.Parameter(self.init_weights.get(self.layer3))

        # apply pruning
        self.layer1 = prune.custom_from_mask(self.layer1, name='weight',  mask=pruned_mask_layer1)
        self.layer2 = prune.custom_from_mask(self.layer2, name='weight',  mask=pruned_mask_layer2)
        self.layer3 = prune.custom_from_mask(self.layer3, name='weight',  mask=pruned_mask_layer3)

    def sparsity_report(self):
        """ Generate a list with sparsities for the whole network and per layer."""
        unpruned_weights_layer1 = int(self.layer1.weight.nonzero().numel()/2)
        unpruned_weights_layer2 = int(self.layer2.weight.nonzero().numel()/2)
        unpruned_weights_layer3 = int(self.layer3.weight.nonzero().numel()/2)
        sparsity_layer1 = unpruned_weights_layer1 / self.init_weight_count1
        sparsity_layer2 = unpruned_weights_layer2 / self.init_weight_count2
        sparsity_layer3 = unpruned_weights_layer3 / self.init_weight_count3
        sparsity = (unpruned_weights_layer1 + unpruned_weights_layer2 + unpruned_weights_layer3) / self.init_weight_count_all_layers
        return np.round([sparsity, sparsity_layer1, sparsity_layer2, sparsity_layer3], 4)
