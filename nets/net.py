import numpy as np
import torch
import torch.nn as nn

from pruning.magnitude_pruning import prune_layer


class Net(nn.Module):
    """
    Superclass for a trainable and prunable network.
    Its architecture can be specified via sizes in plan_conv and plan_fc.
    Initial weights for each layer are stored as buffers after applying the weight initialization.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.init_weight_count_net = dict([('conv', 0), ('fc', 0)])
        self.conv = None
        self.fc = None
        self.criterion = None

    def store_initial_weights(self):
        """ Store initial weights as buffer in each layer. """
        pass

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        pass

    def prune_net(self, prune_rate_conv, prune_rate_fc, reset=True):
        """ Prune all layers with the given prune rate using weight masks (use half of it for the output layer).
        If 'reset' is True, the unpruned weights are set to their initial values after pruning. """
        for layer in self.conv:
            prune_layer(layer, prune_rate_conv, reset)
        for layer in self.fc:
            prune_layer(layer, prune_rate_fc, reset)

        # prune output-layer with half of the fc pruning rate
        prune_layer(self.out, prune_rate_fc / 2, reset)

    @staticmethod
    def sparsity_layer(layer):
        """ Calculate sparsity and counts unpruned weights for given layer. """
        if isinstance(layer, nn.Linear):
            unpr_weight_count = int(layer.weight.nonzero().numel() / 2)
            init_weight_count = layer.in_features * layer.out_features
        elif isinstance(layer, nn.Conv2d):
            unpr_weight_count = int(layer.weight.nonzero().numel() / 4)
            init_weight_count = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            raise AssertionError(f"Could not calculate sparsity for layer of type {type(layer)}.")

        sparsity = unpr_weight_count / init_weight_count
        return sparsity, unpr_weight_count

    def sparsity_report(self):
        """ Generate a list with sparsity for the whole network and per layer. """
        unpr_weight_counts = 0
        sparsity_list = []

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                curr_sparsity, curr_unpr_weight_count = self.sparsity_layer(layer)

                sparsity_list.append(curr_sparsity)
                unpr_weight_counts += curr_unpr_weight_count
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                curr_sparsity, curr_unpr_weight_count = self.sparsity_layer(layer)

                sparsity_list.append(curr_sparsity)
                unpr_weight_counts += curr_unpr_weight_count

        out_sparsity, out_unpr_weight_count = self.sparsity_layer(self.out)
        sparsity_list.append(out_sparsity)
        unpr_weight_counts += out_unpr_weight_count

        sparsity_net = unpr_weight_counts / (self.init_weight_count_net['conv'] + self.init_weight_count_net['fc'])
        sparsity_list.insert(0, sparsity_net)
        return np.round(sparsity_list, decimals=4)

    def equal_layers(self, other):
        """ Returns True, if 'other' has the same types of layers in 'conv' and 'fc', and if all pairs of Linear- and
        Conv2d-layers have equal weight, bias and initial_weight attributes. """
        layer_list_self = [layer for layer in self.conv] + [layer for layer in self.fc]
        layer_list_other = [layer for layer in other.conv] + [layer for layer in other.fc]
        for layer_self, layer_other in zip(layer_list_self, layer_list_other):
            if type(layer_self) is not type(layer_other):
                return False
            if isinstance(layer_self, nn.Linear) or isinstance(layer_other, nn.Conv2d):
                if not torch.equal(layer_self.weight, layer_other.weight):
                    return False
                if not torch.equal(layer_self.bias, layer_other.bias):
                    return False
                if not torch.equal(layer_self.weight_init, layer_other.weight_init):
                    return False
        return True
