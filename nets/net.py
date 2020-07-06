import numpy as np
import torch.nn as nn


class Net(nn.Module):
    """
    A trainable and prunable network.
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

    def prune_net(self, *prune_rates):
        """ Prune all layers with the given prune rates.
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        pass

    @staticmethod
    def sparsity_layer(layer):
        """ Calculates sparsity and counts unpruned weights for given layer. """
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
