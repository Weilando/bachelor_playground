import torch.nn as nn
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.weight_initializer import gaussian_glorot
from nets.net import Net
from nets.plan_check import is_numerical_spec, is_batchnorm_spec
from pruning.magnitude_pruning import prune_layer, setup_masks

class Conv(Net):
    """
    Convolutional network with convolutional layers in the beginning and fully-connected layers afterwards.
    Its architecture can be specified via sizes (positive integers) in plan_conv and plan_fc.
    'A' and and 'M' have special roles in plan_conv, as they generate Average- and Max-Pooling layers.
    It is possible append 'B' to any size in plan_conv to add a batch-norm layer directly behind the convolutional layer.
    If no architecture is specified, a Conv2-architecture is generated.
    Works for the CIFAR-10 dataset with input 32*32*3.
    Initial weights for each layer are stored as buffers after applying the weight initialization with Gaussian Glorot.
    """
    def __init__(self, plan_conv=[64, 64, 'M'], plan_fc=[256, 256]):
        super(Conv, self).__init__()
        # statistics
        self.init_weight_count_net = dict([('conv', 0), ('fc', 0)])

        # create and initialize layers with Gaussian Glorot
        conv_layers = []
        fc_layers = []
        filters = 3
        pooling_count = 0

        for spec in plan_conv:
            if spec == 'A':
                conv_layers.append(nn.AvgPool2d(2))
                pooling_count += 1
            elif spec == 'M':
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pooling_count += 1
            elif is_batchnorm_spec(spec):
                conv_layers.append(nn.Conv2d(filters, spec, kernel_size=3, padding=1))
                conv_layers.append(nn.BatchNorm2d(spec))
                conv_layers.append(nn.ReLU())
                self.init_weight_count_net['conv'] += filters * spec * 9
                filters = spec
            elif is_numerical_spec(spec):
                conv_layers.append(nn.Conv2d(filters, spec, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU())
                self.init_weight_count_net['conv'] += filters * spec * 9
                filters = spec
            else:
                raise AssertionError(f"{spec} from plan_conv is not a numerical spec.")

        # Each Pooling-layer quarters the input size (32*32=1024)
        filters = filters * round(1024 / (4**pooling_count))
        for spec in plan_fc:
            assert is_numerical_spec(spec), f"{spec} from plan_fc is not a numerical spec."
            fc_layers.append(nn.Linear(filters, spec))
            fc_layers.append(nn.Tanh())
            self.init_weight_count_net['fc'] += filters * spec
            filters = spec

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)
        self.out = nn.Linear(filters, 10)
        self.init_weight_count_net['fc'] += filters * 10
        self.crit = nn.CrossEntropyLoss()

        self.apply(gaussian_glorot)
        self.store_initial_weights()
        self.apply(setup_masks)

    def store_initial_weights(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                layer.register_buffer('weight_init', layer.weight.clone())
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.register_buffer('weight_init', layer.weight.clone())
        self.out.register_buffer('weight_init', self.out.weight.clone())

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        return x

    def prune_net(self, prune_rate_conv, prune_rate_fc):
        """ Prune all layers with the given prune rate (use half of it for the output layer).
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        for layer in self.conv:
            prune_layer(layer, prune_rate_conv)
        for layer in self.fc:
            prune_layer(layer, prune_rate_fc)
        # prune output-layer with half of the fc pruning rate
        prune_layer(self.out, prune_rate_fc/2)

    def sparsity_layer(self, layer):
        """ Calculates sparsity and counts unpruned weights for given layer. """
        if isinstance(layer, nn.Linear):
            unpr_weight_count = int(layer.weight.nonzero().numel()/2)
            init_weight_count = layer.in_features * layer.out_features
        elif isinstance(layer, nn.Conv2d):
            unpr_weight_count = int(layer.weight.nonzero().numel()/4)
            init_weight_count = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            raise AssertionError(f"Could not calculate sparsity for layer of type {type(layer)}.")

        sparsity = unpr_weight_count / init_weight_count
        return (sparsity, unpr_weight_count)

    def sparsity_report(self):
        """ Generate a list with sparsities for the whole network and per layer. """
        unpr_weight_counts = 0
        sparsities = []

        for layer in self.conv:
             if isinstance(layer, nn.Conv2d):
                 curr_sparsity, curr_unpr_weight_count = self.sparsity_layer(layer)

                 sparsities.append(curr_sparsity)
                 unpr_weight_counts += curr_unpr_weight_count
        for layer in self.fc:
             if isinstance(layer, nn.Linear):
                 curr_sparsity, curr_unpr_weight_count = self.sparsity_layer(layer)

                 sparsities.append(curr_sparsity)
                 unpr_weight_counts += curr_unpr_weight_count

        out_sparsity, out_unpr_weight_count = self.sparsity_layer(self.out)
        sparsities.append(out_sparsity)
        unpr_weight_counts += out_unpr_weight_count

        sparsity_net = unpr_weight_counts / (self.init_weight_count_net['conv'] + self.init_weight_count_net['fc'])
        sparsities.insert(0, sparsity_net)
        return np.round(sparsities, decimals=4)
