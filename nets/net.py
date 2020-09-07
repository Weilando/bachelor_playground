import numpy as np
import torch
import torch.nn as nn

from data.data_loaders import get_sample_shape
from experiments.experiment_specs import DatasetNames, NetNames
from nets.plan_check import get_activation, get_number_from_batch_norm_spec, get_number_from_numerical_spec, \
    is_batch_norm_spec, is_numerical_spec
from nets.weight_initializer import gaussian_glorot
from pruning.magnitude_pruning import prune_layer, setup_masks


class Net(nn.Module):
    """
    Convolutional neural network with convolutional layers in the beginning and fully-connected layers afterwards.
    Its architecture can be specified via sizes (positive integers) in plan_conv and plan_fc.
    'A' and and 'M' have special roles in plan_conv, as they generate Average- and Max-Pooling layers.
    Append 'B' to any size in plan_conv to add a Batch-Norm layer directly behind the convolutional layer.
    Works for the MNIST dataset with samples of shape [1, 28, 28] and inputs from CIFAR-10 with shape [3, 32, 32].
    'net_name' specifies if this network should be a ReLU-network (CONV) or uses tanh-activations like Lenet (LENET).
    Initial weights for each layer are stored as buffers after applying the weight initialization with Gaussian Glorot.
    """

    def __init__(self, net_name: NetNames, dataset_name: DatasetNames, plan_conv: list, plan_fc: list):
        super(Net, self).__init__()
        assert (net_name == NetNames.CONV) or (net_name == NetNames.LENET), \
            f"Could not initialize net, because the given name {net_name} is invalid."

        self.init_weight_count_net = dict([('conv', 0), ('fc', 0)])
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.plan_conv = plan_conv
        self.plan_fc = plan_fc

        # create and initialize layers with Gaussian Glorot
        conv_layers = []
        fc_layers = []
        sample_shape = get_sample_shape(dataset_name)
        filters = sample_shape[0]
        pooling_count = 0

        for spec in plan_conv:
            if spec == 'A':
                conv_layers.append(nn.AvgPool2d(2))
                pooling_count += 1
            elif spec == 'M':
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pooling_count += 1
            elif is_batch_norm_spec(spec):
                spec_number = get_number_from_batch_norm_spec(spec)
                conv_layers.append(nn.Conv2d(filters, spec_number, kernel_size=3, padding=1))
                conv_layers.append(nn.BatchNorm2d(spec_number))
                conv_layers.append(get_activation(net_name))
                self.init_weight_count_net['conv'] += filters * spec_number * 9
                filters = spec_number
            elif is_numerical_spec(spec):
                spec_number = get_number_from_numerical_spec(spec)
                conv_layers.append(nn.Conv2d(filters, spec_number, kernel_size=3, padding=1))
                conv_layers.append(get_activation(net_name))
                self.init_weight_count_net['conv'] += filters * spec_number * 9
                filters = spec_number
            else:
                raise AssertionError(f"{spec} from plan_conv is an invalid spec.")

        # each Pooling-layer quarters the input size (height * width)
        filters = filters * round((sample_shape[1] * sample_shape[2]) / (4 ** pooling_count))
        for spec in plan_fc:
            assert is_numerical_spec(spec), f"{spec} from plan_fc is not a numerical spec."
            spec_number = get_number_from_numerical_spec(spec)
            fc_layers.append(nn.Linear(filters, spec_number))
            fc_layers.append(get_activation(net_name))
            self.init_weight_count_net['fc'] += filters * spec_number
            filters = spec_number

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)
        self.out = nn.Linear(filters, 10)
        self.init_weight_count_net['fc'] += filters * 10
        self.criterion = nn.CrossEntropyLoss()

        self.apply(gaussian_glorot)
        self.store_initial_weights()
        self.apply(setup_masks)

    def store_initial_weights(self):
        """ Store initial weights as buffer in each layer. """
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                layer.register_buffer('weight_init', layer.weight.clone())
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.register_buffer('weight_init', layer.weight.clone())
        self.out.register_buffer('weight_init', self.out.weight.clone())

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        x = self.conv(x)  # x has shape [batch_size, channels, height, width]
        x = x.view(x.size(0), -1)  # squash all dimensions but batch_size, get [batch_size, channels * height * width]
        x = self.fc(x)
        return self.out(x)

    def prune_net(self, prune_rate_conv, prune_rate_fc, reset=True):
        """ Prune all layers with the given prune rate using weight masks (use half of it for the output layer).
        If 'reset' is True, the unpruned weights are set to their initial values after pruning. """
        for layer in self.conv:
            prune_layer(layer, prune_rate_conv, reset)
        for layer in self.fc:
            prune_layer(layer, prune_rate_fc, reset)
        prune_layer(self.out, prune_rate_fc / 2, reset)  # prune output-layer with half of prune_rate_fc

    def get_new_instance(self, reset_weight=True):
        """ Return a copy of this net with pruned mask.
        If 'reset_weight' is True, the copy is untrained. """
        new_net = Net(self.net_name, self.dataset_name, self.plan_conv, self.plan_fc)
        new_net.load_state_dict(self.state_dict())
        new_net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.0, reset=reset_weight)  # reapply pruned mask
        return new_net

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
        """ Return True, if 'other' has the same types of layers in 'conv' and 'fc', and if all pairs of Linear- and
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
