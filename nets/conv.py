import torch.nn as nn

from nets.net import Net
from nets.plan_check import is_numerical_spec, is_batch_norm_spec, get_number_from_batch_norm_spec, \
    get_number_from_numerical_spec
from nets.weight_initializer import gaussian_glorot
from pruning.magnitude_pruning import setup_masks


class Conv(Net):
    """
    Convolutional network with convolutional layers in the beginning and fully-connected layers afterwards.
    Its architecture can be specified via sizes (positive integers) in plan_conv and plan_fc.
    'A' and and 'M' have special roles in plan_conv, as they generate Average- and Max-Pooling layers.
    Append 'B' to any size in plan_conv to add a Batch-Norm layer directly behind the convolutional layer.
    If no architecture is specified, a Conv-2 architecture is generated.
    Works for the CIFAR-10 dataset with input 32*32*3.
    Initial weights for each layer are stored as buffers after applying the weight initialization with Gaussian Glorot.
    """

    def __init__(self, plan_conv=None, plan_fc=None):
        super(Conv, self).__init__()
        # create Conv-2 if no plans are given
        if plan_conv is None:
            plan_conv = [64, 64, 'M']
        if plan_fc is None:
            plan_fc = [256, 256]

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
            elif is_batch_norm_spec(spec):
                spec_number = get_number_from_batch_norm_spec(spec)
                conv_layers.append(nn.Conv2d(filters, spec_number, kernel_size=3, padding=1))
                conv_layers.append(nn.BatchNorm2d(spec_number))
                conv_layers.append(nn.ReLU())
                self.init_weight_count_net['conv'] += filters * spec_number * 9
                filters = spec_number
            elif is_numerical_spec(spec):
                spec_number = get_number_from_numerical_spec(spec)
                conv_layers.append(nn.Conv2d(filters, spec_number, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU())
                self.init_weight_count_net['conv'] += filters * spec_number * 9
                filters = spec
            else:
                raise AssertionError(f"{spec} from plan_conv is an invalid spec.")

        # Each Pooling-layer quarters the input size (32*32=1024)
        filters = filters * round(1024 / (4 ** pooling_count))
        for spec in plan_fc:
            assert is_numerical_spec(spec), f"{spec} from plan_fc is not a numerical spec."
            spec_number = get_number_from_numerical_spec(spec)
            fc_layers.append(nn.Linear(filters, spec_number))
            fc_layers.append(nn.ReLU())
            self.init_weight_count_net['fc'] += filters * spec_number
            filters = spec_number

        self.plan_conv = plan_conv
        self.plan_fc = plan_fc
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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out(x)

    def get_new_instance(self, reset_weight=True):
        """ Return a copy of this net with pruned mask.
        If 'reset_weight' is True, the copy is untrained. """
        new_net = Conv(self.plan_conv, self.plan_fc)
        new_net.load_state_dict(self.state_dict())
        new_net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.0, reset=reset_weight)  # reapply pruned mask
        return new_net
