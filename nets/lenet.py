import torch.nn as nn

from nets.net import Net
from nets.plan_check import is_numerical_spec, get_number_from_numerical_spec
from nets.weight_initializer import gaussian_glorot
from pruning.magnitude_pruning import setup_masks


class Lenet(Net):
    """
    Lenet with FC layers for the MNIST dataset with input 28*28.
    Layer sizes can be set via argument plan_fc.
    Create Lenet 300-100 if no plan is specified.
    The neural network is prunable using iterative magnitude pruning (IMP).
    Initial weights for each layer are stored as buffers after applying the weight initialization with Gaussian Glorot.
    """
    def __init__(self, plan_fc=None):
        super(Lenet, self).__init__()
        # create Lenet 300-100 if no plan is given
        if plan_fc is None:
            plan_fc = [300, 100]

        # create and initialize layers with Gaussian Glorot
        fc_layers = []
        input_features = 784  # 28*28=784, dimension of samples in MNIST

        for spec in plan_fc:
            assert is_numerical_spec(spec), f"{spec} from plan_fc is not a numerical spec."
            spec_number = get_number_from_numerical_spec(spec)
            fc_layers.append(nn.Linear(input_features, spec_number))
            fc_layers.append(nn.Tanh())
            self.init_weight_count_net['fc'] += input_features * spec_number
            input_features = spec_number

        self.conv = []
        self.fc = nn.Sequential(*fc_layers)
        self.plan_fc = plan_fc
        self.out = nn.Linear(input_features, 10)
        self.init_weight_count_net['fc'] += input_features * 10
        self.criterion = nn.CrossEntropyLoss()

        self.apply(gaussian_glorot)
        self.store_initial_weights()
        self.apply(setup_masks)

    def store_initial_weights(self):
        """ Store initial weights as buffer in each layer. """
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.register_buffer('weight_init', layer.weight.clone())
        self.out.register_buffer('weight_init', self.out.weight.clone())

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        x = x.view(-1, 784)  # 28*28=784, dimension of samples in MNIST
        x = self.fc(x)
        return self.out(x)

    def get_untrained_instance(self):
        """ Return a pruned, untrained version of this net, i.e. return this net with initial weights. """
        new_net = Lenet(self.plan_fc)
        new_net.load_state_dict(self.state_dict())
        new_net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.0, reset=True)  # reapply pruned mask
        return new_net
