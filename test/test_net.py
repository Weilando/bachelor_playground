from unittest import TestCase
from unittest import main as unittest_main

import torch
from torch import nn

from nets.net import Net


class TestNet(TestCase):
    """ Tests for the Net class.
    Call with 'python -m test.test_net' from project root '~'.
    Call with 'python -m test_net' from inside '~/test'. """

    def test_equal_layers(self):
        """ Should return True, as the net is equal to itself. """
        net = Net()
        net.conv = [nn.Conv2d(in_channels=9, out_channels=3, kernel_size=(3, 3)),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Tanh()]
        net.fc = [nn.Linear(in_features=4, out_features=2),
                  nn.ReLU()]
        net.conv[0].register_buffer('weight_init', torch.ones(2))
        net.fc[0].register_buffer('weight_init', torch.ones(2))
        self.assertIs(net.equal_layers(other=net), True)

    def test_equal_layers_unequal_types(self):
        """ Should return False, as two layers have unequal types. """
        net0, net1 = Net(), Net()
        net0.conv = [nn.Tanh()]
        net1.conv = [nn.ReLU()]
        net0.fc, net1.fc = [], []
        self.assertIs(net0.equal_layers(other=net1), False)

    def test_equal_layers_unequal_weights(self):
        """ Should return False, as two layers contain unequal 'weight'-attributes. """
        net0, net1 = Net(), Net()
        net0.conv, net1.conv = [], []
        net0.fc = [nn.Linear(3, 2)]
        net1.fc = [nn.Linear(3, 2)]
        weight = torch.ones((3, 2))
        net0.fc[0].weight = nn.Parameter(weight)
        weight[0, 0] = 2
        net1.fc[0].weight = nn.Parameter(weight)
        self.assertIs(net0.equal_layers(other=net1), False)


if __name__ == '__main__':
    unittest_main()
