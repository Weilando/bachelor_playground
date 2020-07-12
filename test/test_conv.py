from unittest import TestCase
from unittest import main as unittest_main

import numpy as np
import torch

from nets.conv import Conv
from nets.net import Net


class TestConv(TestCase):
    """ Tests for the Conv class.
    Call with 'python -m test.test_conv' from project root '~'.
    Call with 'python -m test_conv' from inside '~/test'.
    Inputs are of size [batch_size, 3, 32, 32]. """

    def test_forward_pass_simple_conv(self):
        """ The neural network with small Conv architecture should perform a forward pass without exceptions. """
        net = Conv(plan_conv=[8, 'M', '16B', 'A'], plan_fc=[32, 16])
        input_sample = torch.rand(1, 3, 32, 32)
        net(input_sample)

    def test_raise_error_on_invalid_conv_spec(self):
        """ The network should raise an assertion error, because plan_conv contains an invalid spec. """
        with self.assertRaises(AssertionError):
            Conv(plan_conv=['InvalidSpec!'])

    def test_raise_error_on_sparsity_for_invalid_layer(self):
        """ The network should raise an assertion error, because sparsity is not defined for max-pooling. """
        with self.assertRaises(AssertionError):
            Net.sparsity_layer(torch.nn.MaxPool2d(2))

    def test_weight_count_conv2(self):
        """ The neural network with Conv-2 architecture should have the right weight counts.
        conv = conv1+conv2 = 3*9*64 + 64*9*64 = 38592
        fc = hid1+hid2+out = (16*16*64)*256 + 256*256 + 256*10 = 4262400 """
        net = Conv()
        expected = dict([('conv', 38592), ('fc', 4262400)])
        self.assertEqual(expected, net.init_weight_count_net)

    def test_sparsity_report_initial_weights_simple_conv(self):
        """ The convolutional neural network should be fully connected right after initialization. """
        net = Conv(plan_conv=[8, 'M', 16, 'A'], plan_fc=[32, 16])
        sparsity_report = net.sparsity_report()
        self.assertTrue(([1.0, 1.0, 1.0, 1.0, 1.0, 1.0] == sparsity_report).all())

    def test_sparsity_report_after_single_prune(self):
        """ Should prune each layer with the given pruning rate, except for the last layer.
        The last layer needs to be pruned using half of the fc pruning rate.
        For the whole net's sparsity we get:
        total_weights = conv+fc = 38592+4262400 = 4300992
        sparsity = (38592*0.9 + (16*16*64*256 + 256*256)*0.8 + (256*10)*0.9) / 4300992 ~ 0.8010 """
        net = Conv()
        net.prune_net(prune_rate_conv=0.1, prune_rate_fc=0.2)
        sparsity_report = net.sparsity_report()
        self.assertTrue(np.allclose([0.801, 0.9, 0.9, 0.8, 0.8, 0.9], sparsity_report, atol=1e-03, rtol=1e-03))


if __name__ == '__main__':
    unittest_main()
