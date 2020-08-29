from unittest import TestCase, main as unittest_main

import numpy as np
import torch

from experiments.experiment_specs import DatasetNames, NetNames
from nets.net import Net


class TestNet(TestCase):
    """ Tests for the Net class.
    Call with 'python -m test.test_net' from project root '~'.
    Call with 'python -m test_net' from inside '~/test'. """

    def test_raise_error_on_invalid_conv_spec(self):
        """ The network should raise an assertion error, because plan_conv contains an invalid spec. """
        with self.assertRaises(AssertionError):
            Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=['invalid_spec'], plan_fc=[])

    def test_raise_error_on_invalid_fc_spec(self):
        """ The network should raise an assertion error, because plan_fc contains an invalid spec. """
        with self.assertRaises(AssertionError):
            Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[], plan_fc=['invalid_spec'])

    def test_raise_error_on_invalid_net_name(self):
        """ The network should raise an assertion error, because 'net_name' is invalid. """
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            Net('Invalid name', DatasetNames.MNIST, plan_conv=[], plan_fc=[])

    def test_raise_error_on_invalid_dataset_name(self):
        """ The network should raise an assertion error, because 'dataset_name' is invalid. """
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            Net(NetNames.LENET, 'Invalid name', plan_conv=[], plan_fc=[])

    def test_raise_error_on_sparsity_for_invalid_layer(self):
        """ The network should raise an assertion error, because sparsity is not defined for max-pooling. """
        with self.assertRaises(AssertionError):
            Net.sparsity_layer(torch.nn.MaxPool2d(2))

    def test_weight_count(self):
        """ The CNN should have the right weight counts.
        conv = conv1 + conv2 = 3*9*2 + 2*9*2 = 90
        fc = hid1 + hid2 + out = (16*16*2)*4 + 4*2 + 2*10 = 2076 """
        net = Net(NetNames.CONV, DatasetNames.CIFAR10, plan_conv=[2, 2, 'M'], plan_fc=[4, 2])
        self.assertEqual(dict([('conv', 90), ('fc', 2076)]), net.init_weight_count_net)

    def test_forward_pass_mnist(self):
        """ The neural network with one hidden layer should perform a forward pass for without exceptions. """
        net = Net(NetNames.LENET, DatasetNames.MNIST, plan_conv=[], plan_fc=[2])
        input_sample = torch.rand(1, 1, 28, 28)
        net(input_sample)

    def test_forward_pass_cifar10(self):
        """ The neural network with small Conv architecture should perform a forward pass without exceptions. """
        net = Net(NetNames.LENET, DatasetNames.CIFAR10, plan_conv=[2, '2', 'M', '2B', 'A'], plan_fc=['4', 2])
        input_sample = torch.rand(1, 3, 32, 32)
        net(input_sample)

    def test_sparsity_report_after_single_prune_lenet_300_100(self):
        """ Should prune each layer with the given pruning rate, except for the last layer (half fc pruning-rate).
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9 + (300*100)*0.9 + (100*10)*0.95) / 266200 ~ 0.9002 """
        net = Net(NetNames.LENET, DatasetNames.MNIST, plan_conv=[], plan_fc=[300, 100])
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        np.testing.assert_array_equal(np.array([0.9002, 0.9, 0.9, 0.95]), net.sparsity_report())

    def test_sparsity_report_after_double_prune_lenet_300_100(self):
        """ Should prune each layer with the given pruning rate, except for the last layer (half fc pruning-rate).
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9^2 + (300*100)*0.9^2 + (100*10)*0.95^2) / 266200 ~ 0.8103 """
        net = Net(NetNames.LENET, DatasetNames.MNIST, plan_conv=[], plan_fc=[300, 100])
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        np.testing.assert_array_equal(np.array([0.8103, 0.81, 0.81, 0.902]), net.sparsity_report())

    def test_sparsity_report_after_single_prune_conv2(self):
        """ Should prune each layer with the given pruning rate, except for the last layer (half fc pruning-rate).
        total_weights = conv+fc = 38592+4262400 = 4300992
        sparsity = (38592*0.9 + (16*16*64*256 + 256*256)*0.8 + (256*10)*0.9) / 4300992 ~ 0.8010 """
        net = Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[64, 64, 'M'], plan_fc=[256, 256])
        net.prune_net(prune_rate_conv=0.1, prune_rate_fc=0.2)
        np.testing.assert_almost_equal(np.array([0.801, 0.9, 0.9, 0.8, 0.8, 0.9]), net.sparsity_report(), decimal=3)

    def test_get_untrained_instance(self):
        """ The pruned and trained network should return an untrained copy of itself, i.e. with initial values. """
        net = Net(NetNames.CONV, DatasetNames.CIFAR10, plan_conv=[2, 'M'], plan_fc=[2])
        net.conv[0].weight.add_(0.5)
        net.fc[0].weight.add_(0.5)
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1, reset=False)

        new_net = net.get_new_instance(reset_weight=True)

        np.testing.assert_array_equal(net.sparsity_report(), new_net.sparsity_report())
        self.assertEqual(NetNames.CONV, new_net.net_name)
        self.assertEqual(DatasetNames.CIFAR10, new_net.dataset_name)
        self.assertIs(torch.equal(new_net.conv[0].weight, net.conv[0].weight_init.mul(net.conv[0].weight_mask)), True)
        self.assertIs(torch.equal(new_net.fc[0].weight, net.fc[0].weight_init.mul(net.fc[0].weight_mask)), True)

    def test_get_trained_instance(self):
        """ The pruned and trained network should return a trained copy of itself. """
        net = Net(NetNames.CONV, DatasetNames.CIFAR10, plan_conv=[2, 'M'], plan_fc=[2])
        net.conv[0].weight.add_(0.5)
        net.fc[0].weight.add_(0.5)
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1, reset=False)

        new_net = net.get_new_instance(reset_weight=False)

        np.testing.assert_array_equal(net.sparsity_report(), new_net.sparsity_report())
        self.assertEqual(NetNames.CONV, new_net.net_name)
        self.assertEqual(DatasetNames.CIFAR10, new_net.dataset_name)
        self.assertIs(torch.equal(new_net.conv[0].weight, net.conv[0].weight.mul(net.conv[0].weight_mask)), True)
        self.assertIs(torch.equal(new_net.fc[0].weight, net.fc[0].weight.mul(net.fc[0].weight_mask)), True)

    def test_sparsity_report_initial_weights(self):
        """ The convolutional neural network should be fully connected right after initialization. """
        net = Net(NetNames.CONV, DatasetNames.CIFAR10, plan_conv=[8, 'M', 16, 'A'], plan_fc=[32, 16])
        np.testing.assert_array_equal(np.ones(6, dtype=float), net.sparsity_report())

    def test_equal_layers(self):
        """ Should return True, as the net is equal to itself. """
        net = Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[2, 'M'], plan_fc=[2])
        self.assertIs(net.equal_layers(other=net), True)

    def test_equal_layers_unequal_types(self):
        """ Should return False, as two layers have unequal activation functions. """
        net0 = Net(NetNames.LENET, DatasetNames.MNIST, plan_conv=[2, 'M'], plan_fc=[2])
        net1 = Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[2, 'M'], plan_fc=[2])
        self.assertIs(net0.equal_layers(other=net1), False)

    def test_equal_layers_unequal_weights(self):
        """ Should return False, as two layers contain unequal 'weight'-attributes. """
        torch.manual_seed(0)
        net0 = Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[2, 'M'], plan_fc=[2])
        torch.manual_seed(1)
        net1 = Net(NetNames.CONV, DatasetNames.MNIST, plan_conv=[2, 'M'], plan_fc=[2])
        self.assertIs(net0.equal_layers(other=net1), False)


if __name__ == '__main__':
    unittest_main()
