from unittest import TestCase
from unittest import main as unittest_main

import torch

from nets.lenet import Lenet


class TestLenet(TestCase):
    """ Tests for the Lenet class.
    Call with 'python -m test.test_lenet' from project root '~'.
    Call with 'python -m test_lenet' from inside '~/test'. """

    def test_forward_pass_simple_architecture(self):
        """ The neural network with one hidden layer should perform a forward pass for without exceptions. """
        net = Lenet(plan_fc=[5, '10'])
        input_sample = torch.rand(28, 28)
        net(input_sample)

    def test_weight_count_simple_architecture(self):
        """ The neural network with one hidden layer should return the correct weight count.
        It holds weight_count = 28*28*5 + 5*10 + 10*10 = 4070. """
        net = Lenet(plan_fc=[5, 10])
        expected_weight_count = dict([('conv', 0), ('fc', 4070)])
        self.assertEqual(expected_weight_count, net.init_weight_count_net)

    def test_sparsity_report_initial_weights(self):
        """ The neural network should be fully connected right after initialization. """
        net = Lenet()
        sparsity_report = net.sparsity_report().tolist()  # convert np.array to list
        self.assertListEqual([1.0, 1.0, 1.0, 1.0], sparsity_report)

    def test_sparsity_report_after_single_prune(self):
        """ Should prune each layer with the given pruning rate, except for the last layer.
        The last layer needs to be pruned using half of the pruning rate.
        For the whole net's sparsity we get:
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9 + (300*100)*0.9 + (100*10)*0.95) / 266200 ~ 0.9002 """
        net = Lenet()
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        sparsity_report = net.sparsity_report().tolist()  # convert np.array to list
        self.assertListEqual([0.9002, 0.9, 0.9, 0.95], sparsity_report)

    def test_sparsity_report_after_double_prune(self):
        """ Should prune each layer with the given pruning rate, except for the last layer.
        The last layer needs to be pruned using half of the pruning rate.
        For the whole net's sparsity we get:
        total_weights = (28*28*300) + (300*100) + (100*10) = 266200
        sparsity = ((28*28*300)*0.9^2 + (300*100)*0.9^2 + (100*10)*0.95^2) / 266200 ~ 0.8103 """
        net = Lenet()
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1)
        sparsity_report = net.sparsity_report().tolist()  # convert np.array to list
        self.assertListEqual([0.8103, 0.81, 0.81, 0.902], sparsity_report)

    def test_get_untrained_instance(self):
        """ The pruned and trained network should return an untrained copy of itself, i.e. with initial values. """
        net = Lenet([10])
        net.fc[0].weight.add_(0.5)
        net.prune_net(prune_rate_conv=0.0, prune_rate_fc=0.1, reset=False)

        new_net = net.get_untrained_instance()

        self.assertListEqual(net.sparsity_report().tolist(), new_net.sparsity_report().tolist())
        self.assertIs(torch.equal(new_net.fc[0].weight, net.fc[0].weight_init.mul(net.fc[0].weight_mask)), True)


if __name__ == '__main__':
    unittest_main()
