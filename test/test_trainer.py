from unittest import TestCase
from unittest import main as unittest_main

import torch
import torch.nn as nn

from training.trainer import TrainerAdam, calc_hist_length


def generate_single_layer_net():
    """ Setup a neural network with one linear layer for test purposes. """
    net = nn.Linear(4, 2)
    net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
    net.bias = nn.Parameter(torch.zeros(2))
    net.criterion = nn.CrossEntropyLoss()
    return net


def generate_fake_data_loader():
    """" Generate fake-DataLoader with four batches, i.e. a list with sub-lists of samples and labels. """
    samples1 = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
    samples2 = torch.tensor([[1., 2., 3., 4.], [1., 1., 2., 2.], [2., 2., 2., 2.]])
    labels1 = torch.tensor([0, 0, 1])
    labels2 = torch.tensor([1, 1, 0])
    return [[samples1, labels1], [samples1, labels2], [samples2, labels1], [samples2, labels2]]


class TestTrainer(TestCase):
    """ Tests for the trainer module.
    Call with 'python -m test.test_trainer' from project root '~'.
    Call with 'python -m test_trainer' from inside '~/test'. """

    def test_calculate_correct_hist_length(self):
        """ Should calculate the correct length for history arrays.
        History is saved at the following combinations of epochs and iterations: 0,4; 1,4. """
        self.assertEqual(2, calc_hist_length(5, 2, 5))

    def test_calculate_correct_hist_length_rounding(self):
        """ Should calculate the correct length for history arrays.
        History is saved at the following combinations of epochs and iterations: 0,3; 1,2; 2,1. """
        self.assertEqual(3, calc_hist_length(5, 3, 4))

    def test_execute_training(self):
        """ The training should be executed without errors and results should have correct shapes.
        Use a simple net with one linear layer and fake-data_loaders.
        Inputs have shape (1,4). """
        net = generate_single_layer_net()

        # setup trainer with fake-DataLoader (use the same loader for training, validation and test)
        fake_loader = generate_fake_data_loader()
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        expected_hist_shape = (2,)

        net, loss_hist, val_acc_hist, test_acc_hist = trainer.train_net(net, epoch_count=2, plot_step=3)

        self.assertTrue(net is not None)
        self.assertEqual(expected_hist_shape, loss_hist.shape)
        self.assertEqual(expected_hist_shape, val_acc_hist.shape)
        self.assertEqual(expected_hist_shape, test_acc_hist.shape)
        self.assertTrue(all(loss_hist > 0))
        self.assertTrue(all(val_acc_hist > 0))
        self.assertTrue(all(test_acc_hist > 0))

    def test_execute_training_rounding(self):
        """ Should execute training without errors and save results with correct shapes.
        Use a simple net with one linear layer and fake-data_loaders.
        Inputs have shape (1,4). """
        net = generate_single_layer_net()

        # setup trainer with fake-DataLoader (use the same loader for training, validation and test)
        fake_loader = generate_fake_data_loader()
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        expected_hist_shape = (2,)

        net, loss_hist, val_acc_hist, test_acc_hist = trainer.train_net(net, epoch_count=2, plot_step=4)

        self.assertTrue(net is not None)
        self.assertEqual(expected_hist_shape, loss_hist.shape)
        self.assertEqual(expected_hist_shape, val_acc_hist.shape)
        self.assertEqual(expected_hist_shape, test_acc_hist.shape)
        self.assertTrue(all(loss_hist > 0))
        self.assertTrue(all(val_acc_hist > 0))
        self.assertTrue(all(test_acc_hist > 0))

    def test_compute_correct_test_acc(self):
        """ Should calculate the correct test-accuracy.
        The fake-net with one linear layer classifies half of the fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        net = generate_single_layer_net()

        # setup trainer and fake-DataLoader with two batches (use the same samples for both batches)
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels_batch1 = torch.tensor([0, 0, 1])
        labels_batch2 = torch.tensor([1, 1, 0])
        test_loader = [[samples, labels_batch1], [samples, labels_batch2]]
        trainer = TrainerAdam(0., None, None, test_loader=test_loader)

        self.assertEqual(0.5, trainer.compute_acc(net, test=True))

    def test_compute_correct_val_acc(self):
        """ Should calculate the correct validation-accuracy.
        The fake-net with one linear layer classifies all fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        net = generate_single_layer_net()

        # setup trainer and fake-DataLoader with one batch
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels = torch.tensor([0, 0, 1])
        val_loader = [[samples, labels]]
        trainer = TrainerAdam(0., None, val_loader, test_loader=None)

        self.assertEqual(1., trainer.compute_acc(net, test=False))


if __name__ == '__main__':
    unittest_main()
