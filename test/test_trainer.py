import unittest

import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.trainer import TrainerAdam, calc_hist_length

class Test_trainer(unittest.TestCase):
    """ Tests for the trainer module.
    Call with 'python -m test.test_trainer' from project root '~'.
    Call with 'python -m test_trainer' from inside '~/test'. """
    def test_calculate_correct_hist_length(self):
        """ The correct length for history arrays should be calculated.
        History is saved at the following combinations of epochs and iterations: 0,0; 2,0. """
        self.assertEqual(2, calc_hist_length(5, 4, 10))

    def test_calculate_correct_hist_length_rounding(self):
        """ The correct length for history arrays should be calculated.
        History is saved at the following combinations of epochs and iterations: 0,0; 2,4; 5,0. """
        self.assertEqual(2, calc_hist_length(4, 4, 10))

    def test_execute_training(self):
        """ The training should be executed without errors and results should have correct shapes.
        Use a simple net with one linear layer and fake-data_loaders.
        Inputs have shape (1,4). """
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))
        net.crit = nn.CrossEntropyLoss()

        # setup trainer and fake-dataloader with four batches (use the same loader for training, validation and test)
        samples1 = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        samples2 = torch.tensor([[1., 2., 3., 4.], [1., 1., 2., 2.], [2., 2., 2., 2.]])
        labels1 = torch.tensor([0,0,1])
        labels2 = torch.tensor([1,1,0])
        fake_loader = [[samples1, labels1], [samples1, labels2], [samples2, labels1], [samples2, labels2]]
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        expected_hist_shape = (3,)
        expected_hist_epoch_shape = (2,)

        net, loss_hist, val_acc_hist, test_acc_hist, val_acc_hist_epoch, test_acc_hist_epoch = trainer.train_net(net, epoch_count=2, plot_step=3)

        self.assertTrue(net is not None)
        self.assertEqual(expected_hist_shape, loss_hist.shape)
        self.assertEqual(expected_hist_shape, val_acc_hist.shape)
        self.assertEqual(expected_hist_shape, test_acc_hist.shape)
        self.assertEqual(expected_hist_epoch_shape, val_acc_hist_epoch.shape)
        self.assertEqual(expected_hist_epoch_shape, test_acc_hist_epoch.shape)
        self.assertTrue(all(loss_hist > 0))
        self.assertTrue(all(val_acc_hist > 0))
        self.assertTrue(all(test_acc_hist > 0))
        self.assertTrue(all(val_acc_hist_epoch > 0))
        self.assertTrue(all(test_acc_hist_epoch > 0))

    def test_execute_training_rounding(self):
        """ The training should be executed without errors and results should have correct shapes.
        Use a simple net with one linear layer and fake-data_loaders.
        Inputs have shape (1,4). """
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))
        net.crit = nn.CrossEntropyLoss()

        # setup trainer and fake-dataloader with four batches (use the same loader for training, validation and test)
        samples1 = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        samples2 = torch.tensor([[1., 2., 3., 4.], [1., 1., 2., 2.], [2., 2., 2., 2.]])
        labels1 = torch.tensor([0,0,1])
        labels2 = torch.tensor([1,1,0])
        fake_loader = [[samples1, labels1], [samples1, labels2], [samples2, labels1], [samples2, labels2]]
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        expected_hist_shape = (2,)
        expected_hist_epoch_shape = (2,)

        net, loss_hist, val_acc_hist, test_acc_hist, val_acc_hist_epoch, test_acc_hist_epoch = trainer.train_net(net, epoch_count=2, plot_step=4)

        self.assertTrue(net is not None)
        self.assertEqual(expected_hist_shape, loss_hist.shape)
        self.assertEqual(expected_hist_shape, val_acc_hist.shape)
        self.assertEqual(expected_hist_shape, test_acc_hist.shape)
        self.assertEqual(expected_hist_epoch_shape, val_acc_hist_epoch.shape)
        self.assertEqual(expected_hist_epoch_shape, test_acc_hist_epoch.shape)
        self.assertTrue(all(loss_hist > 0))
        self.assertTrue(all(val_acc_hist > 0))
        self.assertTrue(all(test_acc_hist > 0))
        self.assertTrue(all(val_acc_hist_epoch > 0))
        self.assertTrue(all(test_acc_hist_epoch > 0))

    def test_compute_correct_test_acc(self):
        """ The correct test-accuracy should be calculated.
        The fake-net with one linear layer classifies half of the fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))

        # setup trainer and fake-dataloader with two batches (use the same samples for both batches)
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels_batch1 = torch.tensor([0,0,1])
        labels_batch2 = torch.tensor([1,1,0])
        test_loader = [[samples, labels_batch1], [samples, labels_batch2]]
        trainer = TrainerAdam(0., None, None, test_loader=test_loader)

        self.assertEqual(0.5, trainer.compute_acc(net, test=True))

    def test_compute_correct_val_acc(self):
        """ The correct validation-accuracy should be calculated.
        The fake-net with one linear layer classifies all fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))

        # setup trainer and fake-dataloader with one batch
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels = torch.tensor([0,0,1])
        val_loader = [[samples, labels]]
        trainer = TrainerAdam(0., None, val_loader, test_loader=None)

        self.assertEqual(1., trainer.compute_acc(net, test=False))

if __name__ == '__main__':
    unittest.main()
