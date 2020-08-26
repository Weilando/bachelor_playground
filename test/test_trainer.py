from unittest import TestCase, main as unittest_main, mock

import numpy as np
import torch
import torch.nn as nn

from experiments.experiment_histories import calc_hist_length_per_net
from training.trainer import TrainerAdam


def generate_single_layer_net():
    """ Setup a neural network with one linear layer for test purposes. """
    net = nn.Linear(4, 2)
    net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
    net.bias = nn.Parameter(torch.zeros(2))
    net.criterion = nn.CrossEntropyLoss()
    return net


def generate_fake_data_loader():
    """" Generate fake-DataLoader with four batches, i.e. a list with sub-lists of samples and labels.
    It has four batches with three samples each. """
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
        self.assertEqual(2, calc_hist_length_per_net(5, 2, 5))

    def test_calculate_correct_hist_length_rounding(self):
        """ Should calculate the correct length for history arrays.
        History is saved at the following combinations of epochs and iterations: 0,3; 1,2; 2,1. """
        self.assertEqual(3, calc_hist_length_per_net(5, 3, 4))

    def test_execute_training(self):
        """ The training should be executed without errors and results should have correct shapes.
        Use a simple net with one linear layer and fake-data_loaders.
        Inputs have shape (1,4). """
        net = generate_single_layer_net()

        # setup trainer with fake-DataLoader (use the same loader for training, validation and test)
        fake_loader = generate_fake_data_loader()
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        net, train_loss_hist, val_loss_hist, val_acc_hist, test_acc_hist, _, _ = \
            trainer.train_net(net, epoch_count=2, plot_step=4)
        zero_history = np.zeros(2, dtype=float)  # has expected shape (2,) and is used to check for positive entries

        self.assertIs(net is not None, True)
        np.testing.assert_array_less(zero_history, train_loss_hist)
        np.testing.assert_array_less(zero_history, val_loss_hist)
        np.testing.assert_array_less(zero_history, val_acc_hist)
        np.testing.assert_array_less(zero_history, test_acc_hist)

    def test_execute_training_with_early_stopping(self):
        """ Should execute training without errors and save results with correct shapes.
        It should also return a valid checkpoint and early-stopping index.
        Use a simple net with one linear layer and fake-data_loaders with samples of shape (1,4). """
        # create net and setup trainer with fake-DataLoader (use the same loader for training, validation and test)
        net = generate_single_layer_net()
        fake_loader = generate_fake_data_loader()
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader, save_early_stop=True)

        # perform training with mocked validation-loss
        with mock.patch('training.trainer.TrainerAdam.compute_val_loss',
                        side_effect=[2.0, 1.0, 0.5, 1.0]) as mocked_val_loss:
            net, train_loss_hist, val_loss_hist, val_acc_hist, test_acc_hist, early_stop_index, early_stop_cp = \
                trainer.train_net(net, epoch_count=2, plot_step=2)  # 8 batches (iterations), 4 early-stop evaluations
            self.assertEqual(4, mocked_val_loss.call_count)

        # early-stopping criterion is True for first three calls (last one counts), thus 5 is the 'early_stop_index'
        self.assertEqual(5, early_stop_index)
        net.load_state_dict(early_stop_cp)  # check if the checkpoint can be loaded without errors

        self.assertIs(net is not None, True)
        zero_history = np.zeros(4, dtype=float)  # has expected shape (4,) and is used to check for positive entries
        np.testing.assert_array_less(zero_history, train_loss_hist)
        np.testing.assert_array_less(zero_history, val_loss_hist)
        np.testing.assert_array_less(zero_history, val_acc_hist)
        np.testing.assert_array_less(zero_history, test_acc_hist)

    def test_compute_test_acc(self):
        """ Should calculate the correct test-accuracy.
        The fake-net with one linear layer classifies half of the fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        net = generate_single_layer_net()

        # setup trainer and fake-DataLoader with two batches (use the same samples for both batches)
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels_batch1 = torch.tensor([0, 0, 1])
        labels_batch2 = torch.tensor([1, 1, 0])
        test_loader = [[samples, labels_batch1], [samples, labels_batch2]]
        trainer = TrainerAdam(0., [], [], test_loader=test_loader)

        self.assertEqual(0.5, trainer.compute_acc(net, test=True))

    def test_compute_val_acc(self):
        """ Should calculate the correct validation-accuracy.
        The fake-net with one linear layer classifies all fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. """
        net = generate_single_layer_net()

        # setup trainer and fake-DataLoader with one batch
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels = torch.tensor([0, 0, 1])
        val_loader = [[samples, labels]]
        trainer = TrainerAdam(0., [], val_loader, test_loader=[])

        self.assertEqual(1., trainer.compute_acc(net, test=False))

    def test_compute_val_loss(self):
        """ Should calculate a positive validation loss. """
        net = generate_single_layer_net()

        # setup trainer with fake-DataLoader (use the same loader for training, validation and test)
        fake_loader = generate_fake_data_loader()
        trainer = TrainerAdam(0., fake_loader, fake_loader, fake_loader)

        val_loss = trainer.compute_val_loss(net)
        self.assertLessEqual(0.0, val_loss)

    def test_should_save_early_stop_checkpoint_no_evaluation(self):
        """ Should return False, because the early-stopping criterion should not be evaluated. """
        trainer = TrainerAdam(0., [], [], [], save_early_stop=False)
        self.assertIs(trainer.should_save_early_stop_checkpoint(0.5, 0.2), False)

    def test_should_save_early_stop_checkpoint_no_new_minimum_greater(self):
        """ Should return False, because the current validation-loss is greater than the minimum. """
        trainer = TrainerAdam(0., [], [], [], save_early_stop=True)
        self.assertIs(trainer.should_save_early_stop_checkpoint(0.5, 0.2), False)

    def test_should_save_early_stop_checkpoint_no_new_minimum_equal(self):
        """ Should return False, because the current validation-loss is equal to the the minimum. """
        trainer = TrainerAdam(0., [], [], [], save_early_stop=True)
        self.assertIs(trainer.should_save_early_stop_checkpoint(0.2, 0.2), False)

    def test_should_save_early_stop_checkpoint_new_checkpoint(self):
        """ Should return True, because the validation-accuracy reached a new minimum. """
        trainer = TrainerAdam(0., [], [], [], save_early_stop=True)
        self.assertIs(trainer.should_save_early_stop_checkpoint(0.1, 0.2), True)


if __name__ == '__main__':
    unittest_main()
