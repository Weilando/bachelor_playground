from copy import deepcopy

import numpy as np
import time
import torch

from data.plotter_evaluation import format_time
from experiments.experiment_histories import calc_hist_length_per_net
from experiments.experiment_specs import VerbosityLevel
from training.logger import log_detailed_only


class TrainerAdam(object):
    """ Class for training a neural network 'net' with the adam-optimizer.
    The network is trained on batches from 'train_loader'.
    Performance is evaluated with batches from 'val_loader' or 'test_loader'.
    'save_early_stop' specifies, if the early-stopping criterion should be evaluated during training.
    If 'save_early_stop' is True, a checkpoint of net is saved at the iteration with minimal validation-loss. """

    def __init__(self, learning_rate, train_loader, val_loader, test_loader, device=torch.device('cpu'),
                 save_early_stop=False, verbosity=VerbosityLevel.SILENT):
        super(TrainerAdam, self).__init__()
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_early_stop = save_early_stop
        self.verbosity = verbosity

        self.train_loader_len = len(train_loader)
        self.val_loader_len = len(val_loader)

    def train_net(self, net, epoch_count=3, plot_step=100):
        """ Train the given model 'net' with a new instance of Adam-optimizer for 'epoch_count' epochs.
        Save accuracies and loss after every 'plot_step' iterations. """
        # initialize histories (one entry per plot_step iteration)
        hist_length = calc_hist_length_per_net(self.train_loader_len, epoch_count, plot_step)
        train_loss_hist = np.zeros(hist_length, dtype=float)
        val_loss_hist = np.zeros_like(train_loss_hist, dtype=float)
        val_acc_hist = np.zeros_like(train_loss_hist, dtype=float)
        test_acc_hist = np.zeros_like(train_loss_hist, dtype=float)

        # setup training
        net.to(self.device)  # push model to device
        net.train(True)  # set model to training mode (important for batch-norm/dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)  # instantiate optimizer
        running_train_loss = 0
        hist_count = 0

        early_stop_checkpoint = None
        early_stop_index = -1
        min_val_loss = float('inf')

        for e in range(0, epoch_count):
            log_detailed_only(self.verbosity, f"epoch: {(e + 1):3d} ", False)
            tic = time.time()
            epoch_base = e * self.train_loader_len

            for j, data in enumerate(self.train_loader, epoch_base):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # push inputs and targets to device
                opt.zero_grad()  # zero the parameter gradients

                # forward pass
                outputs = net(inputs)
                train_loss = net.criterion(outputs, labels)

                # backward pass
                train_loss.backward()
                opt.step()

                # evaluate accuracies, save accuracies and loss
                running_train_loss += train_loss.item()
                if (j % plot_step) == (plot_step - 1):
                    train_loss_hist[hist_count] = running_train_loss / plot_step
                    val_loss_hist[hist_count] = self.compute_val_loss(net)
                    val_acc_hist[hist_count] = self.compute_acc(net, test=False)
                    test_acc_hist[hist_count] = self.compute_acc(net, test=True)

                    net.train(True)  # set model to training mode (important for batch-norm/dropout)
                    if self.should_save_early_stop_checkpoint(val_loss_hist[hist_count], min_val_loss):
                        min_val_loss = val_loss_hist[hist_count]
                        early_stop_index = j
                        early_stop_checkpoint = deepcopy(net.state_dict())
                    hist_count += 1
                    running_train_loss = 0

            toc = time.time()
            log_detailed_only(self.verbosity, f"train-loss: {(train_loss_hist[hist_count - 1]):6.4f} "
                                              f"val-loss: {(val_loss_hist[hist_count - 1]):6.4f} "
                                              f"val-acc: {(val_acc_hist[hist_count - 1]):6.4f} "
                                              f"(took {format_time(toc - tic)}).", True)
        return net, train_loss_hist, val_loss_hist, val_acc_hist, test_acc_hist, early_stop_index, early_stop_checkpoint

    def compute_acc(self, net, test=True):
        """ Compute the accuracy for 'net'.
        'test' indicates whether the test- or validation-accuracy should be calculated. """
        net.train(False)  # set model to evaluation mode (important for batch-norm/dropout)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in (self.test_loader if test else self.val_loader):
                # push inputs and targets to device
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return correct / total

    def compute_val_loss(self, net):
        """ Compute the validation-loss for 'net'. """
        net.train(False)  # set model to evaluation mode (important for batch-norm/dropout)

        running_val_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                # push inputs and targets to device
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # forward pass
                outputs = net(inputs)
                val_loss = net.criterion(outputs, labels)
                running_val_loss += val_loss.item()

        return running_val_loss / self.val_loader_len

    def should_save_early_stop_checkpoint(self, val_loss, min_val_loss):
        """ Evaluate the early-stopping criterion, if it is active (return False if inactive).
        The criterion is fulfilled, if the validation-loss reached a new minimum. """
        return self.save_early_stop and (val_loss < min_val_loss)
