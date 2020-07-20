import math
import time

import numpy as np
import torch

from data import plotter
from experiments.experiment_settings import VerbosityLevel
from training.logger import log_from_medium, log_detailed_only


def calc_hist_length(batch_count, epoch_count, plot_step):
    """ Calculate length of history arrays based on batch_count, epoch_count and plot_step. """
    return math.floor((batch_count * epoch_count) / plot_step)


class TrainerAdam(object):
    """ Class for training a neural network 'net' with the adam-optimizer.
    The network is trained on batches from 'train_loader'.
    They are evaluated with batches from val_loader or test_loader. """

    def __init__(self, learning_rate, train_loader, val_loader, test_loader, device=torch.device('cpu'),
                 verbosity=VerbosityLevel.SILENT):
        super(TrainerAdam, self).__init__()
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.verbosity = verbosity

        self.train_loader_len = len(train_loader)
        self.val_loader_len = len(val_loader)

    def train_net(self, net, epoch_count=3, plot_step=100):
        """ Train the given model 'net' with optimizer 'opt' for given epochs.
        Save accuracies and loss every 'plot_step' iterations and after each epoch.
        'reg_factor' adds L1-regularization. """
        net.to(self.device)  # push model to device
        net.train(True)  # set model to training mode (important for batch-norm/dropout)

        # initialize histories (one entry per plot_step iteration)
        hist_length = calc_hist_length(self.train_loader_len, epoch_count, plot_step)
        train_loss_hist = np.zeros(hist_length, dtype=float)
        val_loss_hist = np.zeros_like(train_loss_hist, dtype=float)
        val_acc_hist = np.zeros_like(train_loss_hist, dtype=float)
        test_acc_hist = np.zeros_like(train_loss_hist, dtype=float)

        # setup training
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)  # instantiate optimizer
        running_train_loss = 0
        hist_count = 0

        for e in range(0, epoch_count):
            log_from_medium(self.verbosity, f"epoch: {(e + 1):3d} ", False)
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

                    hist_count += 1
                    running_train_loss = 0
                    net.train(True)  # set model to training mode (important for batch-norm/dropout)

            toc = time.time()
            log_detailed_only(self.verbosity, f"train-loss: {(train_loss_hist[hist_count - 1]):6.4f} "
                                              f"val-loss: {(val_loss_hist[hist_count - 1]):6.4f} ", False)
            log_from_medium(self.verbosity,
                            f"val-acc: {(val_acc_hist[hist_count - 1]):6.4f} (took {plotter.format_time(toc - tic)})")
        return net, train_loss_hist, val_loss_hist, val_acc_hist, test_acc_hist

    def compute_acc(self, net, test=True):
        """ Compute the given net's accuracy.
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
                correct += (predicted == labels).sum().item()

        return correct / total

    def compute_val_loss(self, net):
        """ Compute the given net's validation loss. """
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
