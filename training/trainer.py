import math
import time

import numpy as np
import torch
import torch.optim as optim

from experiments.experiment_settings import VerbosityLevel
from training import plotter


def calc_hist_length(batch_count, epoch_count, plot_step):
    """ Calculate length of history arrays based on batch_count, epoch_count and plot_step. """
    return math.ceil((batch_count * epoch_count) / plot_step)


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

    def train_net(self, net, epoch_count=3, plot_step=100):
        """ Train the given model 'net' with optimizer 'opt' for given epochs.
        Save accuracies and loss every 'plot_step' iterations and after each epoch.
        'reg_factor' adds L1-regularization. """
        net.to(self.device)  # push model to device

        # initialize histories
        hist_length = calc_hist_length(len(self.train_loader), epoch_count, plot_step)
        loss_hist = np.zeros(hist_length, dtype=float)  # save each plot_step iterations
        val_acc_hist = np.zeros_like(loss_hist, dtype=float)  # save each plot_step iterations
        test_acc_hist = np.zeros_like(loss_hist, dtype=float)  # save each plot_step iterations
        val_acc_hist_epoch = np.zeros(epoch_count, dtype=float)  # save per epoch
        test_acc_hist_epoch = np.zeros_like(val_acc_hist_epoch, dtype=float)  # save per epoch

        # setup training
        opt = optim.Adam(net.parameters(), lr=self.learning_rate)  # instantiate optimizer
        hist_count = 0
        tic = 0

        for e in range(0, epoch_count):
            if self.verbosity != VerbosityLevel.SILENT:
                print(f"epoch: {(e + 1):2} ", end="")
                tic = time.time()
            epoch_base = e * len(self.train_loader)

            for j, data in enumerate(self.train_loader):
                # set model to training mode (important for batch-norm/dropout)
                net.train(True)
                # push inputs and targets to device
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                opt.zero_grad()  # zero the parameter gradients

                # forward pass
                outputs = net(inputs)
                loss = net.criterion(outputs, labels)

                # backward pass
                loss.backward()
                opt.step()

                # evaluate accuracies, save accuracies and loss
                if ((epoch_base + j) % plot_step) == 0:
                    loss_hist[hist_count] = loss.item()
                    val_acc_hist[hist_count] = self.compute_acc(net, test=False)
                    test_acc_hist[hist_count] = self.compute_acc(net, test=True)
                    hist_count += 1
                    if self.verbosity == VerbosityLevel.DETAILED:
                        print(f"-", end="")
                    net.train(True)

            # evaluate and save accuracies after each epoch
            val_acc_hist_epoch[e] = self.compute_acc(net, test=False)
            test_acc_hist_epoch[e] = self.compute_acc(net, test=True)

            # print progress
            if self.verbosity != VerbosityLevel.SILENT:
                toc = time.time()
                print(f"val-acc: {(val_acc_hist_epoch[e]):1.4} (took {plotter.format_time(toc - tic)})")
        return net, loss_hist, val_acc_hist, test_acc_hist, val_acc_hist_epoch, test_acc_hist_epoch

    def compute_acc(self, net, test=True):
        """ Compute the given net's accuracy.
        'test' indicates whether the test- or validation-accuracy should be calculated. """
        net.train(False)  # set model to evaluation mode (important for batch-norm/dropout)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in (self.test_loader if test else self.val_loader):
                # Push inputs and targets to device
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
