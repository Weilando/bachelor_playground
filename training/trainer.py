import time # for "stop watch"
import math
import numpy as np

import torch
import torch.optim as optim

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training import plotter

class TrainerAdam(object):
    """ Class for training a neural network 'net' with the adam-optimizer.
    The network is trained on batches from 'train_loader'.
    They are evaluated with batches from val_loader or test_loader. """
    def __init__(self, learning_rate, train_loader, val_loader, test_loader):
        super(TrainerAdam, self).__init__()
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Enable CUDA
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def train_net(self, net, epoch_count=3, loss_plot_step=100, verbose=False):
        """ Train the given model 'net' with optimizer 'opt' for given epochs.
        Save the loss every 'loss_plot_step' iterations. """
        net.to(self.device) # push model to GPU

        # initialize histories
        loss_history_epoch_length = math.ceil(len(self.train_loader) / loss_plot_step)
        loss_history = np.zeros((loss_history_epoch_length * epoch_count), dtype=float)
        val_acc_history = np.zeros((epoch_count), dtype=float)
        test_acc_history = np.zeros((epoch_count), dtype=float)

        # setup training
        net.train(True) # set model to training mode (important for batchnorm/dropout)
        opt = optim.Adam(net.parameters(), lr=self.learning_rate) # instantiate optimizer

        for e in range(0, epoch_count):
            if verbose:
                print(f"epoch: {(e+1):2} (", end="")
            tic = time.time()

            for j, data in enumerate(self.train_loader):
                # Push inputs and targets to GPU
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                opt.zero_grad()

                # regularization loss
                reg_loss = 0
                for param in net.parameters():
                    reg_loss += torch.sum(torch.abs(param))

                # forward pass
                outputs = net(inputs)

                # training loss
                train_loss = net.crit(outputs, labels)

                # calculate total loss
                loss = train_loss + 0.00005*reg_loss
                if (j % loss_plot_step) == 0:
                    loss_history[e*loss_history_epoch_length + int(j/100)] = loss.item()
                    if verbose:
                        print(f"-", end="")

                # backward pass
                loss.backward()
                opt.step()

            if verbose:
                print(f")", end="")
            # accuracies are evaluated after each epoch to increase training speed
            val_acc_history[e] = self.compute_accuracy(net, test=False)
            if verbose:
                print(f"v", end="")
            test_acc_history[e] = self.compute_accuracy(net, test=True)

            # print progress
            toc = time.time()
            if verbose:
                print(f"t val-acc: {(val_acc_history[e]):1.4} (took {plotter.format_time(toc-tic)})")
            else:
                print(f"epoch: {(e+1):2}, val-acc: {(val_acc_history[e]):1.4} (took {plotter.format_time(toc-tic)})")
        return net, loss_history, val_acc_history, test_acc_history

    def compute_accuracy(self, net, test=True):
        """ Compute the given net's accuracy.
        'test' indicates wheter the test- or validation-accuracy should be calculated. """
        net.train(False) # set model to evaluation mode (important for batchnorm/dropout)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in (self.test_loader if test else self.val_loader):
                # Push inputs and targets to GPU
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
