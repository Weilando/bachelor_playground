import math
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments import experiment_settings, experiment
from nets.conv import Conv
from data.dataloaders import get_cifar10_dataloaders
from training.trainer import TrainerAdam

class Experiment_Conv_CIFAR10(experiment.Experiment):
    def __init__(self, args):
        super(Experiment_Conv_CIFAR10, self).__init__(args)
        self.prune_rate_conv = args['prune_rate_conv']
        self.prune_rate_fc = args['prune_rate_fc']
        self.prune_count = args['prune_count']

    def setup_experiment(self):
        """ Load dataset, initialize trainer, create np.arrays for histories and initialize nets. """
        # load dataset
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(self.args['device'])

        # initialize trainer
        self.trainer = TrainerAdam(self.learning_rate, train_loader, val_loader, test_loader, self.device)

        # create histories
        loss_history_epoch_length = math.ceil(len(train_loader) / self.loss_plot_step)
        self.loss_histories = np.zeros((self.net_count, self.prune_count+1, loss_history_epoch_length*self.epoch_count), dtype=float)
        self.val_acc_histories = np.zeros((self.net_count, self.prune_count+1, self.epoch_count), dtype=float)
        self.test_acc_histories = np.zeros((self.net_count, self.prune_count+1, self.epoch_count), dtype=float)
        self.sparsity_history = np.ones((self.prune_count+1), dtype=float)

        # initialize neural networks
        self.nets = [None] * self.net_count
        for n in range(self.net_count):
            self.nets[n] = Conv(self.args['plan_conv'], self.args['plan_fc'])
        print(self.nets[0])

    def execute_experiment(self):
        """ Execute experiment, i.e. perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        print(f"Prune networks with rates {self.prune_rate_conv} (conv) and {self.prune_rate_fc} (fc).")
        for n in range(self.net_count):
            print(f"Train network #{n} (unpruned).")
            self.nets[n], self.loss_histories[n,0], self.val_acc_histories[n,0], self.test_acc_histories[n,0] = self.trainer.train_net(self.nets[n], self.epoch_count, self.loss_plot_step)
            print(f"Final test-accuracy: {(self.test_acc_histories[n][0][-1]):1.4}")

            for p in range(1, self.prune_count+1):
                print(f"Prune network #{n} in round {p}", end="")
                self.nets[n].prune_net(self.prune_rate_conv, self.prune_rate_fc)

                if n==0:
                    self.sparsity_history[p] = self.nets[0].sparsity_report()[0]
                print(f" (sparsity {self.sparsity_history[p]:.6}).", end="")

                print(" Train network.")
                self.nets[n], self.loss_histories[n][p], self.val_acc_histories[n][p], self.test_acc_histories[n][p] = self.trainer.train_net(self.nets[n], self.epoch_count, self.loss_plot_step)

                print(f"Final test-accuracy: {(self.test_acc_histories[n][p][-1]):1.4}")
            print()
