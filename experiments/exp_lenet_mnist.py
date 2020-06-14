import time
import json
import numpy as np
import math
import matplotlib.pyplot as plt

import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nets.lenet import Lenet
from data.dataloaders import get_mnist_dataloaders
from training import plotter
from training.trainer import TrainerAdam

class Exp_Lenet_MNIST(object):
    def __init__(self, args):
        super(Exp_Lenet_MNIST, self).__init__()
        self.args = args

        self.net_count = args['net_count']
        self.epoch_count = args['epoch_count']
        self.learning_rate = args['learning_rate']
        self.prune_rate = args['prune_rate']
        self.prune_count = args['prune_count']
        self.loss_plot_step = args['loss_plot_step']

    def setup_experiment(self):
        """ Load dataset, initialize trainer, create np.arrays for histories and initialize nets. """
        # load dataset
        train_loader, val_loader, test_loader = get_mnist_dataloaders()

        # initialize trainer
        self.trainer = TrainerAdam(self.learning_rate, train_loader, val_loader, test_loader)

        # create histories
        loss_history_epoch_length = math.ceil(len(train_loader) / self.loss_plot_step)
        self.loss_histories = np.zeros((self.net_count, self.prune_count+1, loss_history_epoch_length*self.epoch_count), dtype=float)
        self.val_acc_histories = np.zeros((self.net_count, self.prune_count+1, self.epoch_count), dtype=float)
        self.test_acc_histories = np.zeros((self.net_count, self.prune_count+1, self.epoch_count), dtype=float)
        self.sparsity_history = np.ones((self.prune_count+1), dtype=float)

        # initialize neural networks
        self.nets = [None] * self.net_count
        for n in range(self.net_count):
            self.nets[n] = Lenet()
        print(self.nets[0])

    def run_experiment(self):
        """ Run experiment, i.e. perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        experiment_start = time.time() # start clock for experiment duration

        self.setup_experiment()

        # IMP
        print(f"Prune networks with rate {self.prune_rate}.")
        for n in range(self.net_count):
            print(f"Train network #{n} (unpruned).")
            self.nets[n], self.loss_histories[n,0], self.val_acc_histories[n,0], self.test_acc_histories[n,0] = self.trainer.train_net(self.nets[n], self.epoch_count, self.loss_plot_step)
            print(f"Final test-accuracy: {(self.test_acc_histories[n][0][-1]):1.4}")

            for p in range(1, self.prune_count+1):
                print(f"Prune network #{n} in round {p}", end="")
                self.nets[n].prune_net(self.prune_rate)

                if n==0:
                    self.sparsity_history[p] = self.nets[0].sparsity_report()[0]
                print(f" (sparsity {self.sparsity_history[p]:.6}).", end="")

                print(" Train network.")
                self.nets[n], self.loss_histories[n][p], self.val_acc_histories[n][p], self.test_acc_histories[n][p] = self.trainer.train_net(self.nets[n], self.epoch_count, self.loss_plot_step)

                print(f"Final test-accuracy: {(self.test_acc_histories[n][p][-1]):1.4}")
            print()

        # stop clock for experiment duration
        experiment_stop = time.time()
        duration = plotter.format_time(experiment_stop-experiment_start)
        print(f"Experiment duration: {duration}")
        self.args['duration'] = duration

        self.save_results()

    def setup_results_path(self):
        """ Generate absolute path to subdirectory 'results' and create it, if necessary. """
        results_path = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        return results_path

    def generate_file_prefix(self, save_time):
        return f"{save_time}-{self.args['net']}-{self.args['dataset']}"

    def save_specs(self, results_path, file_prefix):
        """ Save experiment's specs in json-file. """
        # create path to json-file
        json_path = os.path.join(results_path, f"{file_prefix}-specs.json")

        # write dict 'args' to json-file
        description_json = json.dumps(self.args)
        f = open(json_path, "w")
        f.write(description_json)
        f.close()

    def save_nets(self, results_path, file_prefix):
        """ Save trained networks in pth-files. """
        for num, net in enumerate(self.nets):
            net_path = histories_path = os.path.join(results_path, f"{file_prefix}-net{num}.pth")
            torch.save(net, net_path)

    def save_histories(self, results_path, file_prefix):
        """ Save loss-, validation- and test-histories in npz-file.
        The np-arrays can be restored via 'np.load'. """
        histories_path = os.path.join(results_path, f"{file_prefix}-histories") # suffix is added by np.savez
        np.savez(histories_path, loss_histories=self.loss_histories, val_acc_histories=self.val_acc_histories, test_acc_histories=self.test_acc_histories)

    def save_results(self):
        """ Save experiment's specs and results on disk.
        Filenames are 'YYYY_MM_DD-hh_mm_ss-lenet-mnist' + specs/histories/net. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S-", time.localtime())
        file_prefix = self.generate_file_prefix(save_time)

        results_path = self.setup_results_path()
        self.save_specs(results_path, file_prefix)
        self.save_histories(results_path, file_prefix)
        self.save_nets(results_path, file_prefix)
        print("Successfully wrote results on disk.")


if __name__=='__main__':
    experiment_settings = dict()
    experiment_settings['net_count'] = 1
    experiment_settings['epoch_count'] = 2
    experiment_settings['learning_rate'] = 1.2e-3 # page 3, figure 2
    experiment_settings['prune_rate'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 1
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'lenet'
    experiment_settings['dataset'] = 'mnist'

    experiment = Exp_Lenet_MNIST(experiment_settings)
    experiment.run_experiment()
