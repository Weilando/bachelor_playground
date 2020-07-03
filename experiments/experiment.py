import math
import numpy as np
import time
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import plotter
from data import result_saver as rs
from training.trainer import calc_hist_length
from data.dataloaders import get_mnist_dataloaders, get_cifar10_dataloaders
from experiments.experiment_settings import VerbosityLevel, DatasetNames
from training.trainer import TrainerAdam

class Experiment(object):
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args

        self.net_count = args['net_count']
        self.epoch_count = args['epoch_count']
        self.learning_rate = args['learning_rate']
        self.plot_step = args['plot_step']
        self.device = torch.device(args['device'])
        self.verbosity = args['verbosity']
        if self.verbosity != VerbosityLevel.SILENT:
            print(args)

    def setup_experiment(self):
        """ Load dataset, initialize trainer, create np.arrays for histories and initialize nets. """
        self.load_data_and_setup_trainer()
        self.create_histories()
        self.init_nets()

    def load_data_and_setup_trainer(self):
        """ Load dataset and initialize trainer.
        Store the length of the training-loader into 'self.epoch_length' to initialize histories. """
        # load dataset
        if self.args['dataset'] == DatasetNames.MNIST:
            train_loader, val_loader, test_loader = get_mnist_dataloaders(device=self.args['device'], verbosity=self.verbosity)
        elif self.args['dataset'] == DatasetNames.CIFAR10:
            train_loader, val_loader, test_loader = get_cifar10_dataloaders(device=self.args['device'], verbosity=self.verbosity)

        self.epoch_length = len(train_loader)

        # initialize trainer
        self.trainer = TrainerAdam(self.learning_rate, train_loader, val_loader, test_loader, self.device, self.verbosity)

    def create_histories(self):
        """ Create np-arrays containing values from the training process.
        Loss and accuracies are saved at each plot_step iteration.
        Accuracies and the nets' sparsity are saved after each epoch. """
        # calculate amount of iterations to save at
        history_length = calc_hist_length(self.epoch_length, self.epoch_count, self.plot_step)
        self.loss_hists = np.zeros((self.net_count, self.prune_count+1, history_length), dtype=float)
        self.val_acc_hists = np.zeros_like(self.loss_hists, dtype=float)
        self.test_acc_hists = np.zeros_like(self.loss_hists, dtype=float)

        self.val_acc_hists_epoch = np.zeros((self.net_count, self.prune_count+1, self.epoch_count), dtype=float)
        self.test_acc_hists_epoch = np.zeros_like(self.val_acc_hists_epoch, dtype=float)
        self.sparsity_hist = np.ones((self.prune_count+1), dtype=float)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the exeperiment. """
        self.nets = [None] * self.net_count
        pass

    def execute_experiment(self):
        """ Execute all actions for experiment and save accuracy- and loss-histories. """
        pass

    def run_experiment(self):
        """ Run experiment, i.e. setup and execute it and store the results. """
        experiment_start = time.time() # start clock for experiment duration

        self.setup_experiment()
        self.execute_experiment()

        experiment_stop = time.time() # stop clock for experiment duration
        duration = plotter.format_time(experiment_stop-experiment_start)
        if self.verbosity != VerbosityLevel.SILENT:
            print(f"Experiment duration: {duration}")
        self.args['duration'] = duration

        self.save_results()

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = rs.generate_file_prefix(self.args, save_time)

        results_path = rs.setup_results_path()
        rs.save_specs(self.args, results_path, file_prefix)
        rs.save_histories(self, results_path, file_prefix)
        rs.save_nets(self, results_path, file_prefix)
        if self.verbosity != VerbosityLevel.SILENT:
            print("Successfully wrote results on disk.")
