import time

import numpy as np
import torch

from data import result_saver as rs
from data.dataloaders import get_mnist_dataloaders, get_cifar10_dataloaders
from experiments.experiment_settings import VerbosityLevel, DatasetNames
from nets.net import Net
from training import plotter
from training.trainer import TrainerAdam, calc_hist_length


class Experiment(object):
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args

        if self.args.verbosity != VerbosityLevel.SILENT:
            print(args)

        self.device = torch.device(args.device)

        # Setup nets in init_nets()
        self.nets = [Net] * self.args.net_count

        # Setup epoch_length and trainer in load_data_and_setup_trainer()
        self.trainer = None
        self.epoch_length = 0

        # Setup history-arrays in create_histories()
        self.loss_hists, self.val_acc_hists, self.test_acc_hists = None, None, None
        self.val_acc_hists_epoch, self.test_acc_hists_epoch, self.sparsity_hist = None, None, None

    def setup_experiment(self):
        """ Load dataset, initialize trainer, create np.arrays for histories and initialize nets. """
        self.load_data_and_setup_trainer()
        self.create_histories()
        self.init_nets()

    def load_data_and_setup_trainer(self):
        """ Load dataset and initialize trainer.
        Store the length of the training-loader into 'self.epoch_length' to initialize histories. """
        # load dataset
        if self.args.dataset == DatasetNames.MNIST:
            train_loader, val_loader, test_loader = get_mnist_dataloaders(device=self.args.device,
                                                                          verbosity=self.args.verbosity)
        elif self.args.dataset == DatasetNames.CIFAR10:
            train_loader, val_loader, test_loader = get_cifar10_dataloaders(device=self.args.device,
                                                                            verbosity=self.args.verbosity)
        else:
            raise AssertionError(f"Could not load datasets, because the given name {self.args.dataset} is invalid.")

        self.epoch_length = len(train_loader)

        # initialize trainer
        self.trainer = TrainerAdam(self.args.learning_rate, train_loader, val_loader, test_loader, self.device,
                                   self.args.verbosity)

    def create_histories(self):
        """ Create np-arrays containing values from the training process.
        Loss and accuracies are saved at each plot_step iteration.
        Accuracies and the nets' sparsity are saved after each epoch. """
        # calculate amount of iterations to save at
        history_length = calc_hist_length(self.epoch_length, self.args.epoch_count, self.args.plot_step)
        self.loss_hists = np.zeros((self.args.net_count, self.args.prune_count + 1, history_length), dtype=float)
        self.val_acc_hists = np.zeros_like(self.loss_hists, dtype=float)
        self.test_acc_hists = np.zeros_like(self.loss_hists, dtype=float)

        self.val_acc_hists_epoch = np.zeros((self.args.net_count, self.args.prune_count + 1, self.args.epoch_count),
                                            dtype=float)
        self.test_acc_hists_epoch = np.zeros_like(self.val_acc_hists_epoch, dtype=float)
        self.sparsity_hist = np.ones((self.args.prune_count + 1), dtype=float)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the experiment. """
        pass

    def execute_experiment(self):
        """ Execute all actions for experiment and save accuracy- and loss-histories. """
        pass

    def run_experiment(self):
        """ Run experiment, i.e. setup and execute it and store the results. """
        experiment_start = time.time()  # start clock for experiment duration

        self.setup_experiment()
        self.execute_experiment()

        experiment_stop = time.time()  # stop clock for experiment duration
        duration = plotter.format_time(experiment_stop - experiment_start)
        if self.args.verbosity != VerbosityLevel.SILENT:
            print(f"Experiment duration: {duration}")
        self.args.duration = duration

        self.save_results()

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = rs.generate_file_prefix(self.args, save_time)

        results_path = rs.setup_results_path()
        rs.save_specs(self.args, results_path, file_prefix)
        rs.save_histories(self, results_path, file_prefix)
        rs.save_nets(self, results_path, file_prefix)
        if self.args.verbosity != VerbosityLevel.SILENT:
            print("Successfully wrote results on disk.")
