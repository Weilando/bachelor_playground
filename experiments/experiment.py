import time

import numpy as np
import torch

from data import result_saver as rs
from data.data_loaders import get_mnist_data_loaders, get_cifar10_data_loaders, get_toy_data_loaders
from experiments.experiment_settings import DatasetNames, NetNames
from nets.conv import Conv
from nets.lenet import Lenet
from nets.net import Net
from training import plotter
from training.logger import log_from_medium, log_detailed_only
from training.trainer import TrainerAdam, calc_hist_length


class Experiment(object):
    def __init__(self, args, result_path='../data/results'):
        super(Experiment, self).__init__()
        self.args = args
        self.result_path = result_path

        log_from_medium(self.args.verbosity, args)

        self.device = torch.device(args.device)

        # Setup nets in init_nets()
        self.nets = [Net] * self.args.net_count

        # Setup epoch_length and trainer in load_data_and_setup_trainer()
        self.trainer = None
        self.epoch_length = 0

        # Setup history-arrays in create_histories()
        self.loss_hists, self.val_acc_hists, self.test_acc_hists, self.sparsity_hist = None, None, None, None

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
            train_loader, val_loader, test_loader = get_mnist_data_loaders(device=self.args.device,
                                                                           verbosity=self.args.verbosity)
        elif self.args.dataset == DatasetNames.CIFAR10:
            train_loader, val_loader, test_loader = get_cifar10_data_loaders(device=self.args.device,
                                                                             verbosity=self.args.verbosity)
        elif self.args.dataset in [DatasetNames.TOY_MNIST, DatasetNames.TOY_CIFAR10]:
            train_loader, val_loader, test_loader = get_toy_data_loaders(self.args.dataset)
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

        self.sparsity_hist = np.ones((self.args.prune_count + 1), dtype=float)

    # noinspection PyTypeChecker
    def init_nets(self):
        """ Initialize nets which are used during the experiment. """
        for n in range(self.args.net_count):
            if self.args.net == NetNames.LENET:
                self.nets[n] = Lenet(self.args.plan_fc)
            elif self.args.net == NetNames.CONV:
                self.nets[n] = Conv(self.args.plan_conv, self.args.plan_fc)
            else:
                raise AssertionError(f"Could not initialize net, because the given name {self.args.net} is invalid.")

        log_detailed_only(self.args.verbosity, self.nets[0])

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
        self.args.duration = duration
        log_from_medium(self.args.verbosity, f"Experiment duration: {duration}")

        self.save_results()

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = rs.generate_file_prefix(self.args, save_time)

        results_path = rs.setup_and_get_result_path(self.result_path)
        rs.save_specs(results_path, file_prefix, self.args)
        rs.save_histories(results_path, file_prefix, self.loss_hists, self.val_acc_hists, self.test_acc_hists,
                          self.sparsity_hist)
        rs.save_nets(results_path, file_prefix, self.nets)
        log_from_medium(self.args.verbosity, "Successfully wrote results on disk.")
