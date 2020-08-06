import time
import torch

from data import plotter
from data.data_loaders import get_mnist_data_loaders, get_cifar10_data_loaders, get_toy_data_loaders
from experiments.early_stop_histories import EarlyStopHistoryList
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_specs import DatasetNames
from training.logger import log_from_medium
from training.trainer import TrainerAdam


class Experiment(object):
    """
    Superclass for experiments, providing logic for setup, execution and result storing.
    """

    def __init__(self, specs):
        super(Experiment, self).__init__()
        self.specs = specs

        log_from_medium(self.specs.verbosity, specs)

        self.device = torch.device(specs.device)

        # setup epoch_length and trainer in load_data_and_setup_trainer()
        self.trainer = None
        self.epoch_length = 0

        # setup history-arrays in setup_experiment()
        self.hists = ExperimentHistories()
        self.stop_hists = EarlyStopHistoryList()

    def setup_experiment(self):
        """ Load dataset, initialize trainer and setup histories. """
        pass

    def load_data_and_setup_trainer(self):
        """ Load dataset and initialize trainer.
        Store the length of the training-loader into 'self.epoch_length' to initialize histories. """
        # load dataset
        if self.specs.dataset == DatasetNames.MNIST:
            train_loader, val_loader, test_loader = get_mnist_data_loaders(device=self.specs.device,
                                                                           verbosity=self.specs.verbosity)
        elif self.specs.dataset == DatasetNames.CIFAR10:
            train_loader, val_loader, test_loader = get_cifar10_data_loaders(device=self.specs.device,
                                                                             verbosity=self.specs.verbosity)
        elif self.specs.dataset in [DatasetNames.TOY_MNIST, DatasetNames.TOY_CIFAR10]:
            train_loader, val_loader, test_loader = get_toy_data_loaders(self.specs.dataset)
        else:
            raise AssertionError(f"Could not load datasets, because the given name {self.specs.dataset} is invalid.")

        self.epoch_length = len(train_loader)

        # initialize trainer
        self.trainer = TrainerAdam(self.specs.learning_rate, train_loader, val_loader, test_loader, self.device,
                                   self.specs.save_early_stop, self.specs.verbosity)

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
        self.specs.duration = duration
        log_from_medium(self.specs.verbosity, f"Experiment duration: {duration}")

        self.save_results()

    def save_results(self):
        """ Save experiment's specs and histories on disk. """
        pass
