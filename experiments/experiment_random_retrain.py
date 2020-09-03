import os

import time

from data import result_saver as rs
from data.plotter_evaluation import format_time
from data.result_loader import extract_experiment_path_prefix, generate_absolute_specs_path, \
    get_early_stop_history_from_file, get_specs_from_file, random_histories_file_exists
from experiments.experiment import Experiment
from experiments.experiment_specs import ExperimentSpecs
from nets.net import Net
from nets.weight_initializer import gaussian_glorot
from training.logger import log_from_medium


class ExperimentRandomRetrain(Experiment):
    """
    Experiment which trains randomly reinitialized subnetworks from a previous IMP-experiment.
    The original nets with sparsity 100% are not retrained.
    """

    def __init__(self, relative_path_original_specs, original_net_number, retrain_net_count):
        self.experiment_path_prefix = extract_experiment_path_prefix(relative_path_original_specs)
        absolute_path_specs_path = generate_absolute_specs_path(relative_path_original_specs)
        original_specs = get_specs_from_file(absolute_path_specs_path)
        self.early_stop_history = get_early_stop_history_from_file(self.experiment_path_prefix, original_specs,
                                                                   original_net_number)

        assert original_specs.save_early_stop, "No EarlyStopHistory was created by the original IMP-experiment."
        assert not random_histories_file_exists(self.experiment_path_prefix, original_net_number), \
            f"A random-histories file exists for net_number {original_net_number} and would be overwritten."

        super(ExperimentRandomRetrain, self).__init__(original_specs)

        self.retrain_net_count = retrain_net_count
        self.original_net_number = original_net_number
        self.specs.save_early_stop = False  # irrelevant during random retraining

    def setup_experiment(self):
        """ Load dataset, initialize trainer and setup histories. """
        self.load_data_and_setup_trainer()
        self.hists.setup(self.retrain_net_count, self.specs.prune_count - 1, self.epoch_length, self.specs.epoch_count,
                         self.specs.plot_step)

    @staticmethod
    def generate_randomly_reinitialized_net(specs, state_dict):
        """ Build a net from 'state_dict' and randomly reinitialize its weights.
        The net has the same masks like the net specified by 'state_dict'. """
        assert isinstance(specs, ExperimentSpecs), f"'specs' needs to be ExperimentSpecs, but is {type(specs)}."
        net = Net(specs.net, specs.dataset, specs.plan_conv, specs.plan_fc)
        net.load_state_dict(state_dict)
        net.apply(gaussian_glorot)
        net.store_initial_weights()
        net.prune_net(0.0, 0.0)
        return net

    def execute_experiment(self):
        """ Randomly reinitialize and train 'retrain_net_count' instances per level of pruning.
        Do not retrain the original network with sparsity 100%.
        Generate nets with correct architecture and masks from parameters in specs and state_dicts. """
        for prune_level, state_dict in enumerate(self.early_stop_history.state_dicts[1:]):
            for net_number in range(self.retrain_net_count):
                tic = time.time()
                net = ExperimentRandomRetrain.generate_randomly_reinitialized_net(self.specs, state_dict)

                if net_number == 0:
                    self.hists.sparsity[prune_level] = net.sparsity_report()[0]
                    log_from_medium(self.specs.verbosity,
                                    f"Level of pruning: {prune_level + 1}/{self.specs.prune_count} "
                                    f"(sparsity {self.hists.sparsity[prune_level]:6.4f}).")
                log_from_medium(self.specs.verbosity, f"Train network #{net_number + 1}/{self.retrain_net_count} ",
                                False)

                (_, self.hists.train_loss[net_number, prune_level], self.hists.val_loss[net_number, prune_level],
                 self.hists.val_acc[net_number, prune_level], self.hists.test_acc[net_number, prune_level], _, _) \
                    = self.trainer.train_net(net, self.specs.epoch_count, self.specs.plot_step)

                toc = time.time()
                log_from_medium(self.specs.verbosity, f"(took {format_time(toc - tic)}).")

    def save_results(self):
        """ Save generated histories for randomly reinitialized models on disk. """
        results_path, file_prefix = os.path.split(self.experiment_path_prefix)
        rs.save_experiment_histories_random_retrain(results_path, file_prefix, self.original_net_number, self.hists)
        log_from_medium(self.specs.verbosity, "Successfully wrote results on disk.")
