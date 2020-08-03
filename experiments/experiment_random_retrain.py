import os

from data import result_saver as rs
from data.result_loader import extract_experiment_path_prefix, generate_absolute_specs_path, get_specs_from_file, \
    get_early_stop_history_from_file, random_histories_file_exists
from experiments.experiment import Experiment
from experiments.experiment_settings import ExperimentSettings, NetNames
from nets.conv import Conv
from nets.lenet import Lenet
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
        assert isinstance(specs, ExperimentSettings), f"'specs' needs to be ExperimentSettings, but is {type(specs)}."

        if specs.net == NetNames.LENET:
            net = Lenet(specs.plan_fc)
        elif specs.net == NetNames.CONV:
            net = Conv(specs.plan_conv, specs.plan_fc)
        else:
            raise AssertionError(f"Could not initialize net, because the given name {specs.net} is invalid.")

        net.load_state_dict(state_dict)
        net.apply(gaussian_glorot)
        net.store_initial_weights()
        net.prune_net(0.0, 0.0)
        return net

    def execute_experiment(self):
        """ Retrain in the end to check the last nets' accuracies. """
        for prune_level, state_dict in enumerate(self.early_stop_history.state_dicts[1:]):
            log_from_medium(self.specs.verbosity,
                            f"Pruning level {prune_level + 1} (sparsity {self.hists.sparsity[prune_level]:6.4f}).")
            for net_number in range(self.retrain_net_count):
                log_from_medium(self.specs.verbosity, f"Train network #{net_number} / {self.retrain_net_count}.")
                net = ExperimentRandomRetrain.generate_randomly_reinitialized_net(self.specs, state_dict)

                _, self.hists.train_loss[net_number, prune_level], self.hists.val_loss[net_number, prune_level], \
                self.hists.val_acc[net_number, prune_level], self.hists.test_acc[net_number, prune_level], _, _ \
                    = self.trainer.train_net(net, self.specs.epoch_count, self.specs.plot_step)

    def save_results(self):
        """ Save generated histories for randomly reinitialized models on disk. """
        results_path, file_prefix = os.path.split(self.experiment_path_prefix)
        rs.save_experiment_histories_random_retrain(results_path, file_prefix, self.original_net_number, self.hists)
        log_from_medium(self.specs.verbosity, "Successfully wrote results on disk.")
