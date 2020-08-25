import time

from data import result_saver as rs
from experiments.experiment import Experiment
from nets.net import Net
from training.logger import log_from_medium, log_detailed_only


class ExperimentPruning(Experiment):
    """
    Super class for pruning experiments which implements setup and save routines.
    Sub classes need to implement the 'execute_experiment(...)'.
    """

    def __init__(self, specs, result_path='../data/results'):
        super(ExperimentPruning, self).__init__(specs)
        self.result_path = result_path

        # setup nets in init_nets()
        self.nets = [Net] * self.specs.net_count

    def init_nets(self):
        """ Initialize nets which are used during the experiment. """
        for n in range(self.specs.net_count):
            # noinspection PyTypeChecker
            self.nets[n] = Net(self.specs.net, self.specs.dataset, self.specs.plan_conv, self.specs.plan_fc)

        log_detailed_only(self.specs.verbosity, self.nets[0])

    def setup_experiment(self):
        """ Load dataset, initialize trainer, setup histories and initialize nets. """
        self.load_data_and_setup_trainer()
        self.hists.setup(self.specs.net_count, self.specs.prune_count, self.epoch_length, self.specs.epoch_count,
                         self.specs.plot_step)
        self.stop_hists.setup(self.specs.net_count, self.specs.prune_count)
        self.init_nets()

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = rs.generate_file_prefix(self.specs, save_time)

        results_path = rs.setup_and_get_result_path(self.result_path)
        rs.save_specs(results_path, file_prefix, self.specs)
        rs.save_experiment_histories(results_path, file_prefix, self.hists)
        rs.save_nets(results_path, file_prefix, self.nets)
        if self.specs.save_early_stop:
            rs.save_early_stop_history_list(results_path, file_prefix, self.stop_hists)
        log_from_medium(self.specs.verbosity, "Successfully wrote results on disk.")
