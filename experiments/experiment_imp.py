import time

from data import result_saver as rs
from experiments.experiment import Experiment
from experiments.experiment_settings import NetNames
from nets.conv import Conv
from nets.lenet import Lenet
from nets.net import Net
from training.logger import log_from_medium, log_detailed_only


class ExperimentIMP(Experiment):
    """
    Experiment with iterative magnitude pruning with resetting (IMP).
    """

    def __init__(self, specs, result_path='../data/results'):
        super(ExperimentIMP, self).__init__(specs)
        self.result_path = result_path

        # setup nets in init_nets()
        self.nets = [Net] * self.specs.net_count

    # noinspection PyTypeChecker
    def init_nets(self):
        """ Initialize nets which are used during the experiment. """
        for n in range(self.specs.net_count):
            if self.specs.net == NetNames.LENET:
                self.nets[n] = Lenet(self.specs.plan_fc)
            elif self.specs.net == NetNames.CONV:
                self.nets[n] = Conv(self.specs.plan_conv, self.specs.plan_fc)
            else:
                raise AssertionError(f"Could not initialize net, because the given name {self.specs.net} is invalid.")

        log_detailed_only(self.specs.verbosity, self.nets[0])

    def setup_experiment(self):
        """ Load dataset, initialize trainer, setup histories and initialize nets. """
        self.load_data_and_setup_trainer()
        self.hists.setup(self.specs.net_count, self.specs.prune_count, self.epoch_length, self.specs.epoch_count,
                         self.specs.plot_step)
        self.stop_hists.setup(self.specs.net_count, self.specs.prune_count)
        self.init_nets()

    def execute_experiment(self):
        """ Perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        for n in range(self.specs.net_count):
            for p in range(0, self.specs.prune_count + 1):
                if p > 0:
                    log_from_medium(self.specs.verbosity, f"Prune network #{n} in round {p}. ", False)
                    self.nets[n].prune_net(self.specs.prune_rate_conv, self.specs.prune_rate_fc, reset=True)

                if n == 0:
                    self.hists.sparsity[p] = self.nets[0].sparsity_report()[0]

                log_from_medium(self.specs.verbosity, f"Train network #{n} (sparsity {self.hists.sparsity[p]:6.4f}).")

                self.nets[n], self.hists.train_loss[n, p], self.hists.val_loss[n, p], self.hists.val_acc[n, p], \
                self.hists.test_acc[n, p], self.stop_hists.histories[n].indices[p], \
                self.stop_hists.histories[n].state_dicts[p] \
                    = self.trainer.train_net(self.nets[n], self.specs.epoch_count, self.specs.plot_step)

                log_from_medium(self.specs.verbosity,
                                f"Final test-accuracy: {(self.hists.test_acc[n, p, -1]):6.4f}")
            log_from_medium(self.specs.verbosity, "")

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
