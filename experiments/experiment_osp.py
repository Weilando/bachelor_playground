import time

from data import plotter
from experiments.experiment_pruning import ExperimentPruning
from training.logger import log_from_medium


class ExperimentOSP(ExperimentPruning):
    """
    Experiment with one-shot pruning.
    """

    def __init__(self, specs, result_path='../data/results'):
        super(ExperimentOSP, self).__init__(specs, result_path)

    def execute_experiment(self):
        """ Perform one-shot pruning and save accuracy- and loss-histories after each training.
        Retrain subnetworks to evaluate accuracies. """
        for n in range(self.specs.net_count):
            for p in range(0, self.specs.prune_count + 1):
                tic = time.time()
                if p > 0:
                    log_from_medium(self.specs.verbosity, f"Prune network #{n} in round {p}. ", False)
                    self.nets[n].load_state_dict(self.nets[0].state_dict())
                    self.nets[n].prune_net(self.specs.prune_rate_conv ** p, self.specs.prune_rate_fc ** p, reset=True)

                if n == 0:
                    self.hists.sparsity[p] = self.nets[0].sparsity_report()[0]

                log_from_medium(self.specs.verbosity, f"Train network #{n} (sparsity {self.hists.sparsity[p]:6.4f}).")

                self.nets[n], self.hists.train_loss[n, p], self.hists.val_loss[n, p], self.hists.val_acc[n, p], \
                self.hists.test_acc[n, p], self.stop_hists.histories[n].indices[p], \
                self.stop_hists.histories[n].state_dicts[p] \
                    = self.trainer.train_net(self.nets[n], self.specs.epoch_count, self.specs.plot_step)

                toc = time.time()
                log_from_medium(self.specs.verbosity,
                                f"Final test-accuracy: {(self.hists.test_acc[n, p, -1]):6.4f} "
                                f"(took {plotter.format_time(toc - tic)}).")
            log_from_medium(self.specs.verbosity, "")
