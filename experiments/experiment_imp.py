import time

from data.plotter_evaluation import format_time
from experiments.experiment_pruning import ExperimentPruning
from training.logger import log_from_medium


class ExperimentIMP(ExperimentPruning):
    """
    Experiment with iterative magnitude pruning with resetting (IMP).
    """

    def __init__(self, specs, result_path='../data/results'):
        super(ExperimentIMP, self).__init__(specs, result_path)

    # noinspection DuplicatedCode
    def execute_experiment(self):
        """ Perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain and evaluate nets after each pruning step. """
        for n in range(self.specs.net_count):
            for p in range(0, self.specs.prune_count + 1):
                tic = time.time()
                if p > 0:
                    log_from_medium(self.specs.verbosity, f"Prune network #{n} in round {p}. ", False)
                    self.nets[n].prune_net(self.specs.prune_rate_conv, self.specs.prune_rate_fc, reset=True)

                if n == 0:
                    self.hists.sparsity[p] = self.nets[0].sparsity_report()[0]

                log_from_medium(self.specs.verbosity, f"Train network #{n} (sparsity {self.hists.sparsity[p]:6.4f}).")
                (self.nets[n], self.hists.train_loss[n, p], self.hists.val_loss[n, p], self.hists.val_acc[n, p],
                 self.hists.test_acc[n, p], self.stop_hists.histories[n].indices[p],
                 self.stop_hists.histories[n].state_dicts[p]) \
                    = self.trainer.train_net(self.nets[n], self.specs.epoch_count, self.specs.plot_step)

                toc = time.time()
                log_from_medium(self.specs.verbosity,
                                f"Final test-accuracy: {(self.hists.test_acc[n, p, -1]):6.4f} "
                                f"(took {format_time(toc - tic)}).")
            log_from_medium(self.specs.verbosity, "")
