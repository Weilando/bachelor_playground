from experiments.experiment import Experiment
from experiments.experiment_settings import VerbosityLevel


class ExperimentIMP(Experiment):
    def __init__(self, args):
        super(ExperimentIMP, self).__init__(args)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the experiment. """
        pass

    def prune_net(self, net):
        """ Prune given net via its 'prune_net' method. """
        pass

    def execute_experiment(self):
        """ Perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        for n in range(self.args.net_count):
            for p in range(0, self.args.prune_count + 1):
                if p > 0:
                    if self.args.verbosity != VerbosityLevel.SILENT:
                        print(f"Prune network #{n} in round {p}.", end=" ")
                    self.prune_net(self.nets[n])

                if n == 0:
                    self.sparsity_hist[p] = self.nets[0].sparsity_report()[0]

                if self.args.verbosity != VerbosityLevel.SILENT:
                    print(f"Train network #{n} (sparsity {self.sparsity_hist[p]:.6}).")

                self.nets[n], self.loss_hists[n, p], self.val_acc_hists[n, p], self.test_acc_hists[n, p], \
                    self.val_acc_hists_epoch[n, p], self.test_acc_hists_epoch[n, p] \
                    = self.trainer.train_net(self.nets[n], self.args.epoch_count, self.args.plot_step)

                if self.args.verbosity != VerbosityLevel.SILENT:
                    print(f"Final test-accuracy: {(self.test_acc_hists_epoch[n, p, -1]):1.4}")
            if self.args.verbosity != VerbosityLevel.SILENT:
                print()
