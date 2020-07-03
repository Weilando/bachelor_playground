import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.experiment_settings import VerbosityLevel
from experiments.experiment import Experiment
from nets.conv import Conv

class Experiment_Conv_CIFAR10(Experiment):
    def __init__(self, args):
        super(Experiment_Conv_CIFAR10, self).__init__(args)
        self.prune_rate_conv = args['prune_rate_conv']
        self.prune_rate_fc = args['prune_rate_fc']
        self.prune_count = args['prune_count']

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the exeperiment. """
        self.nets = [None] * self.net_count
        for n in range(self.net_count):
            self.nets[n] = Conv(self.args['plan_conv'], self.args['plan_fc'])
        if self.verbosity == VerbosityLevel.DETAILED:
            print(self.nets[0])

    def execute_experiment(self):
        """ Execute experiment, i.e. perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        for n in range(self.net_count):
            for p in range(0, self.prune_count+1):
                if p > 0:
                    if self.verbosity != VerbosityLevel.SILENT:
                        print(f"Prune network #{n} in round {p}", end=" ")
                    self.nets[n].prune_net(self.prune_rate_conv, self.prune_rate_fc)

                if n==0:
                    self.sparsity_hist[p] = self.nets[0].sparsity_report()[0]

                if self.verbosity != VerbosityLevel.SILENT:
                    print(f"Train network #{n} (sparsity {self.sparsity_hist[p]:.6}).")

                self.nets[n], self.loss_hists[n,p], self.val_acc_hists[n,p], self.test_acc_hists[n,p], self.val_acc_hists_epoch[n,p], self.test_acc_hists_epoch[n,p] = self.trainer.train_net(self.nets[n], self.epoch_count, self.plot_step)

                if self.verbosity != VerbosityLevel.SILENT:
                    print(f"Final test-accuracy: {(self.test_acc_hists_epoch[n,p,-1]):1.4}")
            if self.verbosity != VerbosityLevel.SILENT:
                print()
