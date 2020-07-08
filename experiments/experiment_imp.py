from experiments.experiment import Experiment
from nets.conv import Conv
from nets.lenet import Lenet
from training.logger import log_from_medium


class ExperimentIMP(Experiment):
    def __init__(self, args, result_path='../data/results'):
        super(ExperimentIMP, self).__init__(args, result_path)

    def prune_net(self, net):
        """ Prune given net via its 'prune_net' method. """
        if isinstance(net, Lenet):
            net.prune_net(self.args.prune_rate_fc)
        elif isinstance(net, Conv):
            net.prune_net(self.args.prune_rate_conv, self.args.prune_rate_fc)

    def execute_experiment(self):
        """ Perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        for n in range(self.args.net_count):
            for p in range(0, self.args.prune_count + 1):
                if p > 0:
                    log_from_medium(self.args.verbosity, f"Prune network #{n} in round {p}.", False)
                    self.prune_net(self.nets[n])

                if n == 0:
                    self.sparsity_hist[p] = self.nets[0].sparsity_report()[0]

                log_from_medium(self.args.verbosity, f"Train network #{n} (sparsity {self.sparsity_hist[p]:.6}).")

                self.nets[n], self.loss_hists[n, p], self.val_acc_hists[n, p], self.test_acc_hists[n, p], \
                self.val_acc_hists_epoch[n, p], self.test_acc_hists_epoch[n, p] \
                    = self.trainer.train_net(self.nets[n], self.args.epoch_count, self.args.plot_step)

                log_from_medium(self.args.verbosity,
                                f"Final test-accuracy: {(self.test_acc_hists_epoch[n, p, -1]):1.4}")
            log_from_medium(self.args.verbosity, "")
