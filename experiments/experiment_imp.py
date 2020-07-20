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
                    log_from_medium(self.args.verbosity, f"Prune network #{n} in round {p}. ", False)
                    self.prune_net(self.nets[n])

                if n == 0:
                    self.hists.sparsity[p] = self.nets[0].sparsity_report()[0]

                log_from_medium(self.args.verbosity, f"Train network #{n} (sparsity {self.hists.sparsity[p]:6.4f}).")

                self.nets[n], self.hists.train_loss[n, p], self.hists.val_loss[n, p], \
                self.hists.val_acc[n, p], self.hists.test_acc[n, p] \
                    = self.trainer.train_net(self.nets[n], self.args.epoch_count, self.args.plot_step)

                log_from_medium(self.args.verbosity,
                                f"Final test-accuracy: {(self.hists.test_acc[n, p, -1]):6.4f}")
            log_from_medium(self.args.verbosity, "")
