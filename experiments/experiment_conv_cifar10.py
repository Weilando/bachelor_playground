from experiments.experiment_imp import ExperimentIMP
from experiments.experiment_settings import VerbosityLevel
from nets.conv import Conv


class ExperimentConvCIFAR10(ExperimentIMP):
    def __init__(self, args):
        super(ExperimentConvCIFAR10, self).__init__(args)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the experiment. """
        for n in range(self.args.net_count):
            self.nets[n] = Conv(self.args.plan_conv, self.args.plan_fc)
        if self.args.verbosity == VerbosityLevel.DETAILED:
            print(self.nets[0])

    def prune_net(self, net):
        """ Prune given net via its 'prune_net' method. """
        net.prune_net(self.args.prune_rate_conv, self.args.prune_rate_fc)
