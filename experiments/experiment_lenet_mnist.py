from experiments.experiment_imp import ExperimentIMP
from experiments.experiment_settings import VerbosityLevel
from nets.lenet import Lenet


class ExperimentLenetMNIST(ExperimentIMP):
    def __init__(self, args):
        super(ExperimentLenetMNIST, self).__init__(args)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the experiment. """
        for n in range(self.args.net_count):
            self.nets[n] = Lenet(self.args.plan_fc)
        if self.args.verbosity == VerbosityLevel.DETAILED:
            print(self.nets[0])

    def prune_net(self, net):
        """ Prune given net via its 'prune_net' method. """
        net.prune_net(self.args.prune_rate_fc)
