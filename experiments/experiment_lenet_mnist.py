import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments import experiment_settings, experiment
from nets.lenet import Lenet
from data.dataloaders import get_mnist_dataloaders
from training.trainer import TrainerAdam

class Experiment_Lenet_MNIST(experiment.Experiment):
    def __init__(self, args):
        super(Experiment_Lenet_MNIST, self).__init__(args)
        self.prune_rate_fc = args['prune_rate_fc']
        self.prune_count = args['prune_count']

    def load_data_and_setup_trainer(self):
        """ Load dataset and initialize trainer in 'self.trainer'.
        Store the length of the training-loader into 'self.epoch_length' to initialize histories. """
        # load dataset
        train_loader, val_loader, test_loader = get_mnist_dataloaders(device=self.args['device'])
        self.epoch_length = len(train_loader)

        # initialize trainer
        self.trainer = TrainerAdam(self.learning_rate, train_loader, val_loader, test_loader, self.device)

    def init_nets(self):
        """ Initialize nets in list 'self.nets' which should be trained during the exeperiment. """
        self.nets = [None] * self.net_count
        for n in range(self.net_count):
            self.nets[n] = Lenet(self.args['plan_fc'])
        print(self.nets[0])

    def execute_experiment(self):
        """ Execute experiment, i.e. perform iterative magnitude pruning and save accuracy- and loss-histories after each training.
        Retrain in the end to check the last nets' accuracies. """
        print(f"Prune networks with rate {self.prune_rate_fc}.")
        for n in range(self.net_count):
            print(f"Train network #{n} (unpruned).")
            self.nets[n], self.loss_hists[n,0], self.val_acc_hists[n,0], self.test_acc_hists[n,0], self.val_acc_hists_epoch[n,0], self.test_acc_hists_epoch[n,0] = self.trainer.train_net(self.nets[n], self.epoch_count, self.plot_step)
            print(f"Final test-accuracy: {(self.test_acc_hists[n,0,-1]):1.4}")

            for p in range(1, self.prune_count+1):
                print(f"Prune network #{n} in round {p}", end="")
                self.nets[n].prune_net(self.prune_rate_fc)

                if n==0:
                    self.sparsity_hist[p] = self.nets[0].sparsity_report()[0]
                print(f" (sparsity {self.sparsity_hist[p]:.6}).", end="")

                print(" Train network.")
                self.nets[n], self.loss_hists[n,p], self.val_acc_hists[n,p], self.test_acc_hists[n,p], self.val_acc_hists_epoch[n,p], self.test_acc_hists_epoch[n,p] = self.trainer.train_net(self.nets[n], self.epoch_count, self.plot_step)

                print(f"Final test-accuracy: {(self.test_acc_hists[n,p,-1]):1.4}")
            print()
