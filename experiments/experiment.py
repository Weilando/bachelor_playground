import json
import numpy as np
import time
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import plotter

class Experiment(object):
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args

        self.net_count = args['net_count']
        self.epoch_count = args['epoch_count']
        self.learning_rate = args['learning_rate']
        self.prune_rate = args['prune_rate']
        self.prune_count = args['prune_count']
        self.loss_plot_step = args['loss_plot_step']

    def setup_experiment(self):
        """ Load dataset, initialize trainer, create np.arrays for histories and initialize nets. """
        pass

    def execute_experiment(self):
        """ Execute all actions for experiment and save accuracy- and loss-histories. """
        pass

    def run_experiment(self):
        """ Run experiment, i.e. setup and execute it and store the results. """
        experiment_start = time.time() # start clock for experiment duration

        self.setup_experiment()
        self.execute_experiment()

        experiment_stop = time.time() # stop clock for experiment duration
        duration = plotter.format_time(experiment_stop-experiment_start)
        print(f"Experiment duration: {duration}")
        self.args['duration'] = duration

        self.save_results()

    def setup_results_path(self):
        """ Generate absolute path to subdirectory 'results' and create it, if necessary. """
        results_path = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        return results_path

    def generate_file_prefix(self, save_time):
        """ Generate file names' prefix. """
        return f"{save_time}-{self.args['net']}-{self.args['dataset']}"

    def save_specs(self, results_path, file_prefix):
        """ Save experiment's specs in json-file. """
        # create path to json-file
        json_path = os.path.join(results_path, f"{file_prefix}-specs.json")

        # write dict 'args' to json-file
        description_json = json.dumps(self.args)
        with open(json_path, "w") as f:
            f.write(description_json)

    def save_nets(self, results_path, file_prefix):
        """ Save trained networks in pth-files. """
        for num, net in enumerate(self.nets):
            net_path = histories_path = os.path.join(results_path, f"{file_prefix}-net{num}.pth")
            net.train(True)
            net.to(torch.device("cpu"))
            torch.save(net, net_path)

    def save_histories(self, results_path, file_prefix):
        """ Save loss-, validation- and test-histories and sparsity-history in npz-file. """
        histories_path = os.path.join(results_path, f"{file_prefix}-histories") # suffix is added by np.savez
        np.savez(histories_path,
                loss_histories=self.loss_histories,
                val_acc_histories=self.val_acc_histories,
                test_acc_histories=self.test_acc_histories,
                sparsity_history=self.sparsity_history)

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = self.generate_file_prefix(save_time)

        results_path = self.setup_results_path()
        self.save_specs(results_path, file_prefix)
        self.save_histories(results_path, file_prefix)
        self.save_nets(results_path, file_prefix)
        print("Successfully wrote results on disk.")
