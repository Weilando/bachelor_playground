import time

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import plotter
from data import result_saver as rs

class Experiment(object):
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args

        self.net_count = args['net_count']
        self.epoch_count = args['epoch_count']
        self.learning_rate = args['learning_rate']
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

    def save_results(self):
        """ Save experiment's specs, histories and models on disk. """
        save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
        file_prefix = rs.generate_file_prefix(self.args, save_time)

        results_path = rs.setup_results_path()
        rs.save_specs(self.args, results_path, file_prefix)
        rs.save_histories(self, results_path, file_prefix)
        rs.save_nets(self, results_path, file_prefix)
        print("Successfully wrote results on disk.")
