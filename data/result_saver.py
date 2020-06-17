import json
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_results_path():
    """ Generate absolute path to subdirectory 'results' and create it, if necessary. """
    results_path = os.path.join(os.getcwd(), '../data/results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    return results_path

def generate_file_prefix(specs, save_time):
    """ Generate file names' prefix. """
    return f"{save_time}-{specs['net']}-{specs['dataset']}"

def save_specs(specs, results_path, file_prefix):
    """ Save experiment's specs in json-file. """
    # create path to json-file
    json_path = os.path.join(results_path, f"{file_prefix}-specs.json")

    # write dict 'args' to json-file
    description_json = json.dumps(specs)
    with open(json_path, "w") as f:
        f.write(description_json)

def save_nets(experiment, results_path, file_prefix):
    """ Save state_dicts from trained networks in single pth-files. """
    for num, net in enumerate(experiment.nets):
        net.train(True)
        net.to(torch.device("cpu"))
        net_path = histories_path = os.path.join(results_path, f"{file_prefix}-net{num}.pth")
        torch.save(net.state_dict(), net_path)

def save_histories(experiment, results_path, file_prefix):
    """ Save loss-, validation- and test-histories and sparsity-history in npz-file. """
    histories_path = os.path.join(results_path, f"{file_prefix}-histories") # suffix is added by np.savez
    np.savez(histories_path,
            loss_histories=experiment.loss_histories,
            val_acc_histories=experiment.val_acc_histories,
            test_acc_histories=experiment.test_acc_histories,
            sparsity_history=experiment.sparsity_history)
