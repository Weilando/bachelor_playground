import glob
import json
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_spec_files():
    """ Print all relative paths for files ending with '-specs.json' from subdirectory '/results'. """
    for experiment_file in sorted(glob.glob("results/*-specs.json")):
        print(experiment_file)

def extract_experiment_prefix(specs_relative_path):
    """ Given a relative path to a specs-file, extract the experiment's relative prefix. """
    return specs_relative_path.replace('-specs.json', '')

def get_specs_from_file(specs_relative_path):
    """ Read the specs-file (.json) specified by the given relative path and return it as dict. """
    specs_absolute_path = os.path.join(os.getcwd(), specs_relative_path)
    with open(specs_absolute_path, 'r') as f:
        specs = json.load(f)
    return specs

def get_histories_from_file(experiment_prefix):
    """ Read histories from the histories-file (.npz) specified by the given experiment_prefix and return them as np.arrays. """
    histories_path = os.path.join(os.getcwd(), f"{experiment_prefix}-histories.npz")
    histories_file = np.load(histories_path)
    return histories_file['loss_histories'], histories_file['val_acc_histories'], histories_file['test_acc_histories'], histories_file['sparsity_history']

def get_models_from_file(experiment_prefix):
    """ Read models from their model-files (.pth) specified by the given experiment_prefix and return them as array of nets. """
    nets = []
    for model_file in sorted(glob.glob(f"{experiment_prefix}-net[0-9]*.pth")):
        nets.append(torch.load(model_file, map_location=torch.device("cpu")))
    return nets
