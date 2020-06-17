import glob
import json
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets import lenet, conv

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

def get_models_from_file(experiment_prefix, specs):
    """ Read models' state_dicts from files (.pth) specified by the given experiment_prefix and return an array of nets with the loaded states. """
    nets = []
    for model_file in sorted(glob.glob(f"{experiment_prefix}-net[0-9]*.pth")):
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))

        if specs['net'] == 'lenet':
            net = lenet.Lenet(specs['plan_fc'])
            net.load_state_dict(checkpoint)
            net.prune_net(0.) # apply pruned masks, but do not modify the masks
        else:
            net = conv.Conv(specs['plan_conv'], specs['plan_fc'])
            net.load_state_dict(checkpoint)
            net.prune_net(0., 0.) # apply pruned masks, but do not modify the masks

        nets.append(net)
    return nets
