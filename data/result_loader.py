import glob
import json
import os

import numpy as np
import torch

from experiments.experiment_settings import NetNames
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
    """ Read histories from the npz-file specified by the given experiment_prefix and return them as np.arrays. """
    h_path = os.path.join(os.getcwd(), f"{experiment_prefix}-histories.npz")
    hf = np.load(h_path)
    return hf['loss_h'], hf['val_acc_h'], hf['test_acc_h'], hf['val_acc_ep_h'], hf['test_acc_ep_h'], hf['sparsity_h']


def get_models_from_file(experiment_prefix, specs):
    """ Read models' state_dicts from pth-files specified by the given experiment_prefix.
    Return an array of nets with the loaded states. """
    nets = []
    for model_file in sorted(glob.glob(f"{experiment_prefix}-net[0-9]*.pth")):
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))

        if specs.net == NetNames.LENET:
            net = lenet.Lenet(specs.plan_fc)
            net.load_state_dict(checkpoint)
            net.prune_net(0.)  # apply pruned masks, but do not modify the masks
        elif specs.net == NetNames.CONV:
            net = conv.Conv(specs.plan_conv, specs.plan_fc)
            net.load_state_dict(checkpoint)
            net.prune_net(0., 0.)  # apply pruned masks, but do not modify the masks
        else:
            raise AssertionError(f"Could not rebuild net because name {specs.net} is invalid.")

        nets.append(net)
    return nets
