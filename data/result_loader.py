import glob
import json
import os

import numpy as np
import torch

from experiments.experiment_settings import NetNames
from nets import lenet, conv


# helper functions
def is_valid_specs_path(absolute_specs_path):
    """ Check if the given absolute path exists and if the file ends with '-specs.json'. """
    return os.path.exists(absolute_specs_path) and absolute_specs_path.endswith('-specs.json')


def generate_absolute_specs_path(relative_specs_path):
    """ Generate absolute specs path from given relative specs path. """
    return os.path.join(os.getcwd(), relative_specs_path)


def generate_experiment_path_prefix(absolute_specs_path):
    """ Given a relative path to a specs-file, extract the experiment's absolute path prefix. """
    return absolute_specs_path.replace('-specs.json', '')


def generate_histories_file_path(experiment_path_prefix):
    """ Given an experiment path prefix, append '-histories.npz'. """
    return f"{experiment_path_prefix}-histories.npz"


def generate_net_file_paths(experiment_path_prefix, net_count):
    """ Given an experiment path prefix, return an array of net-file paths. """
    assert net_count > 0, f"Net count needs to be greater than zero, but was {net_count}."
    return [f"{experiment_path_prefix}-net{n}.pth" for n in range(net_count)]


# higher level functions
def print_relative_spec_file_paths():
    """ Print all relative paths for files ending with '-specs.json' from subdirectory '/results'. """
    for experiment_file in sorted(glob.glob("results/*-specs.json")):
        print(experiment_file)


def extract_experiment_path_prefix(relative_specs_path):
    """ Get and verify experiment_path_prefix from relative path to specs. """
    absolute_specs_path = generate_absolute_specs_path(relative_specs_path)
    assert is_valid_specs_path(absolute_specs_path), "The given specs path is invalid."
    return generate_experiment_path_prefix(absolute_specs_path)


def get_specs_from_file(absolute_specs_path):
    """ Read the specs-file (.json) specified by the given relative path and return it as dict. """
    with open(absolute_specs_path, 'r') as f:
        specs = json.load(f)
    return specs


def get_histories_from_file(experiment_path_prefix):
    """ Read histories from the npz-file specified by the given experiment_path_prefix and return them as np.arrays. """
    histories_file_path = generate_histories_file_path(experiment_path_prefix)
    with np.load(histories_file_path) as f:
        return f['loss_h'], f['val_acc_h'], f['test_acc_h'], f['val_acc_ep_h'], f['test_acc_ep_h'], f['sparsity_h']


def get_models_from_files(experiment_path_prefix, specs):
    """ Read models' state_dicts from pth-files specified by the given experiment_path_prefix.
    Return an array of nets with the loaded states. """
    nets = []
    net_file_paths = generate_net_file_paths(experiment_path_prefix, specs['net_count'])
    for model_file in net_file_paths:
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
        if specs['net'] == NetNames.LENET:
            net = lenet.Lenet(specs['plan_fc'])
            net.load_state_dict(checkpoint)
            net.prune_net(0.)  # apply pruned masks, but do not modify the masks
        elif specs['net'] == NetNames.CONV:
            net = conv.Conv(specs['plan_conv'], ['specs.plan_fc'])
            net.load_state_dict(checkpoint)
            net.prune_net(0., 0.)  # apply pruned masks, but do not modify the masks
        else:
            raise AssertionError(f"Could not rebuild net because name {specs['net']} is invalid.")

        nets.append(net)
    return nets
