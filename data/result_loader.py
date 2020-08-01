import glob
import json
import os

import numpy as np
import torch

from experiments.early_stop_histories import EarlyStopHistoryList
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_settings import NetNames, ExperimentSettings
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


def generate_experiment_histories_file_path(experiment_path_prefix):
    """ Given an experiment path prefix, append '-histories.npz'. """
    return f"{experiment_path_prefix}-histories.npz"


def generate_early_stop_file_path(experiment_path_prefix, net_number):
    """ Given an experiment path prefix, return an early-stop-file path with 'net_number'. """
    return f"{experiment_path_prefix}-early-stop{net_number}.pth"


def generate_net_file_paths(experiment_path_prefix, net_count):
    """ Given an experiment path prefix, return an array of net-file paths. """
    assert net_count > 0, f"'net_count' needs to be greater than zero, but was {net_count}."
    return [f"{experiment_path_prefix}-net{n}.pth" for n in range(net_count)]


# higher level functions
def get_relative_spec_file_paths(sub_dir='results'):
    """ Return all relative paths with suffix '-specs.json' from subdirectory 'sub_dir' in ascending order. """
    pattern = os.path.join(sub_dir, "*-specs.json")
    return sorted(glob.glob(pattern))


def extract_experiment_path_prefix(relative_specs_path):
    """ Generate and verify experiment_path_prefix from 'relative_specs_path'. """
    absolute_specs_path = generate_absolute_specs_path(relative_specs_path)
    assert is_valid_specs_path(absolute_specs_path), "The given specs path is invalid."
    return generate_experiment_path_prefix(absolute_specs_path)


def get_specs_from_file(absolute_specs_path, as_dict=False):
    """ Read the specs-file (.json) specified by 'absolute_specs_path'.
    Return result as dict or ExperimentSettings object. """
    with open(absolute_specs_path, 'r') as specs_file:
        specs_dict = json.load(specs_file)
    if as_dict:
        return specs_dict
    return ExperimentSettings(**specs_dict)


def get_experiment_histories_from_file(experiment_path_prefix):
    """ Read history-arrays from the npz-file specified by 'experiment_path_prefix' and return them as
    ExperimentHistory object. """
    histories_file_path = generate_experiment_histories_file_path(experiment_path_prefix)
    with np.load(histories_file_path) as histories_file:
        return ExperimentHistories(**histories_file)  # unpack dict-like histories-file


def get_early_stop_history_from_file(experiment_path_prefix, specs, net_number):
    """ Read EarlyStopHistory from file specified by 'experiment_path_prefix', 'specs' and the corresponding
    'net_number'. """
    assert isinstance(specs, ExperimentSettings), f"Expected specs of type ExperimentSettings, but got {type(specs)}."
    assert specs.save_early_stop, f"'save_early_stop' is False in given 'specs', i.e. no EarlyStopHistoryList exists."
    assert 0 <= net_number < specs.net_count, \
        f"'net_number' needs to be between 0 and {specs.net_count - 1}, but is {net_number}."

    early_stop_file_path = generate_early_stop_file_path(experiment_path_prefix, net_number)
    return torch.load(early_stop_file_path, map_location=torch.device("cpu"))


def get_early_stop_history_list_from_files(experiment_path_prefix, specs):
    """ Read all EarlyStopHistory objects corresponding to 'specs' from their files and return them as one
    EarlyStopHistoryList. """
    assert isinstance(specs, ExperimentSettings), f"Expected specs of type ExperimentSettings, but got {type(specs)}."
    assert specs.save_early_stop, f"'save_early_stop' is False in given 'specs', i.e. no EarlyStopHistoryList exists."

    history_list = EarlyStopHistoryList()
    history_list.setup(specs.net_count, specs.prune_count)

    for net_number in range(specs.net_count):
        early_stop_file_path = generate_early_stop_file_path(experiment_path_prefix, net_number)
        history_list.histories[net_number] = torch.load(early_stop_file_path, map_location=torch.device("cpu"))

    return history_list


def get_models_from_files(experiment_path_prefix, specs):
    """ Read models' state_dicts from pth-files specified by the given experiment_path_prefix.
    Return an array of nets with the loaded states. """
    assert isinstance(specs, ExperimentSettings), f"Expected specs of type ExperimentSettings, but got {type(specs)}."
    nets = []
    net_file_paths = generate_net_file_paths(experiment_path_prefix, specs.net_count)
    for model_file in net_file_paths:
        if specs.net == NetNames.LENET:
            net = lenet.Lenet(specs.plan_fc)
        elif specs.net == NetNames.CONV:
            net = conv.Conv(specs.plan_conv, specs.plan_fc)
        else:
            raise AssertionError(f"Could not rebuild net because name {specs.net} is invalid.")

        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint)
        net.prune_net(0., 0.)  # apply pruned masks, but do not modify the masks
        nets.append(net)
    return nets
