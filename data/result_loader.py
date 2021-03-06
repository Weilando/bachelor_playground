import glob
import json
import os

import numpy as np
import torch

from experiments.early_stop_histories import EarlyStopHistoryList
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_specs import ExperimentSpecs
from nets.net import Net


# helper functions
def is_valid_specs_path(absolute_specs_path):
    """ Check if the given absolute path exists and if the file ends with '-specs.json'. """
    return os.path.exists(absolute_specs_path) and absolute_specs_path.endswith('-specs.json')


def generate_absolute_specs_path(relative_specs_path):
    """ Generate absolute specs path from given relative specs path. """
    return os.path.join(os.getcwd(), relative_specs_path)


def generate_experiment_path_prefix(absolute_specs_path):
    """ Given a relative path to a specs-file, extract the experiment's absolute path prefix. """
    assert absolute_specs_path.endswith('-specs.json') and (absolute_specs_path.count('specs.json') == 1)
    return absolute_specs_path.replace('-specs.json', '')


def generate_experiment_histories_file_path(experiment_path_prefix):
    """ Given an 'experiment_path_prefix', append '-histories.npz'. """
    return f"{experiment_path_prefix}-histories.npz"


def generate_random_experiment_histories_file_path(experiment_path_prefix, net_number):
    """ Given an 'experiment_path_prefix', return random-histories-file path with 'net_number'. """
    return f"{experiment_path_prefix}-random-histories{net_number}.npz"


def generate_early_stop_file_path(experiment_path_prefix, net_number):
    """ Given an 'experiment_path_prefix', return an early-stop-file path with 'net_number'. """
    return f"{experiment_path_prefix}-early-stop{net_number}.pth"


def generate_net_file_paths(experiment_path_prefix, net_count):
    """ Given an 'experiment_path_prefix', return an array of net-file paths. """
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
    """ Read the specs-file (.json) specified by 'absolute_specs_path' and return a dict or ExperimentSpecs object. """
    with open(absolute_specs_path, 'r') as specs_file:
        specs_dict = json.load(specs_file)
    if as_dict:
        return specs_dict
    return ExperimentSpecs(**specs_dict)


def get_experiment_histories_from_file(experiment_path_prefix):
    """ Read history-arrays from the specified npz-file and return them as ExperimentHistories object. """
    histories_file_path = generate_experiment_histories_file_path(experiment_path_prefix)
    with np.load(histories_file_path) as histories_file:
        return ExperimentHistories(**histories_file)  # unpack dict-like histories-file


def get_random_experiment_histories_from_file(experiment_path_prefix, net_number):
    """ Read history-arrays from the specified npz-file and return them as ExperimentHistories object. """
    histories_file_path = generate_random_experiment_histories_file_path(experiment_path_prefix, net_number)
    with np.load(histories_file_path) as histories_file:
        return ExperimentHistories(**histories_file)  # unpack dict-like histories-file


def get_all_random_experiment_histories_from_files(experiment_path_prefix, net_count):
    """ Read history-arrays from all specified npz-files with net_number from zero to 'net_count' and return them as
    one ExperimentHistories object. """
    assert net_count > 0, f"'net_count' needs to be greater than 0, but is {net_count}."
    histories = get_random_experiment_histories_from_file(experiment_path_prefix, 0)

    for net_number in range(1, net_count):
        current_histories = get_random_experiment_histories_from_file(experiment_path_prefix, net_number)
        histories = histories.stack_histories(current_histories)
    return histories


def get_early_stop_history_from_file(experiment_path_prefix, specs, net_number):
    """ Read EarlyStopHistory from the specified pth-file. """
    assert isinstance(specs, ExperimentSpecs), f"'specs' has invalid type {type(specs)}."
    assert specs.save_early_stop, f"'save_early_stop' is False in given 'specs', i.e. no EarlyStopHistoryList exists."
    assert 0 <= net_number < specs.net_count, \
        f"'net_number' needs to be between 0 and {specs.net_count - 1}, but is {net_number}."

    early_stop_file_path = generate_early_stop_file_path(experiment_path_prefix, net_number)
    return torch.load(early_stop_file_path, map_location=torch.device("cpu"))


def get_early_stop_history_list_from_files(experiment_path_prefix, specs):
    """ Read all EarlyStopHistory objects related to 'specs' from pth-files and return one EarlyStopHistoryList. """
    assert isinstance(specs, ExperimentSpecs), f"'specs' has invalid type {type(specs)}."
    assert specs.save_early_stop, f"'save_early_stop' is False in given 'specs', i.e. no EarlyStopHistoryList exists."

    history_list = EarlyStopHistoryList()
    history_list.setup(specs.net_count, specs.prune_count)

    for net_number in range(specs.net_count):
        early_stop_file_path = generate_early_stop_file_path(experiment_path_prefix, net_number)
        history_list.histories[net_number] = torch.load(early_stop_file_path, map_location=torch.device("cpu"))
    return history_list


def get_models_from_files(experiment_path_prefix, specs):
    """ Read models' state_dicts from specified pth-files  and return an array of nets with the loaded states. """
    assert isinstance(specs, ExperimentSpecs), f"Expected specs of type ExperimentSpecs, but got {type(specs)}."
    nets = []
    net_file_paths = generate_net_file_paths(experiment_path_prefix, specs.net_count)
    for model_file in net_file_paths:
        state_dict = torch.load(model_file, map_location=torch.device("cpu"))
        nets.append(generate_model_from_state_dict(state_dict, specs))
    return nets


def generate_model_from_state_dict(state_dict, specs):
    """ Generate a model specified by 'specs' and load the given 'state_dict'. """
    net = Net(specs.net, specs.dataset, specs.plan_conv, specs.plan_fc)
    net.load_state_dict(state_dict)
    net.prune_net(0., 0., reset=False)  # apply pruned masks, but do not modify the masks
    return net


def random_histories_file_exists(experiment_path_prefix, net_number):
    """ Indicate if a random-histories-file exists for 'net_number'. """
    file_path = generate_random_experiment_histories_file_path(experiment_path_prefix, net_number)
    return len(glob.glob(file_path)) > 0
