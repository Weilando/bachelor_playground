import json
import os
from dataclasses import asdict

import numpy as np
import torch

from experiments.early_stop_histories import EarlyStopHistoryList
from experiments.experiment_histories import ExperimentHistories
from experiments.experiment_settings import ExperimentSettings


def setup_and_get_result_path(relative_result_path='../data/results'):
    """ Generate absolute path to results directory and create it, if necessary. """
    absolute_result_path = os.path.join(os.getcwd(), relative_result_path)
    if not os.path.exists(absolute_result_path):
        os.mkdir(absolute_result_path)
    return absolute_result_path


def generate_file_prefix(specs, save_time):
    """ Generate file names' prefix. """
    return f"{save_time}-{specs.net}-{specs.dataset}"


def generate_specs_file_name(file_prefix):
    """ Generate file name and suffix for specs-file (.json). """
    return f"{file_prefix}-specs.json"


def generate_experiment_histories_file_name(file_prefix):
    """ Generate file name and suffix for histories-file (.npz). """
    return f"{file_prefix}-histories.npz"


def generate_early_stop_file_name(file_prefix, net_number):
    """ Generate file name and suffix for EarlyStopHistory-file (.pth). """
    return f"{file_prefix}-early-stop{net_number}.pth"


def generate_net_file_name(file_prefix, net_number):
    """ Generate file name and suffix for net-file (.pth). """
    return f"{file_prefix}-net{net_number}.pth"


def save_specs(results_path, file_prefix, specs):
    """ Save experiment's 'specs' as dict in json-file. """
    assert isinstance(specs, ExperimentSettings), f"specs must have type ExperimentSettings, but is {type(specs)}."

    file_name = generate_specs_file_name(file_prefix)
    json_path = os.path.join(results_path, file_name)

    description_json = json.dumps(asdict(specs))
    with open(json_path, "w") as f:
        f.write(description_json)


def save_nets(results_path, file_prefix, net_list):
    """ Save state_dicts from trained networks in 'net_list' in single pth-files. """
    for net_number, net in enumerate(net_list):
        net.train(True)
        net.to(torch.device("cpu"))

        file_name = generate_net_file_name(file_prefix, net_number)
        file_path = os.path.join(results_path, file_name)

        with open(file_path, "wb") as f:
            torch.save(net.state_dict(), f)


def save_experiment_histories(results_path, file_prefix, histories):
    """ Save all np.arrays from 'histories' in one npz-file. """
    assert isinstance(histories, ExperimentHistories), \
        f"'histories' must have type ExperimentHistories, but is {type(histories)}."

    file_name = generate_experiment_histories_file_name(file_prefix)
    file_path = os.path.join(results_path, file_name)

    with open(file_path, "wb") as f:
        histories_dict = asdict(histories)  # generate key-value pairs of names and np.arrays
        np.savez(f, **histories_dict)  # unpack key-value pairs to save named arrays


def save_early_stop_history_list(results_path, file_prefix, history_list):
    """ Save each EarlyStopHistory from 'history_list' in a single pth-file. """
    assert isinstance(history_list, EarlyStopHistoryList), \
        f"'checkpoints' must have type EarlyStopHistoryList, but is {type(history_list)}."

    for net_number, history in enumerate(history_list.histories):
        file_name = generate_early_stop_file_name(file_prefix, net_number)
        file_path = os.path.join(results_path, file_name)

        with open(file_path, "wb") as f:
            torch.save(history, f)
