import json
import os
from dataclasses import asdict

import numpy as np
import torch


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


def generate_histories_file_name(file_prefix):
    """ Generate file name and suffix for histories-file (.npz). """
    return f"{file_prefix}-histories.npz"


def generate_net_file_name(file_prefix, net_number):
    """ Generate file name and suffix for net-file (.pth). """
    return f"{file_prefix}-net{net_number}.pth"


def save_specs(results_path, file_prefix, specs):
    """ Save experiment's specs as dict in json-file. """
    file_name = generate_specs_file_name(file_prefix)
    json_path = os.path.join(results_path, file_name)

    description_json = json.dumps(asdict(specs))
    with open(json_path, "w") as f:
        f.write(description_json)


def save_nets(results_path, file_prefix, net_list):
    """ Save state_dicts from trained networks in single pth-files. """
    for net_number, net in enumerate(net_list):
        net.train(True)
        net.to(torch.device("cpu"))

        file_name = generate_net_file_name(file_prefix, net_number)
        net_path = os.path.join(results_path, file_name)

        with open(net_path, "wb") as f:
            torch.save(net.state_dict(), f)


def save_histories(results_path, file_prefix, loss_h, val_acc_h, test_acc_h, val_acc_ep_h, test_acc_ep_h, sparsity_h):
    """ Save loss-, validation- and test-histories and sparsity-history in npz-file. """
    file_name = generate_histories_file_name(file_prefix)
    histories_path = os.path.join(results_path, file_name)

    with open(histories_path, "wb") as f:
        np.savez(f, loss_h=loss_h, val_acc_h=val_acc_h, test_acc_h=test_acc_h,
                 val_acc_ep_h=val_acc_ep_h, test_acc_ep_h=test_acc_ep_h, sparsity_h=sparsity_h)
