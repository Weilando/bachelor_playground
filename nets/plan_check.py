import re

from torch import nn as nn

from experiments.experiment_specs import NetNames


def is_numerical_spec(input_string):
    """ Match all strings containing a positive integer without leading zeros. """
    regex = re.compile('[1-9][0-9]*')
    match = regex.fullmatch(str(input_string))
    return bool(match)


def is_batch_norm_spec(input_string):
    """ Match all strings containing a positive integer without leading zeros followed by one 'B'. """
    regex = re.compile('[1-9][0-9]*[B]')
    match = regex.fullmatch(str(input_string))
    return bool(match)


def get_number_from_numerical_spec(spec):
    """ Return the number from a numerical spec. """
    assert isinstance(spec, int) or isinstance(spec, str), f"'input' has invalid type {type(spec)}."
    if isinstance(spec, str):
        try:
            return int(spec)
        except ValueError:
            raise ValueError("'spec' does not contain an int.")
    return spec


def get_number_from_batch_norm_spec(spec):
    """ Return the number from a batch-norm spec. """
    try:
        return int(spec[:-1])
    except ValueError:
        raise ValueError("'spec' does not contain an int.")


def get_activation(net_name: NetNames):
    """ Return tanh for Lenet and ReLU otherwise. """
    return nn.Tanh() if (net_name == NetNames.LENET) else nn.ReLU()
