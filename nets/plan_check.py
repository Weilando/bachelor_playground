import re


def is_numerical_spec(input_string):
    """ Matches all strings containing a positive integer without leading zeros. """
    regex = re.compile('[1-9][0-9]*')
    match = regex.fullmatch(str(input_string))
    return bool(match)


def is_batch_norm_spec(input_string):
    """ Matches all strings containing a positive integer without leading zeros followed by one 'B'. """
    regex = re.compile('[1-9][0-9]*[B]')
    match = regex.fullmatch(str(input_string))
    return bool(match)
