import torch.nn as nn


def gaussian_glorot(module):
    """ Recursively apply Gaussian Glorot initialization to all linear and convolutional layers. """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
