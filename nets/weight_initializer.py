import torch.nn as nn

def gaussian_glorot(layer):
    """ Recursively apply Gaussian Glorot initialization to all linear and convolutional layers. """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
