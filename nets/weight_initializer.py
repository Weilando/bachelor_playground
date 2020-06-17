import torch

def gaussian_glorot(layer):
    """ Recursively apply Gaussian Glorot initialization to all linear and convolutional layers. """
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(layer.weight)
