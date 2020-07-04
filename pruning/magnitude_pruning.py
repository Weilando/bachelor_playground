import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import math

def setup_masks(layer):
    """ Setup a mask of ones for all linear and convolutional layers. """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        prune.custom_from_mask(layer, name='weight', mask=torch.ones_like(layer.weight))

def prune_mask(layer, prune_rate):
    """ Prune mask for given layer, i.e. set its entries to zero for weights with smallest magnitudes.
    The amount of pruning is the product of the pruning rate and the number of the layer's unpruned weights. """
    if isinstance(layer, nn.Linear):
        init_weight_count = layer.in_features * layer.out_features
    elif isinstance(layer, nn.Conv2d):
        init_weight_count = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
    else:
        raise AssertionError(f"Could not prune mask for layer of type {type(layer)}.")

    unpruned_weight_count = int(layer.weight_mask.flatten().sum())
    pruned_weight_count = int(init_weight_count - unpruned_weight_count)
    prune_amount = math.ceil(unpruned_weight_count * prune_rate)

    sorted_weight_indices = layer.weight.flatten().abs().argsort()

    new_mask = layer.weight_mask.clone()
    new_mask.flatten()[sorted_weight_indices[pruned_weight_count:(pruned_weight_count+prune_amount)]] = 0.
    return new_mask

def prune_layer(layer, prune_rate):
    """ Prune given layer with certain prune_rate and reset surviving weights to their initial values. """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        pruned_mask = prune_mask(layer, prune_rate)

        # temporarily remove pruning
        prune.remove(layer, name='weight')
        # set weights to initial weights
        layer.weight = nn.Parameter(layer.weight_init.clone())

        # apply pruned mask
        prune.custom_from_mask(layer, name='weight',  mask=pruned_mask)
