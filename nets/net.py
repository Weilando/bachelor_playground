import torch.nn as nn

class Net(nn.Module):
    """
    A trainable and prunable network.
    Its architecture can be specified via sizes in plan_conv and plan_fc.
    Initial weights for each layer are stored as buffers after applying the weight initialization.
    """
    def __init__(self):
        super(Net, self).__init__()
        pass

    def store_initial_weights(self):
        """ Store initial weights as buffer in each layer. """
        pass

    def forward(self, x):
        """ Calculate forward pass for tensor x. """
        pass

    def prune_net(self, prune_rate_conv, prune_rate_fc):
        """ Prune all layers with the given prune rates.
        Use weight masks and reset the unpruned weights to their initial values after pruning. """
        pass

    def sparsity_layer(self, layer):
        """ Calculates sparsity and counts unpruned weights for given layer. """
        pass

    def sparsity_report(self):
        """ Generate a list with sparsities for the whole network and per layer. """
        pass
