import unittest

import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.trainer import TrainerAdam

class Test_trainer(unittest.TestCase):
    """ Tests for the trainer module.
    Call with 'python -m test.test_trainer' from project root '~'.
    Call with 'python -m test_trainer' from inside '~/test'. """
    def test_calculate_correct_test_acc(self):
        ''' The correct test-accuracy should be calculated.
        The fake-net with one linear layer classifies half of the fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. '''
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))

        # setup trainer and fake-dataloader with two batches (use the same samples for both batches)
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels_batch1 = torch.tensor([0,0,1])
        labels_batch2 = torch.tensor([1,1,0])
        test_loader = [[samples, labels_batch1], [samples, labels_batch2]]
        trainer = TrainerAdam(0., None, None, test_loader=test_loader)

        self.assertEqual(0.5, trainer.compute_acc(net, test=True))

    def test_calculate_correct_val_acc(self):
        ''' The correct validation-accuracy should be calculated.
        The fake-net with one linear layer classifies all fake-samples correctly.
        Use a fake-val_loader with one batch to validate the result. '''
        # setup net
        net = nn.Linear(4,2)
        net.weight = nn.Parameter(torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]))
        net.bias = nn.Parameter(torch.zeros(2))

        # setup trainer and fake-dataloader with one batch
        samples = torch.tensor([[2., 2., 2., 2.], [2., 2., 0., 0.], [0., 0., 2., 2.]])
        labels = torch.tensor([0,0,1])
        val_loader = [[samples, labels]]
        trainer = TrainerAdam(0., None, val_loader, test_loader=None)

        self.assertEqual(1., trainer.compute_acc(net, test=False))

if __name__ == '__main__':
    unittest.main()
