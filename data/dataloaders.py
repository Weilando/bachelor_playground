import torch.utils.data as d_utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_mnist_dataloaders(batch_size=60, train_len=55000, val_len=5000, path='../data/datasets', device='cpu'):
    """ Load the MNIST-dataset and provide DataLoaders with given properties.
    The training-set contains 60000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len+val_len) == 60000, f"Can not apply split into training- and validation-set, as {train_len}+{val_len}!=60000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=tr)
    train_data, val_data = d_utils.random_split(train_data, [train_len, val_len])
    train_loader = d_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device!='cpu'))
    val_loader = d_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device!='cpu'))

    test_data = datasets.MNIST(root=path, train=False, download=True, transform=tr)
    test_loader = d_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device!='cpu'))

    print(len(train_data), len(train_loader), len(val_data), len(val_loader), len(test_data), len(test_loader))
    return train_loader, val_loader, test_loader

def get_cifar10_dataloaders(batch_size=60, train_len=45000, val_len=5000, path='../data/datasets', device='cpu'):
    """ Load the CIFAR-10-dataset and provide DataLoaders with given properties.
    The training-set contains 50000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len+val_len) == 50000, f"Can not apply split into training- and validation-set, as {train_len}+{val_len}!=50000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=tr)
    train_data, val_data = d_utils.random_split(train_data, [45000, 5000])
    train_loader = d_utils.DataLoader(train_data, batch_size=60, shuffle=True, num_workers=4, pin_memory=(device!='cpu'))
    val_loader = d_utils.DataLoader(val_data, batch_size=60, shuffle=False, num_workers=2, pin_memory=(device!='cpu'))

    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=tr)
    test_loader = d_utils.DataLoader(test_data, batch_size=60, shuffle=False, num_workers=2, pin_memory=(device!='cpu'))

    print(len(train_data), len(train_loader), len(val_data), len(val_loader), len(test_data), len(test_loader))
    return train_loader, val_loader, test_loader
