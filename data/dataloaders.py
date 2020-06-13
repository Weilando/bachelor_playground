import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_mnist_dataloaders(batch_size=60, train_len=55000, val_len=5000, path='../data/datasets'):
    """ Load the MNIST-dataset and provide DataLoaders with given properties.
    The training-set contains 60000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len+val_len) == 60000, f"Can not apply split into training- and validation-set, as {train_len}+{val_len}!=60000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    training_data = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=tr)
    training_data, validation_data = torch.utils.data.random_split(training_data, [train_len, val_len])
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=1)

    test_data = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=tr)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    print(len(training_data), len(training_loader), len(validation_data), len(validation_loader), len(test_data), len(test_loader))
    return training_loader, validation_loader, test_loader

def get_cifar10_dataloaders(batch_size=60, train_len=45000, val_len=5000, path='../data/datasets'):
    """ Load the CIFAR-10-dataset and provide DataLoaders with given properties.
    The training-set contains 50000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len+val_len) == 50000, f"Can not apply split into training- and validation-set, as {train_len}+{val_len}!=50000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_data = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=tr)
    training_data, validation_data = torch.utils.data.random_split(training_data, [45000, 5000])
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=60, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=60, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=tr)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=60, shuffle=True, num_workers=2)

    print(len(training_data), len(training_loader), len(validation_data), len(validation_loader), len(test_data), len(test_loader))
    return training_loader, validation_loader, test_loader
