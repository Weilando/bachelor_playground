import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

from experiments.experiment_settings import VerbosityLevel, DatasetNames
from training.logger import log_from_medium


def generate_data_loaders(train_data, val_data, test_data, batch_size, device, verbosity: VerbosityLevel):
    """ Generate data loaders from given data sets. """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=(device != 'cpu'), drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device != 'cpu'))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2,
                             pin_memory=(device != 'cpu'))

    log_from_medium(verbosity, (len(train_data), len(train_loader), len(val_data), len(val_loader), len(test_data),
                                len(test_loader)))

    return train_loader, val_loader, test_loader


def get_mnist_data_loaders(batch_size=60, train_len=55000, val_len=5000, path='../data/datasets', device='cpu',
                           verbosity=VerbosityLevel.SILENT):
    """ Load the MNIST-dataset and provide DataLoaders with given properties.
    The training-set contains 60000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len + val_len) == 60000, f"Invalid split into train- and val-set, as {train_len}+{val_len}!=60000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=tr)
    train_data, val_data = random_split(train_data, [train_len, val_len])

    test_data = datasets.MNIST(root=path, train=False, download=True, transform=tr)

    return generate_data_loaders(train_data, val_data, test_data, batch_size, device, verbosity)


def get_cifar10_data_loaders(batch_size=60, train_len=45000, val_len=5000, path='../data/datasets', device='cpu',
                             verbosity=VerbosityLevel.SILENT):
    """ Load the CIFAR-10-dataset and provide DataLoaders with given properties.
    The training-set contains 50000 samples which can be split into two for training and validation.
    The test-set contains 10000 samples. """
    assert (train_len + val_len) == 50000, f"Invalid split into train- and val-set, as {train_len}+{val_len}!=50000."

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=tr)
    train_data, val_data = random_split(train_data, [train_len, val_len])

    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=tr)

    return generate_data_loaders(train_data, val_data, test_data, batch_size, device, verbosity)


def get_toy_data_loaders(dataset_name, path='../data/datasets'):
    """ Load a part of the MNIST-dataset and provide DataLoaders for use in fast tests.
    The training-set contains 6, the validation-set 4 and the test set 4 samples.
    The batch size is 2. """
    # load from subset from MNIST test-set, as it is smaller than the training set
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if dataset_name == DatasetNames.TOY_MNIST:
        data = datasets.MNIST(root=path, train=False, download=True, transform=tr)
    elif dataset_name == DatasetNames.TOY_CIFAR10:
        data = datasets.CIFAR10(root=path, train=False, download=True, transform=tr)
    else:
        raise AssertionError(f"Could not load toy-datasets, because the given name {dataset_name} is invalid.")

    data = Subset(data, range(0, 14))
    train_data, val_data, test_data = random_split(data, [6, 4, 4])

    train_loader = DataLoader(train_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
