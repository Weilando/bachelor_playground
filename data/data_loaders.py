import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from experiments.experiment_specs import DatasetNames, VerbosityLevel
from training.logger import log_from_medium


def get_sample_shape(dataset_name: DatasetNames):
    """ Return a tuple with the shape (channels, height, width) of a single sample in a batch from 'dataset_name'. """
    if dataset_name == DatasetNames.MNIST:
        return 1, 28, 28
    elif dataset_name == DatasetNames.CIFAR10:
        return 3, 32, 32
    raise AssertionError(f"The given dataset_name {dataset_name} is invalid.")


def generate_data_loaders(train_data, val_data, test_data, batch_size, device, verb: VerbosityLevel):
    """ Generate DataLoaders from given data sets. """
    train_ld = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device != 'cpu'),
                          drop_last=True)
    val_ld = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device != 'cpu'))
    test_ld = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device != 'cpu'))

    log_from_medium(verb, (len(train_data), len(train_ld), len(val_data), len(val_ld), len(test_data), len(test_ld)))
    return train_ld, val_ld, test_ld


def get_mnist_data_loaders(batch_size=60, train_len=55000, val_len=5000, path='../data/datasets', device='cpu',
                           verb=VerbosityLevel.SILENT):
    """ Load the MNIST-dataset (60,000 training samples, 10,000 test samples) and provide DataLoaders.
    The training-set is split into a training- and validation-set. """
    assert (train_len + val_len) == 60000, f"Invalid split into train- and val-set, as {train_len}+{val_len}!=60000."
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=tr)
    train_data, val_data = random_split(train_data, [train_len, val_len])
    test_data = datasets.MNIST(root=path, train=False, download=True, transform=tr)

    return generate_data_loaders(train_data, val_data, test_data, batch_size, device, verb)


def get_cifar10_data_loaders(batch_size=60, train_len=45000, val_len=5000, path='../data/datasets', device='cpu',
                             verb=VerbosityLevel.SILENT):
    """ Load the CIFAR-10-dataset (50,000 training samples, 10,000 test samples) and provide DataLoaders.
    The training-set is split into a training- and validation-set. """
    assert (train_len + val_len) == 50000, f"Invalid split into train- and val-set, as {train_len}+{val_len}!=50000."
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=tr)
    train_data, val_data = random_split(train_data, [train_len, val_len])
    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=tr)

    return generate_data_loaders(train_data, val_data, test_data, batch_size, device, verb)
