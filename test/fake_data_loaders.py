import torch


def generate_fake_mnist_data_loaders():
    """" Generate fake-DataLoaders with fake batches, i.e. a list with sub-lists of samples and labels.
    Each batch holds three pairs of samples and labels. """
    torch.manual_seed(123)
    samples1 = torch.rand((3, 28, 28))
    samples2 = torch.rand((3, 28, 28))
    labels1 = torch.tensor([0, 0, 1])
    labels2 = torch.tensor([1, 1, 0])
    train = [[samples1, labels1], [samples1, labels2], [samples2, labels1], [samples2, labels2]]
    val = [[samples2, labels1]]
    test = [[samples1, labels2], [samples2, labels1]]
    return train, val, test


def generate_fake_cifar10_data_loaders():
    """" Generate fake-DataLoaders with fake batches, i.e. a list with sub-lists of samples and labels.
    Each batch holds three pairs of samples and labels. """
    torch.manual_seed(123)
    samples1 = torch.rand((3, 3, 32, 32))
    samples2 = torch.rand((3, 3, 32, 32))
    labels1 = torch.tensor([0, 0, 1])
    labels2 = torch.tensor([1, 1, 0])
    train = [[samples1, labels1], [samples1, labels2], [samples2, labels1], [samples2, labels2]]
    val = [[samples2, labels1]]
    test = [[samples1, labels2], [samples2, labels1]]
    return train, val, test
