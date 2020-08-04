from unittest import TestCase
from unittest import main as unittest_main

from data.data_loaders import get_toy_data_loaders
from experiments.experiment_specs import DatasetNames


class TestDataLoaders(TestCase):
    """ Tests for the data_loader module.
    Call with 'python -m test.test_data_loader' from project root '~'.
    Call with 'python -m test_data_loader' from inside '~/test'. """

    def test_get_toy_mnist_data_loaders(self):
        """ Should load toy MNIST dataset with expected batch counts and sample shapes. """
        train_loader, val_loader, test_loader = get_toy_data_loaders(DatasetNames.TOY_MNIST)

        sample, _ = next(iter(val_loader))  # get one sample
        self.assertEqual(sample[0].shape, (1, 28, 28))

        self.assertEqual(len(train_loader), 3)
        self.assertEqual(len(val_loader), 2)
        self.assertEqual(len(test_loader), 2)

    def test_get_toy_cifar10_data_loaders(self):
        """ Should load toy CIFAR-10 dataset with expected batch counts. """
        train_loader, val_loader, test_loader = get_toy_data_loaders(DatasetNames.TOY_CIFAR10)

        sample, _ = next(iter(val_loader))  # get one sample
        self.assertEqual(sample[0].shape, (3, 32, 32))

        self.assertEqual(len(train_loader), 3)
        self.assertEqual(len(val_loader), 2)
        self.assertEqual(len(test_loader), 2)

    def test_raise_error_on_invalid_toy_name(self):
        """ Should raise assertion error if the specified name is invalid. """
        with self.assertRaises(AssertionError):
            get_toy_data_loaders(DatasetNames.MNIST)


if __name__ == '__main__':
    unittest_main()
