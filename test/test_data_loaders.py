from unittest import TestCase
from unittest import main as unittest_main

from data.data_loaders import get_mnist_data_loaders, get_cifar10_data_loaders, get_sample_shape
from experiments.experiment_specs import DatasetNames


class TestDataLoaders(TestCase):
    """ Tests for the data_loader module.
    Call with 'python -m test.test_data_loader' from project root '~'.
    Call with 'python -m test_data_loader' from inside '~/test'. """

    def test_get_sample_shape_mnist(self):
        """ Should return the correct sample_shape for a single sample from MNIST. """
        self.assertEqual((1, 28, 28), get_sample_shape(DatasetNames.MNIST))

    def test_get_sample_shape_cifar10(self):
        """ Should return the correct sample_shape for a single sample from CIFAR-10. """
        self.assertEqual((3, 32, 32), get_sample_shape(DatasetNames.CIFAR10))

    def test_get_sample_shape_invalid_name(self):
        """ Should return the correct sample_shape for a single sample from CIFAR-10. """
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            get_sample_shape('Invalid name')

    def test_get_mnist_data_loaders(self):
        """ Should load toy MNIST dataset with expected batch counts and sample shapes. """
        train_loader, val_loader, test_loader = get_mnist_data_loaders()

        sample, _ = next(iter(val_loader))  # get one sample
        self.assertEqual(sample[0].shape, (1, 28, 28))

        self.assertEqual(len(train_loader), 916)
        self.assertEqual(len(val_loader), 84)
        self.assertEqual(len(test_loader), 167)

    def test_get_cifar10_data_loaders(self):
        """ Should load toy CIFAR-10 dataset with expected batch counts. """
        train_loader, val_loader, test_loader = get_cifar10_data_loaders()

        sample, _ = next(iter(val_loader))  # get one sample
        self.assertEqual(sample[0].shape, (3, 32, 32))

        self.assertEqual(len(train_loader), 750)
        self.assertEqual(len(val_loader), 84)
        self.assertEqual(len(test_loader), 167)


if __name__ == '__main__':
    unittest_main()
