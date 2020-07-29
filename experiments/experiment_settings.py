from dataclasses import dataclass
from enum import Enum, IntEnum


class VerbosityLevel(IntEnum):
    """ Enum to define the level of verbosity.
    'SILENT' gives no output at all.
    'MEDIUM' gives status messages.
    'DETAILED' gives more status and progress messages. """
    SILENT = 0
    MEDIUM = 1
    DETAILED = 2


class DatasetNames(str, Enum):
    """ Enum to define available datasets. """
    MNIST = "MNIST"
    CIFAR10 = "CIFAR-10"
    TOY_MNIST = "Toy-MNIST"
    TOY_CIFAR10 = "Toy-CIFAR"


class NetNames(str, Enum):
    """ Enum to define available network architectures. """
    LENET = "Lenet"
    CONV = "Conv"


class PruningMethodNames(str, Enum):
    """ Enum to define available pruning methods. """
    IMP_LAYER = "IMP-per-layer"
    # IMP_GLOBAL = "IMP-global"


class ExperimentNames(str, Enum):
    """ Enum to define available experiments. """
    LENET_MNIST = "lenet-mnist"
    CONV2_CIFAR10 = "conv2-cifar10"
    CONV4_CIFAR10 = "conv4-cifar10"
    CONV6_CIFAR10 = "conv6-cifar10"

    @staticmethod
    def get_value_list():
        # noinspection PyUnresolvedReferences
        return [name.value for name in ExperimentNames]


@dataclass
class ExperimentSettings:
    net: NetNames
    plan_conv: list
    plan_fc: list
    net_count: int

    dataset: DatasetNames

    epoch_count: int
    learning_rate: float
    plot_step: int
    verbosity: VerbosityLevel

    prune_method: PruningMethodNames
    prune_count: int
    prune_rate_conv: float
    prune_rate_fc: float

    duration: str = "not finished"
    device: str = "cpu"
    device_name: str = "cpu"

    save_early_stop = False


def get_settings(experiment):
    """ Load ExperimentSettings for experiment specified by ExperimentName. """
    if experiment == ExperimentNames.LENET_MNIST:
        return get_settings_lenet_mnist()
    elif experiment == ExperimentNames.CONV2_CIFAR10:
        return get_settings_conv2_cifar10()
    elif experiment == ExperimentNames.CONV4_CIFAR10:
        return get_settings_conv4_cifar10()
    elif experiment == ExperimentNames.CONV6_CIFAR10:
        return get_settings_conv6_cifar10()
    else:
        raise AssertionError(f"{experiment} is an invalid experiment name.")


def get_settings_lenet_mnist():
    """ Original experiment with Lenet 300-100 on MNIST. """
    return ExperimentSettings(
        net=NetNames.LENET,
        net_count=3,
        plan_conv=[],
        plan_fc=[300, 100],
        dataset=DatasetNames.MNIST,
        epoch_count=60,  # 55000 iterations / 916 batches ~ 60 epochs
        learning_rate=1.2e-3,  # page 3, figure 2
        plot_step=100,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=3,
        prune_rate_fc=0.2,  # page 3, figure 2
        prune_rate_conv=0.0
    )


def get_settings_conv2_cifar10():
    """ Original experiment with Conv-6 on CIFAR-10. """
    return ExperimentSettings(
        net=NetNames.CONV,
        net_count=3,
        plan_conv=[64, 64, 'M'],
        plan_fc=[256, 256],
        dataset=DatasetNames.CIFAR10,
        epoch_count=27,  # 20000 iterations / 750 batches ~ 26.6 epochs
        learning_rate=2e-4,  # page 3, figure 2
        plot_step=100,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=3,
        prune_rate_conv=0.1,  # page 3, figure 2
        prune_rate_fc=0.2  # page 3, figure 2
    )


def get_settings_conv4_cifar10():
    """ Original experiment with Conv-4 on CIFAR-10. """
    experiment_settings = get_settings_conv2_cifar10()
    experiment_settings.plan_conv = [64, 64, 'M', 128, 128, 'M']
    experiment_settings.plan_fc = [256, 256]
    experiment_settings.epoch_count = 34  # 25000 iterations / 750 batches ~ 33.3 epochs
    experiment_settings.learning_rate = 3e-4  # page 3, figure 2
    experiment_settings.prune_rate_conv = 0.1  # page 3, figure 2
    experiment_settings.prune_rate_fc = 0.2  # page 3, figure 2
    return experiment_settings


def get_settings_conv6_cifar10():
    """ Original experiment with Conv-6 on CIFAR-10. """
    experiment_settings = get_settings_conv2_cifar10()
    experiment_settings.net_count = 3
    experiment_settings.plan_conv = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
    experiment_settings.plan_fc = [256, 256]
    experiment_settings.epoch_count = 40  # 30000 iterations / 750 batches = 40 epochs
    experiment_settings.learning_rate = 3e-4  # page 3, figure 2
    experiment_settings.prune_rate_conv = 0.15  # page 3, figure 2
    experiment_settings.prune_rate_fc = 0.2  # page 3, figure 2
    return experiment_settings


def get_settings_lenet_toy():
    """ Toy settings for a toy Lenet on a toy-MNIST dataset. """
    return ExperimentSettings(
        net=NetNames.LENET,
        net_count=2,
        plan_conv=[],
        plan_fc=[3],
        dataset=DatasetNames.TOY_MNIST,
        epoch_count=2,
        learning_rate=2e-4,
        plot_step=2,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=1,
        prune_rate_conv=0.0,
        prune_rate_fc=0.2
    )


def get_settings_conv_toy():
    """ Toy settings for a toy Conv on a toy-CIFAR-10 dataset. """
    return ExperimentSettings(
        net=NetNames.CONV,
        net_count=2,
        plan_conv=[4, 'M', 'A'],
        plan_fc=[3],
        dataset=DatasetNames.TOY_CIFAR10,
        epoch_count=2,
        learning_rate=2e-4,
        plot_step=2,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=1,
        prune_rate_conv=0.0,
        prune_rate_fc=0.2
    )
