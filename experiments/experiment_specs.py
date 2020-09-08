from dataclasses import dataclass
from enum import Enum, IntEnum


class VerbosityLevel(IntEnum):
    """ Define the level of verbosity. """
    SILENT = 0  # no output at all
    MEDIUM = 1  # status messages
    DETAILED = 2  # more status and progress messages


class DatasetNames(str, Enum):
    """ Define available datasets. """
    MNIST = "MNIST"
    CIFAR10 = "CIFAR-10"


class NetNames(str, Enum):
    """ Define available network architectures. """
    LENET = "Lenet"
    CONV = "Conv"


class PruningMethodNames(str, Enum):
    """ Define available pruning methods. """
    IMP_LAYER = "IMP-per-layer"
    # IMP_GLOBAL = "IMP-global"


class ExperimentNames(str, Enum):
    """ Define available experiments. """
    IMP = "imp"  # iterative magnitude pruning
    OSP = "osp"  # one-shot pruning
    RR = "rr"  # random retraining

    @staticmethod
    def get_value_list():
        # noinspection PyUnresolvedReferences
        return [name.value for name in ExperimentNames]


class ExperimentPresetNames(str, Enum):
    """ Define available presets for pruning-experiments. """
    LENET_MNIST = "lenet-mnist"
    CONV2_CIFAR10 = "conv2-cifar10"
    CONV4_CIFAR10 = "conv4-cifar10"
    CONV6_CIFAR10 = "conv6-cifar10"
    TINY_CIFAR10 = "tiny-cifar10"

    @staticmethod
    def get_value_list():
        # noinspection PyUnresolvedReferences
        return [name.value for name in ExperimentPresetNames]


@dataclass
class ExperimentSpecs:
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

    save_early_stop: bool = False
    experiment_name: ExperimentNames = ExperimentNames.IMP


def get_specs(experiment_name):
    """ Load ExperimentSpecs for experiment_name (options specified by ExperimentPresetNames). """
    if experiment_name == ExperimentPresetNames.LENET_MNIST:
        return get_specs_lenet_mnist()
    elif experiment_name == ExperimentPresetNames.CONV2_CIFAR10:
        return get_specs_conv2_cifar10()
    elif experiment_name == ExperimentPresetNames.CONV4_CIFAR10:
        return get_specs_conv4_cifar10()
    elif experiment_name == ExperimentPresetNames.CONV6_CIFAR10:
        return get_specs_conv6_cifar10()
    elif experiment_name == ExperimentPresetNames.TINY_CIFAR10:
        return get_specs_tiny_cnn_cifar10()
    else:
        raise AssertionError(f"{experiment_name} is an invalid 'experiment_name'.")


def get_specs_lenet_mnist():
    """ Original experiment with Lenet 300-100 on MNIST. """
    return ExperimentSpecs(
        net=NetNames.LENET,
        net_count=3,
        plan_conv=[],
        plan_fc=[300, 100],
        dataset=DatasetNames.MNIST,
        epoch_count=55,  # 50000 iterations / 916 batches ~ 55 epochs
        learning_rate=1.2e-3,  # page 3, figure 2
        plot_step=100,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=3,
        prune_rate_fc=0.2,  # page 3, figure 2
        prune_rate_conv=0.0
    )


def get_specs_conv2_cifar10():
    """ Original experiment with Conv-2 on CIFAR-10. """
    return ExperimentSpecs(
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


def get_specs_conv4_cifar10():
    """ Original experiment_name with Conv-4 on CIFAR-10. """
    experiment_specs = get_specs_conv2_cifar10()
    experiment_specs.plan_conv = [64, 64, 'M', 128, 128, 'M']
    experiment_specs.plan_fc = [256, 256]
    experiment_specs.epoch_count = 34  # 25000 iterations / 750 batches ~ 33.3 epochs
    experiment_specs.learning_rate = 3e-4  # page 3, figure 2
    experiment_specs.prune_rate_conv = 0.1  # page 3, figure 2
    experiment_specs.prune_rate_fc = 0.2  # page 3, figure 2
    return experiment_specs


def get_specs_conv6_cifar10():
    """ Original experiment with Conv-6 on CIFAR-10. """
    experiment_specs = get_specs_conv2_cifar10()
    experiment_specs.net_count = 3
    experiment_specs.plan_conv = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
    experiment_specs.plan_fc = [256, 256]
    experiment_specs.epoch_count = 40  # 30000 iterations / 750 batches = 40 epochs
    experiment_specs.learning_rate = 3e-4  # page 3, figure 2
    experiment_specs.prune_rate_conv = 0.15  # page 3, figure 2
    experiment_specs.prune_rate_fc = 0.2  # page 3, figure 2
    return experiment_specs


def get_specs_tiny_cnn_cifar10():
    """ Experiment with Tiny CNN on CIFAR-10. """
    experiment_specs = get_specs_conv2_cifar10()
    experiment_specs.net_count = 1
    experiment_specs.plan_conv = [8, 8, 'M', 8, 'M']
    experiment_specs.plan_fc = [300, 100]
    experiment_specs.learning_rate = 8e-4
    experiment_specs.prune_rate_conv = 0.1
    experiment_specs.prune_rate_fc = 0.2
    experiment_specs.plot_step = 150
    return experiment_specs
