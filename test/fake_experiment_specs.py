from experiments.experiment_specs import ExperimentSpecs, NetNames, DatasetNames, VerbosityLevel, PruningMethodNames


def get_specs_lenet_toy():
    """ Toy specs for a toy Lenet on a toy-MNIST dataset. """
    return ExperimentSpecs(
        net=NetNames.LENET,
        net_count=2,
        plan_conv=[],
        plan_fc=[3],
        dataset=DatasetNames.MNIST,
        epoch_count=2,
        learning_rate=2e-4,
        plot_step=2,
        verbosity=VerbosityLevel.SILENT,
        prune_method=PruningMethodNames.IMP_LAYER,
        prune_count=1,
        prune_rate_conv=0.0,
        prune_rate_fc=0.2
    )


def get_specs_conv_toy():
    """ Toy specs for a toy Conv on a toy-CIFAR-10 dataset. """
    experiment_specs = get_specs_lenet_toy()
    experiment_specs.net = NetNames.CONV
    experiment_specs.plan_conv = [4, 'M', 'A']
    experiment_specs.plan_fc = [3]
    experiment_specs.dataset = DatasetNames.CIFAR10
    experiment_specs.prune_rate_conv = 0.2
    return experiment_specs
