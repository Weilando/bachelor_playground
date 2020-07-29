import os
from argparse import ArgumentParser

import torch.cuda

from experiments.experiment_imp import ExperimentIMP
from experiments.experiment_settings import get_settings, ExperimentNames, VerbosityLevel
from training.logger import log_from_medium

current_path = os.path.dirname(__file__)
os.chdir(os.path.join(current_path, 'experiments'))


def should_override_arg_positive_int(value, debug_name):
    """ Check if the flag was set (i.e. 'value'!=None) and if 'value' is valid (i.e. positive integer).
    The error message contains 'debug_name', if 'value' is invalid. """
    if value is not None:
        assert isinstance(value, int) and (value > 0), f"{debug_name} needs to be a positive integer, but is {value}."
        return True
    return False


def should_override_arg_rate(value, debug_name):
    """ Check if the flag was set (i.e. 'value'!=None) and if 'value' is valid (i.e. float from zero to one).
    The error message contains 'debug_name', if 'value' is invalid. """
    if value is not None:
        assert isinstance(value, float) and (value >= 0) and (value <= 1), \
            f"{debug_name} needs to be a float between zero and one, but is {value}."
        return True
    return False


def should_override_arg_plan(plan, debug_name):
    """ Check if the flag was set (i.e. 'plan'!=None) and if the given plan is valid (i.e. list). """
    if plan is not None:
        assert isinstance(plan, list), f"{debug_name} needs to be list, but is of type {type(plan)}."
        return True
    return False


def setup_cuda(cuda_wanted):
    """ Returns cuda-device and its name, if cuda is preferred and available.
    Return "cpu" otherwise. """
    use_cuda = cuda_wanted and torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    return device, device_name


def main(experiment, epochs, nets, prunes, learn_rate, prune_rate_conv, prune_rate_fc, plan_conv, plan_fc, cuda,
         verbose, listing, early_stop):
    assert verbose in VerbosityLevel.__members__.values()
    log_from_medium(verbose, "Welcome to bachelor_playground.")

    settings = get_settings(experiment)

    settings.device, settings.device_name = setup_cuda(cuda)
    log_from_medium(verbose, settings.device_name)

    if should_override_arg_positive_int(epochs, 'Epoch count'):
        settings.epoch_count = epochs
    if should_override_arg_positive_int(nets, 'Net count'):
        settings.net_count = nets
    if should_override_arg_positive_int(prunes, 'Prune count'):
        settings.prune_count = prunes
    if should_override_arg_rate(learn_rate, 'Learning-rate'):
        settings.learning_rate = learn_rate
    if should_override_arg_rate(prune_rate_conv, 'Pruning-rate for convolutional layers'):
        settings.prune_rate_conv = prune_rate_conv
    if should_override_arg_rate(prune_rate_fc, 'Pruning-rate for fully-connected layers'):
        settings.prune_rate_fc = prune_rate_fc
    if should_override_arg_plan(plan_conv, 'Convolutional plan'):
        settings.plan_conv = plan_conv
    if should_override_arg_plan(plan_fc, 'Fully connected plan'):
        settings.plan_fc = plan_fc
    settings.verbosity = VerbosityLevel(verbose)
    settings.save_early_stop = early_stop

    if listing:
        print(settings)
    else:
        experiment = ExperimentIMP(settings)
        experiment.run_experiment()


if __name__ == '__main__':
    p = ArgumentParser(description="bachelor_playground is a framework for pruning-experiments."
                                   "Please choose an available experiment and specify options via flags."
                                   "You can find further information in the README.")

    p.add_argument('experiment', choices=ExperimentNames.get_value_list(),
                   help="choose experiment")
    p.add_argument('-c', '--cuda', action='store_true', default=False, help="use cuda, if available")
    p.add_argument('-v', '--verbose', action='count', default=0,
                   help="activate output, use twice for more detailed output at higher frequency (i.e. -vv)")
    p.add_argument('-l', '--listing', action='store_true', default=False,
                   help="list loaded settings, but do not run the experiment")
    p.add_argument('-e', '--epochs', type=int, default=None, metavar='E', help="specify number of epochs")
    p.add_argument('-n', '--nets', type=int, default=None, metavar='N', help="specify number of trained networks")
    p.add_argument('-p', '--prunes', type=int, default=None, metavar='P', help="specify number of pruning steps")
    p.add_argument('-lr', '--learn_rate', type=float, default=None, metavar='R', help="specify learning-rate")
    p.add_argument('-prc', '--prune_rate_conv', type=float, default=None, metavar='R',
                   help="specify pruning-rate for convolutional layers")
    p.add_argument('-prf', '--prune_rate_fc', type=float, default=None, metavar='R',
                   help="specify pruning-rate for fully-connected layers")
    p.add_argument('-es', '--early_stop', action='store_true', default=False,
                   help="evaluate early-stopping criterion during training "
                        "and save checkpoints per net and level of pruning")
    p.add_argument('--plan_conv', type=str, nargs='+', default=None, metavar='SPEC',
                   help="specify convolutional layers as list of output-sizes (as int or string); "
                        "special layers: 'A' for average-pooling, 'M' for max-pooling, 'iB' with int i for batch-norm")
    p.add_argument('--plan_fc', type=str, nargs='+', default=None, metavar='SPEC',
                   help="specify fully-connected layers as list of output-sizes (as int or string)")

    args = p.parse_args()
    main(**vars(args))
