import os
from argparse import ArgumentParser

import sys
import torch.cuda

from experiments.experiment_imp import ExperimentIMP
from experiments.experiment_osp import ExperimentOSP
from experiments.experiment_random_retrain import ExperimentRandomRetrain
from experiments.experiment_specs import ExperimentNames, ExperimentPresetNames, VerbosityLevel, get_specs
from training.logger import log_from_medium

current_path = os.path.dirname(__file__)
os.chdir(os.path.join(current_path, 'experiments'))


def should_override_arg_positive_int(value, debug_name):
    """ Check if the flag was set and if 'value' is a positive integer. """
    if value is not None:
        assert isinstance(value, int) and (value > 0), f"{debug_name} needs to be a positive integer, but is {value}."
        return True
    return False


def should_override_arg_rate(value, debug_name):
    """ Check if the flag was set and 'value' is a float from zero to one. """
    if value is not None:
        assert isinstance(value, float) and (0 <= value <= 1), f"0.0<={value}<=1.0 is invalid for {debug_name}."
        return True
    return False


def should_override_arg_plan(plan, debug_name):
    """ Check if the flag was set and if 'plan' is a list. """
    if plan is not None:
        assert isinstance(plan, list), f"{debug_name} has invalid type {type(plan)}."
        return True
    return False


def setup_cuda(cuda_wanted):
    """ Returns cuda-device and its name, if cuda is preferred and available.
    Return "cpu" otherwise. """
    use_cuda = cuda_wanted and torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    return device, device_name


def parse_arguments(args):
    """ Creates an ArgumentParser with help messages. """
    # top-level parser
    parser = ArgumentParser(description="bachelor_playground is a framework for pruning-experiments. "
                                        "Please choose an available experiment_name and specify options via flags. "
                                        "You can find further information in the README.",
                            epilog="Use 'python -m playground <experiment> -h' to get help for a specific experiment.")
    subparsers = parser.add_subparsers(title='Experiments',
                                       description='These experiments are available as subcommands:',
                                       dest='experiment_name')

    # parent parser for pruning experiments, defines common flags like 'net_count' and 'cuda'
    parser_pr = ArgumentParser(add_help=False)
    parser_pr.add_argument('experiment_preset', choices=ExperimentPresetNames.get_value_list(),
                           help="choose a preset; Lenet uses tanh-activations and Conv uses ReLU-activations")
    parser_pr.add_argument('-c', '--cuda', action='store_true', default=False, help="use cuda, if available")
    parser_pr.add_argument('-es', '--early_stop', action='store_true', default=False,
                           help="evaluate early-stopping criterion during training "
                                "and save checkpoints per net and level of pruning")
    parser_pr.add_argument('-l', '--listing', action='store_true', default=False,
                           help="list loaded settings, but do not run the experiment_name")
    parser_pr.add_argument('-ps', '--plot_step', type=int, default=None, metavar='N',
                           help="specify the number of iterations between history-entries")
    parser_pr.add_argument('-v', '--verbose', action='count', default=0,
                           help="activate output, use twice for more detailed output at higher frequency (i.e. -vv)")

    parser_pr.add_argument('-e', '--epochs', type=int, default=None, metavar='N', help="specify number of epochs")
    parser_pr.add_argument('-n', '--nets', type=int, default=None, metavar='N',
                           help="specify number of trained networks")
    parser_pr.add_argument('-p', '--prunes', type=int, default=None, metavar='N',
                           help="specify number of pruning steps")
    parser_pr.add_argument('-lr', '--learn_rate', type=float, default=None, metavar='R', help="specify learning-rate")
    parser_pr.add_argument('-prc', '--prune_rate_conv', type=float, default=None, metavar='R',
                           help="specify pruning-rate for convolutional layers")
    parser_pr.add_argument('-prf', '--prune_rate_fc', type=float, default=None, metavar='R',
                           help="specify pruning-rate for fully-connected layers")
    parser_pr.add_argument('--plan_conv', type=str, nargs='*', default=None, metavar='SPEC',
                           help="specify convolutional layers as list of output-sizes (as int or string); "
                                "special layers: 'A' for average-pooling, 'M' for max-pooling,"
                                "'iB' for convolution of size i with batch-norm")
    parser_pr.add_argument('--plan_fc', type=str, nargs='*', default=None, metavar='SPEC',
                           help="specify fully-connected layers as list of output-sizes (as int or string)")

    # parser for main IMP-experiments
    subparsers.add_parser(ExperimentNames.IMP.value,
                          description='Choose a preset and adapt parameters with flags.',
                          help='Iterative magnitude pruning. This is the main experiment.',
                          parents=[parser_pr])

    # parser for main OSP-experiments
    subparsers.add_parser(ExperimentNames.OSP.value,
                          description='Choose a preset and adapt parameters with flags.',
                          help='One-shot pruning.',
                          parents=[parser_pr])

    # parser for retraining-experiments
    parser_rr = subparsers.add_parser(ExperimentNames.RR.value,
                                      description="Specify the original IMP- or OSP-experiment and the number of nets"
                                                  "to train per level of sparsity. "
                                                  "Reuse parameters like 'epoch_count' and 'cuda' from original specs. "
                                                  "Load state_dicts from EarlyStopHistory to get pruned masks.",
                                      help='Training randomly reinitialized networks from a previous IMP-experiment. '
                                           'An EarlyStopHistory-file must exist.')
    parser_rr.add_argument('specs_path', type=str, default=None,
                           help="specify the relative path to the original specs-file (from main-package)")
    parser_rr.add_argument('net_number', type=int, default=None,
                           help="specify the original net_number to load its EarlyStopHistory")
    parser_rr.add_argument('net_count', type=int, default=None,
                           help="specify the number of random initializations per level of pruning")

    if len(args) < 1:  # show help, if no arguments are given
        parser.print_help(sys.stderr)
        sys.exit()
    return parser.parse_args(args)


def setup_pruning(args):
    """ Setup IMP- or OSP-experiment and print the specs or execute it. """
    assert args.verbose in VerbosityLevel.__members__.values()

    specs = get_specs(args.experiment_preset)

    specs.device, specs.device_name = setup_cuda(args.cuda)
    log_from_medium(args.verbose, specs.device_name)

    if should_override_arg_positive_int(args.epochs, 'Epoch count'):
        specs.epoch_count = args.epochs
    if should_override_arg_positive_int(args.nets, 'Net count'):
        specs.net_count = args.nets
    if should_override_arg_positive_int(args.prunes, 'Prune count'):
        specs.prune_count = args.prunes
    if should_override_arg_rate(args.learn_rate, 'Learning-rate'):
        specs.learning_rate = args.learn_rate
    if should_override_arg_rate(args.prune_rate_conv, 'Pruning-rate Conv'):
        specs.prune_rate_conv = args.prune_rate_conv
    if should_override_arg_rate(args.prune_rate_fc, 'Pruning-rate FC'):
        specs.prune_rate_fc = args.prune_rate_fc
    if should_override_arg_plan(args.plan_conv, 'Convolutional plan'):
        specs.plan_conv = args.plan_conv
    if should_override_arg_plan(args.plan_fc, 'Fully connected plan'):
        specs.plan_fc = args.plan_fc
    if should_override_arg_positive_int(args.plot_step, 'Plot-step'):
        specs.plot_step = args.plot_step
    specs.verbosity = VerbosityLevel(args.verbose)
    specs.save_early_stop = args.early_stop
    specs.experiment_name = args.experiment_name

    if args.listing:
        print(specs)
    elif args.experiment_name == ExperimentNames.IMP:
        experiment = ExperimentIMP(specs)
        experiment.run_experiment()
    elif args.experiment_name == ExperimentNames.OSP:
        experiment = ExperimentOSP(specs)
        experiment.run_experiment()


def setup_random_retrain(args):
    """ Setup and run retraining experiment for specs from a loaded OSP- or IMP-experiment. """
    # working directory of experiment is ./experiment, but path is relative to main package
    relative_specs_path = os.path.join('..', args.specs_path)

    experiment = ExperimentRandomRetrain(relative_specs_path, args.net_number, args.net_count)
    experiment.run_experiment()


def main(args):
    parsed_args = parse_arguments(args)

    if parsed_args.experiment_name in [ExperimentNames.IMP, ExperimentNames.OSP]:
        setup_pruning(parsed_args)
    elif parsed_args.experiment_name == ExperimentNames.RR:
        setup_random_retrain(parsed_args)


if __name__ == '__main__':
    main(sys.argv[1:])
