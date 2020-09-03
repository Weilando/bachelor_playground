from experiments.experiment_specs import VerbosityLevel


def print_message(message, new_line):
    """ Print 'message' and append a new line, if 'new_line' is True. """
    print(message, end="\n" if new_line else "")


def log_from_medium(verbosity_level, message, new_line=True):
    """ Print 'message', if verbosity_level is at least medium, and append a new line, if 'new_line' is True. """
    if verbosity_level > VerbosityLevel.SILENT:
        print_message(message, new_line)


def log_detailed_only(verbosity_level, message, new_line=True):
    """ Print 'message', if verbosity_level is detailed, and append a new line, if 'new_line' is True. """
    if verbosity_level == VerbosityLevel.DETAILED:
        print_message(message, new_line)
