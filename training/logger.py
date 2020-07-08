from experiments.experiment_settings import VerbosityLevel


def print_message(message, new_line):
    """ Print the given message and append a new line, if new_line is True. """
    if new_line:
        print(message)
    else:
        print(message, end="")


def log_from_medium(verbosity_level, message, new_line=True):
    """ Print the given message, if verbosity_level is at least medium.
    A new_line is appended, if line_break is True. """
    if verbosity_level > VerbosityLevel.SILENT:
        print_message(message, new_line)


def log_detailed_only(verbosity_level, message, new_line=True):
    """ Print the given message, if verbosity_level is detailed.
    A new_line is appended, if line_break is True. """
    if verbosity_level == VerbosityLevel.DETAILED:
        print_message(message, new_line)
