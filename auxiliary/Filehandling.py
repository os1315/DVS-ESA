import parse
import sys


def readinConfig():
    try:
        config_file = open("config.txt")
    except FileNotFoundError:
        config_file = open("../config.txt")
    config_string = config_file.read()
    config_directory = parse.search("target_dir {};", config_string)[0]
    config_file.close()

    return config_directory


class ProgressTracker:

    def __init__(self, total_items):
        self.percentage = 1
        self.total_items = total_items

    def update(self, state):
        if state > self.percentage * self.total_items / 100:
            sys.stdout.write('\rProgress: {:d}%'.format(self.percentage))
            sys.stdout.flush()
            self.percentage = self.percentage + 1

    def complete(self, message=None):
        sys.stdout.write('\rProgress: {:d}% ---> '.format(100))
        sys.stdout.flush()
        if message is not None:
            sys.stdout.write('{0}\n'.format(message))
        self.percentage = 1

