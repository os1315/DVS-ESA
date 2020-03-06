import parse
import sys
import os


def readinConfig():
    try:
        config_file = open("config.txt")
    except FileNotFoundError:
        config_file = open("../config.txt")
    config_string = config_file.read()
    config_directory = parse.search("target_dir {};", config_string)[0]
    config_file.close()

    return config_directory


def IsNotInt(s):
    try:
        int(s)
        return False
    except ValueError:
        return True


def readinFrameRate(test_name=""):
    try:
        fli_file = open("frames/" + test_name + "/test_traj.fli")
        first_str = fli_file.readline()

        frame_rate_str = parse.search("# Frame rate: {:d}\n", first_str)[0]
        fli_file.close()

        return int(frame_rate_str)

    except FileNotFoundError:
        print("FAILED TO READ IN FRAME RATE FOR LOG FILE")

        while True:
            frame_rate_str = input("Pls insert frame rate (suggested 2000): \n\n")

            try:
                frame_rate = int(frame_rate_str)
                return frame_rate
            except ValueError:
                pass


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
