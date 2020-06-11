import numpy as np
import parse
import sys
import os


def isNotInt(s):
    try:
        int(s)
        return False
    except ValueError:
        return True


def readinConfig():
    try:
        config_file = open("config.txt")
    except FileNotFoundError:
        config_file = open("../config.txt")
    config_string = config_file.read()
    config_directory = parse.search("target_dir {};", config_string)[0]
    config_file.close()

    return config_directory


def readinFrameRate(test_name=""):
    try:
        fli_file = open("test_traj.fli")
        first_str = fli_file.readline()

        frame_rate_str = parse.search("# Frame rate: {:d}\n", first_str)[0]
        fli_file.close()

        return int(frame_rate_str)

    except FileNotFoundError:
        print("FAILED TO READ IN FRAME RATE FROM LOG FILE")

        while True:
            frame_rate_str = input("Pls insert frame rate (suggested 2000): \n\n")

            try:
                frame_rate = int(frame_rate_str)
                return frame_rate
            except ValueError:
                pass


def readinFlightTrajectory(test_name=""):
    fli_file = open("frames/" + test_name + "/test_traj.fli")

    # Read lines until you reach trajectory
    while True:
        junk_line = fli_file.readline()
        if junk_line == "view craft\n":
            junk_line = fli_file.readline()
            break

    # Read in trajectory
    trajectory = []
    idx = 0

    while True:
        str_position = fli_file.readline()
        packet = str_position.split()[1:7]
        idx += 1

        if len(packet) > 0:
            trajectory.append(packet)
        else:
            break

    fli_file.close()

    return np.array(trajectory).astype(float)


class ProgressTracker:
    """
    Use to print out progress message to cmd prompt
    """

    def __init__(self, total_items: int):
        self.percentage = 1
        self.total_items = total_items

    def update(self, state: int) -> None:
        if state > self.percentage * self.total_items / 100:
            sys.stdout.write('\rProgress: {:d}%'.format(self.percentage))
            sys.stdout.flush()
            self.percentage = self.percentage + 1

    def complete(self, message: str = None) -> None:
        sys.stdout.write('\rProgress: {:d}% ---> '.format(100))
        sys.stdout.flush()
        if message is not None:
            sys.stdout.write('{0}\n'.format(message))
        self.percentage = 1
