# This file HAS to be run from cmd line, since PyCharm does not support running bash scripts.
# Tested with CygWin64 by running: py -u PANGU/controllerSingle.py
# Without the -u flag the script will not properly echo to Ubuntu type consoles

import os
import subprocess
import sys

sys.path.append('./resources')
sys.path.append('./auxiliary')
from Filehandling import readinConfig


# # Variables from Python
# NEW_FILE=$1
# TARG_DIR=$2
#
# # Camera params
# EXP1=$3
# QE1=$4
# BIAS1=$5
# GAIN1=$6
# RMS1=$7
# DC1=$8
#
# EXP1=$9
# QE1=$10
# BIAS1=$11
# GAIN1=$12
# RMS1=$13
# DC1=$14

# Use to format log file string


class TestContainer:

    def __init__(self):
        self.param_bright = None
        self.param_dim = None
        self.message = " "

    def strName(self, count):
        test_name = "\ntest" + str(count) + ";"
        chunk1 = "\t\nEXP1: " + str(self.param_bright[0]) + "; QE1: " + str(self.param_bright[1]) + "; BIAS1: " + str(self.param_bright[2])
        chunk2 = "; GAIN1: " + str(self.param_bright[3]) + "; RMS1: " + str(self.param_bright[4]) + "; DC1: " + str(self.param_bright[5]) + ";"
        chunk3 = "\t\nEXP2: " + str(self.param_dim[0]) + "; QE2: " + str(self.param_dim[1]) + "; BIAS2: " + str(self.param_dim[2])
        chunk4 = "; GAIN2: " + str(self.param_dim[3]) + "; RMS2: " + str(self.param_dim[4]) + "; DC2: " + str(self.param_dim[5]) + ";"

        return test_name + chunk1 + chunk2 + chunk3 + chunk4


############################################################
############ BELOW ARE THE SCRIPT SETTINGS #################
############################################################

# Change dir
target_dir = readinConfig()

#  Location of shell shell script
bashCommand = "./shell/compound.sh"

#  Name of test
testName = "/sineHover"


# Container for all tests
testVector = []
test_container = TestContainer

# LIST ALL TESTS HERE
# test_bri = [EXP1,     QE1,   BIAS1, GAIN1,   RMS1,    DC1]
# test_dim = [EXp2,     QE2,   BIAS2, GAIN2,   RMS2,    DC2]

test_container.message = "3 orders of difference between bright and dim"
test_container.param_bright = [" 1.000", " 0.09", " 0", " 1000000", " 0.055", " 24.966"]
test_container.param_dim = [" 0.015", " 0.09", " 0", " 1000", " 0.055", " 24.966"]
testVector.append(test_container)

# Keep track of number of performed tests
counter = 1

while os.path.isdir(target_dir + "/frames" + testName + str(counter)):
    print(target_dir + "/frames" + testName + str(counter))
    counter = counter + 1

for test_iterator in testVector:
    echo = test_iterator.strName(test_iterator, counter)
    print(echo + "\n")

    test_tag = testName + str(counter)

    # Running script, the .sh will create its own dir before opening PANGU
    try:
        process = subprocess.run(['bash', bashCommand, test_tag, target_dir, test_iterator.param_bright, test_iterator.param_dim])
        test_logfile = open(target_dir + "/frames" + test_tag + "/log.txt", "w")
        test_logfile.write(test_iterator.message + "\n\n")
        test_logfile.write(echo)
        test_logfile.close()

    # Upon failure this script will create the dir with the log indicating failure
    except:

        # Probably doesn't work either, can't be bother to fix this now
        subprocess.run(['mkdir frames/', test_tag])

        # Save log file
        test_logfile = open(target_dir + "/frames" + test_tag + "/log.txt", "w")
        test_logfile.write(echo)
        test_logfile.write("\n\nTEST FAILED")
        test_logfile.close()

    counter = counter + 1
