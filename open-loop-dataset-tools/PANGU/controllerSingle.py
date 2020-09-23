# This file HAS to be run from cmd line, since PyCharm does not support running bash scripts.
# Tested with CygWin64 by running: py -u PANGU/controllerSingle.py
# Without the -u flag the script will not properly echo to Ubuntu type consoles

import os
import subprocess
import sys

sys.path.append('./resources')
sys.path.append('./auxiliary')
from Filehandling import readinConfig, readinFrameRate


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

# Use to format log file string
class TestContainer:

    def __init__(self):
        self.parameters = None
        self.message = " "

    def strName(self, count):
        test_name = "\ntest" + str(count) + ";"
        chunk1 = "\t\nEXP1: " + str(self.parameters[0]) + "; QE1: " + str(self.parameters[1]) + "; BIAS1: " + str(self.parameters[2])
        chunk2 = "; GAIN1: " + str(self.parameters[3]) + "; RMS1: " + str(self.parameters[4]) + "; DC1: " + str(self.parameters[5]) + ";"

        return test_name + chunk1 + chunk2


############################################################
############ BELOW ARE THE SCRIPT SETTINGS #################
############################################################

# Change dir
target_dir = readinConfig()
# frame_rate = readinFrameRate(target_dir)

#  Location of shell shell script
bashCommand = "./shell/singleNB.sh"

#  Name of test
testName = "/constDescent"

# Container for all tests
testVector = []
test_container = TestContainer

# LIST ALL TESTS HERE
# test = [EXP1,     QE1,   BIAS1, GAIN1,   RMS1,    DC1]

test_container.message = "*no message*"
test_container.parameters = [" 1.000", " 0.09", " 0", " 100000", " 0.055", " 24.966"]
testVector.append(test_container)

############################################################
############# BELOW IS THE ACTUAL SCRIPT ###################
############################################################

# Keep track of number of performed tests
counter = 1

while os.path.isdir(target_dir + "/frames" + testName + str(counter)):
    print(target_dir + "/frames/" + testName + str(counter))
    counter = counter + 1

for test_iterator in testVector:
    echo = test_iterator.strName(test_iterator, counter)
    print(echo + "\n")

    test_tag = testName + str(counter)

    # Running script, the .sh will create its own dir before opening PANGU
    try:
        process = subprocess.run(['bash', bashCommand, test_tag, target_dir, test_iterator.parameters])
        test_logfile = open(target_dir + "/frames/" + test_tag + "/log.txt", "w")
        test_logfile.write(test_iterator.message + "\n\n")  # Write in comment
        # test_logfile.write('# Frame rate: {}\n\r'.format(frame_rate))   # Wrint in frame rate!!
        test_logfile.write(echo)    # Echo contains all modified parameters
        test_logfile.close()

    # Upon failure this script will create the dir with the log indicating failure
    except:

        #  Ok this is never gonna work, cause redirection only happens within the nash script
        os.mkdir("/frames/" + test_tag)

        # Save log file
        test_logfile = open(target_dir + "/frames/" + test_tag + "/log.txt", "w")
        test_logfile.write(echo)
        test_logfile.write("\n\nTEST FAILED")
        test_logfile.close()

    # Increment to next test
    counter = counter + 1
