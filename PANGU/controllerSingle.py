# This file HAS to be run from cmd line, since PyCharm does not support running bash scripts.
# Tested with CygWin64 by running: py -u PANGU/controllerSingle.py
# Without the -u flag the script will not properly echo to Ubuntu type consoles

import os
import subprocess
import sys
sys.path.append('./resources')
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

# Use to format log file string
def strName(vector, counter):
    test_name = "\ntest" + str(counter) + ";"
    chunk1 = "\t\nEXP1: " + str(vector[0]) + "; QE1: " + str(vector[1]) + "; BIAS1: " + str(vector[2])
    chunk2 = "; GAIN1: " + str(vector[3]) + "; RMS1: " + str(vector[4]) + "; DC1: " + str(vector[5]) + ";"

    return test_name + chunk1 + chunk2


############################################################
############ BELOW ARE THE SCRIPT SETTINGS #################
############################################################

# Change dir
target_dir = readinConfig()

#  Location of shell shell script
bashCommand = "./shell/singleNB.sh"

#  Name of test
testName = "constDescent"

# Container for all tests
testVector = []

# LIST ALL TESTS HERE
# test_bri = [EXP1,     QE1,   BIAS1, GAIN1,   RMS1,    DC1]
# test_dim = [EXp2,     QE2,   BIAS2, GAIN2,   RMS2,    DC2]

test_bri = [" 1.000", " 0.09", " 0", " 1000000", " 0.055", " 24.966"]
# test_bri = [" 0.015", " 0.09", " 0", " 10" , " 0.055", " 24.966"]
testVector.append(test_bri)

############################################################
############# BELOW IS THE ACTUAL SCRIPT ###################
############################################################

# Keep track of number of performed tests
counter = 1

while os.path.isdir(target_dir + "/frames/" + testName + str(counter)):
    print(target_dir + "/frames/" + testName + str(counter))
    counter = counter + 1

for testIterator in testVector:
    echo = strName(testIterator, counter)
    print(echo + "\n")

    testTag = testName + str(counter)

    # Running script, the .sh will create its own dir before opening PANGU
    try:
        process = subprocess.run(['bash', bashCommand, testTag, target_dir, testIterator])
        flight_file = open(target_dir + "/frames/" + testTag + "/log.txt", "w")
        flight_file.write(echo)
        flight_file.close()

    # Upon failure this script will create the dir with the log indicating failure
    except:

        #  Ok this is never gonna work, cause redirection only happens within the nash script
        os.mkdir("/frames/" + testTag)

        # Save log file
        flight_file = open(target_dir + "/frames/" + testTag + "/log.txt", "w")
        flight_file.write(echo)
        flight_file.write("\n\nTEST FAILED")
        flight_file.close()

    # Increment to next test
    counter = counter + 1
