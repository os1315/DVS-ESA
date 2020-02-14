import subprocess
import os

# Use to format log file string
def strName(vector, counter):
    test_name = "\ntest" + str(counter) + ";"
    chunk1 = "\t\nEXP1: " + str(vector[0]) +"; QE1: " + str(vector[1]) + "; BIAS1: " + str(vector[2])
    chunk2 =  "; GAIN1: " + str(vector[3]) + "; RMS1: " + str(vector[4]) + "; DC1: " + str(vector[5]) + ";"

    return test_name + chunk1 + chunk2

#  Location of shell shell script
bashCommand = "./shell/singleNB.sh"

#  Name of test
testName = "linearFlyby"


# Container for all tests
testVector = []

# List all tests here
# test_bri = [EXP1,     QE1,   BIAS1, GAIN1,   RMS1,    DC1]
# test_dim = [EXp2,     QE2,   BIAS2, GAIN2,   RMS2,    DC2]

test_bri = [" 1.000", " 0.09", " 0", " 1000000" , " 0.055", " 24.966"]
# test_bri = [" 0.015", " 0.09", " 0", " 10" , " 0.055", " 24.966"]
testVector.append(test_bri)

# Keep track of number of performed tests
counter = 1
while(os.path.isdir("frames/" + testName + str(counter))):
    counter = counter + 1
    print("frames/" + testName + str(counter))

for x in testVector:
    echo = strName(x,counter)
    print(echo)

    test = testName + str(counter)

    # Running script, the .sh will create its own dir before opening PANGU
    try:
        process = subprocess.run(['bash', bashCommand, test, x])
        flight_file = open("frames/" + test + "/log.txt","w")
        flight_file.write(echo)
        flight_file.close()

    # Upon failure this script will create the dir with the log indicating failure
    except:
        subprocess.run(['mkdir frames/', test])

        # Save log file
        flight_file = open("frames/" + test + "/log.txt","w")
        flight_file.write(echo)
        flight_file.write("\n\nTEST FAILED")
        flight_file.close()

    # Increment to next test
    counter = counter + 1