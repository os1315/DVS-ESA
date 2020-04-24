# Math imports

# Numpy imports

# Matplotlib imports

# Other imports
import os

# This is the actually file being tested

from testBench import TestBench
from auxiliary.Filehandling import readinConfig


TB = TestBench("constDescent3", target_dir=readinConfig())

while True:

    print("\nNext command?\n\tr - reload images \n\ti - process images \n\tv - play processed images \n\tp - process "
          "& play \n\tt - play raw\n\tn - exit")

    echo = input("Input:   ")

    if echo == 'R' or echo == 'r':
        echo = input("Image name: ")
        if os.path.exists(readinConfig() + "/" + echo):
            del TB
            TB = TestBench(echo)
        else:
            print("\nNo such test exists!")
            print("Images have NOT been deleted.\n")

    elif echo == 'I' or echo == 'i':
        cap = input('Cap frames?    ')
        if cap == 'n' or cap == 'N':
            pass
        elif cap == '':
            TB.processImages()
        else:
            TB.processImages(int(cap))

    elif echo == 'V' or echo == 'v':
        TB.playProcessed()

    elif echo == 'P' or echo == 'p':
        cap = input('Cap frames?    ')
        if cap == 'n' or cap == 'N':
            pass
        elif cap == '':
            TB.playImport()
        else:
            TB.playImport(int(cap))

    elif echo == 'N' or echo == 'n':
        break

    elif echo == 'T' or echo == 't':
        TB.playRaw()
