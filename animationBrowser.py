# Math imports
from math import pi
from math import sqrt
from math import cos
from math import sin
from math import floor
from math import ceil

# Numpy imports
import numpy as np
from numpy import array
from numpy import matmul

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

# Other imports
import struct
import parse
import time
import importlib
import os

# This is the actually file being tested
import event_creation
import playProcessedModule

from testBench import TestBench
from Filehandling import readinConfig


TB = TestBench("constDescent2", target_dir=readinConfig())

run_module = True

while run_module:

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
        run_module = False

    elif echo == 'T' or echo == 't':
        TB.playRaw()
