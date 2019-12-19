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

# Pillow imports
from PIL import Image

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

# Other imports
import struct
import time
import importlib
import os

# This is the actualy file being tested
# from event_creation_TB import runMe
import event_creation
import playProcessedModule


class TestBench:

    def __init__(self, testName):
    # READ IN ALL IMAGES
    # Try is file images had already been opened:
        try:
            # The image itself
            self.raw_images = np.load("frames/" + testName + "/" + testName + "_ABR.npy")

            try:
                self.raw_R = np.load("frames/" + testName + "/" + testName + "_R.npy")
                self.raw_G = np.load("frames/" + testName + "/" + testName + "_G.npy")
                self.raw_B = np.load("frames/" + testName + "/" + testName + "_B.npy")
            except Exception:
                print("\nNo RBG files, only grayscale")
                self.raw_R = 1
                self.raw_G = 1
                self.raw_B = 1

            # Image params
            self.x_size = self.raw_images.shape[0]
            self.y_size = self.raw_images.shape[1]
            self.frames = self.raw_images.shape[2]

            self.isProcessed = False

            print("\nRead-in from saved numpy array.")
            # Do sth

        except IOError:
            print("\nFirst time opening, converting to numpy array")

            # Image params
            self.x_size = 128   # Least square value (???)
            self.y_size = 128   # Least square value (???)
            self.frames = 60   # Should find no. of frams from file count

            # Preallocate array for all images
            self.raw_R = np.zeros((self.x_size,self.y_size,self.frames), dtype='float32')
            self.raw_G = np.zeros((self.x_size,self.y_size,self.frames), dtype='float32')
            self.raw_B = np.zeros((self.x_size,self.y_size,self.frames), dtype='float32')
            self.raw_images = np.zeros((self.x_size,self.y_size,self.frames), dtype='float32')



            if os.path.isdir("frames/" + testName + "/" + "raw/"):
                print("Constructing from single image")
                importlib.reload(event_creation)
                self.raw_images = event_creation.convertFromSingle(testName, self.x_size, self.y_size, self.frames)

            elif (os.path.isdir("frames/" + testName + "/" + "raw_dim/") and os.path.isdir("frames/" + testName + "/" + "raw_bright/")):
                print("Constructing from multiple images")
                self.raw_images = event_creation.convertFromCompound(testName, self.x_size, self.y_size, self.frames)

        # Mark that these are new and not processed
        self.isProcessed = False

    def __del__(self):
        print("Images cleared from memory.")

    def testImport(self, frame_cap = None):
        if frame_cap is None:
            self.processedFrames = self.frames
        elif (frame_cap > self.frames):
            self.processedFrames = self.frames
        else:
            self.processedFrames = frame_cap

        try:
            importlib.reload(event_creation)
            self.prc_images = event_creation.runMe(self.raw_images, self.x_size, self.y_size, self.processedFrames)
            self.isProcessed = True
        except Exception as e:
            print("\nIMPORT FAILED! -> Processing")
            print(str(e) + '\n')


    def playProcessed(self):
        if (self.isProcessed == True):

            try:
                importlib.reload(playProcessedModule)
                playProcessedModule.playProcessed(self.prc_images, self.processedFrames)
            except Exception as e:
                print("\nIMPORT FAILED! -> Visualising")
                print(str(e) + '\n')

        else:
            print("\nData was not processed")

    def playImport(self, frame_cap = None):
        try:
            self.testImport(frame_cap)
        except Exception as e:
            print("\nIMPORT FAILED! -> Processing")
            print(str(e) + '\n')
        try:
            self.playProcessed()
        except Exception as e:
            print("\nIMPORT FAILED! -> Visualising")
            print(str(e) + '\n')

    def playRaw(self):
        # PRESENT THE IMAGES
        print(self.raw_images)

        fig1 = plt.figure()
        im = plt.imshow(self.raw_images[:,:,0])
        im_ani = animation.FuncAnimation(fig1,lambda j: im.set_array(self.raw_images[:,:,j]),frames=range(self.frames),interval=100, repeat_delay=3000)
        plt.show()

TB = TestBench("test11")

run_module = True

while (run_module == True):

    print("\nNext command?\n\tr - reload images \n\ti - process images \n\tv - view processed images \n\tp - process&play \n\tn - exit")

    echo = input("Input:   ")

    if (echo == 'R' or echo == 'r'):
        del TB
        echo = input("Image name: ")
        TB = TestBench(echo)

    elif (echo == 'I' or echo == 'i'):
        cap = input('Cap frames?    ')
        if (cap == 'n' or cap == 'N'):
            pass
        elif (cap == ''):
            TB.testImport()
        else:
            TB.testImport(int(cap))

    elif (echo == 'V' or echo == 'v'):
        TB.playProcessed()

    elif (echo == 'P' or echo == 'p'):
        cap = input('Cap frames?    ')
        if (cap == 'n' or cap == 'N'):
            pass
        elif (cap == ''):
            TB.playImport()
        else:
            TB.playImport(int(cap))

    elif (echo == 'N' or echo == 'n'):
        run_module = False

    elif (echo == 'T' or echo == 't'):
        TB.playRaw()
