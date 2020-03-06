# Math imports

# Numpy imports
import numpy as np

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Other imports
import time
import importlib
import os
import traceback
import sys

from Filehandling import readinFrameRate

sys.path.append('./resources')

# My other imports

# This is the actually file being tested
import event_creation
import playProcessedModule


class TestBench:

    def __init__(self, testName, target_dir=" "):

        # Switch directory to where the data is
        if target_dir == " ":
            pass
        else:
            os.chdir(target_dir)

        self.test_name = testName

        # Extract frame rate from fli file
        self.frame_rate = readinFrameRate(self.test_name)
        print("Frame rate: ", self.frame_rate)

        # READ IN ALL IMAGES
        # Try is file images had already been opened:
        try:
            start_time = time.time()

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

            # Echo success
            run_time = time.time() - start_time
            print('\nRead-in from saved numpy array in {} s'.format(run_time/1000))

        except IOError:
            print("\nFirst time opening, converting to numpy array")

            # Image params
            self.x_size = 128   # Least square value (???)
            self.y_size = 128   # Least square value (???)
            try:
                self.frames = len(os.listdir(target_dir + "/frames/" + testName + "/raw"))   # Find # of frams from file count
            except FileNotFoundError:
                self.frames = len(os.listdir(target_dir + "/frames/" + testName + "/raw_bright"))  # Find # of frams from file count

            # Preallocate array for all images
            # self.raw_R = np.zeros((self.x_size, self.y_size, self.frames), dtype='float32')
            # self.raw_G = np.zeros((self.x_size, self.y_size, self.frames), dtype='float32')
            # self.raw_B = np.zeros((self.x_size, self.y_size, self.frames), dtype='float32')
            self.raw_images = np.zeros((self.x_size, self.y_size, self.frames), dtype='float32')

            # Verify is it's single picture or compound
            if os.path.isdir("frames/" + testName + "/" + "raw/"):
                print("Constructing from single image")
                importlib.reload(event_creation)
                self.raw_images = event_creation.convertWithNoisebank(testName, self.x_size, self.y_size, self.frames, NBnum=1) # Note you need NBnum specifying bank otherwise it will attempt compound

            elif os.path.isdir("frames/" + testName + "/" + "raw_dim/") and os.path.isdir("frames/" + testName + "/" + "raw_bright/"):
                print("Constructing from multiple images")
                importlib.reload(event_creation)
                self.raw_images = event_creation.convertFromCompound(testName, self.x_size, self.y_size, self.frames)

        # Mark that these are new and not processed
        self.isProcessed = False

    def __del__(self):
        print("Images cleared from memory.")

    def processImages(self, frame_cap=None):

        # Capping frames
        if frame_cap is None:
            self.processedFrames = self.frames
        elif frame_cap > self.frames:
            self.processedFrames = self.frames
        else:
            self.processedFrames = frame_cap

        # Launching converter
        try:
            importlib.reload(event_creation)
            self.prc_images, EVENT_LIST = event_creation.createEvents(self.raw_images, self.frame_rate)
            self.isProcessed = True
        except Exception as e:
            print("\nIMPORT FAILED! -> Processing")
            print(traceback.format_exc() + '\n')

        # Saving event list
        try:
            EVENT_FILE = open("frames/" + self.test_name + "/eventlist.txt", 'w')
            EVENT_FILE.write("T: " + str(50) + ";\n")
            for event in EVENT_LIST:
                EVENT_FILE.write("x: " + str(event[0]) + "; y: " + str(event[1]) + "; t: " + "{:.0f}".format(event[2]) + "; p: " + str(event[3]) + "\n" )
            EVENT_FILE.close()
        except Exception as e:
            print("\nIMPORT FAILED! -> Processing -> Saving event list")
            print(traceback.format_exc() + '\n')

    def playProcessed(self):
        if self.isProcessed == True:

            try:
                importlib.reload(playProcessedModule)
                playProcessedModule.playProcessed(self.prc_images, self.processedFrames)
            except Exception as e:
                print("\nIMPORT FAILED! -> Visualising")
                print(traceback.format_exc() + '\n')

        else:
            print("\nData was not processed")

    def playRaw(self):

        fig1 = plt.figure()
        im = plt.imshow(self.raw_images[:, :, 0])
        im_ani = animation.FuncAnimation(fig1, lambda j: im.set_array(self.raw_images[:, :, j]), frames=range(self.frames), interval=100, repeat_delay=3000)
        plt.show()

    def playImport(self, frame_cap = None):
        try:
            self.processImages(frame_cap)
        except Exception as e:
            print("\n\nIMPORT FAILED! -> Processing\n")
            print(traceback.format_exc() + '\n')
        try:
            self.playProcessed()
        except Exception as e:
            print("\n\nIMPORT FAILED! -> Visualising\n")
            print(traceback.format_exc() + '\n')
