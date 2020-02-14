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

import struct
import parse
import time
import importlib

# My imports (have to be reloded for top level reloadto take effect)
# import singlePixelTransform
import convInterpolate as CI
importlib.reload(CI)



def plotEvents(EVENT_LIST, theta, T, all_images_log, j, k):

    # EVENT_LIST [[y,x,t,p]] or [[x,y,t,diff]]

    eventList = np.ones([len(EVENT_LIST),3])

    for n in range(len(EVENT_LIST)):
        eventList[n,0] = EVENT_LIST[n][2]   # Event Time
        eventList[n,1] = EVENT_LIST[n][3]   # Differentiator
        # eventList[n,2] = EVENT_LIST[n][5]
        print('{:4.1f} {:4.1f}'.format(eventList[n,0],eventList[n,1]))

    figaux, ax = plt.subplots()

    line1 = ax.scatter(np.linspace(0,T*(all_images_log.shape[2]-1),all_images_log.shape[2]),all_images_log[j,k], color='r')
    line1.set_label('Illumination')

    # ax3 = plt.subplot(212,sharex=ax1, sharey=ax1)
    line3 = ax.scatter(eventList[:,0], eventList[:,1])
    line3.set_label("Differentiator state")

    ax.grid(True)
    ax.legend()
    plt.show()


def createEvents(raw_images,x_size,y_size,frames_count):

    # List of events
    EVENT_LIST = []
    PARTIAL_LIST = []
    j = 70
    k = 14

    # Camera params
    theta = 0.4
    plot_count = 3
    T = 2000          # Sampling period in us (1/frame rate)
    latency = 15    # Length of a pixels refractory period in us

    # Arificially add dark current
    all_images = raw_images

    image_out = np.ones([x_size,y_size,frames_count,plot_count])
    image_out[:,:,:,0] = all_images[:,:,:frames_count]
    image_out[:,:,:,2] = np.log10(all_images[:,:,:frames_count])

    print("\n\nEvaluating plot 1:")

    initial_image = all_images[:,:,0]

    converter = CI.convInterpolate(x_size, y_size, theta, T, latency,initial_image)

    EVENT_LIST = []

    for n in range(1,frames_count):
        print(" Frame: ",n+1,'/',frames_count, end='\r')
        PARTIAL_LIST, image_out[:,:,n,1] = converter.update(all_images[:,:,n])
        EVENT_LIST = EVENT_LIST + PARTIAL_LIST

    print(" ")

    # print("\n\nEvaluating plot 2:")

    # theta = 0.9
    #
    # for n in range(1,frames_count):
    #
    #     print("Frame: ",n,'/',frames_count, end='\r')
    #
    #     delta = current_states-all_images_log[:,:,n]
    #     threshold = current_states * theta
    #
    #     for x in range(x_size):
    #         for y in range(y_size):
    #
    #             if (abs(delta[x,y]) > abs(threshold[x,y]) and delta[x,y] < 0):
    #                 image_out[x,y,n,2] = 1.0
    #                 current_states[x,y] = all_images_log[x,y,n]
    #                 # current_states[x,y] = current_states[x,y] + current_states[x,y] * theta
    #             elif (abs(delta[x,y]) > abs(threshold[x,y]) and delta[x,y] > 0):
    #                 image_out[x,y,n,2] = 0.0
    #                 current_states[x,y] = all_images_log[x,y,n]
    #                 # current_states[x,y] = current_states[x,y] - current_states[x,y] * theta
    #             else:
    #                 image_out[x,y,n,2] = 0.5

        # Save the delta for visualisation
    # image_out[:,:,n,2] = all_images_log
        # image_out[:,:,n,3] = threshold

    return image_out, EVENT_LIST

# ______________________________________________________________________________

def readRatio(testName):
    log_file = open("frames/" + testName + "/" + "log.txt" , "r")
    log_string = log_file.read()

    GAIN1 = parse.search("GAIN1: {:d};", log_string)[0]
    GAIN2 = parse.search("GAIN2: {:d};", log_string)[0]

    log_file.close()

    return GAIN1/GAIN2


def convertFromCompound(testName,x_size,y_size,frames):

    start_time = time.time()

    # Preallocate array for all images
    raw_R = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_G = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_B = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_images = np.zeros((x_size,y_size,frames), dtype='float32')

    # Finds gain values from log file and adjusts relative image brightness.
    contrastRatio = readRatio(testName)

    for n in range(frames):
        image_file_bright = open("frames/" + testName + "/" + "raw_bright/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        image_file_dim = open("frames/" + testName + "/" + "raw_dim/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        # print(image_file_bright)

        x = y = 0
        b = True

        while b:
            ### Redaing single channel ###
            bright = image_file_bright.read(4)
            dim = image_file_dim.read(4)
            raw_images[y,x,n] = (struct.unpack('>f',bright)[0] + struct.unpack('>f',dim)[0]/contrastRatio)/1

            # Shift to next RGB triplet
            bright = image_file_bright.read(8)
            dim = image_file_dim.read(8)

            # New row or break at end
            x = x + 1
            if (x > x_size-1):
                x = 0
                y = y+1
                if (y > y_size-1):
                    break

        image_file_bright.close()
        image_file_dim.close()

    # raw_images = (raw_R + raw_G + raw_B) / 3

    # CREATE FILE WITH ALL DATA
    # np.save("frames/" + testName + "/" + testName + "_R.npy",raw_R)
    # np.save("frames/" + testName + "/"  + testName + "_G.npy",raw_G)
    # np.save("frames/" + testName + "/"  + testName + "_B.npy",raw_B)
    # np.save("frames/" + testName + "/"  + testName + "_ABR.npy",raw_images)

    run_time = time.time() - start_time

    print("File saved!")
    print('Runtime: {:3.2f} for {:d} images or {:3.2f} ms per image'.format(run_time, raw_images.shape[2], 1000*run_time/raw_images.shape[2] ))

    return raw_images

def convertWithNoisebank(testName,x_size,y_size,frames):

    start_time = time.time()

    # Noise bank number
    NBnum = 1

    # Preallocate array for all images
    raw_R = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_G = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_B = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_images = np.zeros((x_size,y_size,frames), dtype='float32')

    # Finds gain values from log file and adjusts relative image brightness.
    contrastRatio = readRatio(testName)

    for n in range(frames):
        image_file_bright = open("frames/" + testName + "/" + "raw_bright/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        image_file_dim = open("frames/" + testName + "/" + "raw_dim/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        # print(image_file_bright)

        x = y = 0
        b = True

        while b:

            b = image_file_bright.read(12)
            R_bright, G_bright, B_bright = struct.unpack('>fff',b)
            b = image_file_dim.read(12)
            R_dim, G_dim, B_dim = struct.unpack('>fff',b)

            raw_R[y,x,n] = R_bright + R_dim/contrastRatio
            raw_G[y,x,n] = G_bright + G_dim/contrastRatio
            raw_B[y,x,n] = B_bright + B_dim/contrastRatio

            # New row or break at end
            x = x + 1
            if (x > x_size-1):
                x = 0
                y = y+1
                if (y > y_size-1):
                    break

        image_file_bright.close()
        image_file_dim.close()

    raw_images = (raw_R + raw_G + raw_B) / 3

    # Add noise from noisebank
    for image in range(raw_images.shape[2]):

        # Randomly pick noise from noisebank
        random.randrange(299)
        noise_image = open("frames/" + noiseBank + str(NBnumber) + "/" + "raw_dim/" + noiseBank + '_{:03d}'.format(n) + ".img" , "rb")

        x = y = 0
        b = True

        while b:

            b = noise_image.read(4)
            raw_images[y,x,image] = raw_images[y,x,image] + struct.unpack('>f',b)

            # Shift to next triplet
            b = noise_image.read(8)

            # New row or break at end
            x = x + 1
            if (x > x_size-1):
                x = 0
                y = y+1
                if (y > y_size-1):
                    break

        noise_image.close()

    # CREATE FILE WITH ALL DATA
    np.save("frames/" + testName + "/" + testName + "_R.npy",raw_R)
    np.save("frames/" + testName + "/"  + testName + "_G.npy",raw_G)
    np.save("frames/" + testName + "/"  + testName + "_B.npy",raw_B)
    np.save("frames/" + testName + "/"  + testName + "_ABR.npy",raw_images)

    run_time = time.time() - start_time

    print("File saved!")
    print('Runtime: {:3.2f} for {:d} images or {:3.2f} ms per image'.format(run_time, raw_images.shape[2], 1000*run_time/raw_images.shape[2] ))

    return raw_images


def convertFromSingle(testName,x_size,y_size,frames):

    # Preallocate array for all images
    raw_R = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_G = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_B = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_images = np.zeros((x_size,y_size,frames), dtype='float32')


    for n in range(frames):
        # image_file = open("frames/" + testName + "/" + "raw/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        image_file = open("frames/" + testName + "/" + "raw/" + "noisetest_0" + '_{:03d}'.format(n) + ".img" , "rb")
        print(image_file)

        x = y = 0
        b = True

        while b:
            # b = image_file.read(12)
            # self.raw_R[y,x,n], self.raw_G[y,x,n], self.raw_B[y,x,n] = struct.unpack('>fff',b)
            b = image_file.read(4)
            raw_images[y,x,n] = struct.unpack('>f',b)[0]
            # b = image_file.read(4)
            # self.raw_images[y,x,n] = self.raw_images[y,x,n] + struct.unpack('>f',b)[0]
            # b = image_file.read(4)
            # self.raw_images[y,x,n] = self.raw_images[y,x,n] + struct.unpack('>f',b)[0]
            # self.raw_images[y,x,n] = self.raw_images[y,x,n] / 3

            x = x + 1

            # New row or break at end
            if (x > x_size-1):
                x = 0
                y = y+1
                if (y > y_size-1):
                    break

            # Shift to next RGB triplet
            b = image_file.read(8)

        image_file.close()

    # self.raw_images = (self.raw_R + self.raw_G + self.raw_B) / 3

    # CREATE FILE WITH ALL DATA
    # np.save("frames/" + testName + "/" + testName + "_R.npy",self.raw_R)
    # np.save("frames/" + testName + "/"  + testName + "_G.npy",self.raw_G)
    # np.save("frames/" + testName + "/"  + testName + "_B.npy",self.raw_B)
    # np.save("frames/" + testName + "/"  + testName + "_ABR.npy",self.raw_images)
    print("File saved!")

    return raw_images
