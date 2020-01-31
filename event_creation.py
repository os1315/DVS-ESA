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

# My imports
# import singlePixelTransform

def plotEvents(EVENT_LIST, theta, T, all_images_log, j, k):

    # EVENT_LIST [[x,y,t,p]] or [[x,y,t,diff]]

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
    j = 70
    k = 14

    # Camera params
    theta = 0.2
    plot_count = 4
    T = 50          # Sampling period in us (1/frame rate)
    latency = 15    # Length of a pixels refractory period in us

    # Arificially add dark current
    all_images = raw_images

    # 4D matrix containining raw and processed data at all times
    image_out = np.ones((x_size,y_size,frames_count,plot_count), dtype='float32') * 0.5

    # zeroth index on 4th dimension is always reserved for raw data
    image_out[:,:,:,0] = all_images[:,:,:frames_count]

    # DATA TRANSFORM
    all_images_log = np.log10(all_images)                   # Create matrix of logged values
    current_states = all_images_log[:,:,0]                  # Stores the values that pixel differentiators are zeroed at
    refract_period = np.zeros((x_size,y_size), dtype='float32')   # Stores ENDS of refractive periods for the pixels
    time = 0        # Time at which frame was recorded
    quick_burst = np.zeros((x_size,y_size))

    # quick_burst = 0   ->  no burst
    # quick_burst = 1   ->  positive
    # quick_burst = -1  ->  negative

    threshold_p = current_states + theta   # Constant threshold
    threshold_n = current_states - theta   # Constant threshold

    # Echo to console
    print("\nEvaluating plot 1:")

    # Iterate through frames
    for n in range(1,frames_count):

        print("Frame: ",n+1,'/',frames_count, end='\r')

        delta = all_images_log[:,:,n] - all_images_log[:,:,n-1]
        # threshold = current_states * theta        # Dynamic threshold

        for x in range(x_size):     # Iterate over x dimension
            for y in range(y_size): # Iterate over y dimension

                # Slope used for piecewise linear interpolation of pixel intensity
                slope = (all_images_log[x,y,n] - all_images_log[x,y,n-1])/T

                # Do quick burst if flag is up
                if quick_burst[x,y] > 0 :
                    event_time = refract_period[x,y]
                    # Update pixel state and set new end for refractory period
                    current_states[x,y] = all_images_log[x,y,n-1] + slope*(event_time-time)
                    refract_period[x,y] = event_time + latency
                    threshold_p[x,y] = current_states[x,y] + theta
                    threshold_n[x,y] = current_states[x,y] - theta

                    if quick_burst[x,y] == 1 :
                        if x == j and y == k:
                            EVENT_LIST.append([x,y,event_time,current_states[x,y]])
                    if quick_burst[x,y] == 2 :
                        if x == j and y == k:
                            EVENT_LIST.append([x,y,event_time,current_states[x,y]])
                    quick_burst[x,y] = 0

                # Keeps looking for events until pixel refrect period extends into next frame
                while(refract_period[x,y] < (time+T)):

                    # Case for increasing brightness
                    if threshold_p[x,y] < all_images_log[x,y,n] and slope > 0 :

                        # print("I'm here")

                        # Dummy value event visualisation, might delete later
                        image_out[x,y,n,1] = 1.0
                        # Linear estimate of threshold crossing instance
                        dt = abs((threshold_p[x,y] - all_images_log[x,y,n-1])/slope)

                        # This section calculates the registration of the event depending on the pixels refractory period
                        if refract_period[x,y] > time + dt :
                            event_time = refract_period[x,y]
                            current_states[x,y] = all_images_log[x,y,n-1] + slope*(event_time-time)
                            refract_period[x,y] = event_time + latency
                        else:
                            event_time = ceil(time + dt)
                            current_states[x,y] = current_states[x,y] + theta
                            refract_period[x,y] = event_time + latency

                        # Append event to event list and update thresholds
                        EVENT_LIST.append([x,y,event_time,1])
                        threshold_p[x,y] = current_states[x,y] + theta
                        threshold_n[x,y] = current_states[x,y] - theta

                        # Case for decreasing brightness
                    elif threshold_n[x,y] > all_images_log[x,y,n] and slope < 0 :

                        image_out[x,y,n,1] = 0.0
                        # Linear estimate of threshold crossing instance
                        dt = abs((threshold_n[x,y]-all_images_log[x,y,n-1])/slope)

                        # This section calculates the registration of the event depending on the pixels refractory period
                        if refract_period[x,y] > time + dt :
                            event_time = refract_period[x,y]
                            # Update pixel state and set new end for refractory period
                            current_states[x,y] = all_images_log[x,y,n-1] + slope*(event_time-time)
                            refract_period[x,y] = event_time + latency
                        else:
                            event_time = ceil(time + dt)
                            # Update pixel state and set new end for refractory period
                            current_states[x,y] = current_states[x,y] - theta
                            refract_period[x,y] = event_time + latency


                        EVENT_LIST.append([x,y, event_time,-1])
                        threshold_p[x,y] = current_states[x,y] + theta
                        threshold_n[x,y] = current_states[x,y] - theta

                    else:
                        break

                # track if pixel should fire immidiately at the end of refractory period
                if ((refract_period[x,y] > (time+T)) and (threshold_p[x,y] < all_images_log[x,y,n])) :
                    quick_burst[x,y] = 1
                elif ((refract_period[x,y] > (time+T)) and (threshold_n[x,y] > all_images_log[x,y,n])) :
                    quick_burst[x,y] = 2

        # Update time
        time = time + T

        image_out[:,:,n,2] = current_states

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
    #
    for n in range(1,frames_count):
        # image_out[:,:,n,2] = current_states
        image_out[:,:,n,3] = all_images_log[:,:,n]
        print("Frame: ",n+1,'/',frames_count, end='\r')

    print("Frame: ",n+1,'/',frames_count)

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
            # b = image_file.read(12)
            # self.raw_R[y,x,n], self.raw_G[y,x,n], self.raw_B[y,x,n] = struct.unpack('>fff',b)
            bright = image_file_bright.read(4)
            dim = image_file_dim.read(4)
            raw_images[y,x,n] = (struct.unpack('>f',bright)[0] + struct.unpack('>f',dim)[0]/contrastRatio)/1
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
            bright = image_file_bright.read(8)
            dim = image_file_dim.read(8)

        image_file_bright.close()
        image_file_dim.close()

    # self.raw_images = (self.raw_R + self.raw_G + self.raw_B) / 3

    # CREATE FILE WITH ALL DATA
    # np.save("frames/" + testName + "/" + testName + "_R.npy",self.raw_R)
    # np.save("frames/" + testName + "/"  + testName + "_G.npy",self.raw_G)
    # np.save("frames/" + testName + "/"  + testName + "_B.npy",self.raw_B)
    # np.save("frames/" + testName + "/"  + testName + "_ABR.npy",self.raw_images)
    print("File saved!")

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
