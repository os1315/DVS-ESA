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

def runMe(raw_images,x_size,y_size,frames_count):

    # Camera params
    theta = 0.2
    plot_count = 3

    # Arificially add dark current
    all_images = raw_images

    # New expaded matrix just for visualisation
    image_out = np.ones((x_size,y_size,frames_count,plot_count), dtype='float32') * 0.5

    # image_out[0:x_size,:,:] = all_images
    image_out[:,:,:,0] = all_images[:,:,:frames_count]

    # DATA TRANSFORM
    all_images_log = np.log10(all_images)    # Create matrix of logged values
    current_states = all_images_log[:,:,0] # Stores the values that pixel differentiators are zeroed at

    print("\nEvaluating plot 1:")

    for n in range(1,frames_count):

        print("Frame: ",n,'/',frames_count, end='\r')

        delta = current_states-all_images_log[:,:,n]
        threshold = current_states * theta

        for x in range(x_size):
            for y in range(y_size):

                if (abs(delta[x,y]) > abs(threshold[x,y]) and delta[x,y] < 0):
                    image_out[x,y,n,1] = 1.0
                    current_states[x,y] = all_images_log[x,y,n]
                    # current_states[x,y] = current_states[x,y] + current_states[x,y] * theta
                elif (abs(delta[x,y]) > abs(threshold[x,y]) and delta[x,y] > 0):
                    image_out[x,y,n,1] = 0.0
                    current_states[x,y] = all_images_log[x,y,n]
                    # current_states[x,y] = current_states[x,y] - current_states[x,y] * theta
                else:
                    image_out[x,y,n,1] = 0.5

        # Different theta

    print("\n\nEvaluating plot 2:")

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

    for n in range(1,frames_count):
        image_out[:,:,n,2] = all_images_log[:,:,n]


    print("Frame: ",n+1,'/',frames_count)

    return image_out

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

def convertFromCompound(testName,x_size,y_size,frames):

    # Preallocate array for all images
    raw_R = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_G = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_B = np.zeros((x_size,y_size,frames), dtype='float32')
    raw_images = np.zeros((x_size,y_size,frames), dtype='float32')

    for n in range(frames):
        image_file_bright = open("frames/" + testName + "/" + "raw_bright/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        image_file_dim = open("frames/" + testName + "/" + "raw_dim/" + testName + '_{:03d}'.format(n) + ".img" , "rb")
        print(image_file_bright)

        x = y = 0
        b = True

        while b:
            # b = image_file.read(12)
            # self.raw_R[y,x,n], self.raw_G[y,x,n], self.raw_B[y,x,n] = struct.unpack('>fff',b)
            bright = image_file_bright.read(4)
            dim = image_file_dim.read(4)
            raw_images[y,x,n] = (struct.unpack('>f',bright)[0] + 0.01*struct.unpack('>f',dim)[0])/1 + 1**(-12)
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
