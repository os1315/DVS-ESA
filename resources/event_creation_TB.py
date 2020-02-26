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


# # READ IN ALL IMAGES
# # Image params
# x_size = 300
# y_size = 300
# frames_count = 120

def runMe(raw_images,x_size,y_size,frames_count):

    # Camera params
    theta = 0.2

    # # Load data
    # all_images = np.load("frames/test1_ABR.npy")
    # print("Data loaded!")

    # Arificially add dark current
    all_images = raw_images + (2**(-32))

    # New expaded matrix just for visualisation
    image_out = np.ones((x_size,y_size*2 + 10,frames_count), dtype='float32') * 0.1
    # image_out[0:x_size,:,:] = all_images
    image_out[:,y_size+10:y_size*2 + 10,:] = all_images[:,:,:frames_count]

    # DATA TRANSFORM
    all_images_log = np.log(all_images)    # Create matrix of logged values
    # image_out[0:x_size,:,0] = all_images_log[:,:,0] #First frame
    current_states = all_images_log[:,:,0] # Stores the values that pixel differentiators are zeroed at

    print("Images preprocessed, calculating events...")

    for n in range(1,frames_count):
        print("Frame: ",n)
        for x in range(x_size):
            for y in range(y_size):
                delta = current_states[x,y]-all_images_log[x,y,n]
                threshold = current_states[x,y] * theta

                if (abs(delta) > threshold and delta < 0):
                    image_out[x,y,n] = 1.0
                    current_states[x,y] = all_images_log[x,y,n]
                    # current_states[x,y] = current_states[x,y] + current_states[x,y] * theta
                elif (abs(delta) > threshold and delta > 0):
                    image_out[x,y,n] = 0.0
                    current_states[x,y] = all_images_log[x,y,n]
                    # current_states[x,y] = current_states[x,y] - current_states[x,y] * theta
                else:
                    image_out[x,y,n] = 0.1
                    # print(".")

                    # image_out[0:x_size,:,0] = all_images_log[:,:,0]
    #
    # # PRESENT THE IMAGES
    # fig1 = plt.figure()
    # im = plt.imshow(image_out[:,:,0])
    # im_ani = animation.FuncAnimation(fig1,lambda j: im.set_array(image_out[:,:,j]),frames=range(frames_count),interval=100, repeat_delay=3000)
    # plt.show()

# plt.ion()
# plt.figure()
#
# plt.imshow(all_images[:,:,0])
# plt.show(block=False)
# start_time = time.time()
# plt.pause(0.05)
#
#
# for n in range(frames):
#     plt.imshow(all_images[:,:,n])
#     plt.draw()
#     elapsed_time = time.time() - start_time
#     print("Frame: ", n, " Elapsed Time: ", elapsed_time)
#     start_time = time.time()
#     plt.pause(0.05)


# Opening 100x100 images
# image_file = open("frames/float_00.img", "rb")
#
# #  This is the image array, notice that indexing goes from left-top corner, contrary to image convention
# image_np = np.zeros((100,100))
# x = 0
# y = 0
#
#
# b = True
# while b:
#     b = image_file.read(4)
#     image_np[y,x] = floor(struct.unpack('>f',b)[0]*255)
#
#     image_np[y,x] = floor(struct.unpack('>f',b)[0]*255)
#
#     x = x+1
#
#     if (x>99):
#         x = 0
#         y = y+1
#         if (y>99):
#             break
#
#     # Shift to next RGB triplet
#     b = image_file.read(8)
#
# frame = Image.fromarray(image_np)
# print(image_np[:,40])
# frame.show()
