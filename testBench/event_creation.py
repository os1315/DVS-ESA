# Math imports

import importlib
import random
import struct
import time

# Matplotlib imports
import matplotlib.pyplot as plt

# Numpy imports
import numpy as np
import parse

# My imports (have to be reloaded for top level reload to take effect)
# import singlePixelTransform
import convInterpolate as CI
from auxiliary.Filehandling import ProgressTracker

importlib.reload(CI)


def plotEvents(EVENT_LIST, T, all_images_log, j, k):
    """This is a testing function left over from when the createEvents functions was first created"""

    # EVENT_LIST [[y,x,t,p]] or [[x,y,t,diff]]

    eventList = np.ones([len(EVENT_LIST), 3])

    for n in range(len(EVENT_LIST)):
        eventList[n, 0] = EVENT_LIST[n][2]  # Event Time
        eventList[n, 1] = EVENT_LIST[n][3]  # Differentiator
        # eventList[n,2] = EVENT_LIST[n][5]
        print('{:4.1f} {:4.1f}'.format(eventList[n, 0], eventList[n, 1]))

    figaux, ax = plt.subplots()

    line1 = ax.scatter(np.linspace(0, T * (all_images_log.shape[2] - 1), all_images_log.shape[2]), all_images_log[j, k],
                       color='r')
    line1.set_label('Illumination')

    # ax3 = plt.subplot(212,sharex=ax1, sharey=ax1)
    line3 = ax.scatter(eventList[:, 0], eventList[:, 1])
    line3.set_label("Differentiator state")

    ax.grid(True)
    ax.legend()
    plt.show()


def createEvents(raw_images, frame_rate):
    # List of events
    EVENT_LIST = []

    x_size, y_size, frames_count = raw_images.shape
    # Camera params
    theta = 0.2
    plot_count = 3
    T = 1 / frame_rate  # Sampling period in us (1/frame rate)
    latency = 15  # Length of a pixels refractory period in us

    # Artificially add dark current
    all_images = raw_images

    image_out = np.ones([x_size, y_size, frames_count, plot_count])
    image_out[:, :, :, 0] = all_images[:, :, :frames_count]
    image_out[:, :, :, 2] = np.log10(all_images[:, :, :frames_count])

    print("\n\nEvaluating plot 1:")

    initial_image = all_images[:, :, 0]

    converter = CI.convInterpolate(theta, T, latency, initial_image)

    for n in range(1, frames_count):
        print(" Frame: ", n + 1, '/', frames_count)
        PARTIAL_LIST, image_out[:, :, n, 1] = converter.update(all_images[:, :, n])
        EVENT_LIST = EVENT_LIST + PARTIAL_LIST

    print(" ")

    return image_out, EVENT_LIST


# ______________________________________________________________________________

def readRatio(testName, NB=None):
    if NB is None:
        log_file = open("frames/" + testName + "/" + "log.txt", "r")
        log_string = log_file.read()

        GAIN1 = parse.search("GAIN1: {:d};", log_string)[0]
        GAIN2 = parse.search("GAIN2: {:d};", log_string)[0]

        log_file.close()

    else:
        log_file = open("frames/" + testName + "/" + "log.txt", "r")
        log_string = log_file.read()
        GAIN1 = parse.search("GAIN1: {:d};", log_string)[0]
        log_file.close()

        log_file = open("frames/noiseBank" + "/" + "log.txt", "r")
        log_string = log_file.read()
        GAIN2 = parse.search("GAIN1: {:d};", log_string)[0]
        log_file.close()

    return GAIN1 / GAIN2

def readImageIdxFormat(testName):
    return


def convertFromCompound(testName, x_size, y_size, frames):
    # Measure time
    start_time = time.time()

    # Preallocate array for all images
    raw_R = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_G = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_B = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_images = np.zeros((x_size, y_size, frames), dtype='float32')

    # Finds gain values from log file and adjusts relative image brightness.    (FIX IT!!!!)
    # contrast_ratio_noise = readRatio(testName)

    # Finds the values from log file and adjusts the relative image brightness
    contrast_ratio_image = readRatio(testName)

    # For progress echos to console
    tracker = ProgressTracker(frames)

    for n in range(frames):

        # Echo progress to console
        tracker.update(n)

        image_file_bright = open("frames/" + testName + "/" + "raw_bright/" + testName + '_{:05d}'.format(n) + ".img",
                                 "rb")
        image_file_dim = open("frames/" + testName + "/" + "raw_dim/" + testName + '_{:05d}'.format(n) + ".img", "rb")
        # print(image_file_bright)

        x = y = 0
        b = True

        while b:

            b = image_file_bright.read(12)
            R_bright, G_bright, B_bright = struct.unpack('>fff', b)

            b = image_file_dim.read(12)
            R_dim, G_dim, B_dim = struct.unpack('>fff', b)

            if (R_bright + G_bright + B_bright)/3 < 0.001:
                raw_R[y, x, n] = R_dim/contrast_ratio_image
                raw_G[y, x, n] = G_dim/contrast_ratio_image
                raw_B[y, x, n] = B_dim/contrast_ratio_image
            else:
                raw_R[y, x, n] = R_bright
                raw_G[y, x, n] = G_bright
                raw_B[y, x, n] = B_bright

            raw_images[y, x, n] = (raw_R[y, x, n] + raw_G[y, x, n] + raw_B[y, x, n]) / 3

            # New row or break at end
            x = x + 1
            if x > x_size - 1:
                x = 0
                y = y + 1
                if y > y_size - 1:
                    break

        image_file_bright.close()
        image_file_dim.close()


    ## Add noise from noise bank
    # NOT YET IMPLEMENTED

    # Echo of no noise
    tracker.complete("No noise selected, adding padding...")

    for image in range(raw_images.shape[2]):

        tracker.update(image)

        subset = raw_R[:, :, image]
        padding = np.min(subset[np.nonzero(subset[:, :])]) * 100  # Arbitrarily decided what the padding is

        print("\nPadding: ", padding)
        raw_R[:, :, image] = raw_R[:, :, image] + padding
        raw_G[:, :, image] = raw_G[:, :, image] + padding
        raw_B[:, :, image] = raw_B[:, :, image] + padding
        raw_images[:, :, image] = raw_images[:, :, image] + padding
        print("Min value: ", np.min(raw_images[:, :, image]))

    tracker.complete("Saving...")

    # CREATE FILE WITH ALL DATA
    np.save("frames/" + testName + "/" + testName + "_R.npy", raw_R)
    np.save("frames/" + testName + "/" + testName + "_G.npy", raw_G)
    np.save("frames/" + testName + "/" + testName + "_B.npy", raw_B)
    np.save("frames/" + testName + "/" + testName + "_ABR.npy", raw_images)

    run_time = time.time() - start_time

    print("File saved!")
    print('Runtime: {:3.2f} for {:d} images or {:3.2f} ms per image'.format(run_time, raw_images.shape[2],
                                                                            1000 * run_time / raw_images.shape[2]))

    return raw_images


def convertWithNoisebank(testName, x_size, y_size, frames, NBnum=None):
    # Measure time
    start_time = time.time()

    # Preallocate array for all images
    raw_R = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_G = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_B = np.zeros((x_size, y_size, frames), dtype='float32')
    raw_images = np.zeros((x_size, y_size, frames), dtype='float32')

    # Finds gain values from log file and adjusts relative image brightness.
    if NBnum is not None:
        contrastRatio = readRatio(testName, NB=NBnum)

    # For progress echos to console
    tracker = ProgressTracker(frames)

    # Read in frames
    for n in range(frames):
        image_file_bright = open("frames/" + testName + "/" + "raw/" + testName + '_{:05d}'.format(n) + ".img",
                                 "rb")

        # Echo progress to console
        tracker.update(n)

        x = y = 0
        b = True

        while b:

            b = image_file_bright.read(12)
            R_bright, G_bright, B_bright = struct.unpack('>fff', b)

            raw_R[y, x, n] = R_bright
            raw_G[y, x, n] = G_bright
            raw_B[y, x, n] = B_bright

            # New row or break at end
            x = x + 1
            if x > x_size - 1:
                x = 0
                y = y + 1
                if y > y_size - 1:
                    break

        image_file_bright.close()

    raw_images = (raw_R + raw_G + raw_B) / 3

    ## Add noise from noise bank
    # Pre-allocate array for noise
    noise_R = np.zeros((x_size, y_size), dtype='float32')
    noise_G = np.zeros((x_size, y_size), dtype='float32')
    noise_B = np.zeros((x_size, y_size), dtype='float32')
    noise_image = np.zeros((x_size, y_size), dtype='float32')

    # Verify if it should add noise
    if NBnum is not None:

        # Echo complete to console
        tracker.complete("Images read-in from file, adding noise...")

        # Iterate through read images
        for image in range(raw_images.shape[2]):

            # Echo progress to console
            tracker.update(image)

            # Randomly pick noise from noise bank
            rand = random.randrange(299)
            noise_file = open("frames/noiseBank" + str(NBnum) + "/" + "raw/noiseBank" + '_{:03d}'.format(rand) + ".img", "rb")
            noise_file = open("frames/noiseBank" + str(NBnum) + "/" + "raw/noiseBank" + '_{:03d}'.format(rand) + ".img", "rb")

            # Iterate through file in 12 byte batches (RGB triplets)
            x = y = 0
            b = True

            while b:

                b = noise_file.read(12)
                R, G, B = struct.unpack('>fff', b)

                noise_R[x, y] = R
                noise_G[x, y] = G
                noise_B[x, y] = B
                noise_image[x, y] = (R + G + B) / 3

                # New row or break at end
                x = x + 1
                if x > x_size - 1:
                    x = 0
                    y = y + 1
                    if y > y_size - 1:
                        break

            noise_file.close()

            # This is just a workaround cause noise generation is shit
            padding = np.min(noise_image[np.nonzero(noise_image)])/(2*contrastRatio)

            raw_R[:, :, image] = raw_B[:, :, image] + noise_R / contrastRatio + padding
            raw_G[:, :, image] = raw_G[:, :, image] + noise_G / contrastRatio + padding
            raw_B[:, :, image] = raw_B[:, :, image] + noise_B / contrastRatio + padding
            raw_images[:, :, image] = raw_images[:, :, image] + noise_image / contrastRatio + padding

        # Echo complete to console
        tracker.complete("Noise added, saving...\n")

    # This is here just to allow conversion of images that contain zeroes
    else:
        # Echo of no noise
        tracker.complete("No noise selected, adding padding and saving...")

        for image in range(raw_images.shape[2]):
            padding = np.min(raw_R)/10  # Arbitrarily decided what the padding is
            raw_R[:, :, image] = raw_B[:, :, image] + padding
            raw_G[:, :, image] = raw_G[:, :, image] + padding
            raw_B[:, :, image] = raw_B[:, :, image] + padding
            raw_images[:, :, image] = raw_images[:, :, image] + padding

    # CREATE FILE WITH ALL DATA
    np.save("frames/" + testName + "/" + testName + "_R.npy", raw_R)
    np.save("frames/" + testName + "/" + testName + "_G.npy", raw_G)
    np.save("frames/" + testName + "/" + testName + "_B.npy", raw_B)
    np.save("frames/" + testName + "/" + testName + "_ABR.npy", raw_images)

    run_time = time.time() - start_time

    print("File saved!")
    print('Runtime: {:3.2f} for {:d} images or {:3.2f} ms per image'.format(run_time, raw_images.shape[2],
                                                                            1000 * run_time / raw_images.shape[2]))

    return raw_images
