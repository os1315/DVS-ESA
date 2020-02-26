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

# Other imports
import struct
import parse
import time as tm
import os
import traceback

# This is the actualy file being tested
import event_creation

x_size = 128
y_size = 128
frames = 60

j = 51
k = 88

# Tested file
file = "testLogged1"

# Initialize array for raw images
raw_images = np.zeros((x_size,y_size,frames), dtype='float32')

# Read gains from log file, we need this to reconstruct the images
log_file = open("frames/" + file + "/log.txt" , "r")
log_string = log_file.read()

GAIN1 = parse.search("GAIN1: {:d};", log_string)[0]
GAIN2 = parse.search("GAIN2: {:d};", log_string)[0]

log_file.close()

# Reconstruct images
raw_images = event_creation.convertFromCompound("testLogged1",x_size,y_size,frames)
testVector = np.log10(raw_images[j,k,:])

# Track conversion execution time
start_time = tm.time()

# Camera / simulation parameters
theta = 0.2
T = 50
latency = 15
refract_period = 0
time = 0

# Pixel states
quick_burst = 0
current_states = testVector[0]
EVENT_LIST = []

threshold_p = current_states + theta   # Constant threshold
threshold_n = current_states - theta   # Constant threshold

# Iterate through frames
for n in range(1,testVector.shape[0]):

    delta = testVector[n] - testVector[n-1]

    # Slope used for piecewise linear interpolation of pixel intensity
    slope = (testVector[n] - testVector[n-1])/T
    print('\ntime: {} \ncurrent {:01.4f} illum: {:01.4f}'.format(time, current_states,testVector[n]))

    # Determine if quick burst
    if quick_burst > 0 :
        event_time = refract_period
        # Update pixel state and set new end for refractory period
        current_states = testVector[n-1] + slope*(event_time-time)
        refract_period = event_time + latency
        threshold_p = current_states + theta
        threshold_n = current_states - theta
        if quick_burst == 1 :
            EVENT_LIST.append([event_time,1, current_states])
            print("\nQuick Burst: 1")
        if quick_burst == 2 :
            EVENT_LIST.append([event_time,-1, current_states])
            print("\nQuick Burst: -1")
        quick_burst = 0

    # Keeps looking for events until pixel refract_period extends into next frame
    while(refract_period < (time+T)):
        # Case for increasing brightness
        print("ref_perdiod: ", refract_period)
        if threshold_p < testVector[n] and slope > 0 :

            # Linear estimate of threshold crossing instance
            dt = abs((current_states + theta  - testVector[n-1])/slope)
            print('\ndt: {} theta: {} slope: {}'.format(dt,theta,slope))

            # This section calculates the registration of the event depending on the pixels refractory period
            if refract_period > time + dt :
                event_time = refract_period
                # Update pixel state and set new end for refractory period
                current_states = testVector[n-1] + slope*(event_time-time)
                refract_period = event_time + latency
                print("\nRP limited")
            else:
                event_time = ceil(time + dt)
                # Update pixel state and set new end for refractory period
                current_states = current_states + theta
                refract_period = event_time + latency
                print(" ")

            EVENT_LIST.append([event_time,1, current_states])
            threshold_p = current_states + theta
            threshold_n = current_states - theta
            print('event_time: {} \ncurrent {:01.4f} ref_period: {:01.4f}'.format(event_time, current_states,refract_period))
            print('th_p: {:01.4f} th_n: {:01.4f}'.format(threshold_p, threshold_n))

        # Case for decreasing brightness
        elif threshold_n > testVector[n] and slope < 0 :

            # image_out[x,y,n,1] = 0.0
            # Linear estimate of threshold crossing instance
            dt = abs((current_states - theta - testVector[n-1])/slope)
            print('\ndt: {} theta: {} slope: {}'.format(dt,theta,slope))

            # This section calculates the registration of the event depending on the pixels refractory period
            if refract_period > time + dt :
                event_time = refract_period
                # Update pixel state and set new end for refractory period
                current_states = testVector[n-1] + slope*(event_time-time)
                refract_period = event_time + latency
                print("\nRP limited")
            else:
                event_time = ceil(time + dt)
                # Update pixel state and set new end for refractory period
                current_states = current_states - theta
                refract_period = event_time + latency
                print(" ")

            EVENT_LIST.append([event_time,-1, current_states])
            threshold_p = current_states + theta
            threshold_n = current_states - theta
            print('event_time: {} \ncurrent {:01.4f} ref_period: {:01.4f}'.format(event_time, current_states,refract_period))
            print('th_p: {:01.4f} th_n: {:01.4f}'.format(threshold_p, threshold_n))

        else:
            print("broke")
            break

    print("\n_________")
    print(refract_period > (time+T))
    print(threshold_p < testVector[n])
    print("_________\n")

    print("\n_________")
    print(refract_period > (time+T))
    print(threshold_n > testVector[n])
    print("_________\n")

    if quick_burst >= 0:
        if ((refract_period >= (time+T)) and (threshold_p < testVector[n])) :
            quick_burst = 1
        elif ((refract_period >= (time+T)) and (threshold_n > testVector[n])) :
            quick_burst = 2

    # Update time
    time = time + T

print('\n\nMetod 2 runtine: {:5.0f} [ms]\n\n'.format(1000*(tm.time() - start_time)))

if len(EVENT_LIST) == 0:
    print("no events found\n\n")
else:

    eventList = np.ones([len(EVENT_LIST),3])

    for n in range(len(EVENT_LIST)):
        eventList[n,0] = EVENT_LIST[n][0]   # Event Time
        eventList[n,1] = EVENT_LIST[n][1]   # Polarity
        eventList[n,2] = EVENT_LIST[n][2]   # Differentiator
        print('{:4.1f} {:4.1f} {:1.3f}'.format(eventList[n,0],eventList[n,1],eventList[n,2]))

        eventInt = eventList
        eventInt[0,1] = testVector[0] + eventList[0,1]*theta

        # Integrate the event list
    for n in range(1,len(EVENT_LIST)):
        eventInt[n,1] = eventInt[n-1,1] + eventList[n,1]*theta


##########################################################
################## Event list reading ####################
##########################################################
try:
    file_location = "frames/" + file + "/eventlist.txt"

    EVENT_FILE = open(file_location, 'r')

    # Extract meta data about file (for now only period)
    T = parse.search("T: {:d};", EVENT_FILE.readline())[0]

    time = 0
    events_from_list_t = []
    events_from_list_p = []

    # This will scan entire file
    for event in EVENT_FILE:

        x, y, t, p = parse.search("x: {:d}; y: {:d}; t: {:d}; p: {:d}",event)

        # NOTE: The loop below only works because the event_list is already sorted chronologically by frames.

        # If still within this period then stack events...
        if x==j and y==k:
            events_from_list_t.append([t])
            events_from_list_p.append([p])
        # ...else rescale frame to [0:1], push to list, update timer and reset.

    EVENT_FILE.close()
    plot_from_list = 1

except Exception as e:
    print("Reading event list failed")
    print(traceback.format_exec() + '\n')
    plot_from_list = 0


##########################################################
######################## Plotting ########################
##########################################################

fig, ax = plt.subplots()

# ax[1].scatter(range(current_states.shape[0]),testVector - current_states)

line1 = ax.scatter(np.linspace(0,T*(testVector.shape[0]-1),testVector.shape[0]),testVector, color='r')
line1.set_label('Illumination')

line1 = ax.scatter(eventInt[:,0], eventInt[:,1], color='b')
line1.set_label("Integrated events")

line3 = ax.scatter(eventInt[:,0], eventInt[:,2], color='orange')
line3.set_label("Differentiator state")

if plot_from_list == 1:

    myline = np.mean(testVector)

    line4 = ax.scatter(events_from_list_t, myline + events_from_list_p, color='g')
    line4.set_label("Events from list")

ax.grid(True)
ax.legend()

plt.show()
