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

# This is the actualy file being tested
import event_creation

x_size = 128
y_size = 128
frames = 60


raw_images = np.zeros((x_size,y_size,frames), dtype='float32')

log_file = open("frames/testLogged1/" + "log.txt" , "r")
log_string = log_file.read()

GAIN1 = parse.search("GAIN1: {:d};", log_string)[0]
GAIN2 = parse.search("GAIN2: {:d};", log_string)[0]

log_file.close()

raw_images = event_creation.convertFromCompound("testLogged1",x_size,y_size,frames)
testVector = np.log10(raw_images[70,14,10:30])

# Method 2

start_time = tm.time()

theta = 0.2
T = 50
latency = 15
refract_period = 0
time = 0

quick_burst = 0
current_states = testVector[0]
EVENT_LIST = []


threshold_p = current_states + theta   # Constant threshold
threshold_n = current_states - theta   # Constant threshold

for n in range(1,testVector.shape[0]):

    delta = testVector[n] - testVector[n-1]
    # threshold = current_states * theta        # Dynamic threshold

    # Slope used for piecewise linear interpolation of pixel intensity
    slope = (testVector[n] - testVector[n-1])/T
    print('\ntime: {} \ncurrent {:01.4f} illum: {:01.4f}'.format(time, current_states,testVector[n]))

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

    # Keeps looking for events until pixel refrect period extends into next frame
    while(refract_period < (time+T)):
        # Case for increasing brightness
        print("ref_perdiod: ", refract_period)
        if threshold_p < testVector[n] and slope > 0 :

            # Dummy value event visualisation, might delete later
            # image_out[x,y,n,1] = 1.0
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
        if ((refract_period > (time+T)) and (threshold_p < testVector[n])) :
            quick_burst = 1
        elif ((refract_period > (time+T)) and (threshold_n > testVector[n])) :
            quick_burst = 2

    # Update time
    time = time + T

print('\n\nMetod 2 runtine: {:5.0f} [ms]\n\n'.format(1000*(tm.time() - start_time)))

if len(EVENT_LIST) == 0:
    print("no events\n\n")
else:

    eventList = np.ones([len(EVENT_LIST),3])

    for n in range(len(EVENT_LIST)):
        eventList[n,0] = EVENT_LIST[n][0]   # Event Time
        eventList[n,1] = EVENT_LIST[n][1]   # Polarity
        eventList[n,2] = EVENT_LIST[n][2]   # Differentiator
        print('{:4.1f} {:4.1f} {:1.3f}'.format(eventList[n,0] + 1029,eventList[n,1],eventList[n,2]))

        eventInt = eventList
        eventInt[0,1] = testVector[0] + eventList[0,1]*theta

        # Integrate the event list
    for n in range(1,len(EVENT_LIST)):
        eventInt[n,1] = eventInt[n-1,1] + eventList[n,1]*theta



    fig, ax = plt.subplots()

    # ax[1].scatter(range(current_states.shape[0]),testVector - current_states)

    line1 = ax.scatter(np.linspace(0,T*(testVector.shape[0]-1),testVector.shape[0]),testVector, color='r')
    line1.set_label('Illumination')

    line1 = ax.scatter(eventInt[:,0], eventInt[:,1])
    line1.set_label("Integrated events")


    # ax3 = plt.subplot(212,sharex=ax1, sharey=ax1)
    line3 = ax.scatter(eventInt[:,0], eventInt[:,2])
    line3.set_label("Differentiator state")
    ax.grid(True)

    ax.legend()

    plt.show()
