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
import parse
import time as tm
import importlib
import os

# This is the actualy file being tested
import event_creation

class Pixel:

    def __init__(self, init_ilum, quick_burst=False):

        # Pixel parameters
        self.theta = 0.2
        self.T = 50
        self.latency = 15
        self.refract_period = 0
        self.time = 0
        self.quick_burst = 0

        # Ilumination and Differentiator state
        self.previous_ilum = init_ilum
        self.current_states = init_ilum
        self.threshold_p = self.current_states + self.theta
        self.threshold_n = self.current_states - self.theta

    def __del__(self):
        pass

    def Update(self, ilum):

        # This sholdn't be here, but I can't be bothered
        testVector = [self.previous_ilum, ilum]
        n = 1
        EVENT_LIST = []

        delta = testVector[n] - testVector[n-1]
        # threshold = current_states * theta        # Dynamic threshold

        # Slope used for piecewise linear interpolation of pixel intensity
        slope = (testVector[n] - testVector[n-1])/self.T

        print('\ntime: {} \ncurrent {:01.4f} illum: {:01.4f}'.format(self.time, self.current_states,testVector[n]))

        if self.quick_burst > 0 :
            event_time = self.refract_period
            # Update pixel state and set new end for refractory period
            self.current_states = testVector[n-1] + slope*(event_time-self.time)
            self.refract_period = event_time + self.latency
            if self.quick_burst == 1 :
                EVENT_LIST.append([event_time,1, self.current_states])
                print("\nQuick Burst: 1")
            if self.quick_burst == 2 :
                EVENT_LIST.append([event_time,-1, self.current_states])
                print("\nQuick Burst: -1")
            self.quick_burst = 0

        # Keeps looking for events until pixel refrect period extends into next frame
        while(self.refract_period < (self.time+self.T)):
            # Case for increasing brightness
            print("ref_perdiod: ", self.refract_period)
            if self.threshold_p < testVector[n] and slope > 0 :

                # Linear estimate of threshold crossing instance
                dt = abs((self.current_states + self.theta  - testVector[n-1])/slope)
                print('\ndt: {} theta: {} slope: {}'.format(dt,self.theta,slope))

                # This section calculates the registration of the event depending on the pixels refractory period
                if self.refract_period > self.time + dt :
                    event_time = self.refract_period
                    # Update pixel state and set new end for refractory period
                    self.current_states = testVector[n-1] + slope*(event_time-self.time)
                    self.refract_period = event_time + self.latency
                    print("\nRP limited")
                else:
                    event_time = ceil(self.time + dt)
                    # Update pixel state and set new end for refractory period
                    self.current_states = self.current_states + self.theta
                    self.refract_period = event_time + self.latency
                    print(" ")

                EVENT_LIST.append([event_time,1, self.current_states])
                self.threshold_p = self.current_states + self.theta
                self.threshold_n = self.current_states - self.theta
                print('event_time: {} \ncurrent {:01.4f} ref_period: {:01.4f}'.format(event_time, self.current_states,self.refract_period))
                print('th_p: {:01.4f} th_n: {:01.4f}'.format(self.threshold_p, self.threshold_n))

            # Case for decreasing brightness
            elif self.threshold_n > testVector[n] and slope < 0 :

                # Linear estimate of threshold crossing instance
                dt = abs((self.current_states - self.theta - testVector[n-1])/slope)
                print('\ndt: {} theta: {} slope: {}'.format(dt,self.theta,slope))

                # This section calculates the registration of the event depending on the pixels refractory period
                if self.refract_period >self. time + dt :
                    event_time = self.refract_period
                    # Update pixel state and set new end for refractory period
                    self.current_states = testVector[n-1] + slope*(event_time-self.time)
                    self.refract_period = event_time + self.latency
                    print("\nRP limited")
                else:
                    event_time = ceil(self.time + dt)
                    # Update pixel state and set new end for refractory period
                    self.current_states = self.current_states - self.theta
                    self.refract_period = event_time + self.latency
                    print(" ")

                EVENT_LIST.append([event_time,-1, self.current_states])
                self.threshold_p = self.current_states + self.theta
                self.threshold_n = self.current_states - self.theta
                print('event_time: {} \ncurrent {:01.4f} ref_period: {:01.4f}'.format(event_time, self.current_states,self.refract_period))
                print('th_p: {:01.4f} th_n: {:01.4f}'.format(self.threshold_p, self.threshold_n))

            else:
                print("broke")
                break

        print("\n_________")
        print(self.refract_period > (self.time+self.T))
        print(self.threshold_p < testVector[n])
        print("_________\n")

        print("\n_________")
        print(self.refract_period > (self.time+self.T))
        print(self.threshold_n > testVector[n])
        print("_________\n")

        if ((self.refract_period > (self.time+self.T)) and (self.threshold_p < testVector[n])) :
            self.quick_burst = 1
        elif ((self.refract_period > (self.time+self.T)) and (self.threshold_n > testVector[n])) :
            self.quick_burst = 2

        # Update time
        self.time = self.time + self.T
        self.previous_ilum = ilum

        return(EVENT_LIST)


##########################################################################
######################## Code using Pixel object #########################
##########################################################################

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
testVector = np.log10(raw_images[70,16,20:40])
# testVector = raw_images[70,16,:]

# Method 1

start_time = tm.time()

pixel = Pixel(testVector[0])
EVENT_LIST = []

for n in range(1,testVector.shape[0]):
    events = pixel.Update(testVector[n])

    for x in events:
        EVENT_LIST.append(x)

print('\n\nMetod 2 runtine: {:5.0f} [ms]\n\n'.format(1000*(tm.time() - start_time)))

#########################################################################
############################# Plotting code #############################
#########################################################################

# Repacking data

eventList = np.ones([len(EVENT_LIST),3])

for n in range(len(EVENT_LIST)):
    eventList[n,0] = EVENT_LIST[n][0]
    eventList[n,1] = EVENT_LIST[n][1]
    eventList[n,2] = EVENT_LIST[n][2]
    print(eventList[n,:])

eventInt = eventList
eventInt[0,1] = testVector[0] + eventList[0,1]*pixel.theta

# Integrate the event list
for n in range(1,len(EVENT_LIST)):
    eventInt[n,1] = eventInt[n-1,1] + eventList[n,1]*pixel.theta

# Plotting
fig, ax = plt.subplots()

line1 = ax.scatter(np.linspace(0,pixel.T*(testVector.shape[0]-1),testVector.shape[0]),testVector, color='r')
line1.set_label('Illumination')

line1 = ax.scatter(eventInt[:,0], eventInt[:,1])
line1.set_label("Integrated events")

line3 = ax.scatter(eventInt[:,0], eventInt[:,2])
line3.set_label("Differentiator state")

ax.grid(True)
ax.legend()
plt.show()
