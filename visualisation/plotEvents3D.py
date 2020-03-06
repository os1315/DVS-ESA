

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import parse

test_tag = "/constDescent4"

str_address = "C:/PANGU/PANGU_5.00/models/lunar_OFlanding/frames" + test_tag + "/eventlist.txt"
# str_address = "test.txt"

event_file = open(str_address, 'r')

# First line is not a point
str_event = event_file.readline()

all_packets = []

while True:
    str_event = event_file.readline()
    packet = parse.parse("x: {:d}; y: {:d}; t: {:d}; p: {:d}", str_event)

    # This is really annoying, but it returns this ".Results" object that I have to unpack
    if packet is not None:
        x = packet[0]
        y = packet[1]
        t = packet[2]
        p = packet[3]
        all_packets.append([x, y, t, p])
    else:
        break

data = np.array(all_packets)

data[:, 2] = data[:, 2]/1000

print('This data set contains {} data points'.format(data.shape[0]))


# Select section
beg = 0
end = 2000000
subset = data[np.where((data[:, 2] < end) & (data[:, 2] > beg))]

# Split according to polarity
subset_neg = subset[subset[:, 3] == -1]
subset_pos = subset[subset[:, 3] == 1]

# Decimation factor
dec_fac = 1

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(subset_neg[::dec_fac, 0], subset_neg[::dec_fac, 1], subset_neg[::dec_fac, 2])
ax.scatter(subset_pos[::dec_fac, 0], subset_pos[::dec_fac, 1], subset_pos[::dec_fac, 2])


ax.set_xlabel('x pixel')
ax.set_ylabel('y pixel')
ax.set_zlabel('time [ms]')

plt.show()


# with open(str_address, 'r') as event_file:
#     str_event = event_file.readline()
#     print(str_event)
#     packet = parse.parse("x: {}; y: {}; t: {}; p: {}", str_event)
#     print(packet)