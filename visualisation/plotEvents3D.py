

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import parse

# test_tag = "/constDescent6"
# str_address = "C:/PANGU/PANGU_5.00/models/lunar_OFlanding/frames" + test_tag + "/eventlist_050.txt"
# # Decimation factor
# dec_fac = 2

test_tag = "/sineHover2_ABR"
str_address = "C:/PANGU/PANGU_5.00/models/lunar_OFlanding/frames" + test_tag + "/eventlist.txt"
# Decimation factor
dec_fac = 2

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

# data[:, 2] = data[:, 2]/1000

print('This data set contains {} data points'.format(data.shape[0]))


# Select section
beg = 0
end = 250000
subset = data[np.where((data[:, 2] < end) & (data[:, 2] > beg))]

# Split according to polarity
subset_neg = subset[subset[:, 3] == -1]
subset_pos = subset[subset[:, 3] == 1]

# Plot
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(subset_neg[::dec_fac, 0], subset_neg[::dec_fac, 1], subset_neg[::dec_fac, 2], s=1)


ax.set_xlabel('pixel x-coordinate', fontsize='x-large', fontweight='bold')
ax.set_ylabel('pixel y-coordinate', fontsize='x-large', fontweight='bold')
ax.set_zlabel('time [s]', fontsize='x-large', fontweight='bold')

plt.show()