import numpy as np
import parse


# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Dir to event list
file = "testLogged1"
file_location = "frames/" + file + "/eventlist.txt"

EVENT_FILE = open(file_location, 'r')

# Extract meta data about file (for now only period)
T = parse.search("T: {:d};", EVENT_FILE.readline())[0]
x_size = 128
y_size = 128

time = 0
FRAME_LIST = []

frame = np.zeros((128,128)) # repplace with metadata read

# while event = EVENT_FILE.readline() :

for event in EVENT_FILE:

    x, y, t, p = parse.search("x: {:d}; y: {:d}; t: {:d}; p: {:d}",event)

    # NOTE: The loop below only works because the event_list is already sorted chronologically by frames.

    # If still within this period then stack events...
    if t < time + T :
        frame[x,y] = frame[x,y] + p
    # ...else rescale frame to [0:1], push to list, update timer and reset.
    else:
        print('TIME: {:d} EVENTS: {:d}'.format(time, np.count_nonzero(np.sign(frame))))
        frame = (np.sign(frame) + 1)/2
        FRAME_LIST.append(frame)
        frame = np.zeros((x_size,y_size))
        time = time + T

EVENT_FILE.close()


##########################################################
######################## Plotting ########################
##########################################################


# Create color map
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> G -> B
n_bins = [0.25, 0.75]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)


fig, ax = plt.subplots()

ims = []

for n in range(len(FRAME_LIST)) :
    im = plt.imshow(FRAME_LIST[n], animated = True,vmin=0,vmax=1, cmap=cm)
    ims.append([im])

# Add legend
legend_elements = [Line2D([0],[0],marker='o',color='w',label='ON event',markerfacecolor='r',markersize=5), Line2D([0],[0],marker='o',color='w',label='OFF event',markerfacecolor='b',markersize=5)]
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),handles=legend_elements, loc = 'upper center', ncol=2)


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)

plt.show()
# PPM.displaySingle(FRAME_LIST)
