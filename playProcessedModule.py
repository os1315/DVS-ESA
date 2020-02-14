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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Replace this with a lambda later

def animateTwo(j,eve_images,raw_images, im):

    new_im = []
    new_im.append(im[0].set_array(eve_images[:,:,j]))
    new_im.append(im[1].set_array(raw_images[:,:,j]))

    return new_im

def animateMore(j, fig, eve_images,raw_images, im,extra_images):

    new_im = []
    new_im.append(im[0].set_array(eve_images[:,:,j]))
    new_im.append(im[1].set_array(raw_images[:,:,j]))
    # new_im.append(im[2].set_array(extra_images[:,:,j,1]))
    new_im.append(im[2].set_array(extra_images[:,:,j]))

    fig.suptitle('Frame:{:d}'.format(j))

    return new_im

def playProcessed(prc_images, frame_count):

    # How many plots
    plot_count = prc_images.shape[3]

    if (plot_count < 3):
        displayTwo(prc_images, frame_count)
    elif (plot_count == 3):
        displayThree(prc_images, frame_count)
    elif (plot_count == 4):
        displayFour(prc_images, frame_count)
    elif (plot_count == 5):
        displayFive(prc_images, frame_count)
    elif (plot_count == 6):
        displaySix(prc_images, frame_count)

#  This is fucking shit and I'm sick and tired of it
def displaySingle(eve_images):

    # Create color map
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> G -> B
    n_bins = [0.25, 0.75]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

    ims = []

    for im in eve_images:
        # ims.append([plt.imshow(im, animated = True)])
        image = plt.imshow(im, animated = True,vmin=0,vmax=1, cmap=cm)
        ims.append([image])

    print(len(ims))

    # Initialize plot and image artist list
    fig1= plt.figure()

    #  First subplot
    # event_ax.imshow(eve_images[:,:,0],vmin=0,vmax=1, cmap=cm)
    # legend_elements = [Line2D([0],[0],marker='o',color='w',label='ON event',markerfacecolor='r',markersize=5), Line2D([0],[0],marker='o',color='w',label='OFF event',markerfacecolor='b',markersize=5)]
    # event_ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),handles=legend_elements, loc = 'upper center', ncol=2)

    #  Colorbar
    im_ani = animation.FuncAnimation(fig1,ims,interval=500, blit=True,repeat_delay=1)

    plt.show()

def displayTwo(prc_images, frame_count):

    # Create color map
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> G -> B
    n_bins = [0.25, 0.75]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    # How many plots
    plot_count = prc_images.shape[3]

    # Subset of plot
    up = prc_images.shape[0]-1
    down = 0
    # up = 180
    # down = 120

    # Regroup images (should be deleted)
    raw_images = prc_images[down:up,down:up,:,0]
    eve_images = prc_images[down:up,down:up,:,1]

    # Initialize plot and image artist list
    fig1, axes = plt.subplots(1,plot_count)
    event_ax = axes[0]
    raw_ax = axes[1]

    im = []

    #  First subplot
    im.append(event_ax.imshow(eve_images[:,:,0],vmin=0,vmax=1, cmap=cm))
    legend_elements = [Line2D([0],[0],marker='o',color='w',label='ON event',markerfacecolor='r',markersize=5), Line2D([0],[0],marker='o',color='w',label='OFF event',markerfacecolor='b',markersize=5)]
    event_ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),handles=legend_elements, loc = 'upper center', ncol=2)


    # Second subplot
    im.append(raw_ax.imshow(raw_images[:,:,0],vmin=0,vmax=1, cmap='gray'))
    divider = make_axes_locatable(raw_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[1], cax=cax, orientation='vertical')

    #  Colorbar
    im_ani = animation.FuncAnimation(fig1,lambda j: animateTwo(j,eve_images,raw_images,im), frames=range(frame_count),interval=2, repeat_delay=3000)

    plt.show()

def displayThree(prc_images, frame_count):

    # Create color map
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> G -> B
    n_bins = [0.25, 0.75]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    # How many plots
    plot_count = prc_images.shape[3]

    # Subset of plot
    up = prc_images.shape[0]-1
    down = 0
    # up = 180
    # down = 120

    # Regroup images
    raw_images = prc_images[down:up,down:up,:,0]
    eve_images = prc_images[down:up,down:up,:,1]
    extra_images = prc_images[down:up,down:up,:,2]

    # Initialize plot and image artist list
    fig1, axes = plt.subplots(1,3)
    fig1.suptitle("Frame: 1")
    event_ax = axes[0]
    raw_ax = axes[1]


    im = []

    #  First subplot
    im.append(event_ax.imshow(eve_images[:,:,0],vmin=0,vmax=1, cmap=cm))
    event_ax.title.set_text('Event data')
    legend_elements = [Line2D([0],[0],marker='o',color='w',label='ON event',markerfacecolor='r',markersize=5), Line2D([0],[0],marker='o',color='w',label='OFF event',markerfacecolor='b',markersize=5)]
    event_ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),handles=legend_elements, loc = 'upper center', ncol=2)

    # Second subplot
    im.append(raw_ax.imshow(raw_images[:,:,0],vmin=0,vmax=1, cmap='gray'))
    divider = make_axes_locatable(raw_ax)
    raw_ax.title.set_text('Raw data')
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[1], cax=cax, orientation='vertical')

    # Third subplot
    im.append(axes[2].imshow(extra_images[:,:,0],vmin=0,vmax=-8))
    divider = make_axes_locatable(axes[2])
    axes[2].title.set_text('Log10 of raw data')     # Ok, this is not a good move if I ever want to automatically slace for more plots
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[2], cax=cax, orientation='vertical')

    #  Colorbar
    im_ani = animation.FuncAnimation(fig1,lambda j: animateMore(j,fig1,eve_images,raw_images,im,extra_images), frames=range(frame_count),interval=50, repeat_delay=3000)

    plt.show()


def displayFour(prc_images, frame_count):

    # Create color map
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> G -> B
    n_bins = [0.25, 0.75]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    # How many plots
    plot_count = prc_images.shape[3]

    # Subset of plot
    up = prc_images.shape[0]-1
    down = 0
    # up = 180
    # down = 120

    # Regroup images
    raw_images = prc_images[down:up,down:up,:,0]
    eve_images = prc_images[down:up,down:up,:,1]
    extra_images = prc_images[down:up,down:up,:,2:]

    # Initialize plot and image artist list
    fig1, axes = plt.subplots(2,2)
    fig1.suptitle("Frame: 1")
    event_ax = axes[0,0]
    raw_ax = axes[1,0]

    im = []

    #  First subplot
    im.append(event_ax.imshow(eve_images[:,:,0],vmin=0,vmax=1, cmap=cm))
    event_ax.title.set_text('Event data')
    legend_elements = [Line2D([0],[0],marker='o',color='w',label='ON event',markerfacecolor='r',markersize=5), Line2D([0],[0],marker='o',color='w',label='OFF event',markerfacecolor='b',markersize=5)]
    event_ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),handles=legend_elements, loc = 'upper center', ncol=2)

    # Second subplot
    im.append(raw_ax.imshow(raw_images[:,:,0],vmin=0,vmax=1, cmap='gray'))
    divider = make_axes_locatable(raw_ax)
    raw_ax.title.set_text('Raw data')
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[1], cax=cax, orientation='vertical')

    # Third subplot
    im.append(axes[0,1].imshow(extra_images[:,:,0,0],vmin=0,vmax=1, cmap=cm))
    divider = make_axes_locatable(axes[0,1])
    axes[0,1].title.set_text('Log10 of raw data')     # Ok, this is not a good move if I ever want to automatically slace for more plots
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[2], cax=cax, orientation='vertical')

    # Fourth subplot
    im.append(axes[1,1].imshow(extra_images[:,:,0,1],vmin=0,vmax=-8))
    divider = make_axes_locatable(axes[1,1])
    axes[1,1].title.set_text('Current stored state')     # Ok, this is not a good move if I ever want to automatically slace for more plots
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im[3], cax=cax, orientation='vertical')

    #  Colorbar
    im_ani = animation.FuncAnimation(fig1,lambda j: animateMore(j,fig1,eve_images,raw_images,im,extra_images), frames=range(frame_count),interval=100, repeat_delay=3000)

    plt.show()
