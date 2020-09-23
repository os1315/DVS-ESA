import os

from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.cbook import get_sample_data

import Filehandling
from EventDataHandlers import readEventList
from EventDataHandlers import splitByPolarity
from Filehandling import readinConfig

import scipy.ndimage as ndi

import DivergenceEstimator

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from matplotlib import cm


tar_dir = readinConfig()
data = readEventList(tar_dir + "/frames/sineHover2/eventlist_ABR.txt")

dec_fac = 1

print(f'This data set contains {data.shape[0]} data points')

# Select section
beg = 2000
end = 2500
subset = data[np.where((data[:, 2] < end) & (data[:, 2] > beg))]

# Split according to polarity
subset_neg = subset[subset[:, 3] == -1]
subset_neg[:, 2] = (subset_neg[:, 2] - 2000)/2 + 1600

# Create projections
# Init estimator
camera_dimensions = {
    'x_size': 128,
    'y_size': 128
}

estimator_settings = {
    'tau': 500,
    'min_points': 15,
    'min_features': 3,
    'r': 4,
    'centroid_seperation': 0.4,
    'time_dimension': 1
}
est = DivergenceEstimator.MeanShiftSingleEstimator(**camera_dimensions, **estimator_settings)

# Time_now = (2500-2000)/2 + 1600 = 1850
est.update(subset_neg, 1850)

img_raw = est.getStoredEventsProjection()
img_blurred = ndi.convolve(img_raw, est.convMask, mode='constant')
centroids = est.centroids_OLD

# temp = plt.figure()
# tempx = temp.add_subplot(1, 1, 1)
# tempx.imshow(img_blurred)
# for c in range(centroids.shape[0]):
#     p = Circle((centroids[c, 1], centroids[c, 0]), 1, color='r')
#     tempx.add_patch(p)

# Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Add cloud of points
ax.scatter(subset_neg[::dec_fac, 0], subset_neg[::dec_fac, 1], subset_neg[::dec_fac, 2], s=1)

# Plot image
# im = Image.fromarray(np.uint8(cm.gist_earth(img_raw)*255))
xx, yy = np.meshgrid(np.linspace(0,128,128), np.linspace(0,128,128))
ax.plot_surface(xx, yy, 1500*np.ones((128,128)), rstride=1, cstride=1, facecolors=cm.PiYG(img_raw/img_raw.max()))
ax.plot_surface(xx, yy, 800*np.ones((128,128)), rstride=1, cstride=1, facecolors=cm.PiYG(img_blurred/img_blurred.max()))
ax.plot_surface(xx, yy, np.zeros((128,128)), rstride=1, cstride=1, facecolors=cm.PiYG(img_blurred/img_blurred.max()))

# Add centroids
circle_color = 'b'
for c in range(4):
    p = Circle((centroids[c, 1], centroids[c, 0]), 2, color=circle_color)
    print(f'x:{centroids[c, 0]}, y:{centroids[c, 1]}')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=3, zdir="z")

p = Circle((109, 69), 2, color=circle_color)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=3, zdir="z")
p = Circle((17, 112), 2, color=circle_color)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=3, zdir="z")
p = Circle((117, 17), 2, color=circle_color)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=3, zdir="z")

ax.set_xlabel('pixel x-coordinate', fontsize='x-large', fontweight='bold')
ax.set_ylabel('pixel y-coordinate', fontsize='x-large', fontweight='bold')
ax.set_zlim([0,2000])

# Remove grid lines
ax.grid(False)
ax.set_zticks([])

# Transparent panes
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.view_init(azim=-59, elev=18)
fig.subplots_adjust(bottom=0.0, top=1.0)

plt.show()
