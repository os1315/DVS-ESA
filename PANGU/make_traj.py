# Math imports
import math
from math import pi
from math import sqrt
from math import cos
from math import sin
from math import atan

# Numpy imports
import numpy as np

# Matplotlib imports
import matplotlib.pyplot as plt


################################
######## Traj functions ########
################################

# Notes:
#   1. All values in bare SI (m,s,etc.)

# Returns a est of coordinates for location at each time step
# Should be changed to return all coordinates
def linearFlyby(frame_rate):
    R = 1.5 * 1000  # How close do we fly-by
    x0 = 10 * 1000  # Starting x
    y0 = -5 * 1000  # Starting y
    v = 7 * 1000  # Speed in asteroid coordinate frame

    v_intend = v / frame_rate

    R0 = x0 * x0 + y0 * y0
    b = (-1) * 2 * R * R * x0
    c = R ** 4 - R ** 2 * y0 ** 2

    xf = (-b + sqrt(b ** 2 - 4 * R0 * c)) / (2 * R0)

    b = (-1) * 2 * R * R * y0
    c = R ** 4 - R ** 2 * x0 ** 2

    yf = (-b + sqrt(b ** 2 - 4 * R0 * c)) / (2 * R0)

    m = (yf - y0) / (xf - x0)
    b = y0 - m * x0
    yaw = 90 - 360 * atan(m) / (2 * pi)

    t_norm = v_intend / sqrt(1 + m ** 2)

    # print('x: {:6.2f}, y: {:6.2f}, t*: {:6.4f}, m: {:6.6f}'.format(xf, yf, t_norm, yaw))

    if xf - x0 > 0:
        x = np.arange(x0, xf + t_norm, t_norm)
    else:
        x = np.arange(xf - t_norm, x0, t_norm)
        m = m * (-1)
        x = x + (x0 - x[x.shape[0] - 1])
        x = np.flip(x, 0)

    if yf - y0 > 0:
        y = np.arange(y0, yf + t_norm * m, t_norm * m)
    else:
        m = m * (-1)
        y = np.arange(yf - t_norm * m, y0, t_norm * m)
        y = y + (y0 - y[y.shape[0] - 1])
        y = np.flip(y, 0)

    return yaw, x, y


def rotationTraj():
    # Variables
    R = 900
    angle = 120
    step = 3

    # Rotation mat:
    # cos(x) -sin(x)
    # sin(x)  cos(x)

    # Initial point and preallocate vector
    x0 = np.array([[800], [0]])
    x = np.zeros([2, int(angle / step + 1)])

    for n in range(int(angle / step + 1)):
        rot = np.array([[cos(n * step * 2 * pi / 360), -sin(n * step * 2 * pi / 360)],
                        [sin(n * step * 2 * pi / 360), cos(n * step * 2 * pi / 360)]])
        a = np.transpose(np.matmul(rot, x0))
        x[:, n] = a
        # print(x[:,n].shape)
        # print(a.shape)

    return x


def linearDescent(frame_rate):
    # Tweaked values
    init_h = 3000
    end_h = 500
    t_total = 250
    v_avg = (end_h - init_h) / t_total
    steps = t_total * frame_rate + 1

    # Default values
    x0 = 0
    y0 = 0
    yaw0 = 0
    pitch0 = -90
    roll0 = 0

    coord_set = np.zeros([steps, 6])
    print('v_avg: {} m/s'.format(v_avg))
    coord_set[0, :] = [x0, y0, init_h, yaw0, pitch0, roll0]

    for n in range(1, steps):
        # indexes = [z, y, x , yaw, pitch, roll]
        z_next = coord_set[n - 1, 2] + v_avg / frame_rate
        coord_set[n, :] = [x0, y0, z_next, yaw0, pitch0, roll0]

    return coord_set


def sineDivergence(frame_rate):
    # Tweaked values
    z_init = 1000
    D_amp = 0.08
    D_w = math.pi/10
    t_total = 100

    # Default values
    x0 = 0
    y0 = 0
    yaw0 = 0
    pitch0 = -90
    roll0 = 0

    t = np.arange(0, t_total, 1/frame_rate)
    D = -D_amp*np.sin(D_w*t)

    z = [z_init]
    for n in range(D.shape[0]):
        z.append(z[-1]/(1 - D[n]*1/frame_rate))

    z = np.array(z)

    coord_set = np.zeros([z.shape[0], 6])
    coord_set[:, 0] = x0
    coord_set[:, 1] = y0
    coord_set[:, 2] = z
    coord_set[:, 3] = yaw0
    coord_set[:, 4] = pitch0
    coord_set[:, 5] = roll0

    return coord_set


################################
######### Main program #########
################################

FRAME_RATE = 100

points = sineDivergence(FRAME_RATE)

print('Duration: {}'.format(points.shape[0] / FRAME_RATE))
print('Number of points: {}'.format(points.shape[0]))

# Save to file
flight_file = open("test_traj.fli", "w")

flight_file.write('# Frame rate: {}\n\r\n\r'.format(FRAME_RATE))
flight_file.write('# Start in craft (yaw/pitch/roll) mode.\n\r')
flight_file.write('view craft\n\r')

for m in range(points.shape[0]):
    output_string = 'start {:4.2f} {:4.2f} {:4.2f} {:4.2f} {:4.2f} {:4.2f} \n'.format(points[m, 0], points[m, 1],
                                                                                      points[m, 2], points[m, 3],
                                                                                      points[m, 4], points[m, 5])
    flight_file.write(output_string)
