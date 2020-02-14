# Math imports
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

def linearTraj(R,x0,y0,v_intend):

    R0 = x0*x0 + y0*y0
    b = (-1)*2*R*R*x0
    c = R**4 - R**2 * y0**2

    xf = ( -b + sqrt( b**2 - 4*R0*c) ) / (2*R0)

    b = (-1)*2*R*R*y0
    c = R**4 - R**2 * x0**2

    yf = ( -b + sqrt( b**2 - 4*R0*c) ) / (2*R0)

    m = (yf-y0)/(xf-x0)
    b = y0 - m * x0
    yaw = 90 - 360*atan(m)/(2*pi)

    t_norm = v_intend / sqrt(1+m**2)

    # print('x: {:6.2f}, y: {:6.2f}, t*: {:6.4f}, m: {:6.6f}'.format(xf, yf, t_norm, yaw))

    if xf-x0 > 0:
        x = np.arange(x0,xf+t_norm,t_norm)
    else:
        x = np.arange(xf-t_norm,x0,t_norm)
        m = m*(-1)
        x = x + (x0 - x[x.shape[0]-1])
        x = np.flip(x,0)


    if yf-y0 > 0:
        y = np.arange(y0,yf+t_norm*m,t_norm*m)
    else:
        m = m*(-1)
        y = np.arange(yf-t_norm*m,y0,t_norm*m)
        y = y + (y0 - y[y.shape[0]-1])
        y = np.flip(y,0)

    return yaw,x,y

def rotationTraj():

    # Variables
    R = 900
    angle = 120
    step  = 3

    # Rotation mat:
    # cos(x) -sin(x)
    # sin(x)  cos(x)

    # Initial point and preallocate vector
    x0 = array([[800],[0]])
    x = np.zeros([2,int(angle/step+1)])

    for n in range(int(angle/step+1)):
        rot = array([[cos(n*step*2*pi/360),-sin(n*step*2*pi/360)],[sin(n*step*2*pi/360),cos(n*step*2*pi/360)]])
        a = np.transpose(matmul(rot,x0))
        x[:,n] = a
        # print(x[:,n].shape)
        # print(a.shape)

    return x,y


################################
######### Main program #########
################################

R  = 1.5 * 1000        # How close do we fly-by
x0 = 10 * 1000   # Starting x
y0 = -5 * 1000  # Starting y
frame_rate = 500   # How many frames per second
v = 7 * 1000     # Speed in asteroid coordinate frame


yaw1,x1,y1 = linearTraj(R,x0,y0,v/frame_rate)

# for n in range(x1.shape[0]):
#     print('x: {:2.2f}, y: {:2.2f}'.format(x1[n], y1[n]))

print('Duration: {}'.format(x1.shape[0]/frame_rate))
print('Number of points: {}'.format(x1.shape[0]))

# plt.scatter(x1,y1)
# plt.show()


# Save to file
flight_file = open("test_traj.fli","w")

flight_file.write('# Start in craft (yaw/pitch/roll) mode.\n\r')
flight_file.write('view craft\n\r\n\r')


for m in range(x1.shape[0]):
    # flight_file.write('start 0 {0:4.2f} {1:4.2f} 0 -90 0\n\r'.format(x[0,m], x[1,m]) )
    flight_file.write( 'start {0:3.2f} '.format(x1[m]))
    flight_file.write( '{0:4.2f} 0 '.format(y1[m]))
    flight_file.write( '{0:4.2f} 0 0\n\r'.format(60) )



# for m in range(300):
#     flight_file.write("start 0 9999 9999 0 -90 0\n" )

#
# plt.scatter(x[0],x[1])
# plt.scatter(0,0)
# plt.grid(True)
#
# plt.axes().set_aspect('equal', 'datalim')
# plt.show()
#
# # Save to file
# flight_file = open("test_traj.fli","w")
#
# flight_file.write('# Start in craft (yaw/pitch/roll) mode.\n\r')
# flight_file.write('view model\n\r\n\r')
#
# for m in range(120):
#     # flight_file.write('start 0 {0:4.2f} {1:4.2f} 0 -90 0\n\r'.format(x[0,m], x[1,m]) )
#     flight_file.write('start 0 0 0 1500 {0:4.2f} 0\n\r'.format(m) )
#






















# Placeholder
