# Math imports
from math import pi
from math import sqrt
from math import cos
from math import sin

# Numpy imports
import numpy as np
from numpy import array
from numpy import matmul

# Matplotlib imports
import matplotlib.pyplot as plt


################################
######## Traj functions ########
################################

def linearTraj(R,x0,y0):

    R0 = x0*x0 + y0*y0
    b = (-1)*2*R*R*x0
    c = R**4 - R**2 * y0**2

    xf = ( -b + sqrt( b**2 - 4*R0*c) ) / (2*R0)

    b = (-1)*2*R*R*y0
    c = R**4 - R**2 * x0**2

    yf = ( -b + sqrt( b**2 - 4*R0*c) ) / (2*R0)

    return xf,yf

xf,yf = linearTraj(2,5,3)

print('xf: {}, yf: {}'.format(xf,yf))

# # Variables
# R = 900
# angle = 120
# step  = 3
#
# # Rotation mat:
# # cos(x) -sin(x)
# # sin(x)  cos(x)
#
# # Initial point and preallocate vector
# x0 = array([[800],[0]])
# x = np.zeros([2,int(angle/step+1)])
#
# for n in range(int(angle/step+1)):
#     rot = array([[cos(n*step*2*pi/360),-sin(n*step*2*pi/360)],[sin(n*step*2*pi/360),cos(n*step*2*pi/360)]])
#     a = np.transpose(matmul(rot,x0))
#     x[:,n] = a
#     # print(x[:,n].shape)
#     # print(a.shape)
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























# Placeholder
