# # Use ODEINT to solve the differential equations defined by the vector field
# from scipy.integrate import odeint
#
#
# def vectorfield(w, t, p):
#     """
#     Defines the differential equations for the coupled spring-mass system.
#
#     Arguments:
#         w :  vector of the state variables:
#                   w = [x1,y1,x2,y2]
#         t :  time
#         p :  vector of the parameters:
#                   p = [m1,m2,k1,k2,L1,L2,b1,b2]
#     """
#     z, v, m, tau = w
#     c, g, Isp, Kp = p
#
#     # Create f = (x1',y1',x2',y2'):
#     f = [v,
#          Kp / m * (tau - z / v),
#          -Kp / Isp * (tau - z / v),
#          -c * c]
#
#     return f
#
#
# # Parameter values
# c = 0.5
# g = 1.623
# Isp = 300
# Kp = 100_000
#
# # Initial conditions
# z0 = 2000
# v0 = -50
# m0 = 10_000
# tau0 = -z0 / v0
#
# # ODE solver parameters
# abserr = 1.0e-8
# relerr = 1.0e-6
# stoptime = 10.0
# numpoints = 250
#
# # Create the time samples for the output of the ODE solver.
# # I use a large number of points, only because I want to make
# # a plot of the solution that looks nice.
# t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
#
# # Pack up the parameters and initial conditions:
# w0 = [z0, v0, m0, tau0]
# p = [c, g, Isp, Kp]
#
# # Call the ODE solver.
# wsol = odeint(vectorfield, w0, t, args=(p,),
#               atol=abserr, rtol=relerr)
#
# print("Printing...")
# for t1, w1 in zip(t, wsol):
#         print(f'{t1}, {w1[0]}, {w1[1]}, {w1[2]}, {w1[3]}')

# figure(1)
#
# xlabel('t')
# grid(True)
# hold(True)
# lw = 1
#
# plot(t, x1, 'b', linewidth=lw)
# plot(t, x2, 'g', linewidth=lw)

# zombie apocalypse modeling
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
#
#
# P = 0  # birth rate
# d = 0.0001  # natural death percent (per day)
# B = 0.0095  # transmission percent  (per day)
# G = 0.0001  # resurect percent (per day)
# A = 0.0001  # destroy percent  (per day)
#
#
# # solve the system dy/dt = f(y, t)
# def f(y, t):
#     Si = y[0]
#     Zi = y[1]
#     Ri = y[2]
#     # the model equations (see Munz et al. 2009)
#     f0 = P - B * Si * Zi - d * Si
#     f1 = B * Si * Zi + G * Ri - A * Si * Zi
#     f2 = d * Si + A * Si * Zi - G * Ri
#     return [f0, f1, f2]
#
#
# # initial conditions
# S0 = 500.  # initial population
# Z0 = 0  # initial zombie population
# R0 = 0  # initial death population
# y0 = [S0, Z0, R0]  # initial condition vector
# t = np.linspace(0, 5., 1000)  # time grid
#
# # solve the DEs
# soln = odeint(f, y0, t)
# S = soln[:, 0]
# Z = soln[:, 1]
# R = soln[:, 2]
#
# # plot results
# plt.figure()
# plt.plot(t, S, label='Living')
# plt.plot(t, Z, label='Zombies')
# plt.xlabel('Days from outbreak')
# plt.ylabel('Population')
# plt.title('Zombie Apocalypse - No Init. Dead Pop.; No New Births.')
# plt.legend(loc=0)
#
# # change the initial conditions
# R0 = 0.01 * S0  # 1% of initial pop is dead
# y0 = [S0, Z0, R0]
#
# # solve the DEs
# soln = odeint(f, y0, t)
# S = soln[:, 0]
# Z = soln[:, 1]
# R = soln[:, 2]
#
# plt.figure()
# plt.plot(t, S, label='Living')
# plt.plot(t, Z, label='Zombies')
# plt.xlabel('Days from outbreak')
# plt.ylabel('Population')
# plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; No New Births.')
# plt.legend(loc=0)
#
# # change the initial conditions
# R0 = 0.01 * S0  # 1% of initial pop is dead
# P = 10  # 10 new births daily
# y0 = [S0, Z0, R0]
#
# # solve the DEs
# soln = odeint(f, y0, t)
# S = soln[:, 0]
# Z = soln[:, 1]
# R = soln[:, 2]
#
# plt.figure()
# plt.plot(t, S, label='Living')
# plt.plot(t, Z, label='Zombies')
# plt.xlabel('Days from outbreak')
# plt.ylabel('Population')
# plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
# plt.legend(loc=0)
#
# plt.show()

# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

# by Andrew D. Straw

import numpy as np
import matplotlib.pyplot as plt
import pickle

filename = 'Dario_demo_unfilt'

with open('closed-loop-simulation/obj/' + filename + '.pkl', 'rb') as f:
    my_dict = pickle.load(f)

# Parse saved data. Settings are printed out, while trajectories are places extracted as dictionary.
file_name = my_dict['name']
print(f'\n== File: {file_name} ==')
for key, item in my_dict.items():
    if key != 'name' and key != 'flight_params':
        print(f'From tile: {key}')
        for subkey, subitem in item.items():
            print(f'\t{subkey}: {subitem}')

flight_params = my_dict['flight_params']

plt.rcParams['figure.figsize'] = (10, 8)

z = flight_params['D']

# intial parameters
n_iter = len(flight_params['time'])
sz = (n_iter,)

Q = 0.001  # process variance

# allocate space for arrays
xhat = np.zeros(sz)  # a posteri estimate of x
P = np.zeros(sz)  # a posteri error estimate
xhatminus = np.zeros(sz)  # a priori estimate of x
Pminus = np.zeros(sz)  # a priori error estimate
K = np.zeros(sz)  # gain or blending factor

R = 0.1 ** 1  # estimate of measurement variance, change to see effect

margin = 0.04

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1, n_iter):
    # time update
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q

    K[k] = Pminus[k] / (Pminus[k] + R)
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]

    # measurement update
    # if z[k] > xhat[k-1] + margin or z[k] < xhat[k-1] - margin:
    #     K[k] = Pminus[k] / (Pminus[k] + R)
    #     xhat[k] = xhat[k-1]
    #     P[k] = (1 - K[k]) * Pminus[k]
    # else:
    #     K[k] = Pminus[k] / (Pminus[k] + R)
    #     xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
    #     P[k] = (1 - K[k]) * Pminus[k]

plt.figure()
plt.plot(z, 'k+', label='noisy measurements')
plt.plot(xhat, 'b-', label='a posteri estimate')
plt.plot(xhat+margin, 'g:',linewidth=0.7)
plt.plot(xhat-margin, 'g:', label='margin', linewidth=0.7)
plt.plot(flight_params['D_real'], 'r')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.show()
