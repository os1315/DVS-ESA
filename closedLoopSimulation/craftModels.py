# A model of a craft lander
from typing import Union
import numpy as np


class Lander:
    U_MAX = 1000  # Max thrust
    g_EARTH = 9.806  # Gravitational acceleration [ms^-2]
    g_LUNAR = 1.623  # Gravitational acceleration [ms^-2]
    Isp = 1000000  # Engine specific impulse

    def __init__(self,
                 init_position: Union[np.array, list] = np.array([0, 0, 0], dtype='float'),
                 init_orientation: Union[np.array, list] = np.array([0, -90, 0], dtype='float'),
                 init_velocity: Union[np.array, list] = np.array([0, 0, 0], dtype='float'),
                 init_mass: float = 1000):

        if type(init_position) is list:
            self.position = np.array(init_position, dtype='float')
        elif type(init_position) is np.array:
            self.position = init_position.astype('float')
        else:
            self.position = init_position.astype(float)

        if type(init_orientation) is list:
            self.orientation = np.array(init_orientation, dtype='float')
        elif type(init_orientation) is np.array:
            self.orientation = init_orientation.astype('float')
        else:
            self.orientation = init_orientation.astype(float)

        if type(init_velocity) is list:
            self.velocity = np.array(init_velocity, dtype='float')
        elif type(init_velocity) is np.array:
            self.velocity = init_velocity.astype('float')
        else:
            self.velocity = init_velocity.astype(float)

        self.mass = float(init_mass)

    def update(self, u: float, dt: float, solver: bool = False) -> np.array:

        # TODO: Generalise to 3d

        if solver:
            raise NotImplementedError
        else:
            self.position = self.position + dt * self.velocity
            self.velocity[2] = self.velocity[2] + dt * (u / self.mass - Lander.g)
            self.mass = self.mass - dt * u / Lander.Isp  # TODO: What is g0 in the paper?

            return self.position

    def verticalTime2Contact(self):
        """Returns the real time to contact at current state of the craft"""

        if self.velocity[2] != 0:
            return self.position[2] / self.velocity

        else:
            return 1000_000


class VerticalLander:
    """
    Models a craft as in "Landing with Time-to-Contact and Ventral Optic Flow Estimates". Contrained only to vertical
    motion.

    Can only decelerate (thrust in opposite direction to gravitational acceleration).
    """

    U_MAX = 100_000  # Max thrust
    Isp = 300  # Engine specific impulse

    g = {'Earth': 9.806,
         'Moon': 1.623,
         'Zero': 0}

    def __init__(self,
                 init_position: float = 0,
                 init_velocity: float = 0,
                 init_mass: float = 10000,
                 body: str = 'Earth'):

        if body in VerticalLander.g:
            self.g = VerticalLander.g[body]
        else:
            raise AttributeError(f'Unspecified body provide: {body}')

        self.position = float(init_position)
        self.velocity = float(init_velocity)
        self.mass = float(init_mass)

    def update(self, u: float, dt: float, solver: bool = False) -> np.array:

        if solver:
            raise NotImplementedError
        else:
            self.position = self.position + dt * self.velocity
            if u < 0:
                self.velocity = self.velocity - dt * self.g
            else:
                self.velocity = self.velocity + dt * (u / self.mass - self.g)
                self.mass = self.mass - dt * u / VerticalLander.Isp  # TODO: What is g0 in the paper?

            return self.position

    def verticalTime2Contact(self):
        """Returns the real time to contact at current state of the craft"""

        try:
            return abs(self.position / self.velocity)
        except ZeroDivisionError:
            return 1_000_000


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    step = 0.1

    time = range(10000)
    time = [t * step for t in time]
    craft = VerticalLander(init_position=3, body='Moon')
    pos = []
    M = []

    for t in time:
        pos.append(craft.update(1000, 0.1))
        M.append(craft.mass)

    z = [-0.5 * VerticalLander.g['Moon'] * t * t for t in time]
    error = [x1 - x2 for x1, x2 in zip(z, pos)]

    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, pos, label='approx')
    ax1.plot(time, z, label='exact')
    ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, error)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time, M)

    plt.show()
