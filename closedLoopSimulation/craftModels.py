# A model of a craft lander
import numpy as np

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
        """
        Updates craft position after time dt.

        :param u:
        :param dt:
        :param solver:
        :return:
        """

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

    def verticalTime2Contact(self) -> float:
        """Returns the real time to contact at current state of the craft"""

        try:
            return abs(self.position / self.velocity)
        except ZeroDivisionError:
            return float(1_000_000)


if __name__ == "__main__":

    # Tests that the craft is falling when repeatedly calling update.

    import matplotlib.pyplot as plt

    step = 0.1

    time = range(10000)
    time = [t * step for t in time]
    craft = VerticalLander(body='Moon')
    pos = []
    M = []

    for t in time:
        pos.append(craft.update(1000, 0.1))
        M.append(craft.mass)

    z = [-0.5 * VerticalLander.g['Moon'] * t * t for t in time]
    deviation = [x1 - x2 for x1, x2 in zip(z, pos)]

    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, pos, label='approx')
    ax1.plot(time, z, label='exact')
    ax1.set_title("Altitude")
    ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, deviation)
    ax1.set_title("Deviation")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time, M)
    ax1.set_title("Mass")

    plt.show()
