# Contains the feedback controllers for the loop.


class PID:
    """
    Simplest possible implementation of a PID controller.
    """

    def __init__(self, Kp: float, Ki: float = 0, Kd: float = 0):
        """
        Inits a PID controller with zeroed intagrator.

        :param Kp:
        :param Ki:
        :param Kd:
        """

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integrator = 0
        self.e_OLD = 0

    def update(self, e, dt=1):

        self.integrator = self.integrator + dt*e
        u = self.Kp*e + self.Ki*self.integrator + self.Kd*(e-self.e_OLD)/dt
        self.e_OLD = e

        return u

    def __str__(self):
        return f'Kp: {self.Kp}; Ki: {self.Ki}; Kd: {self.Kd}, I: {self.integrator}, e_old: {self.e_OLD}'

    def __dict__(self):
        return {'Kp': self.Kp, 'Ki': self.Ki, 'Kd': self.Kd, 'I': self.integrator, 'e_old': self.e_OLD}


class RecurringProportional(PID):
    """
    Pads invalid plant readings with old value. Can only be a proportional controller.
    """

    def __init__(self, Kp, ControllerForTTC):
        super().__init__(Kp=Kp)

        self.TTCC = ControllerForTTC
        self.D_OLD = 0

    def update(self, D, dt=1):

        if D is not None:
            e = -1 / self.TTCC.update(dt) - D
            u = super().update(e, dt=dt)
            self.D_OLD = D
            return u
        else:
            u = super().update(self.TTCC.update(dt)-self.D_OLD, dt=dt)
            return u

    def __str__(self):
        return f'RecurringProportional with {str(self.TTCC)}'


class ProportionalIntegral(PID):
    """
    Implements a PI controller.
    """

    def __init__(self, Kp, Ki, controllerForTTC):
        super().__init__(Kp=Kp, Ki=Ki)

        self.TTCC = controllerForTTC

    def update(self, D, dt=1):
        e = -1 / self.TTCC.update(dt) - D
        u = super().update(e, dt=dt)
        return u

    def __str__(self):
        return f'ProportionalIntegral with {str(self.TTCC)}'


class FilteredProportional(PID):
    """
    A moving average filter working on top of the Proportional controller.
    """

    def __init__(self, Kp, ControllerForTTC, bins=1):
        super().__init__(Kp=Kp)

        self.TTCC = ControllerForTTC
        self.bins = bins
        self.FIFO = [0 for n in range(bins)]

    def update(self, D, dt=1):

        new_FIFO = self.FIFO[1:]
        new_FIFO.append(D)
        self.FIFO = new_FIFO
        e = -1 / self.TTCC.update(dt) - sum(self.FIFO)/self.bins
        u = super().update(e, dt=dt)
        return u

    def __str__(self):
        return f'FilteredProportional with {str(self.TTCC)}'


class cdTTC:

    def __init__(self, c, craft):

        # Initiated at real start
        self.TTC = -craft.position / craft.velocity
        self.c = c

    def update(self, dt):
        self.TTC = self.TTC - dt*self.c*self.c
        return self.TTC

    def __str__(self):
        return f'cdTTC'

