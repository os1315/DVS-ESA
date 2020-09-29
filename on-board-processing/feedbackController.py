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
        return f'ProportionalIntegral (1/D) with {str(self.TTCC)}'


class ConstantDivergence(PID):

    def __init__(self, Kp, Ki, target_div):
        super().__init__(Kp=Kp, Ki=Ki)

        self.target_div = target_div

    def update(self, D, dt=1):
        e = self.target_div - D

        u = super().update(e, dt=dt)
        return u

    def __str__(self):
        return f'ConstantDivergence {str(self.target_div)}'


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

