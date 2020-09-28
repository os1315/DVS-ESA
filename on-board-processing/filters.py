class BaseFilter:
    """
    Base class, basically a pass-through that replaces None with previous value.
    """

    def __init__(self) -> None:
        self.x_0 = 0

    def update(self, x: float) -> float:
        if x is None:
            return self.x_0
        else:
            self.x_0 = x
            return x

    def __str__(self):
        return f'BaseFilter'

    def returnSettings(self):
        return {}


class RollingAverageFilter(BaseFilter):
    """
    Implements a rolling average filter.
    """

    def __init__(self, bin_number: int):
        super().__init__()

        if bin_number < 1:
            raise AttributeError("You need at least one bin")
        self.stored_measurements = []
        for n in range(bin_number):
            self.stored_measurements.append(None)

    def update(self, x: float) -> float:
        """
        Returns last valid measurement if invalid measurements fill all bins.
        :param x:
        :return:
        """

        self.stored_measurements = self.stored_measurements[1:] + [x]
        total = []
        for x in self.stored_measurements:
            if x is not None:
                total.append(x)
        if len(total) < 1:
            return self.x_0
        else:
            self.x_0 = sum(total) / len(total)
            return self.x_0

    def __str__(self):
        return f'RollingAverageFilter_{len(self.stored_measurements)}_bin'

    def returnSettings(self):
        return {'bins': len(self.stored_measurements)}



class RollingAvgWithMargin(RollingAverageFilter):

    def __init__(self, bin_number, margin):
        super().__init__(bin_number)
        self.margin = margin

    def update(self, x: float) -> float:

        if x is not None:
            for s in self.stored_measurements:
                if s is not None:
                    if x < self.x_0 - self.margin or x > self.x_0 + self.margin:
                        x = None

        return super().update(x)

    def __str__(self):
        return f'RollingAveWithMargin_{len(self.stored_measurements)}_bin'

    def returnSettings(self):
        return {'bins': len(self.stored_measurements), 'margin': self.margin}


class KalmanFilter(BaseFilter):

    def __init__(self, margin=None, Q=0.0001, R=0.01, **kwargs):

        super().__init__()

        ## Filter setup
        # Process variance
        self.Q = Q

        # estimate of measurement variance
        self.R = R

        # Margin
        self.margin = margin  # suggested: 0.03

        ## Dynamic variables
        # Stored states
        self.xhat = 0  # a posteri estimate of x
        self.P = 1.0  # a posteri error estimate
        self.xhatminus = 0  # a priori estimate of x
        self.Pminus = 0  # a priori error estimate
        self.K = 0  # gain or blending factor

        print(self.xhat)
        print(self.xhatminus)

    def update(self, x: float) -> float:

        self.xhatminus = self.xhat
        self.Pminus = self.P + self.Q

        if self.margin is None:
            return self._estimate_simple(x)
        else:
            return self._estimate_with_margin(x)

    def _estimate_simple(self, x):

        self.K = self.Pminus / (self.Pminus + self.R)
        self.xhat = self.xhatminus + self.K * (x - self.xhatminus)
        self.P = (1 - self.K) * self.Pminus

        return self.xhat

    def _estimate_with_margin(self, x):

        if x > self.x_0 + self.margin or x < self.x_0 - self.margin:
            self.K = self.Pminus / (self.Pminus + self.R)
            self.xhat = self.x_0
            self.P = (1 - self.K) * self.Pminus
        else:
            self.K = self.Pminus / (self.Pminus + self.R)
            self.xhat = self.xhatminus + self.K * (x - self.xhatminus)
            self.P = (1 - self.K) * self.Pminus

        self.x_0 = self.xhat

        return self.xhat

    def __str__(self):
        return f'Kalman_filter'

    def returnSettings(self):
        if self.margin is not None:
            rtn = {
                'margin': self.margin,
                'Q': self.Q,
                'R': self.R
            }

        else:
            rtn = {
                'Q': self.Q,
                'R': self.R
            }

        return rtn
