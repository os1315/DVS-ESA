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

