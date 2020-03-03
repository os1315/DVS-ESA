# Math imports
from math import ceil

# Numpy imports
import numpy as np


class convInterpolate:

    def __init__(self, x_size, y_size, theta, T, latency, initial_image):
        self.x_size = x_size
        self.y_size = y_size
        self.theta = theta
        self.T = T
        self.latency = latency

        # Has to be a numpy array
        initial_image = np.log10(initial_image)  # Create matrix of logged values

        self.current_states = initial_image  # Stores the values that pixel differentiators are zeroed at
        self.refract_period = np.zeros((x_size, y_size),
                                       dtype='float32')  # Stores ENDS of refractive periods for the pixels
        self.time = 0  # Time at which frame was recorded
        self.quick_burst = np.zeros((x_size, y_size))

        # quick_burst = 0   ->  no burst
        # quick_burst = 1   ->  positive
        # quick_burst = -1  ->  negative

        self.threshold_p = self.current_states + self.theta  # Constant threshold
        self.threshold_n = self.current_states - self.theta  # Constant threshold

        self.previous_image = initial_image

    def __del__(self):
        pass

    def update(self, new_image):

        # returned variables
        PARTIAL_LIST = []
        image_out = np.ones([self.x_size, self.y_size]) * 0.5

        new_image = np.log10(new_image)  # Create matrix of logged values

        delta = new_image - self.previous_image
        # threshold = current_states * theta        # Dynamic threshold

        for x in range(self.x_size):  # Iterate over x dimension
            for y in range(self.y_size):  # Iterate over y dimension

                # Slope used for piecewise linear interpolation of pixel intensity
                slope = delta[x, y] / self.T

                # Do quick burst if flag is up
                if self.quick_burst[x, y] > 0:
                    event_time = self.refract_period[x, y]
                    # Update pixel state and set new end for refractory period
                    self.current_states[x, y] = self.previous_image[x, y] + slope * (event_time - self.time)
                    self.refract_period[x, y] = event_time + self.latency
                    self.threshold_p[x, y] = self.current_states[x, y] + self.theta
                    self.threshold_n[x, y] = self.current_states[x, y] - self.theta

                    if self.quick_burst[x, y] == 1:
                        PARTIAL_LIST.append([x, y, event_time, -1])
                    if self.quick_burst[x, y] == 2:
                        PARTIAL_LIST.append([x, y, event_time, 1])
                    self.quick_burst[x, y] = 0

                # Keeps looking for events until pixel refract period extends into next frame
                while self.refract_period[x, y] < (self.time + self.T):

                    # Case for increasing brightness
                    if self.threshold_p[x, y] < new_image[x, y] and slope > 0:

                        # Dummy value event visualisation, might delete later
                        image_out[x, y] = 1.0
                        # Linear estimate of threshold crossing instance
                        dt = abs((self.threshold_p[x, y] - self.previous_image[x, y]) / slope)

                        # print('\n\n {} \n\n'.format(self.previous_image[x,y]))

                        # This section calculates the registration of the event depending on the pixels refractory period
                        if self.refract_period[x, y] > self.time + dt:
                            event_time = self.refract_period[x, y]
                            self.current_states[x, y] = self.previous_image[x, y] + slope * (event_time - self.time)
                            self.refract_period[x, y] = event_time + self.latency
                        else:
                            event_time = ceil(self.time + dt)
                            self.current_states[x, y] = self.current_states[x, y] + self.theta
                            self.refract_period[x, y] = event_time + self.latency

                        # Append event to event list and update thresholds
                        PARTIAL_LIST.append([x, y, event_time, 1])
                        self.threshold_p[x, y] = self.current_states[x, y] + self.theta
                        self.threshold_n[x, y] = self.current_states[x, y] - self.theta

                        # Case for decreasing brightness
                    elif self.threshold_n[x, y] > new_image[x, y] and slope < 0:

                        image_out[x, y] = 0.0
                        # Linear estimate of threshold crossing instance
                        dt = abs((self.threshold_n[x, y] - self.previous_image[x, y]) / slope)

                        # This section calculates the registration of the event depending on the pixels refractory period
                        if self.refract_period[x, y] > self.time + dt:
                            event_time = self.refract_period[x, y]
                            # Update pixel state and set new end for refractory period
                            self.current_states[x, y] = self.previous_image[x, y] + slope * (event_time - self.time)
                            self.refract_period[x, y] = event_time + self.latency
                        else:
                            event_time = ceil(self.time + dt)
                            # Update pixel state and set new end for refractory period
                            self.current_states[x, y] = self.current_states[x, y] - self.theta
                            self.refract_period[x, y] = event_time + self.latency

                        PARTIAL_LIST.append([x, y, event_time, -1])
                        self.threshold_p[x, y] = self.current_states[x, y] + self.theta
                        self.threshold_n[x, y] = self.current_states[x, y] - self.theta

                    else:
                        break

                # track if pixel should fire immediately at the end of refractory period
                if (self.refract_period[x, y] >= (self.time + self.T)) and (self.threshold_p[x, y] < new_image[x, y]):
                    self.quick_burst[x, y] = 1
                elif (self.refract_period[x, y] >= (self.time + self.T)) and (self.threshold_n[x, y] > new_image[x, y]):
                    self.quick_burst[x, y] = 2

        # Update time and previous image
        self.time = self.time + self.T
        self.previous_image = new_image

        # Sort the generated events and append to returned list
        if len(PARTIAL_LIST) > 0:
            PARTIAL_LIST.sort(key=lambda x: x[2])

        return PARTIAL_LIST, image_out
