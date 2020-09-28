import math
import os
import statistics
import traceback
from time import time

import numpy as np
from matplotlib.patches import Circle
from numpy import array, transpose, matmul, sum, square, concatenate

import numpy.linalg as npl

import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.ndimage as ndi

from collections import deque

import EventDataHandlers
import Filehandling
from Filehandling import readinConfig


class MeanShiftSingleEstimator:
    """
    This class inplements a divergence estimator based on divergence of features on the optic plane, hence sparse 
    optic flow estimation of
    """

    max_in_quad = 15

    def __init__(self, x_size, y_size,
                 tau=600,
                 min_points=15,
                 min_features=5,
                 r=4,
                 centroid_seperation=0.4,
                 time_dimension=1000):

        """
        Stores all settings, inits data containers, generates convolution masks.

        Note: Default values for temporal parameters (tau, time_dimension) assume
        incoming data is in [ms].

        :param x_size: int
        :param y_size: int
            Dimensions of the pixel array
        :param tau:
            Duration of memory of past events (how far back are events considered relevant).
        :param min_points:
            Min number of points for centroid not to be discarded.
        :param min_features:
            Min number of centroids before centroid regen is called.
        :param r:
            Radius of
        :param centroid_seperation:
            Min distance as fraction of image plane for tracked centroids to be considered for divergence estimatino.
        :param time_dimension:
            Normilising factor is t dim is not [s]. E.g. for [ms] time_dimension=1000
        """

        ## STATIC ATTRIBUTES
        # Maxima finding parameters
        self.min_features = min_features
        self.init_maxima = 15
        # self.max_in_quad = self.init_maxima
        self.convMask = self.buildMask(r)

        # Mean shift parameters
        self.centroid_range = r
        self.min_points = min_points

        # Divergence estimation parameters
        self.centroid_seperation = centroid_seperation
        self.time_dimension = time_dimension

        # Camera parameters
        self.x_size = x_size
        self.y_size = y_size

        ## DYNAMIC ATTRIBUTES
        # Time
        self.previous_call = 0

        # Event management
        self.relevant_points = deque()
        self.tau = tau

        # Centroid management
        self.centroid_count = 0
        self.centroids_OLD = None
        self.centroids_NEW = None

    def __str__(self):

        def spc(n):
            s = ''
            if n < 0:
                return s
            for i in range(n):
                s = s + ' '
            return s

        static_HEADER = f'STATIC PARAMETERS\n'
        static_param_str = f'Param: min_points | min_features | centroid_range |' \
                           f' centroid_seperation | time_dimension\n'

        spc_list = [len(x) for x in static_param_str[7:-2].split(' | ')]

        static_value_str = f'Value:' \
                           f' {self.previous_call}{spc(spc_list[0] - len(str(self.previous_call)))} |' \
                           f' {self.min_features}{spc(spc_list[1] - len(str(self.min_features)))} |' \
                           f' {self.centroid_range}{spc(spc_list[2] - len(str(self.centroid_range)))} |' \
                           f' {self.centroid_seperation}{spc(spc_list[3] - len(str(self.centroid_seperation)))} |' \
                           f' {self.time_dimension}{spc(spc_list[4] - len(str(self.time_dimension)))}'

        rtn = static_HEADER + static_param_str + static_value_str + '\n\n'

        dynamic_HEADER = f'DYNAMIC PARAMETERS\n'
        dynamic_param_str = f'Param: time | stored_points | tau   | tracked centroids\n'

        spc_list = [len(x) for x in dynamic_param_str[7:-2].split(' | ')]

        dynamic_value_str = f'Value:' \
                            f' {self.previous_call}{spc(spc_list[0] - len(str(self.previous_call)))} |' \
                            f' {len(self.relevant_points)}{spc(spc_list[1] - len(str(len(self.relevant_points))))} |' \
                            f' {self.tau}{spc(spc_list[2] - len(str(self.tau)))} |' \
                            f' {self.centroid_count}{spc(spc_list[3] - len(str(self.centroid_count)))}'

        rtn = rtn + dynamic_HEADER + dynamic_param_str + dynamic_value_str

        return rtn

    def getStoredEventsProjection(self):
        """
        Returns stored events projection to user.

        :return:
        """

        return self._project2plane()

    def update(self, event_batch: np.array, time_now: float, echo: bool = False) -> float:
        """
        Top level logic of the estimator. Particular phases are broken down into subroutines.

        :param echo:
        :param time_now:
            Real time in the simulation at the moment the estimator is called.
        :param event_batch: np.array
            Events provided by the camera.
        :return D: float
            Divergence estimate.
        """

        self.discardOldEvents(event_batch)

        # If there are no points just return 0. I chose a threshold of 10 completely arbitrarily.
        if len(self.relevant_points) < 10:
            return 0

        # With the exception of the first call the method always begins with shifting centroids to new positions
        if self.centroids_OLD is not None:
            self.meanShift()

            # Standard scenario, just extract div, toss out invalid centroids and return div
            if self.centroid_count > self.min_features:
                D = self._estimateDivergence(dt=time_now - self.previous_call)
                self._updateCentroids()
                if echo:
                    print("normal")
                rtn = D
            # Case that ALL centroids failed to converge, no div can be estimated (hence 0) and new have to be regenerated
            elif self.centroid_count < 1:
                self.findMaxima()
                if echo:
                    print("no div")
                rtn = 0
            # Based on the few centroids left the div can be estimated at this point, but new are regenerated afterwards
            else:
                D = self._estimateDivergence(dt=time_now - self.previous_call)
                self.findMaxima()
                if echo:
                    print("regen")
                rtn = D

        else:
            self.findMaxima()
            rtn = 0

        # Finish with updating time and returning divergance
        self.previous_call = time_now
        return rtn

    def discardOldEvents(self, event_batch: np.array, echo: bool = False) -> None:
        """
        Puts the new batch to container and discards events that are too old.

        :param event_batch:
            Events provided by the camera.
        :param echo:
            Toggles on echo to terminal to debug.
        :return:
        """
        event_count = 0

        for n in range(event_batch.shape[0]):
            self.relevant_points.append(event_batch[n, :])
            event_count += event_batch.shape[0]

        if event_count < 1:
            return

        latest_event_time = self.relevant_points[-1][2]  # Could be also last from self.relevant_points

        if echo:
            print("Batch size: ", len(event_batch))
            print(
                f'First / last: {latest_event_time} / {self.relevant_points[0][2]}  ->  {latest_event_time - self.relevant_points[0][2]}')
            print("Relevant points: ", len(self.relevant_points))

        # Search for old points to remove
        deque_ptr = 0
        while self.relevant_points[deque_ptr][2] < latest_event_time - self.tau:
            deque_ptr += 1

        # Update counter of points
        event_count -= deque_ptr

        if echo:
            print("Points for removal: ", deque_ptr)
            print("Expected:", len(self.relevant_points) - deque_ptr)

        # Remove old points
        while deque_ptr > 0:
            self.relevant_points.popleft()
            deque_ptr += -1

        if echo:
            print("After removal:", len(self.relevant_points))

    def meanShift(self) -> None:
        """
        Assigns all points to nearest centroid as long at its within self.r distance from it and updates centroid
        location. Runs interatively through all points until all centroids converge or are discarded as invalid.
        Assigns previous centroids to self.centroids_OLD and new centroids to self.centroids_NEW.

        Details: centroids are stored in numpy array and faield centroids are tagged with clusteredPoints=-1. The array
        is reformatted in .updateCentroids() method that is called later in .update().

        :return:
        """

        centroid_count = self.centroid_count
        # Build matrix of [newX, newY, oldX, oldY, clusteredPoints]
        centroids = np.zeros([centroid_count, 5])
        centroids[:, :2] = np.zeros([centroid_count, 1])
        centroids[:, 2:4] = self.centroids_OLD

        ## MEAN SHIFTING
        # While loop has to use isclose() cause for floating point numbers they never equal. Atol chosen as 10th of a pixel.
        while np.any(
                np.isclose(centroids[centroids[:, 4] >= 0, 0], centroids[centroids[:, 4] >= 0, 2], atol=0.1, rtol=10e-7)
                & np.isclose(centroids[centroids[:, 4] >= 0, 1], centroids[centroids[:, 4] >= 0, 3], atol=0.1,
                             rtol=10e-7) == False):  # noqa

            # Shift new to old (since you can't do a do-while loop in Python)
            centroids[centroids[:, 4] >= 0, :2] = centroids[centroids[:, 4] >= 0, 2:4]
            # Set new x, y and counter to zero
            centroids[centroids[:, 4] >= 0, 2:] = np.zeros([self.centroid_count, 3])

            # Iterate through all the points searching closest centroid
            for event in self.relevant_points:
                d_min = self.centroid_range
                c_min = 0  # Tbh, this is redundant, just avoids PyCharm syntax warning
                for c in range(centroid_count):
                    if centroids[c, 4] >= 0:
                        d = math.sqrt((centroids[c, 0] - event[0]) ** 2 + (centroids[c, 1] - event[1]) ** 2)
                        # Only care if the centroid is within range of interest
                        if d < d_min:
                            d_min = d
                            c_min = c

                if d_min < self.centroid_range:
                    centroids[c_min, 2] = centroids[c_min, 2] + event[0]
                    centroids[c_min, 3] = centroids[c_min, 3] + event[1]
                    centroids[c_min, 4] = centroids[c_min, 4] + 1

            # Remove centroids with too few points and update centroid count
            centroids[centroids[:, 4] < self.min_points, 4] = -1
            self.centroid_count = centroids[centroids[:, 4] >= 0, :].shape[0]

            # Calculate centroid by dividing summed distances by the number of points
            centroids[centroids[:, 4] >= 0, 2] = centroids[centroids[:, 4] >= 0, 2] / centroids[centroids[:, 4] >= 0, 4]
            centroids[centroids[:, 4] >= 0, 3] = centroids[centroids[:, 4] >= 0, 3] / centroids[centroids[:, 4] >= 0, 4]

            # Remove centroids too close to the edge
            centroids[(centroids[:, 2] < 2) | (centroids[:, 2] > self.x_size), 4] = -1
            centroids[(centroids[:, 3] < 2) | (centroids[:, 3] > self.y_size), 4] = -1
            self.centroid_count = centroids[centroids[:, 4] >= 0, :].shape[0]

        # Store x,y coordinates and state (points count or -1 for failed centroids)
        self.centroids_NEW = np.empty([centroids.shape[0], 3])
        self.centroids_NEW[:, :2] = centroids[:, :2]
        self.centroids_NEW[:, 2] = centroids[:, 4]

    def findMaxima(self, mode: str = 'quad', echo: bool = False) -> None:
        """
        Sums events along t-axis producing a 2D projection, which is then blurred with a Gaussian mask (self.kernal).
        The designated maxima are the n largest local maxima of that "landscape".

        # TODO: Kwarg overriding class attribute? Is that even recommended?
        # TODO: Change to have a self.echo

        :param mode
            Method of chossing which peaks are selected as tracked features.
                global -> all features are arranged according to peak even density and the top n are selected, where n
                          is specified by self.init_maxima
                quad   -> attempts to distribute the peaks among quadrants in the image plane (#TODO What is the rule?)
        :param echo:
            Debuggin option - echos state to terminal
        :return:
        """
        # Project to xy-plane
        slice_projection = self._project2plane()

        projection_convolved = ndi.convolve(slice_projection, self.convMask, mode='constant')
        # local_max = ndi.maximum_filter(projection_convolved, size=5, mode='constant')

        # define an 8-connected neighborhood
        neighborhood = ndi.morphology.generate_binary_structure(2, 2)

        if echo:
            t0 = time()

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = ndi.maximum_filter(projection_convolved, footprint=neighborhood) == projection_convolved

        # we create the mask of the background
        background = (projection_convolved == 0)

        # erode the background in order to successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = ndi.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks, by removing the background from the local_max mask
        # (xor operation), which are then extracted into a list of their x,y coordinates
        detected_peaks = local_max ^ eroded_background
        peak_list = np.where(detected_peaks == True)  # noqa

        # Build array of peaks: [x, y, intensity] sorted by intensity
        max_list = np.vstack((peak_list[0], peak_list[1], projection_convolved[peak_list[0], peak_list[1]])).T
        max_list = max_list[max_list[:, 2].argsort()[::-1]]
        # Discard peaks within 5 pixels from the edge
        max_list = max_list[(max_list[:, 0] > 5) & (max_list[:, 0] < self.x_size - 5), :]
        max_list = max_list[(max_list[:, 1] > 5) & (max_list[:, 1] < self.y_size - 5), :]

        # Container for selected peaks
        selected = np.empty([self.init_maxima, 3])

        if mode == 'globally':
            # TODO: Test this option(?)
            # Avoids indexing error if there are insufficient available peaks.
            if max_list.shape[0] < self.init_maxima:
                if echo:
                    print(f'Insufficient peaks detected. REQUESTED {self.init_maxima}, DETECTED: {max_list.shape[0]}')
                selected = max_list  # Its passing all through since there aren't enough to be picky
            else:
                selected = max_list[:self.init_maxima,:]

        elif mode == 'quad':

            quads = [0, 0, 0, 0, ]
            for n in range(max_list.shape[0]):
                if (max_list[n, 0] > self.x_size / 2) & (max_list[n, 1] > self.y_size / 2):
                    location = 0
                elif (max_list[n, 0] < self.x_size / 2) & (max_list[n, 1] > self.y_size / 2):
                    location = 1
                elif (max_list[n, 0] < self.x_size / 2) & (max_list[n, 1] < self.y_size / 2):
                    location = 2
                else:
                    location = 3

                if quads[location] < MeanShiftSingleEstimator.max_in_quad:
                    quads[location] = quads[location] + 1
                    selected[sum(location)] = max_list[n, :]

                if sum(quads) == self.init_maxima:
                    break

            if echo:
                print(f'Q1: {quads[0]}\n Q2: {quads[1]}\n, Q3: {quads[2]}\n, Q4: {quads[3]}')

            # Stored in OLD as NEW is only a temporary container between methods in update call. NEW gets reset reset at
            # the beginning of update() anyway.
            self.centroids_OLD = selected[:, :2]
            self.centroid_count = self.centroids_OLD.shape[0]

        if echo:
            print("Run time: ", time() - t0)


        # Stores to old, because .update() begins with
        self.centroids_OLD = selected[:, :2]
        self.centroid_count = self.centroids_OLD.shape[0]

    @staticmethod
    def buildMask(r: float, sigma: float = 2, plot: bool = False, mode: str = 'Gaussian') -> np.array:
        """
        Build an convolution mask to blur the image of events projected on the plane. Recommend parameters are the kwarg
        defaults and have been chosen experimentally.

        :param r:
        :param sigma:
        :param plot:
        :param mode:
        :return:
        """

        if (mode != 'Gaussian') and (mode != 'Uniform'):
            raise ValueError('Invalid mode! Permissible: "Gaussian" / "Uniform"')

        array_dim = 2 * math.ceil(r) + 1
        centre = math.ceil(r)
        kernal_array = np.zeros([array_dim, array_dim])

        kernal_array[centre, centre] = 1

        if mode == 'Gaussian':
            if plot:
                fig_MeanShiftKernal, ax_MeanShiftKernal = plt.subplots(2, 2)
                ax_MeanShiftKernal[0, 0].imshow(ndi.filters.gaussian_filter(kernal_array, sigma=2))
                ax_MeanShiftKernal[0, 1].imshow(ndi.filters.gaussian_filter(kernal_array, sigma=3))
                ax_MeanShiftKernal[1, 0].imshow(ndi.filters.gaussian_filter(kernal_array, sigma=4))
                ax_MeanShiftKernal[1, 1].imshow(ndi.filters.gaussian_filter(kernal_array, sigma=5))
                plt.show(block=False)

            kernal_array = ndi.filters.gaussian_filter(kernal_array, sigma=sigma)

            return kernal_array

        elif mode == 'Uniform':
            raise Exception("Not implemented yet")

    def _estimateDivergence(self, dt: float = 0.4) -> float:
        """
        Estimate divergnece from the expansion between event clusters.

        :param dt:
        :return:
        """

        container = []

        old_set = self.centroids_OLD[self.centroids_NEW[:, 2] >= 0, :]
        new_set = self.centroids_NEW[self.centroids_NEW[:, 2] >= 0, :2]

        # Cy
        for j in range(old_set.shape[0]):
            for m in range(old_set.shape[0]):
                if j != m:
                    dist0 = np.linalg.norm(old_set[j, :] - old_set[m, :])
                    dist1 = np.linalg.norm(new_set[j, :] - new_set[m, :])
                    # Add divergence estimate if points aren't too close to each other
                    if (dist0 > self.x_size * self.centroid_seperation) or (
                            dist1 > self.x_size * self.centroid_seperation):
                        container.append((1 - dist1 / dist0) / (dt / self.time_dimension))

        if len(container) > 0:
            return statistics.mean(container)
        else:
            return 0

    def _updateCentroids(self) -> None:
        """
        Updates centroids, for now this means just moving NEW to OLD and reseting NEW to None.
        :return:
        """
        self.centroids_OLD = self.centroids_NEW[self.centroids_NEW[:, 2] >= 0, :2]
        self.centroids_NEW = None

    def _project2plane(self) -> np.array:
        """
        Sums all events along time axis producing an projection of the batch on the xy-plane
        :return: np.array
        """

        slice_projection = np.zeros([128, 128])

        # Project to xy-plane
        for point in self.relevant_points:
            slice_projection[int(point[0]), int(point[1])] = slice_projection[int(point[0]), int(point[1])] + 1

        return slice_projection


class MeanShiftEstimator:
    """
    This class really just wraps around MeanShiftSingleEsimator depending on whether user wants to split events by
    polarity or not.
    """

    def __init__(self, *args, mode='merge', **kwargs):
        """
        Inits one or two instances of MeanShiftSingleEsimator's depending on mode. For *args and **kwargs see docstrings
        in MeanShiftSingleEsimator.__init__()

        :param mode:
        :param args:
        :param kwargs:
        """

        if mode == 'merge':
            self.estimators = (MeanShiftSingleEstimator(*args, **kwargs),)
        elif mode == 'split':
            self.estimators = (MeanShiftSingleEstimator(*args, **kwargs), MeanShiftSingleEstimator(*args, **kwargs))

        # Error if invalid parameters are given
        else:
            raise ValueError('Invalid value assigned to parameter: polarity')

        self.mode = mode

    def __str__(self):
        """
        Print out configuration and state of the estimators
        :return:
        """

        if self.mode == 'merge':
            return f'MODE: {self.mode}\n\n' + f'== First instance ==\n' + self.estimators[0].__str__()
        elif self.mode == 'split':
            rtn = f'MODE: {self.mode}\n\n'
            rtn = rtn + f'== First instance ==\n' + self.estimators[0].__str__()
            rtn = rtn + f'\n\n== Second instance ==\n' + self.estimators[1].__str__()

            return rtn

    def getStoredEventsProjection(self) -> [np.array, np.array]:
        """
        Projects stored (relevant) events along the time axis on 2D plane and returns the result. Returns one or 2
        projects depending on mode of estimator.

        mode='merged' :return: np.array
        mode='split'  :return: (np.array, np.array)
        """

        if self.mode == 'merge':
            return self.estimators[0].getStoredEventsProjection()
        elif self.mode == 'split':
            return self.estimators[0].getStoredEventsProjection(), self.estimators[1].getStoredEventsProjection()

    def update(self, event_batch: np.array, time_now: float) -> [float, float]:
        """
        Analogically to MeanShiftSingleEsimator.update() takes event_batch and current time and passes to relevant case.
        :param event_batch:
        :param time_now:
        :return:
        """

        if self.mode == 'merge':
            return self._mergedUpdate(event_batch, time_now)
        elif self.mode == 'split':
            return self._splitUpdate(event_batch, time_now)

    def _mergedUpdate(self, event_batch: np.array, t: float) -> float:
        """
        Method of all events tracked together, hence both params passed immdiately to single instance method.

        :param event_batch:
        :param t:
        :return:
        """
        return self.estimators[0].update(event_batch, t)

    def _splitUpdate(self, event_batch: np.array, t: float, both: bool = False) -> [float, float]:
        """
        Method for tracking both polaroties separately. Events in batch are first split by polarity and then passed to
        approrpiate estimator instance method. Returned values are averaged.

        :param event_batch:
        :param t:
        mode='merged' :return: Divergence
        mode='split'  :return: pos_estimator_Divergence, neg_estimator_Divergence
        """

        # Case for empty batch, otherwise will raise IndexError on empty array
        if event_batch.shape[0] == 0:
            Dp = self.estimators[0].update(event_batch, t)
            Dn = self.estimators[1].update(event_batch, t)
        # Regular case
        else:
            pos_events = event_batch[event_batch[:, 3] == -1]
            neg_events = event_batch[event_batch[:, 3] == 1]

            Dp = self.estimators[0].update(pos_events, t)
            Dn = self.estimators[1].update(neg_events, t)

        # Return average or both, defaults to average
        rtn = []
        if both:
            return Dp, Dn
        else:
            if Dp is not None:
                if Dn is not None:
                    return (Dp + Dn) / 2
                else:
                    return Dp
            elif Dn is not None:
                return Dn
            else:
                return None

    def centroids(self) -> np.array:
        """
        This extracts centroids from the contained estimators. For 'split mode it return 2 arguments

        mode='merged' :return: estimator.centroids_OLD
        mode='split'  :return: pos_estimator.centroids_OLD, neg_estimator.centroids_OLD
        """

        if self.mode == 'merge':
            return self.estimators[0].centroids_OLD
        elif self.mode == 'split':
            return self.estimators[0].centroids_OLD, self.estimators[1].centroids_OLD

    # TESTSCRIPT
    @staticmethod
    def performanceAssesment(save: bool = True, test_points: list = None) -> None:

        if test_points is None:
            ## Parameter space is split into external parameters (under 'ext' key) and algorithm parameters (under 'object' key).
            test_points = []  # noqa
            # Vary time
            # test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 5},
            #                     'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.4}})
            test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 7},
                                'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4,
                                           'centroid_seperation': 0.4}})
            # test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 10},
            #                     'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.4}})
            # test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 5},
            #                     'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.2}})
            # test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 7},
            #                     'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.2}})
            # test_points.append({'ext': {'batch_size': 600, 'max_in_quad': 10},
            #                     'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.2}})

        ## The heavy lifting aka actual calculations
        # Load in events
        tar_dir = readinConfig()
        event_list = np.array(EventDataHandlers.readEventList(tar_dir + "/frames/sineHover2/eventlist.txt"))

        all_results = []
        all_labels = []

        for itr, test in enumerate(test_points):
            try:
                D = []
                quads = []

                delta = test['ext']['batch_size']
                start = 0
                end = start + delta

                instance = MeanShiftEstimator(128, 128, **test['object'])
                MeanShiftSingleEstimator.max_in_quad = test['ext']['max_in_quad']

                while end < max(event_list[:, 2]):
                    event_batch = event_list[(start < event_list[:, 2]) & (event_list[:, 2] < end), :]

                    print(f'Section: {start} to {end}')

                    D.append(instance.update(event_batch, end))
                    quads.append(instance.estimators[0].quads)

                    start = end
                    end = start + delta

                Didx = range(len(D))
                Didx = [tau * delta for tau in Didx]

                all_results.append((Didx, D))

                if save:
                    # Save for every test point in case it goes to hell
                    file_stream = open(tar_dir + f'/frames/sineHover2/' + f'test{itr}.txt', 'w')
                    quad_stream = open(tar_dir + f'/frames/sineHover2/' + f'test{itr}_QUADS.txt', 'w')
                    # Build string of settings from settings dict
                    settings_line = f''
                    for KEY, VALUE in test.items():
                        settings_line = settings_line + KEY + f': '
                        for key, value in VALUE.items():
                            settings_line = settings_line + f'{key}: {value}, '

                    file_stream.write(settings_line)
                    all_labels.append(settings_line)

                    for i, d in zip(Didx, D):
                        file_stream.write(f'{i},{d}\n')

                    for x in quads:
                        quad_stream.write(f'{x[0]},{x[1]},{x[2]},{x[3]}\n')

                    file_stream.close()
            except Exception as e:
                print(traceback.format_exc())

        ## Plotting:
        # load trajectory file
        test_tag = "sineHover2"

        old = os.getcwd()
        os.chdir(tar_dir)
        frame_rate = Filehandling.readinFrameRate(test_tag)
        trajectory = Filehandling.readinFlightTrajectory(test_tag)[:, 2]
        os.chdir(old)

        trajectory = trajectory[1:trajectory.shape[0]]

        D_true = 1 - (trajectory[0:trajectory.shape[0] - 1] / trajectory[1:trajectory.shape[0]])
        D_true = D_true * frame_rate * 10
        D_idx = np.arange(0, D_true.shape[0]) * 10

        # Plot
        fig_perf, ax_line = plt.subplots()
        ax_line.plot(D_idx, D_true, label=f'Ground truth from trajectory')

        for test, result in zip(test_points, all_results):
            ax_line.plot(result[0], result[1], label=f'batch_size: {test["ext"]["batch_size"]}, '
                                                     f'tau: {test["object"]["tau"]}, '
                                                     f'min_points: {test["object"]["min_points"]}, '
                                                     f'min_features: {test["object"]["min_features"]}, '
                                                     f'r: {test["object"]["r"]} '
                                                     f'centroid seperation: {test["object"]["centroid_seperation"]}')

        ax_line.set_title("Effect of varying minimum centroid separation")
        ax_line.set_ylabel("Divergence [1/s]")
        ax_line.set_xlabel("Time [s]")

        plt.legend()
        plt.show()


def plotArchivedResults() -> None:
    """
    Script for reading in and plotting divergence curves from a performance assessment run.
    :return:
    """

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Directory locations
    test_tag = "sineHover2"
    # subtest_tag = "/MeanShift_test/quad_maxima_distributing"
    subtest_tag = "/MeanShift_test/best"

    tar_dir = readinConfig()

    # Load data
    test_list = os.listdir(tar_dir + '/frames/' + test_tag + subtest_tag)

    curve_set = []
    label_set = []
    title_str = "<no title>"

    for test in test_list:
        test_stream = open(tar_dir + '/frames/' + test_tag + subtest_tag + '/' + test)
        if test == 'title.txt':
            title_str = test_stream.readline()
            test_stream.close()

        else:
            D = []
            Didx = []
            label_set.append(test_stream.readline())

            while True:
                try:
                    didx, d = test_stream.readline().split(',')
                    Didx.append(float(didx))
                    D.append(float(d))
                except ValueError:
                    break

            curve_set.append([Didx, D])
            test_stream.close()

    # load trajectory file
    old = os.getcwd()
    os.chdir(tar_dir)
    frame_rate = Filehandling.readinFrameRate(test_tag)
    trajectory = Filehandling.readinFlightTrajectory(test_tag)[:, 2]
    os.chdir(old)

    trajectory = trajectory[1:trajectory.shape[0]]

    D_true = 1 - (trajectory[0:trajectory.shape[0] - 1] / trajectory[1:trajectory.shape[0]])
    D_true = D_true * frame_rate
    D_idx = np.arange(0, D_true.shape[0]) * 10

    # Plot
    fig_perf, ax_line = plt.subplots()
    ax_line.plot(D_idx, D_true)
    for test_label, test_result in zip(label_set, curve_set):
        ax_line.plot(test_result[0], test_result[1], label=test_label)

    ax_line.legend()
    ax_line.set_title(title_str, fontweight='bold')
    ax_line.set_ylabel("Divergence [1/s]", fontsize='x-large')
    ax_line.set_xlabel("Time [ms]", fontsize='x-large')
    ax_line.set_yticks([-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12])

    plt.show()


if __name__ == "__main__":

    # plotArchivedResults()
    MeanShiftEstimator.performanceAssesment(save=False)
    # plotLocalPlanesEstimatorPerformanceResults('constDescent6')
