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


class LocalPlaneEstimator:
    """This class implements the Local Planes version of Optic Flow estimation"""

    # TODO: Normalize by the number of vectors considered for estimation.
    #  For a sparse optic flow field the div mask will get weird results when zeros are plugged around vector clusters?

    def __init__(self, mode: str, x_size: int, y_size: int, dt: int = 20, dx: int = 3, min_points: int = 3,
                 th1: float = 1 / 10000, th2: float = 0.7):

        # Spaciotemporal neighbourhood parameters
        self.th1 = th1
        self.th2 = th2
        self.bin_num = 10  # Totally arbitrary
        self.dt = dt
        self.dx = dx

        self.x_size = x_size
        self.y_size = y_size

        # Minimum number of points in neighbourhood for fitting
        self.min_points = min_points

        # Do I really need event_count as attribute? Its seem confined to update, check if state carries over.
        self.event_count = 0
        self.latest_event_time = None

        if mode != "REG" or mode != "SVD" or mode != "SVDreg":
            self.mode = mode
        else:
            raise ValueError('Invalid mode')

        self.relevant_points = deque()
        self.div_mask = self.generateDivMask()

        # Should be initialised as np.array
        self.flow_vectors = None

        # Managing instances of figures, how to do that elegantly?
        self.figure_dict = {}

    def __repr__(self):
        return "<Test th1:%s th2:%s min_points:%s dx:%s dt:%s bin_num:%s>" % (
            self.th1, self.th2, self.min_points, self.dx, self.dt, self.bin_num)

    def __str__(self):

        rtn = f'Current settings:\n'
        rtn = rtn + f'th1 ............ {self.th1}\n'
        rtn = rtn + f'th2 ............ {self.th2}\n'
        rtn = rtn + f'min_points ..... {self.min_points}\n'
        rtn = rtn + f'dx ............. {self.dx}\n'
        rtn = rtn + f'dt ............. {self.dt}\n'
        rtn = rtn + f'bin_num ........ {self.bin_num}\n'
        rtn = rtn + f'\nCurrent state:\n'
        rtn = rtn + f'stored points .. {len(self.relevant_points)}\n'
        if self.flow_vectors is not None:
            rtn = rtn + f'stored vectors . {self.flow_vectors.shape[0]}\n'
        else:
            rtn = rtn + f'stored vectors . None\n'
        rtn = rtn + f'mask dimensions: {self.x_size}x{self.y_size}'

        return rtn

    # EXTODO: CALLING THIS FOR EVERY POINT IS A BAD IDEA, ADD SOME GROUPING
    def update(self, event_batch: np.array, echo: bool = False) -> list:

        for n in range(event_batch.shape[0]):
            self.relevant_points.append(event_batch[n, :])
            self.event_count += event_batch.shape[0]

        self.latest_event_time = event_batch[event_batch.shape[0] - 1][2]

        if echo:
            print("Batch size: ", len(event_batch))
            print(
                f'First / last: {self.latest_event_time} / {self.relevant_points[0][2]}  ->  {self.latest_event_time - self.relevant_points[0][2]}')
            print("Relevant points: ", len(self.relevant_points))

        # Search for old points to remove
        deque_ptr = 0
        while self.relevant_points[deque_ptr][2] < self.latest_event_time - self.dt:
            deque_ptr += 1

        # Update counter of points
        self.event_count -= deque_ptr

        if echo:
            print("Points for removal: ", deque_ptr)
            print("Expected:", len(self.relevant_points) - deque_ptr)

        # Remove old points
        while deque_ptr > 0:
            self.relevant_points.popleft()
            deque_ptr += -1

        if echo:
            print("After removal:", len(self.relevant_points))

        # Now apply local plane fit
        self.flow_vectors = self.localPlaneFit()

        # flow vectors should be stored in object and tracked by temporal relevance, similarly to incoming points
        return self.flow_vectors

    def localPlaneFit(self) -> np.array:  # -> list[int, int, int, int, int]:

        t1 = time()

        # list of [x, y, t, vx, vy] elements
        flow_field = []

        # Iterate through ALL points
        for centre_point in self.relevant_points:

            # Initialise / reset container of points selected for fit
            fitting_points = []
            x_centre = centre_point[0]
            y_centre = centre_point[1]
            z_centre = centre_point[2]

            # Establish neighbourhood
            for point in self.relevant_points:
                # noinspection PyPep8
                if x_centre - self.dx < point[0] < x_centre + self.dx and \
                        y_centre - self.dx < point[1] < y_centre + self.dx and \
                        z_centre - self.dt < point[2] < z_centre + self.dt:
                    fitting_points.append(point[:3])  # Excludes polarity

            if len(fitting_points) > self.min_points:
                if self.mode == "REG":
                    new_fit = LocalPlaneEstimator.FitRegression(np.array(fitting_points))
                elif self.mode == "SVD":
                    new_fit = LocalPlaneEstimator.FitSVD(np.array(fitting_points))
                elif self.mode == "SVDreg":
                    new_fit = self.FitSVDRegularised(np.array(fitting_points))
                else:
                    new_fit = None

                if new_fit is not None:
                    flow_field.append(array([x_centre, y_centre, z_centre, new_fit[0], new_fit[1]]))

        print("V. field in: ", time() - t1)
        t1 = time()

        flow_field = self.HistogramFilter(array(flow_field))

        print("Filtered in: ", time() - t1)
        print("Deque length: ", len(self.relevant_points))

        return flow_field  # [x, y, t, dx/dt, dy/dt]

    def returnDiv(self) -> float:
        """Calculates divergance from currently stored flow vectors"""

        # We average over the dot product of all individual vectors with the mask
        D = 0

        # Sum up all the dot products
        for row in range(self.flow_vectors.shape[0]):
            vx = self.flow_vectors[row, 0]
            vy = self.flow_vectors[row, 1]

            mu = self.div_mask[0, int(vx), int(vy)]
            mv = self.div_mask[1, int(vx), int(vy)]

            D = D + mu * self.flow_vectors[row, 0] + mv * self.flow_vectors[row, 1]

        # Return the sum divided by number of flow vectors
        return -D / self.flow_vectors.shape[0]

    def FitSVDRegularised(self, points: np.array) -> np.array:
        """Perform fit on a group of points using repeated SVD with regularisation
            returns numpy array"""

        # subtract out the centroid
        points_norm = points - sum(points, 0) / points.shape[0]

        epsilon = 1000000
        steps = 0

        # SVD method of finding
        _U, _E, V = np.linalg.svd(points_norm)
        plane_vector = np.array([V[2, 0], V[2, 1], V[2, 2]])

        while epsilon > self.th1:
            # Compute distances and select those under threshold
            dst = np.matmul(points_norm, plane_vector.T)

            points_norm = points_norm[np.abs(dst) < self.th2, :]

            # Break if too few points are left
            if points_norm.shape[0] < 5:
                return None

            # Renorm data
            points_norm = points_norm - sum(points_norm, 0) / points_norm.shape[0]

            # Compute new plane
            _U, _E, V = np.linalg.svd(points_norm)
            new_vector = np.array([V[2, 0], V[2, 1], V[2, 2]])

            epsilon = npl.norm(plane_vector - new_vector)

            # Updata plane
            plane_vector = new_vector
            steps = steps + 1

            # Probably params are shit
            if steps > 20:
                return None

        if plane_vector[2] == 0:
            return None

        # The last row of the transposed V matrix is a normal vector to the plane
        # The 1st and 2nd elements of it divided by the 3rd element are dx/dt and dy/dt
        return np.array([-plane_vector[0] / plane_vector[2], -plane_vector[1] / plane_vector[2]])  # , V[2, 2]/V[2, 2]])

    def HistogramFilter(self, input_vectors: np.array, visualise: bool = False) -> np.array:

        # Remove nan's and inf's
        norms = npl.norm(input_vectors[:, 3:], axis=1)

        vector_cleaned = input_vectors[~np.isnan(norms) & ~np.isinf(norms)]

        norms_cleaned = npl.norm(vector_cleaned[:, 3:], axis=1)
        hist, bin_edges = np.histogram(norms_cleaned, bins=self.bin_num)
        # filtered_vectors = vector_cleaned[(norms_cleaned < bin_edges[1])]

        # filtered_vectors = input_vectors[~np.isnan(norms) & ~np.isinf(norms) & (norms < bin_edges[1])]

        # Second hist round
        # norms_cleaned = npl.norm(vector_cleaned[:, 3:], axis=1)
        # hist, bin_edges = np.histogram(norms_cleaned, bins=self.bin_num)
        # filtered_vectors = vector_cleaned[(norms_cleaned < bin_edges[2])]
        filtered_vectors = vector_cleaned[(norms_cleaned < 2000) & (norms_cleaned > 100)]

        if visualise:
            centers = [(bin_edges[idx] + bin_edges[idx + 1]) / 2 for idx in range(len(bin_edges) - 1)]
            plt.bar(centers, hist, align='center', width=bin_edges[1] - bin_edges[0])
            plt.pause(0.5)

        return filtered_vectors

    @staticmethod
    def FitSVD(points: np.array) -> np.array:
        """Perform fit on a group of points using SVD, returns numpy array"""

        # subtract out the centroid
        points_norm = points - sum(points, 0) / points.shape[0]

        # singular value decomposition (note V here is the transpose of the conventional V
        U, E, V = np.linalg.svd(points_norm)

        # The last row of the transposed V matrix is a normal vector to the plane
        # The 1st and 2nd elements of it divided by the 3rd element are dx/dt and dy/dt
        return np.array([-V[2, 0] / V[2, 2], -V[2, 1] / V[2, 2]])  # , V[2, 2]/V[2, 2]])

    @staticmethod
    def FitRegression(points: np.array) -> np.array:
        """Performs linear regression to fit the data, returns numpy array"""

        # subtract out the centroid (is this really necessary?)
        # points[:, 0] = points[:, 0] - mean(points[:, 0])
        # points[:, 1] = points[:, 1] - mean(points[:, 1])
        # points[:, 2] = points[:, 2] - mean(points[:, 2])

        A = np.ones([points.shape[0], 3])
        A[:, 0] = points[:, 0]
        A[:, 1] = points[:, 1]
        b = points[:, 2].transpose()

        xyd = matmul(matmul(np.linalg.inv(matmul(A.transpose(), A)), A.transpose()), b)

        return xyd[0:2]  # Only return dx/dt and dy/dt

    def generateDivMask(self, M: int = None, N: int = None, showmask: bool = False) -> np.array:

        # Check that this mask is properly generated and to element-wise dot product with vector field.

        if M is not None:
            m = M
            if N is None:
                n = m
            else:
                n = N
        else:
            m = self.x_size
            n = self.y_size

        normalizer = 1 / math.sqrt(((m - 1) / 2) ** 2 + ((n - 1) / 2) ** 2)
        mask = np.mgrid[-m / 2:(m - 1) / 2:1, -n / 2:(n - 1) / 2:1] + 0.5
        mask = mask * normalizer

        if showmask:
            chassis = np.mgrid[0:m:1, 0:n:1]
            fig_mask_quiv, ax_mask_quiv = plt.subplots()
            ax_mask_quiv.quiver(chassis[0], chassis[1], mask[0], mask[1])
            ax_mask_quiv.set_title(f'Divergence mask of dim {m}x{n}')
            plt.show()

        return mask

    ### BELOW IS TEST AND DEV CODE ###########################################################################

    @staticmethod
    def createData():
        N_POINTS = 20
        TARGET_X_SLOPE = 2
        TARGET_y_SLOPE = 3
        TARGET_OFFSET = 5
        EXTENTS = 50
        NOISE = 10

        # Error REG with seed=111 is 5.145704395479834
        np.random.seed(111)

        # create random data
        xs = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
        ys = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
        zs = []
        for i in range(N_POINTS):
            zs.append(xs[i] * TARGET_X_SLOPE + \
                      ys[i] * TARGET_y_SLOPE + \
                      TARGET_OFFSET + np.random.normal(scale=NOISE))

        return transpose(array([xs, ys, zs])), EXTENTS

    # TESTSCRIPT
    @staticmethod
    def splitLevelsAndFit(data, estimator_instance, plot=False):
        """Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

        # If this is still useful, change so it plots/returns its own plots

        # Length of section along z
        DELTA = 40

        # Extract mean and state
        level = np.min(data[:, 2]) + DELTA / 2
        slice_count = 10

        # Storage for fit results
        all_fits = np.zeros([slice_count, 2])

        if plot:
            # Plot
            plt.figure()
            ax = plt.subplot(111, projection='3d')
            colors = cm.rainbow(np.linspace(0, 1, slice_count))

        for n in range(slice_count):

            data_slice = LocalPlaneEstimator.takeSlice(data, level + n * DELTA, DELTA)

            if data_slice.shape[0] > 2:
                # Sort the slice, since that is how the data would normally look like
                data_slice = data_slice[data_slice[:, 2].argsort()]
                all_fits[n, :] = estimator_instance.update(data_slice)[0][3:5]
                if plot:
                    ax.scatter(data_slice[:, 0], data_slice[:, 1], data_slice[:, 2], color=colors[n, :])
            else:
                print("No points to fits, section: ", n)

        ideal = np.ones([slice_count, 2])
        ideal[:, 0] = ideal[:, 0] * 2
        ideal[:, 1] = ideal[:, 1] * 3
        error = np.sqrt(np.mean(np.square(all_fits - ideal)))

        for n in range(slice_count):
            print('dx/dt {}, dy/dt: {}'.format(all_fits[n, 0], all_fits[n, 1]))

        print('Error {}: {}'.format(estimator_instance.mode, error))

        if plot:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()

    # TESTSCRIPT
    @staticmethod
    def splitLocallyAndFit(data, estimator_instance, plot=False, coor=None, title=""):
        """TEST SCRIPT: Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

        # If this is still useful, change so it plots/returns its own plots

        # Length of section along z
        DELTA = estimator_instance.dt

        # Extract mean and state
        level = np.min(data[:, 2]) + DELTA / 2
        slice_count = 300

        # Storage for fit results
        all_fits = []

        if plot:
            # Plot
            if coor is None:
                ax = fig.add_subplot(1, 2, 1, projection='3d')
            else:
                ax = fig.add_subplot(coor[0], coor[1], coor[2], projection='3d')
            colors = cm.rainbow(np.linspace(0, 1, slice_count))

        for n in range(slice_count):

            data_slice = LocalPlaneEstimator.takeSlice(data, level + n * DELTA, DELTA)

            if data_slice.shape[0] > 2:
                # Sort the slice, since that is how the data would normally look like
                data_slice = data_slice[data_slice[:, 2].argsort()]
                all_fits.append(estimator_instance.update(data_slice))
                if plot:
                    ax.scatter(data_slice[:, 0], data_slice[:, 1], data_slice[:, 2], color=colors[n, :], s=2)
            else:
                print("No points to fits, section: ", n)

        all_fits_array = None

        for sub_list in all_fits:

            if len(sub_list) > 0:
                if all_fits_array is None:
                    all_fits_array = array(sub_list)
                else:
                    all_fits_array = concatenate((all_fits_array, array(sub_list)))
                print("")

        ideal = array([2, 3])
        error = array([0, 0])

        for n in range(all_fits_array.shape[0]):
            a = all_fits_array[n, 3:5]
            error = error + square(all_fits_array[n, 3:5] - ideal)
            print('dx/dt" {}, dy/dt: {}'.format(all_fits_array[n, 3], all_fits_array[n, 4]))

        print('Error {}: {}'.format(estimator_instance.mode, sum(error / all_fits_array.shape[0])))

        if plot:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            if coor is None:
                ax = fig.add_subplot(1, 2, 2)
            else:
                ax = fig.add_subplot(coor[0], coor[1], coor[2] + 1)
            ax.quiver(all_fits_array[:, 0], all_fits_array[:, 1], all_fits_array[:, 3], all_fits_array[:, 4])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)

    # TESTSCRIPT
    @staticmethod
    def SVDregQuiver(event_list=None, settings=None, plot=False):
        print("TESTSCRIPT: Running SVDregQuiver...")

        if event_list is None:
            tar_dir = readinConfig()
            event_list = np.array(EventDataHandlers.readEventList(tar_dir + "/frames/constDescent6/eventlist_050.txt"))
            print("Loaded data...")
        else:
            print("Data passed in!")

        delta = 1000
        start_at_t0 = 1000
        start = start_at_t0
        end = start_at_t0 + delta

        instance = LocalPlaneEstimator('SVDreg', 128, 128, dx=settings['dx'], dt=settings['dt'])
        print('')
        print(instance)

        div_estimates = []

        if plot:
            fig1 = plt.figure()

            ax = fig1.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlabel('x pixel')
            ax.set_ylabel('y pixel')
            ax.set_zlabel('time [ms]')

            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'Vector field on slice between {start / 1000}[s] and {end / 1000}[s]')

        while end < event_list[event_list.shape[0] - 1, 2]:
            sub_list = event_list[(start < event_list[:, 2]) & (event_list[:, 2] < end), :]

            subset_neg = sub_list[sub_list[:, 3] == -1]
            subset_pos = sub_list[sub_list[:, 3] == 1]

            # print("Is sorted: ", np.all(True == (np.diff(subset_neg[:, 2]) <= 0)))
            # print("Is sorted: ", np.all(True == (np.diff(subset_neg[:, 2]) >= 0)))

            vectors = instance.update(subset_neg)
            vectors = np.array(vectors)

            div_estimates.append(instance.returnDiv())

            # subset_neg = subset_neg[(0 < subset_neg[:, 0]) & (subset_neg[:, 0] < 20)]
            # subset_neg = subset_neg[(110 < subset_neg[:, 1]) & (subset_neg[:, 1] < 128)]

            if plot:
                # Clear graphs
                ax.clear()
                ax1.clear()

                ax.scatter(subset_neg[:, 0], subset_neg[:, 1], subset_neg[:, 2], s=1)
                ax1.quiver(vectors[:, 0], vectors[:, 1], vectors[:, 3], vectors[:, 4])

                plt.pause(0.1)

            # Progress to next slice, CANNOT OVER LAP WITH PREVIOUS SLICE
            start = start + delta
            end = start + delta
            print(f'\nNew limits: {start} - {end}')

            # _temp = input("next slice?")

        xIDX = [10 * a + start_at_t0 / 100 for a in range(len(div_estimates))]
        div_estimates = [d / 10000 for d in div_estimates]

        return xIDX, div_estimates

    # TESTSCRIPT
    @staticmethod
    def performanceAssessment(save=True):

        ## Parameter space:
        test_points = []
        # Vary time
        test_points.append({'dt': 2000, 'dx': 3, 'min_points': 3, 'th1': 0.0001, 'th2': 0.7})
        test_points.append({'dt': 5000, 'dx': 3, 'min_points': 3, 'th1': 0.0001, 'th2': 0.7})

        ## The heavy lifting aka actual calculations
        # Load in events
        tar_dir = readinConfig()
        event_list = np.array(EventDataHandlers.readEventList(tar_dir + "/frames/constDescent6/eventlist_050.txt"))

        all_results = []

        for iter, test in enumerate(test_points):
            try:
                Didx, D = LocalPlaneEstimator.SVDregQuiver(event_list=event_list, settings=test, plot=False)

                all_results.append((Didx, D))

                if save:
                    # Save for every test point in case it goes to hell
                    file_stream = open(tar_dir + f'/frames/constDescent6/' + f'test{iter + 5}.txt', 'w')
                    file_stream.write(
                        f'dt: {test["dt"]}, dx: {test["dx"]}, min_points: {test["min_points"]}, th1: {test["th1"]}, th2: {test["th2"]}\n')
                    for i, d in zip(Didx, D):
                        file_stream.write(f'{i},{d}\n')
                    file_stream.close()
            except Exception as e:
                print(e)

        ## Plotting:
        # load trajectory file
        test_tag = "constDescent6"

        old = os.getcwd()
        os.chdir(tar_dir)
        frame_rate = Filehandling.readinFrameRate(test_tag)
        trajectory = Filehandling.readinFlightTrajectory(test_tag)[:, 2]
        os.chdir(old)

        trajectory = trajectory[1:trajectory.shape[0]]

        D_true = 1 - (trajectory[0:trajectory.shape[0] - 1] / trajectory[1:trajectory.shape[0]])
        D_true = D_true * frame_rate

        # Plot
        fig_perf, ax_line = plt.subplots()
        ax_line.plot(range(D_true.shape[0]), D_true)

        for test, result in zip(test_points, all_results):
            ax_line.plot(result[0], result[1],
                         label=f'dt: {test["dt"]}, dx: {test["dx"]}, min_points: {test["min_points"]}, th1: {test["th1"]}, th2: {test["th2"]}')

        plt.show()

    # TESTSCRIPT
    @staticmethod
    def plotLocalPlanesEstimatorPerformanceResults(test_label: str):

        import os

        tar_dir = readinConfig()
        tar_dir = tar_dir + "/frames/" + test_label + "/LocalPlanes_test/"
        labels = []
        data_sets = []

        # Read in data
        files_list = os.listdir(tar_dir)
        for file in files_list:
            stream = open(tar_dir + file, 'r')
            labels.append(stream.readline())
            T = []
            D = []
            while True:
                try:
                    t, d = stream.readline().split(',')
                    T.append(float(t))
                    D.append(float(d[:-1]))
                except ValueError:
                    break
            data_sets.append([T, D])

        fig, ax = plt.subplots()
        # plt.show()
        for l, data in zip(labels, data_sets):
            ax.plot(data[0], data[1], label=l[:-1])

        # Add ground truth:
        test_tag = "constDescent6"

        old = os.getcwd()
        os.chdir(readinConfig())
        frame_rate = Filehandling.readinFrameRate(test_tag)
        trajectory = Filehandling.readinFlightTrajectory(test_tag)[:, 2]
        os.chdir(old)

        trajectory = trajectory[1:trajectory.shape[0]]
        D_true = 1 - (trajectory[0:trajectory.shape[0] - 1] / trajectory[1:trajectory.shape[0]])
        D_true = D_true * frame_rate

        ax.plot(range(D_true.shape[0]), D_true, c=[0, 0, 0], label='Ground truth')

        ax.set_title('Performance of local planes divergence estimator with various parameter settings')
        ax.legend()
        plt.show()

    @staticmethod
    def takeSlice(data, level, delta):
        selected_points = []
        A = level - delta / 2
        C = level + delta / 2

        for n in range(data.shape[0]):
            if level - delta / 2 <= data[n, 2] < level + delta / 2:
                selected_points.append(data[n, :])

        return np.array(selected_points)


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
        self.kernal = self.buildKernal(r)

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

    def update(self, event_batch: np.array, time_now: float) -> float:
        """
        Top level logic of the estimator. Particular phases are broken down into subroutines.

        :param time_now:
            Real time in the simulation at the moment the estimator is called.
        :param event_batch: np.array
            Events provided by the camera.
        :return D: float
            Divergence estimate.
        """

        self.discardOldEvents(event_batch)

        # With the exception of the first call the method always begins with shifting centroids to new positions
        if self.centroids_OLD is not None:
            self.meanShift()

            # Standard scenario, just extract div, toss out invalid centroids and return div
            if self.centroid_count > self.min_features:
                D = self.estimateDivergence(dt=time_now - self.previous_call)
                self.updateCentroids()
                print("normal")
                rtn = D
            # Case that ALL centroids failed to converge, no div can be estimated (hence 0) and new have to be regenerated
            elif self.centroid_count < 1:
                self.findMaxima()
                print("no div")
                rtn = 0
            # Based on the few centroids left the div can be estimated at this point, but new are regenerated afterwards
            else:
                D = self.estimateDivergence(dt=time_now - self.previous_call)
                self.findMaxima()
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
        while np.any(np.isclose(centroids[centroids[:, 4] >= 0, 0], centroids[centroids[:, 4] >= 0, 2], atol=0.1, rtol=10e-7)
                     & np.isclose(centroids[centroids[:, 4] >= 0, 1], centroids[centroids[:, 4] >= 0, 3], atol=0.1, rtol=10e-7) == False):  # noqa

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

    def findMaxima(self, mode: str = 'quad', plot: bool = False, echo: bool = False) -> None:
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
        :param plot:
            Debugging option - will plot the event density map with selected peaks marked.
        :param echo:
            Debuggin option - echos state to terminal
        :return:
        """

        t0 = 0
        slice_projection = np.zeros([128, 128])

        # Project to xy-plane
        for point in self.relevant_points:
            slice_projection[point[0], point[1]] = slice_projection[point[0], point[1]] + 1

        projection_convolved = ndi.convolve(slice_projection, self.kernal, mode='constant')
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
            # Avoids indexing error if there are insufficient available peaks.
            if max_list.shape[0] < self.init_maxima:
                if echo:
                    print(f'Insufficient peaks detected. REQUESTED {self.init_maxima}, DETECTED: {max_list.shape[0]}')

            selected = max_list  # Its passing all through since there aren't enough to be picky
        elif mode == 'quad':
            # selected = max_list[:maxima_number, :]
            # selected_by_quad = [[max_list[:, (max_list[:, 0] > self.x_size / 2) & (max_list[:, 1] < self.y_size / 2)], 0],
            #                     [max_list[:, (max_list[:, 0] < self.x_size / 2) & (max_list[:, 1] < self.y_size / 2)], 0],
            #                     [max_list[:, (max_list[:, 0] > self.x_size / 2) & (max_list[:, 1] > self.y_size / 2)], 0],
            #                     [max_list[:, (max_list[:, 0] < self.x_size / 2) & (max_list[:, 1] > self.y_size / 2)], 0]]
            #
            # for idx, x in enumerate(selected_by_quad):
            #     if x[0] < 1:
            #         selected_by_quad[idx, 1] = -1
            #     elif x[0] > 4:
            #         selected_by_quad[idx, 1] = 1
            #     else:
            #         selected_by_quad[idx, 1] = 0
            #
            # for idx, x in enumerate(selected_by_quad):
            #     if x[1] == -1:
            #
            #
            #
            # max_list_by_quad = [max_list[:, (max_list[:, 0] > self.x_size / 2) & (max_list[:, 1] < self.y_size / 2)],
            #                     max_list[:, (max_list[:, 0] < self.x_size / 2) & (max_list[:, 1] < self.y_size / 2)],
            #                     max_list[:, (max_list[:, 0] > self.x_size / 2) & (max_list[:, 1] > self.y_size / 2)],
            #                     max_list[:, (max_list[:, 0] < self.x_size / 2) & (max_list[:, 1] > self.y_size / 2)]]

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
            self.quads = quads

            # Stored in OLD as NEW is only a temporary container between methods in update call. NEW gets reset reset at
            # the beginning of update() anyway.
            self.centroids_OLD = selected[:, :2]
            self.centroid_count = self.centroids_OLD.shape[0]

        if echo:
            print("Run time: ", time() - t0)

        if plot:

            fig, ax = plt.subplots(2,2)

            ax[0,0].imshow(slice_projection)
            ax[0,1].imshow(slice_projection)
            ax[1,0].imshow(self.kernal)
            ax[1,1].imshow(projection_convolved)

            ax[0,0].set_title("Projected events")
            ax[0,1].set_title("Marked maxima")
            ax[1,0].set_title("Kernal")
            ax[1,1].set_title("Blurred projection")

            ax[0,0].set_xticks(range(0,129,16))
            ax[0,1].set_xticks(range(0,129,16))
            ax[1,0].set_xticks(range(0,9,2))
            ax[1,1].set_xticks(range(0,129,16))

            ax[0,0].set_yticks(range(0,129,16))
            ax[0,1].set_yticks(range(0,129,16))
            ax[1,0].set_yticks(range(0,9,2))
            ax[1,1].set_yticks(range(0,129,16))

            for n in range(max_list.shape[0]):
                circ = Circle((max_list[n, 1], max_list[n, 0]), 1, color='r')
                ax[0,1].add_patch(circ)

            plt.show()

        # Stores to old, because .update() begins with
        self.centroids_OLD = selected[:, :2]
        self.centroid_count = self.centroids_OLD.shape[0]

    @staticmethod
    def buildKernal(r: float, sigma: float = 2, plot: bool = False, mode: str = 'Gaussian') -> np.array:
        """
        Build an kernal mask to blur the image of events projected on the plane. Recommend parameters are the kwarg
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

    def estimateDivergence(self, dt: float = 0.4) -> float:
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
                    if (dist0 > self.x_size*self.centroid_seperation) or (dist1 > self.x_size*self.centroid_seperation):
                        container.append((1 - dist1 / dist0)/(dt/self.time_dimension))

        if len(container) > 0:
            return statistics.mean(container)
        else:
            return 0

    def updateCentroids(self) -> None:
        """
        Updates centroids, for now this means just moving NEW to OLD and reseting NEW to None.
        :return:
        """
        self.centroids_OLD = self.centroids_NEW[self.centroids_NEW[:, 2] >= 0, :2]
        self.centroids_NEW = None


class MeanShiftEstimator:
    """
    This wrapper class really implements factory pattern to avoid changing the underlying MeanShiftSingleEsimator
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
        Simple case of all events tracked together, hence both params passed immdiately to single instance method.

        :param event_batch:
        :param t:
        :return:
        """
        return self.estimators[0].update(event_batch, t)

    def _splitUpdate(self, event_batch: np.array, t: float, both: bool = False) -> [float, float]:
        """
        Case for tracking both polaroties separately. Events in batch are first split by polarity and then passed to
        approrpiate estimator instance method. Returned values are averaged.

        :param event_batch:
        :param t:
        mode='merged' :return: Divergence
        mode='split'  :return: pos_estimator_Divergence, neg_estimator_Divergence
        """

        pos_events = event_batch[event_batch[:, 3] == -1]
        neg_events = event_batch[event_batch[:, 3] == 1]

            Dp = self.estimators[0].update(pos_events, t)
            Dn = self.estimators[1].update(neg_events, t)

        if both:
            return Dp, Dn
        else:
            return (Dp + Dn)/2

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
                                'object': {'tau': 600, 'min_points': 15, 'min_features': 5, 'r': 4, 'centroid_seperation': 0.4}})
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
    pass

    # plotArchivedResults()
    MeanShiftEstimator.performanceAssesment(save=False)
    # plotLocalPlanesEstimatorPerformanceResults('constDescent6')
