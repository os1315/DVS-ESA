import math
import os
from time import time

import numpy as np
from numpy import array, transpose, matmul, mean, sum, square, concatenate

import numpy.linalg as npl

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from collections import deque

import EventDataHandlers
import Filehandling
from Filehandling import readinConfig


class DivergenceEstimator:

    # This intends to be a base class for all estimators that I come up with or wish to test

    # WHAT SHOULD IT DO?
    # 1. Have a "return Divergence" method
    # 2. Have an update method
    # 3. Have a backdoor/reset method? (That might be better to do for individual inherited classes

    def __del(self):
        pass

    def update(self, data):
        pass

    # This should compute the divergence from all vectors
    def returnDiv(self):
        pass


class LocalPlaneEstimator:
    """This class implements the Local Planes version of Optic Flow estimation"""

    def __init__(self, mode: str, x_size: int, y_size: int, dt: int = 20, dx: int = 3, min_points: int = 3, th1: float = 1 / 10000, th2: float = 0.7):

        # Spaciotemporal neighbourhood parameters
        self.th1 = th1
        self.th2 = th2
        self.bin_num = None
        self.dt = dt
        self.dx = dx

        self.x_size = x_size
        self.y_size = y_size

        # Minimum number of points in neighbourhood for fitting
        self.min_points = min_points

        # TODO: Do I really need event_count as attribute? Its seem confined to update, check if state carries over.
        self.event_count = 0
        self.latest_event_time = None

        if mode != "REG" or mode != "SVD" or mode != "SVDreg":
            self.mode = mode
        else:
            raise ValueError('Invalid mode')

        self.relevant_points = deque()
        self.div_mask = self.generateDivMask()

        # TODO: Should be initialised as np.array
        self.flow_vectors = None

        # TODO: Managing instances of figures, how to do that elegantly?
        self.figure_dict = {}

    def __repr__(self):
        return "<Test th1:%s th2:%s min_points:%s dx:%s dt:%s bin_num:%s>" % (self.th1, self.th2, self.min_points, self.dx, self.dt, self.bin_num)

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

    # TODO: CALLING THIS FOR EVERY POINT IS A BAD IDEA, ADD SOME GROUPING
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

        # TODO: flow vectors should be stored in object and tracked by temporal relevance, similarly to incoming points
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
        hist, bin_edges = np.histogram(norms_cleaned, bins=10)
        # filtered_vectors = vector_cleaned[(norms_cleaned < bin_edges[1])]

        # filtered_vectors = input_vectors[~np.isnan(norms) & ~np.isinf(norms) & (norms < bin_edges[1])]

        # Second hist round
        # norms_cleaned = npl.norm(vector_cleaned[:, 3:], axis=1)
        # hist, bin_edges = np.histogram(norms_cleaned, bins=self.bin_num)
        # filtered_vectors = vector_cleaned[(norms_cleaned < bin_edges[2])]
        filtered_vectors = vector_cleaned[(norms_cleaned < 2000) & (norms_cleaned > 100)]

        if visualise:
            centers = [(bin_edges[idx] + bin_edges[idx + 1]) / 2 for idx in range(len(bin_edges) - 1)]
            plt.bar(centers, hist, align='center', width=bin_edges[1]-bin_edges[0])
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

        # TODO: Check that this mask is properly generated and to element-wise dot product with vector field.

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
def splitLevelsAndFit(data, estimator_instance, plot=False):
    """Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

    # TODO: If this is still useful, change so it plots/returns its own plots

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

        data_slice = takeSlice(data, level + n * DELTA, DELTA)

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
def splitLocallyAndFit(data, estimator_instance, plot=False, coor=None, title=""):
    """TEST SCRIPT: Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

    # TODO: If this is still useful, change so it plots/returns its own plots

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

        data_slice = takeSlice(data, level + n * DELTA, DELTA)

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

    xIDX = [10*a+start_at_t0/100 for a in range(len(div_estimates))]
    div_estimates = [d/10000 for d in div_estimates]

    return xIDX, div_estimates


# TESTSCRIPT
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
            Didx, D = SVDregQuiver(event_list=event_list, settings=test, plot=False)

            all_results.append((Didx, D))

            if save:
                # Save for every test point in case it goes to hell
                file_stream = open(tar_dir + f'/frames/constDescent6/' + f'test{iter+5}.txt', 'w')
                file_stream.write(f'dt: {test["dt"]}, dx: {test["dx"]}, min_points: {test["min_points"]}, th1: {test["th1"]}, th2: {test["th2"]}\n')
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
        ax_line.plot(result[0], result[1], label=f'dt: {test["dt"]}, dx: {test["dx"]}, min_points: {test["min_points"]}, th1: {test["th1"]}, th2: {test["th2"]}')

    plt.show()


# TESTSCRIPT
def plotLocalPlanesEstimatorPerformanceResults(test_label: str):

    import os

    tar_dir = readinConfig()
    tar_dir = tar_dir + "/frames/" + test_label + "/LocalPlanes_test/"
    labels = []
    data_sets = []

    # Read in data
    files_list = os.listdir(tar_dir)
    for file in files_list:
        stream = open(tar_dir+file, 'r')
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


def takeSlice(data, level, delta):
    selected_points = []
    A = level - delta / 2
    C = level + delta / 2

    for n in range(data.shape[0]):
        if level - delta / 2 <= data[n, 2] < level + delta / 2:
            selected_points.append(data[n, :])

    return np.array(selected_points)


# This is a quasi-test to verify functionality
if __name__ == "__main__":

    # SVDregQuiver(plot=False)
    performanceAssessment(save=False)

    # plotLocalPlanesEstimatorPerformanceResults('constDescent6')

