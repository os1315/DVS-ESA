import numpy as np
from numpy import array, transpose, matmul, mean, sum, square, concatenate

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from collections import deque

import EventDataHandlers


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
    # This class implements the Local Planes version of Optic Flow estimation

    def __init__(self, mode, dt=20, dx=3):

        # Spaciotemporal neighbourhood parameters
        self.dt = dt
        self.dx = dx

        self.event_count = 0
        self.latest_event_time = None

        if mode != "REG" or mode != "SVD":
            self.mode = mode
        else:
            raise ValueError('Invalid mode')

        self.relevant_points = deque()

    # CALLING THIS FOR EVERY POINTS IS A BAD IDEA, ADD SOME GROUPING
    def update(self, event_batch):

        for n in range(event_batch.shape[0]):
            self.relevant_points.append(event_batch[n, :])
            self.event_count += event_batch.shape[0]

        self.latest_event_time = event_batch[event_batch.shape[0] - 1][2]

        # Search for old points to remove
        deque_ptr = 0
        while self.relevant_points[deque_ptr][2] < self.latest_event_time - self.dt:
            deque_ptr += 1

        # Update counter of points
        self.event_count -= deque_ptr

        # Remove old points
        while deque_ptr > 0:
            self.relevant_points.popleft()
            deque_ptr += -1

        # Now apply local plane fit
        flow_vectors = self.localPlaneFit()
        return flow_vectors

    def localPlaneFit(self):

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
                    fitting_points.append(point[:])

            # #TODO: Check SVD fit and add option to select

            if len(fitting_points) > 2:
                if self.mode == "REG":
                    new_fit = LocalPlaneEstimator.FitRegression(np.array(fitting_points))
                elif self.mode == "SVD":
                    new_fit = LocalPlaneEstimator.FitSVD(np.array(fitting_points))
                flow_field.append(array([x_centre, y_centre, z_centre, new_fit[0], new_fit[1]]))

        return flow_field

    def returnDiv(self):
        pass

    @staticmethod
    def FitSVD(points):
        """Perform fit on a group of points using SVD, returns numpy array"""

        # subtract out the centroid
        points_norm = points - sum(points, 0) / points.shape[0]

        # singular value decomposition (note V here is the transpose of the conventional V
        U, E, V = np.linalg.svd(points_norm)

        # The last row of the transposed V matrix is a normal vector to the plane
        # The 1st and 2nd elements of it divided by the 3rd element are dx/dt and dy/dt
        return np.array([-V[2, 0] / V[2, 2], -V[2, 1] / V[2, 2]])  # , V[2, 2]/V[2, 2]])

    @staticmethod
    def FitRegression(points):
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


def createData():
    N_POINTS = 200
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


def splitLevelsAndFit(data, estimator_instance, plot=False):
    """Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

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
        print('dx/dt" {}, dy/dt: {}'.format(all_fits[n, 0], all_fits[n, 1]))

    print('Error {}: {}'.format(estimator_instance.mode, error))

    if plot:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def splitLocallyAndFit(data, estimator_instance, plot=False, coor=None, title=""):
    """Roughly splits data set into slices along z-axis and performs separate fits on all of them"""

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
    # Generate data
    DATA, extents = createData()

    # # plot raw data
    # plt.figure()
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(DATA[:, 0], DATA[:, 1], DATA[:, 2], color='b')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    # Initialize estimator instance
    # instance = LocalPlaneEstimator(mode="REG", dx=100, dt=40)
    # splitLevelsAndFit(DATA, instance, plot=True)

    fig = plt.figure()

    # estimator test with local estimation
    instance = LocalPlaneEstimator(mode="SVD", dx=20, dt=40)
    splitLocallyAndFit(DATA, instance, plot=True, coor=(2, 2, 1),
                       title='Planes with {}px neighbourhood'.format(instance.dx))

    instance = LocalPlaneEstimator(mode="SVD", dx=400, dt=40)
    splitLocallyAndFit(DATA, instance, plot=True, coor=(2, 2, 3), title="Planes with infinite neighbourhood")

    plt.show()

    # # Read data from file
    # test_tag = "/constDescent6"
    # str_address = "C:/PANGU/PANGU_5.00/models/lunar_OFlanding/frames" + test_tag + "/eventlist.txt"
    # ALL_DATA = EventDataHandlers.readEventList(str_address)
    # NEG_DATA, POS_DATA = EventDataHandlers.splitByPolarity(ALL_DATA)
    #
    # # estimator test with local estimation
    # instance = LocalPlaneEstimator(mode="SVD", dx=20, dt=1000)
    # splitLocallyAndFit(NEG_DATA, instance, plot=True)
