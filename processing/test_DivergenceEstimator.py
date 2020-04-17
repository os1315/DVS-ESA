import numpy as np
from numpy import transpose, array


class test_DivergenceEstimator:
    """Implements tools for testing and estimating performance of estimators"""

    def __init__(self):
        self.data, self.EXTENTS = test_DivergenceEstimator.createData()

    @staticmethod  # For now a static method, might change later if I have to pass variables around)
    def createData():
        N_POINTS = 50
        TARGET_X_SLOPE = 20
        TARGET_y_SLOPE = 30
        TARGET_OFFSET = 5
        EXTENTS = 5
        NOISE = 50

        # create random data
        xs = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
        ys = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
        zs = []
        for i in range(N_POINTS):
            zs.append(xs[i] * TARGET_X_SLOPE + \
                      ys[i] * TARGET_y_SLOPE + \
                      TARGET_OFFSET + np.random.normal(scale=NOISE))

        return transpose(array([xs, ys, zs])), EXTENTS


if __name__ == "__main__":

    instance = test_DivergenceEstimator
    # Convert to numpy array
    data, EXTENTS = instance.createData()

    # all_fits = localPlaneFit(data)

    # instance.splitLevelsAndFit(data)
    #
    # for next_fit in all_fits:
    #     print("%f x + %f y + %f = z" % (next_fit[0], next_fit[1], next_fit[2]))
    #
    # # plot raw data
    # plt.figure()
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
