# My imports
import statistics

import Filehandling
from Filehandling import readinConfig

import numpy as np
import cv2
import matplotlib.pyplot as plt

# System libs
import os
from contextlib import contextmanager


@contextmanager
def chdir(dirname):
    old = os.getcwd()
    os.chdir(dirname)
    yield old
    os.chdir(old)


class LKEstimator:
    """
    This class is a frame-based diverngence estimator. It uses corner detection functions from OpenCV for Python library
    to find features for tracking and estimates divergence from their motion on the image plane.

    Features are not filtered in any way, nor are they balance across the vision plane. Performance is quite poor.
    """

    def __init__(self, old_frame: np.array, feature_param: dict, visualise: bool = False, playback_speed: float = 5):
        """
        Inits the estimator.

        :param old_frame:
        :param feature_param:
        :param visualise:
        :param playback_speed:
        """
        self.old_frame = old_frame
        self.p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_param)

        self.visualise = visualise
        self.playback_speed = playback_speed

        if visualise:
            # Create a mask image for drawing purposes
            self.mask = np.zeros_like(old_frame)
            # Create some random colors
            self.color = np.random.randint(0, 255, (100, 3))

            # Init windows
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600, 600)

    def __del__(self):
        """
        Destructor will close windows if present.

        :return:
        """

        if self.visualise:
            cv2.destroyAllWindows()

    def update(self, new_frame: np.array) -> float:
        """
        Update of the algorithm, meant to be called upon interation of the environment in the simulation.

        :param new_frame:
        :return:
        """
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, new_frame, self.p0, None, **lk_params)

        if st.size < 10:
            self.p0 = cv2.goodFeaturesToTrack(self.old_frame, mask=None, **feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, new_frame, self.p0, None, **lk_params)
            if self.visualise:
                print("Redrawing features at step ", n)
                print("New feature count: ", st.size)
            DIV = 10 * self.computeFeatureDivergence(self.p0, p1)
        else:
            DIV = 10 * self.computeFeatureDivergence(self.p0, p1)

        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # Now update the previous frame and previous points
        self.old_frame = new_frame.copy()
        self.p0 = good_new.reshape(-1, 1, 2)

        if self.visualise:

            cv2.imshow('frame', new_frame)
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)
                new_frame = cv2.circle(new_frame, (a, b), 5, self.color[i].tolist(), -1)

            lookup_image = cv2.add(new_frame, self.mask)
            cv2.waitKey(self.playback_speed)
            cv2.imshow('frame with mask', lookup_image)

        return DIV

    @staticmethod
    def computeFeatureDivergence(old_set: np.array, new_set: np.array, dt: float = 1):
        """
        Calculates the divergence from a ratio of average distances between features (corners).

        :param old_set:
        :param new_set:
        :param dt:
        :return:
        """
        container = []

        for j in range(old_set.shape[0]):
            for m in range(old_set.shape[0]):
                if j != m:
                    dist0 = np.linalg.norm(old_set[j, :, :] - old_set[m, :, :])
                    dist1 = np.linalg.norm(new_set[j, :, :] - new_set[m, :, :])
                    container.append(1 - dist1 / dist0)

        return statistics.mean(container) / dt


# Running module as main will run a test that loads a set of images from specified directory and treats them as data
# obtained in subsequent iterations of a simulated environment. Default is a set of images generated from camera
# approaching a surface with constant speed.

# If will also load the corresponding trajectory and plot real divergence (velocity/altitude) for reference.

if __name__ == "__main__":

    # SET UP PARAMS
    # params for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=20,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Parameters for divergence estimator
    visualise_simulation = False
    interval_between_frames = 5

    # PREPARE DATA
    # Load test
    target_dir = readinConfig()
    test_tag = "constDescent6"

    with chdir(target_dir + "/frames/" + test_tag):
        print(os.getcwd())
        img = np.load(test_tag + "_ABR.npy")
        img2 = img[:, :, 1]

    # RUN THE OPTIC FLOW ESTIMATION
    START = 2000  # Don't start at 0, cause that frame may be blank
    first_frame = np.floor(img[:, :, START] * 255).astype(dtype='uint8')
    divergence_estimator = LKEstimator(first_frame, feature_params, visualise_simulation, interval_between_frames)

    # Stares divergence
    divergence = []

    for n in range(START + 1, img.shape[2]):
        next_frame = np.floor(img[:, :, n] * 255).astype(dtype='uint8')
        divergence.append(divergence_estimator.update(next_frame))

    # ASSESS ESTIMATOR PERFORMANCE
    # load trajectory file
    with chdir(target_dir):
        frame_rate = Filehandling.readinFrameRate(test_tag)
        trajectory = Filehandling.readinFlightTrajectory(test_tag)[:, 2]

    trajectory = trajectory[START:trajectory.shape[0]]

    D_true = 1 - (trajectory[0:trajectory.shape[0] - 1] / trajectory[1:trajectory.shape[0]])
    D_true = D_true * frame_rate

    # moving average filter
    bin_size = [5, 10, 50, 100]
    divs = []
    for B in bin_size:
        new_divergence = []
        for idx, D in enumerate(divergence[0:len(divergence) - B]):
            new_divergence.append(statistics.mean(divergence[idx:idx + B - 1]))
        divs.append(new_divergence)

    lines = []

    fig, ax = plt.subplots()
    lines.append(ax.plot(range(D_true.shape[0]), D_true)[0])

    for div, B in zip(divs, bin_size):
        lines.append(ax.plot(range(len(div)), div)[0])

    # Styling and legend
    ax.set_xticks(np.linspace(np.min(trajectory), np.max(trajectory), num=5))

    for line, B in zip(lines, bin_size):
        print(type(line))
        line.set_label(f'Bin size: {B}')

    ax.legend()
    plt.show()
