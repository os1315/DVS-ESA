# External import
import os
import pickle
from time import time

import numpy as np
import matplotlib.pyplot as plt

# Project
import DivergenceEstimator
import PyPANGU
import craftModels
import convInterpolate as CI
import feedbackController


class VerticalFreefall:

    def __init__(self, settings=None):

        if settings is None:
            ## Config
            # Environment objects
            self.camera_dimensions = {
                'x_size': 128,
                'y_size': 128
            }

            self.estimator_settings = {
                'tau': 1000,
                'min_points': 15,
                'min_features': 3,
                'r': 4,
                'centroid_seperation': 0.4,
                'time_dimension': 1,
                'mode': 'split'
            }

            self.physical_state = {
                'init_position': 1000,
                'init_velocity': -5,
                'body': 'Moon'
            }

            self.converter_settings = {
                'theta': 0.1,
                'T': 100,
                'latency': 30
            }

            # The steps parameter if for testing how longer waits on algorithm side (through repeated environment
            # updates) affect stability and accuracy
            self.environment_time = {
                'dt': 0.1,
                'steps': [[0, 3], [40, 2]]
            }

        else:
            self.camera_dimensions = settings['camera_dimensions']
            self.estimator_settings = settings['estimator_settings']
            self.physical_state = settings['physical_state']
            self.converter_settings = settings['converter_settings']
            self.environment_time = settings['environment_time']

        if 'steps' in self.environment_time:
            self.multi_step = True
        else:
            self.multi_step = False

        print("self.multi_step: ", self.multi_step)

        # Initialise the environment simulation part
        self.craft = craftModels.VerticalLander(**self.physical_state)
        self.server = PyPANGU.ServerPANGU(**self.camera_dimensions)

        self.server.set_viewpoint_by_degree([0, 0, 1000, 0, -90, 0])
        init_view = self.server.get_image()

        # Initialise the algorithm
        self.converter = CI.convInterpolate(**self.converter_settings, initial_image=init_view.mean(axis=2))
        self.estimator = DivergenceEstimator.MeanShiftEstimator(**self.camera_dimensions, **self.estimator_settings)

    def run(self, save_filename='', save=False, save_dir=None, echo=False):

        if save and save_filename == '':
            raise AttributeError('Save request, but no save filename provided!')

        # Logging
        flight_params = {
            'time': [],
            'position': [],
            'velocity': [],
            'mass': [],
            'D': [],
            'D_real': [],
            'tau': [],
            'incoming_events': [],
            'stored_events': []
        }

        dt = self.environment_time['dt']
        REAL_TIME = 0

        # Visualising
        # online_plot, (stored_events, current_image) = plt.subplots(1,2)
        # plt.show(block=False)

        for i in range(60):

            batches = []
            step_count = 1

            if self.multi_step:
                if i > self.environment_time['steps'][0][0]:
                    step_count = self.environment_time['steps'][0][1]
                if i > self.environment_time['steps'][1][0]:
                    step_count = self.environment_time['steps'][1][1]

            # Take several steps of environment simulation and then provide events
            for steps in range(step_count):
                REAL_TIME = REAL_TIME + dt

                self.craft.update(0, dt)
                self.server.set_viewpoint_by_degree(0, 0, self.craft.position, 0, -90, 0)
                new_view = self.server.get_image()
                new_view = new_view.mean(axis=2)  # Sum the RBG components
                new_batch, _ = self.converter.update(new_view)
                batches = batches + new_batch

                if echo:
                    print(f'Time now: {REAL_TIME:.1f}')
                    print("step: ", steps)
                    print(len(new_batch))

            if len(batches) > 0:
                batches_np = np.array(batches)
            else:
                batches_np = np.array([])

            D = self.estimator.update(np.array(batches_np), REAL_TIME)

            flight_params['time'].append(REAL_TIME)
            flight_params['position'].append(self.craft.position)
            flight_params['velocity'].append(self.craft.velocity)
            flight_params['mass'].append(self.craft.mass)
            flight_params['D'].append(D)
            flight_params['D_real'].append(self.craft.velocity / self.craft.position)
            flight_params['tau'].append(-self.craft.position / self.craft.velocity)
            A, B = self.estimator.getStoredEventsProjection()
            flight_params['stored_events'].append((A + B).sum())
            flight_params['incoming_events'].append(batches_np.shape[0])

            # stored_events.cla()
            # stored_events.imshow(estimator.getStoredEventsProjection())
            #
            # ctrds = estimator.centroids()
            # if ctrds is not None:
            #     for n in range(ctrds.shape[0]):
            #         if ctrds[n,0] > 0.5:
            #             circ = Circle((ctrds[n, 1], ctrds[n, 0]), 1, color='r')
            #             stored_events.add_patch(circ)
            #
            # current_image.imshow(new_view)
            #
            # plt.draw()
            # plt.pause(0.01)

            if echo:
                print(f'z: {self.craft.position:.2f}, '
                      f'v: {self.craft.velocity:.2f}, ')
            # f'features: \n{estimator.centroids()}')

        if save:
            saved_test = {
                'name': save_filename,
                'camera_dimensions': self.camera_dimensions,
                'estimator_settings': self.estimator_settings,
                'physical_state': self.physical_state,
                'converter_settings': self.converter_settings,
                'environment_time': self.environment_time,
                'flight_params': flight_params
            }

            if save_dir is not None:
                with open(save_dir + 'obj\\' + saved_test['name'] + '.pkl', 'wb') as f:
                    pickle.dump(saved_test, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open('obj\\' + saved_test['name'] + '.pkl', 'wb') as f:
                    pickle.dump(saved_test, f, pickle.HIGHEST_PROTOCOL)


class IdealVerticalcdTTC:

    def __init__(self, dt=0.1, Kp=1000):
        self.craft = craftModels.VerticalLander(
            init_position=2000,
            init_velocity=-10, )

        self.flight_params = {
            'time': [],
            'position': [],
            'velocity': [],
            'thrust': [],
            'mass': [],
            'tau': [],
            'tau_target': []
        }

        self.dt = dt
        self.c = 0.5
        self.Kp = Kp
        self.gimbal = feedbackController.PID(Kp=Kp, Ki=100)

    def run(self):
        self.flight_params['tau_target'].append(-self.craft.position / self.craft.velocity)
        self.flight_params['time'].append(self.dt)

        while self.craft.position > 0.1:
            self.flight_params['time'].append(self.flight_params['time'][-1] + self.dt)
            self.flight_params['tau_target'].append(self.flight_params['tau_target'][-1] - self.dt * self.c * self.c)
            self.flight_params['tau'].append(-self.craft.position / self.craft.velocity)
            self.flight_params['thrust'].append(
                # self.craft.g * self.craft.mass +
                # self.gimbal.update((self.flight_params['tau_target'][-1] - self.flight_params['tau'][-1]), dt=self.dt))
                self.Kp * (self.flight_params['tau_target'][-1] - self.flight_params['tau'][-1]))
            self.flight_params['position'].append(self.craft.position)
            self.flight_params['velocity'].append(self.craft.velocity)
            self.flight_params['mass'].append(self.craft.mass)

            self.craft.update(self.flight_params['thrust'][-1], self.dt)

        self.flight_params['tau_target'] = self.flight_params['tau_target'][:-1]
        self.flight_params['time'] = self.flight_params['time'][:-1]

        return self.flight_params

    def plot(self, fig, ax):
        # Position
        ax[0, 0].plot(self.flight_params['time'], self.flight_params['position'])
        ax[0, 0].set_title('Position')

        # Velocity
        ax[1, 0].plot(self.flight_params['time'], self.flight_params['velocity'])
        ax[1, 0].set_title('Velocity')

        # Thrust
        ax[0, 1].plot(self.flight_params['time'], self.flight_params['thrust'])
        ax[0, 1].set_title('Thrust')

        # Mass
        ax[1, 1].plot(self.flight_params['time'], self.flight_params['mass'])
        ax[1, 1].set_title('Mass')

        # Time-to-Contact
        ax[0, 2].plot(self.flight_params['time'], self.flight_params['tau'])
        ax[0, 2].set_title('Time-to-Contact')

        # Time-to-Contact (target)
        ax[1, 2].plot(self.flight_params['time'], self.flight_params['tau_target'])
        ax[1, 2].set_title('Time-to-Contact (target)')

        return fig, ax

    def save(self):
        pass


class VerticalcdTTC:
    """
    Runs a full closed-loop simulation of a vertically constrained landing. Sets up the simulation with a set of
    defuault hardcoded set of settings which can be overwritten by passing alternetive values at instance call.
    """

    setting_label_list = ['name',
                          'camera_dimensions',
                          'estimator_settings',
                          'physical_state',
                          'converter_settings',
                          'environment_time',
                          'controler_settings']

    def __init__(self, user_settings: dict = None) -> None:

        # Init all the settings and manage user inputs
        self._defaultSettingsInit()
        self._overwriteDefaultSettingsWithUserInputs(user_settings)

        # Initialise the environment simulation part
        craft = craftModels.VerticalLander(**self.settings['physical_state'])
        server = PyPANGU.ServerPANGU(**self.settings['camera_dimensions'])

        server.set_viewpoint_by_degree([0, 0, 1000, 0, -90, 0])
        init_view = server.get_image()

        converter = CI.convInterpolate(**self.settings['converter_settings'], initial_image=init_view.mean(axis=2))
        estimator = DivergenceEstimator.MeanShiftEstimator(**self.settings['camera_dimensions'],
                                                           **self.settings['estimator_settings'])
        controller = feedbackController.ProportionalIntegral(self.settings['controller_settings']['Kp'],
                                                             self.settings['controller_settings']['Ki'],
                                                             feedbackController.cdTTC(
                                                                 c=self.settings['controller_settings']['c'],
                                                                 craft=craft))

        # Add controller settings tile once set.
        self.settings['controller_settings']['controller'] = str(controller)
        if hasattr(controller, 'bins'):
            self.settings['controller_settings']['bins'] = controller.bins

        # Run loop
        dt = self.settings['environment_time']['dt']
        REAL_TIME = 0
        u = 0

        for i in range(1000):

            REAL_TIME = REAL_TIME + dt

            # Environment simulation
            start_time = time()
            craft.update(u, dt)
            server.set_viewpoint_by_degree(0, 0, craft.position, 0, -90, 0)
            new_view = server.get_image()
            new_view = new_view.mean(axis=2)  # Sum the RBG components
            new_batch, _ = converter.update(new_view)
            batches_np = np.array(new_batch)
            self.flight_params['calc_time_env'].append(time() - start_time)

            # Estimator runtime
            start_time = time()
            D = estimator.update(np.array(batches_np), REAL_TIME)
            self.flight_params['calc_time_est'].append(time() - start_time)

            # Controller
            start_time = time()
            u = controller.update(D, dt=dt)
            self.flight_params['calc_time_ctrl'].append(time() - start_time)

            # Store all data
            self.flight_params['time'].append(REAL_TIME)
            self.flight_params['position'].append(craft.position)
            self.flight_params['velocity'].append(craft.velocity)
            self.flight_params['mass'].append(craft.mass)
            self.flight_params['D'].append(D)
            self.flight_params['D_real'].append(craft.velocity / craft.position)
            self.flight_params['tau'].append(-craft.position / craft.velocity)
            self.flight_params['tau_target'].append(controller.TTCC.TTC)
            A, B = estimator.getStoredEventsProjection()
            self.flight_params['stored_events'].append((A + B).sum())
            self.flight_params['incoming_events'].append(batches_np.shape[0])

            # Actual thrust is never negative, so:
            if u < 0:
                self.flight_params['thrust'].append(0)
            else:
                self.flight_params['thrust'].append(u)

            # Echo update to terminal
            print(f' == Time now: {REAL_TIME:.1f}\n'
                  f'\tstep: {i}\n'
                  f'\tnew events: {len(new_batch)}\n'
                  f'\trun times:\n',
                  f'\t -> environment: {self.flight_params["calc_time_env"][-1]:.4f}\n',
                  f'\t -> estimator:   {self.flight_params["calc_time_est"][-1]:.4f}\n',
                  f'\t -> controller:  {self.flight_params["calc_time_ctrl"][-1]:.4f}\n',
                  f'\tz: {craft.position:.2f}, v: {craft.velocity:.2f}\n')

            # Check if landed, else loop will run until it exceeds 1000 iterations.
            if craft.position < 10:
                print(f'\nCraft reached ground or crashed')
                break

        self._saveResults()

    def _defaultSettingsInit(self) -> None:
        """
        Inits a set of initials settings of the simulation that will run. These will then be overwritten in the if
        the user provides alternative values for keys.

        :return:
        """

        camera_dimensions = {
            'x_size': 128,
            'y_size': 128
        }

        estimator_settings = {
            'tau': 500,
            'min_points': 15,
            'min_features': 3,
            'r': 4,
            'centroid_seperation': 0.4,
            'time_dimension': 1,
            'mode': 'split'
        }

        physical_state = {
            'init_position': 2000,
            'init_velocity': -50,
            'body': 'Moon'
        }

        converter_settings = {
            'theta': 0.1,
            'T': 100,
            'latency': 30
        }

        controller_settings = {
            'Kp': 1000_000,
            'Ki': 0,
            'c': 0.5
        }

        environment_time = {
            'dt': 0.1
        }

        # Wrap all those in single dict.
        self.settings = {
            'camera_dimensions': camera_dimensions,
            'estimator_settings': estimator_settings,
            'physical_state': physical_state,
            'converter_settings': converter_settings,
            'environment_time': environment_time,
            'controller_settings': controller_settings,
        }

        # Logging
        self.flight_params = {
            'time': [],
            'position': [],
            'velocity': [],
            'mass': [],
            'thrust': [],
            'D': [],
            'D_real': [],
            'tau': [],
            'tau_target': [],
            'incoming_events': [],
            'stored_events': [],
            'calc_time_env': [],
            'calc_time_est': [],
            'calc_time_ctrl': []
        }

        self.name = None

    def _saveResults(self) -> None:
        """
        Shelves results as a pickled dictionary. Consider other formats for long-term storage and sharing.

        :return:
        """

        if self.name is None:
            name = 'test'
        else:
            name = self.name

        # Increment name in counter to prevent overwrite.
        all_files_in_dir = os.listdir('obj')
        for counter in range(1, 1001):
            if f'{name}_{counter}.pkl' not in all_files_in_dir:
                name = f'{name}_{counter}'
                break
        # Place name, settings and results in dict in that order.
        saved_test = {'name': name}
        for key in self.settings:
            saved_test[key] = self.settings[key]
        saved_test['flight_params'] = self.flight_params

        with open('obj\\' + saved_test['name'] + '.pkl', 'wb') as f:
            pickle.dump(saved_test, f, pickle.HIGHEST_PROTOCOL)

        print(f'Test saved as obj/{saved_test["name"]}.pkl')

    def _overwriteDefaultSettingsWithUserInputs(self, user_settings):

        # Nothing provided, return and run with defaults
        if user_settings is None:
            return

        # Check user provided dict in format: {'parameter_name': 'paramter_value'}.
        # Else raise error.
        if type(user_settings) != dict:
            raise AttributeError("Incorrect user settings arguments format, expected dict.")

        # Run through provided dict and replace relevant keys, if 'name' replace self.name attribute.
        for key in user_settings:
            if key in self.settings:

                for sub_key in user_settings[key]:
                    if sub_key in self.settings[key]:
                        self.settings[key][sub_key] = user_settings[key][sub_key]
                    else:
                        raise AttributeError(f'Bad input: \"{sub_key}\" under \"{key}\". No such settings parameter!')

            elif key == 'name':
                self.name = user_settings[key]
            else:
                raise AttributeError(f'Bad input: \"{key}\". No such settings parameter!')


if __name__ == "__main__":

    # test = VerticalFreefall()
    # test.run(save=True, save_filename='test', echo=True)

    # Kp_list = [500, 1000, 1500, 2000]
    # figure, axis = plt.subplots(2, 3)
    #
    # for Kp in Kp_list:
    #     test = IdealVerticalcdTTC(dt=0.01, Kp=Kp)
    #     test.run()
    #     test.plot(figure, axis)
    #
    # plt.show()

    settings = {
        'name': 'another_test',
        'controller_settings': {'Kp': 1000_000}
    }

    VerticalcdTTC(settings)
