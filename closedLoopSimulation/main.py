import numpy as np

import pickle

import PyPANGU
import craftModels
import convInterpolate as CI
import DivergenceEstimator

## Config
# Evironment objects
import feedbackController

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
    'Kp': 1000_00,
    'Ki': 0,
    'c': 0.5
}

# Initialise the environment simulation part
craft = craftModels.VerticalLander(**physical_state)
server = PyPANGU.ServerPANGU(**camera_dimensions)

server.set_viewpoint_by_degree([0, 0, 1000, 0, -90, 0])
init_view = server.get_image()

converter = CI.convInterpolate(**converter_settings, initial_image=init_view.mean(axis=2))
estimator = DivergenceEstimator.MeanShiftEstimator(**camera_dimensions, **estimator_settings)
controller = feedbackController.ProportionalIntegral(controller_settings['Kp'],
                                                     controller_settings['Ki'],
                                                     feedbackController.cdTTC(c=controller_settings['c'],
                                                                              craft=craft))

controller_settings['controller'] = str(controller)
if hasattr(controller, 'bins'):
    controller_settings['bins'] = controller.bins

# Logging
flight_params = {
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
    'calc_time_algo': []
}

# Time dimension and tracking
environment_time = {
    'dt': 0.1
}

dt = environment_time['dt']
REAL_TIME = 0

# Thrust
u = 0

for i in range(1000):

    batches = []

    REAL_TIME = REAL_TIME + dt
    print(f'Time now: {REAL_TIME:.1f}')

    craft.update(u, dt)
    server.set_viewpoint_by_degree(0, 0, craft.position, 0, -90, 0)
    new_view = server.get_image()
    new_view = new_view.mean(axis=2)  # Sum the RBG components
    new_batch, _ = converter.update(new_view)
    print("step: ", i)
    print(len(new_batch))

    batches_np = np.array(new_batch)

    D = estimator.update(np.array(batches_np), REAL_TIME)
    u = controller.update(D, dt=dt)

    flight_params['time'].append(REAL_TIME)
    flight_params['position'].append(craft.position)
    flight_params['velocity'].append(craft.velocity)
    flight_params['mass'].append(craft.mass)
    flight_params['D'].append(D)
    flight_params['D_real'].append(craft.velocity / craft.position)
    flight_params['tau'].append(-craft.position / craft.velocity)
    flight_params['tau_target'].append(controller.TTCC.TTC)
    A, B = estimator.getStoredEventsProjection()
    flight_params['stored_events'].append((A + B).sum())
    flight_params['incoming_events'].append(batches_np.shape[0])

    # Actual thrust is never negative, so:
    if u < 0:
        flight_params['thrust'].append(0)
    else:
        flight_params['thrust'].append(u)

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

    print(f'z: {craft.position:.2f}, '
          f'v: {craft.velocity:.2f}, ')
    # f'features: \n{estimator.centroids()}')

    if abs(craft.position / craft.velocity) < 2 * dt or craft.position < 10:
        print(f'\nCraft reached ground or crashed')
        break
# _____________________________________________________________________________________________________________________
# RETURN RESULTS

if True:
    saved_test = {
        'name': 'P_controller_2',
        'camera_dimensions': camera_dimensions,
        'estimator_settings': estimator_settings,
        'physical_state': physical_state,
        'converter_settings': converter_settings,
        'environment_time': environment_time,
        'controler_settings': controller_settings,
        'flight_params': flight_params,
    }

    with open('obj\\' + saved_test['name'] + '.pkl', 'wb') as f:
        pickle.dump(saved_test, f, pickle.HIGHEST_PROTOCOL)
