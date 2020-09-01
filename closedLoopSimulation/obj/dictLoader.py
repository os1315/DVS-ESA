import math
import pickle
import matplotlib.pyplot as plt

# Open pickled dictionary
# filename = 'Dario_demo_free_fall'
# filename = 'Dario_demo_loop_1'
# filename = 'P_controller_1'
# filename = 'Dario_demo_unfilt'
filename = 'another_test'

with open(filename + '.pkl', 'rb') as f:
    my_dict = pickle.load(f)

# Parse saved data. Settings are printed out, while trajectories are places extracted as dictionary.
file_name = my_dict['name']
print(f'\n== File: {file_name} ==')
for key, item in my_dict.items():
    if key != 'name' and key != 'flight_params':
        print(f'From tile: {key}')
        for subkey, subitem in item.items():
            print(f'\t{subkey}: {subitem}')

flight_params = my_dict['flight_params']

## Plotting the read-in data
fig, ax = plt.subplots(2, 3)
# fig.subplots_adjust(0.5)

# Position
ax[0, 0].plot(flight_params['time'], flight_params['position'])
ax[0, 0].set_title('Position')
ax[0, 0].set_xlabel('Time [s]')
ax[0, 0].set_ylabel('Altitude [m]')

# Velocity
ax[0, 1].plot(flight_params['time'], flight_params['velocity'])
ax[0, 1].set_title('Velocity')
ax[0, 1].set_xlabel('Time [s]')
ax[0, 1].set_ylabel('Velocity [m/s]')

# Mass & thrust
ax[0, 2].plot(flight_params['time'], flight_params['mass'], label='Mass')
ax[0, 2].set_title('Mass')
ax[0, 2].set_xlabel('Time [s]')
ax[0, 2].set_ylabel('Mass [kg]')

if 'thrust' in flight_params:
    twin_ax = ax[0, 2].twinx()
    lns2 = twin_ax.plot(flight_params['time'], flight_params['thrust'], color='r', label='Thrust')
    lns = ax[0,2].lines+lns2
    labs = [l.get_label() for l in lns]
    ax[0, 2].legend(lns, labs)
    twin_ax.set_ylabel('Thrust [N]')


# Time-to-Contact
ax[1, 0].plot(flight_params['time'], flight_params['tau'], label='Tau Actual')
ax[1, 0].set_title('Time-to-Contact')
ax[1, 0].set_xlabel('Time [s]')
ax[1, 0].set_ylabel('TTC [s]')

if 'tau_target' in flight_params:
    ax[1, 0].plot(flight_params['time'], flight_params['tau_target'], color='r', label='Tau Target')
    labs = [l.get_label() for l in ax[1, 0].lines]
    ax[1, 0].legend(ax[1, 0].lines, labs)
    ax[1, 0].set_ylim(flight_params['tau_target'][0]*(-0.5), flight_params['tau_target'][0]*1.5)


# Divergence
# D = flight_params['D']
# bins = 5
# D_filtered = [0 for counter in range(bins)]
# for counter in range(len(D)-bins):
#     D_filtered.append(sum(D[counter:counter+bins])/bins)
# ax[1, 1].plot(flight_params['time'], D_filtered, color='b', label='D Estimated')

ax[1, 1].plot(flight_params['time'], flight_params['D'], color='b', label='D Estimated', linewidth=0.5)
ax[1, 1].plot(flight_params['time'], flight_params['D_real'], color='r', label='D real')
ax[1, 1].set_title('Divergence')
ax[1, 1].set_xlabel('Time [s]')
ax[1, 1].set_ylabel('Divergence [1/s]')

if 'tau_target' in flight_params:
    D_target = [-1/tau for tau in flight_params['tau_target']]
    ax[1, 1].plot(flight_params['time'], D_target, color='g', label='D Target')
    labs = [l.get_label() for l in ax[1, 1].lines]
    ax[1, 1].legend(ax[1, 1].lines, labs)

# Events
try:
    ax[1, 2].plot(flight_params['time'], flight_params['stored_events'], label='stored', color='b')
    ax[1, 2].plot(flight_params['time'], flight_params['incoming_events'], label='incoming', color='r')
    ax[1, 2].set_title('Events')
    ax[1, 2].set_xlabel('Time [s]')
    ax[1, 2].set_ylabel('# of events')
    ax[1, 2].legend()
except KeyError:
    ax[1, 2].set_title('<Nothing to show>')


# plt.tight_layout()
plt.show()
