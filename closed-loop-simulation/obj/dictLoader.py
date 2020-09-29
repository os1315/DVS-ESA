import math
import pickle
import matplotlib.pyplot as plt

# Open pickled dictionary
# filename = 'Dario_demo_free_fall'
# filename = 'Dario_demo_loop_1'
# filename = 'P_controller_1'
# filename = 'Dario_demo_unfilt'
# filename = 'another_test'
# filename = 'highres_1_Kalman_massaware_10'
# filename = 'highres_1_ravg_massaware_11'
# filename = 'preset_int_Kalman_4'
# filename = 'unfilt_P_mass_aware_1'
filename = 'cD_model_1_ravg_massaware_6'
# filename = 'cdTTC_model_1_ravg_massaware_4'

with open(filename + '.pkl', 'rb') as f:
    my_dict = pickle.load(f)

# Parse saved data. Settings are printed out, while trajectories are places extracted as dictionary.
file_name = my_dict['name']
print(f'\n== File: {file_name} ==')
for key, item in my_dict.items():
    if key != 'name' and key != 'flight_params' and key != 'note':
        print(f'From tile: {key}')
        for subkey, subitem in item.items():
            print(f'\t{subkey}: {subitem}')

if 'note' in my_dict:
    print(f'From tile: note')
    print(f'\t{my_dict["note"]}')

flight_params = my_dict['flight_params']

# Cropping the axes
# Crop beginning
start_t = 0.7
idx = 0
while flight_params['time'][idx] < start_t:
    idx = idx + 1
start_t = idx

# Crop end (end_t = -1 for no crop)
end_t = 80000
idx = 0
while flight_params['time'][idx] < end_t:
    idx = idx + 1
    if idx > len(flight_params['time'])-1:
        break
if end_t != -1:
    end_t = idx

## Plotting the read-in data
fig, ax = plt.subplots(2, 3)
# fig.subplots_adjust(0.5)

# Position
ax[0, 0].plot(flight_params['time'][start_t:end_t], flight_params['position'][start_t:end_t])
ax[0, 0].set_title('Position')
ax[0, 0].set_xlabel('Time [s]')
ax[0, 0].set_ylabel('Altitude [m]')

# Velocity
ax[0, 1].plot(flight_params['time'][start_t:end_t], flight_params['velocity'][start_t:end_t])
ax[0, 1].set_title('Velocity')
ax[0, 1].set_xlabel('Time [s]')
ax[0, 1].set_ylabel('Velocity [m/s]')

# Mass & thrust
ax[0, 2].plot(flight_params['time'][start_t:end_t], flight_params['mass'][start_t:end_t], label='Mass')
ax[0, 2].set_title('Mass')
ax[0, 2].set_xlabel('Time [s]')
ax[0, 2].set_ylabel('Mass [kg]')

if 'thrust' in flight_params:
    twin_ax = ax[0, 2].twinx()
    lns2 = twin_ax.plot(flight_params['time'][start_t:end_t], flight_params['thrust'][start_t:end_t], color='r', label='Thrust', linewidth=0.5)
    lns = ax[0,2].lines+lns2
    labs = [l.get_label() for l in lns]
    ax[0, 2].legend(lns, labs)
    twin_ax.set_ylabel('Thrust [N]')


# Time-to-Contact
ax[1, 0].plot(flight_params['time'][start_t:end_t], flight_params['tau'][start_t:end_t], label='Tau Actual')
ax[1, 0].set_title('Time-to-Contact')
ax[1, 0].set_xlabel('Time [s]')
ax[1, 0].set_ylabel('TTC [s]')

if 'tau_target' in flight_params:
    ax[1, 0].plot(flight_params['time'][start_t:end_t], flight_params['tau_target'][start_t:end_t], color='r', label='Tau Target')
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

ax[1, 1].plot(flight_params['time'][start_t:end_t], flight_params['D'][start_t:end_t], color='b', label='D Estimated', linewidth=0.5)
ax[1, 1].plot(flight_params['time'][start_t:end_t], flight_params['D_real'][start_t:end_t], color='r', label='D real')
ax[1, 1].set_title('Divergence')
ax[1, 1].set_xlabel('Time [s]')
ax[1, 1].set_ylabel('Divergence [1/s]')

if 'tau_target' in flight_params:
    D_target = [-1/tau for tau in flight_params['tau_target'][start_t:end_t]]
    ax[1, 1].plot(flight_params['time'][start_t:end_t], D_target, color='g', label='D Target')
    labs = [l.get_label() for l in ax[1, 1].lines]
    ax[1, 1].legend(ax[1, 1].lines, labs)

# Events
try:
    ax[1, 2].plot(flight_params['time'][start_t:end_t], flight_params['stored_events'][start_t:end_t], label='stored', color='b')
    ax[1, 2].plot(flight_params['time'][start_t:end_t], flight_params['incoming_events'][start_t:end_t], label='incoming', color='r')
    ax[1, 2].set_title('Events')
    ax[1, 2].set_xlabel('Time [s]')
    ax[1, 2].set_ylabel('# of events')
    ax[1, 2].legend()
except KeyError:
    ax[1, 2].set_title('<Nothing to show>')


# plt.tight_layout()
plt.show()
