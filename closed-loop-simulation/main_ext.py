import os
import signal
import time

# import PyPANGU
# import matplotlib.pyplot as plt

from closedLoopTestBench import VerticalcdTTC
from closedLoopTestBench import VerticalcD
import subprocess

settings = [
    {
        'name': 'highres_1_Kalman_massaware',
        'controller_settings': {'Kp': 500_000},
        'filter_settings': {'type': 'kalman','Q': 0.001}
    },

    {
        'name': 'highres_1_Kalman_massaware',
        'controller_settings': {'Kp': 1000_000},
        'filter_settings': {'type': 'kalman','Q': 0.001}
    },

    # {
    #     'name': 'highres_1_ravg_massaware',
    #     'controller_settings': {'Kp': 500_000},
    #     'filter_settings': {'type': 'rolling_average_with_margin', 'bins': 5}
    # }
]

settings_full_ravg = [
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 1_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 2_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 3_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 5_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 1_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 2_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 3_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
]

settings_abr_ravg = [
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 5_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },

]

cd_D_settings = [
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 5_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    },

]

rocky_settings = [
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 4_000, 'init_velocity': -100}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 5_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 4_000, 'init_velocity': -100}
    },
]

sparse_settings = [
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 10},
        'physical_state': {'init_position': 4_000, 'init_velocity': -100}
    },
    {
        'name': '_ravg_massaware',
        'controller_settings': {'Kp': 5_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 4_000, 'init_velocity': -100}
    },
]

rocky_models = [
    'model_4',
    'model_5',
    'model_6',
    'model_7',
    ]

sparse_models = [
    'model_8',
    'model_9'
]

camera_models = [
    '50',
    '100'
]

# def showImage():
# #     server = PyPANGU.ServerPANGU(128,128)
# #     server.set_viewpoint_by_degree([0, 0, 2000, 0, -90, 0])
# #     plt.figure()
# #     plt.imshow(server.get_image())
# #     plt.show()

### cdTTC
# for m in camera_models:
#     p = subprocess.Popen(
#         f'C:/cygwin64/bin/bash.exe model_1\server_{m}.sh',
#         cwd='C:\PANGU\PANGU_5.00\models\\test_creator',
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE)
#     time.sleep(5)
#
#     for s in rocky_settings:
#         core_name = s['name']
#         s['name'] = f'cdTTC_res_' + s['name']
#         VerticalcdTTC(s)
#         s['name'] = core_name
#
#     p.terminate()
#     os.system("taskkill /f /im  viewer.exe")
#     time.sleep(5)


### constant Divergence
for m in camera_models:
    p = subprocess.Popen(
        f'C:/cygwin64/bin/bash.exe model_1\server_{m}.sh',
        cwd='C:\PANGU\PANGU_5.00\models\\test_creator',
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    time.sleep(5)

    for s in rocky_settings:
        core_name = s['name']
        s['name'] = f'cdTTC_res' + s['name']
        VerticalcD(s, D_target=-0.03)
        s['name'] = core_name

    p.terminate()
    os.system("taskkill /f /im  viewer.exe")
    time.sleep(5)