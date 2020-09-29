from closedLoopTestBench import VerticalcdTTC
from closedLoopTestBench import VerticalcD

settings = [
    # {
    #     'name': 'highres_1_Kalman_massaware',
    #     'controller_settings': {'Kp': 500_000, 'Ki': 100_000},
    #     'filter_settings': {'type': 'kalman','Q': 0.001},
    #     'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    # },

    # {
    #     'name': 'highres_1_ravg_massaware',
    #     'controller_settings': {'Kp': 500_000},
    #     'filter_settings': {'type': 'rolling_average', 'bins': 5},
    #     'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    # },
    #
    {
        'name': 'test',
        'controller_settings': {'Kp': 4_000_000},
        'filter_settings': {'type': 'rolling_average', 'bins': 5},
        'physical_state': {'init_position': 10_000, 'init_velocity': -200}
    }
]

for s in settings:
    # VerticalcdTTC(s)
    VerticalcD(s)
