from closedLoopTestBench import VerticalcdTTC

settings = [
    {
        'name': 'unfilt_P',
        'controller_settings': {'Kp': 1000_000}
    },

    {
        'name': 'unfilt_P',
        'controller_settings': {'Kp': 10_000_000}
    },

    {
        'name': 'unfilt_P',
        'controller_settings': {'Kp': 100_000_000}
    }
]

for s in settings:
    VerticalcdTTC(s)
