from closedLoopTestBench import VerticalcdTTC

settings = [
    {
        'name': 'preset_int_PI',
        'controller_settings': {'Kp': 1000_000, 'Ki': 10_000}
    }
]

for s in settings:
    VerticalcdTTC(s)
