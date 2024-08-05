PARAM_NAMES = ['alpha_f', 'beta', 'delta_f', 'gamma']
PARAM_MEANS = [9.79e-10, 5.1e-7, 2.27, 2000]  # These will be updated after least squares estimation
PARAM_STDS = [1e-9, 1e-6, 1.0, 1000]

PARAM_BOUNDS = {
    "alpha_f": [0, 5e-9],
    "beta": [0, 5e-6],
    "delta_f": [0, 5],
    "gamma": [0, 4000],
}

STEP_SIZES = {
    "alpha_f": 5e-11,
    "beta": 5e-8,
    "delta_f": 0.1,
    "gamma": 100,
}

FIXED_PARAMS = {
    "f1_0": 0.0047,
    "f2_0": 0.9953,
    "V_0": 3000
}