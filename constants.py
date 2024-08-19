PARAM_NAMES = ['alpha_f', 'beta', 'delta_f', 'gamma', 'f2_0', 'V_0']
PARAM_STDS = [1e-9, 1e-6, 1.0, 1000, 0.005, 5000]  # Increased std for alpha_f, beta, f2_0

PARAM_BOUNDS = {
    "alpha_f": [5e-11, 1e-7],  # Extended upper bound
    "beta": [5e-9, 1e-4],      # Extended upper bound
    "delta_f": [0, 5],
    "gamma": [0, 4000],
    "f2_0": [0.8, 0.9999],     # Adjusted both bounds
    "V_0": [20000, 40000]
}

STEP_SIZES = {
    "alpha_f": 2e-10,  # Reduced from 1e-9
    "beta": 2e-7,      # Reduced from 1e-6
    "delta_f": 0.1,    # Reduced from 0.5
    "gamma": 100,      # Reduced from 500
    "f2_0": 0.001,     # Reduced from 0.005
    "V_0": 1000        # Reduced from 5000
}