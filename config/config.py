import os

class Config:
    PARAM_NAMES = ['alpha_f', 'beta', 'delta_f', 'gamma', 'f2_0', 'V_0']
    PARAM_STDS = [1e-9, 1e-6, 1.0, 1000, 0.005, 3000]

    PARAM_BOUNDS = {
        "alpha_f": [5e-11, 1e-7],
        "beta": [5e-9, 1e-4],
        "delta_f": [0, 5],
        "gamma": [0, 4000],
        "f2_0": [0.8, 0.9999],
        "V_0": [20000, 40000]
    }

    STEP_SIZES = {
        "alpha_f": 2e-10,
        "beta": 2e-7,
        "delta_f": 0.1,
        "gamma": 100,
        "f2_0": 0.001,
        "V_0": 1000
    }

    BASE_OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
    NUM_ITERATIONS = 10
    BURN_IN_PERIOD = int(NUM_ITERATIONS * 0.2)
    TRANSITION_PERIOD = int(NUM_ITERATIONS * 0.3)
    NUM_CHAINS = 4
    EPSILON_VALUES = [0.0, 0.3, 0.6, 0.9]
    P_FATAL_VALUES = [0.0, 0.4, 0.6, 0.8, 1.0]
    T_STAR_VALUES = [1, 3, 5, 7]
    VIRAL_LOAD_THRESHOLDS = [3, 4, 5]  # log10 viral load thresholds
    ISOLATION_PERIODS = list(range(31))  # 0 to 30 days
