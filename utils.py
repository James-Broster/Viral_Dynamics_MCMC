import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from virus_model import VirusModel
from constants import FIXED_PARAMS, PARAM_BOUNDS

def load_data():
    data = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'virusload': [32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, None, 141253.8]
    })
    data = data.dropna()
    time = data['time'].values
    observed_data_fatal = np.log10(data['virusload'].values).reshape(-1, 1)  # Convert to log10
    return time, observed_data_fatal

def calculate_parameter_statistics(chains, burn_in_period):
    latter_chains = chains[:, burn_in_period::100, :]  # Thinning: use every 100th set after burn-in
    flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
    
    medians = np.median(flattened_chains, axis=0)
    lower_ci = np.percentile(flattened_chains, 2.5, axis=0)
    upper_ci = np.percentile(flattened_chains, 97.5, axis=0)
    
    return medians, lower_ci, upper_ci

def estimate_params_least_squares(time, observed_data):
    def residuals(params):
        alpha_f, beta, delta_f, gamma = params
        model_data = VirusModel.solve(np.concatenate([params, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]]), time)
        return (observed_data.flatten() - model_data[:, 2]).flatten()

    initial_guess = [1e-9, 1e-6, 2, 2000]
    bounds = ([PARAM_BOUNDS['alpha_f'][0], PARAM_BOUNDS['beta'][0], PARAM_BOUNDS['delta_f'][0], PARAM_BOUNDS['gamma'][0]],
              [PARAM_BOUNDS['alpha_f'][1], PARAM_BOUNDS['beta'][1], PARAM_BOUNDS['delta_f'][1], PARAM_BOUNDS['gamma'][1]])
    
    result = least_squares(residuals, initial_guess, bounds=bounds)
    return result.x