import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
from virus_model import VirusModel
from scipy.stats import gamma
from config import PARAM_NAMES, PARAM_BOUNDS
from scipy.optimize import minimize


def create_directory_structure(base_path):
    for case in ['fatal', 'non_fatal']:
        case_path = os.path.join(base_path, case)
        os.makedirs(case_path, exist_ok=True)
        
        os.makedirs(os.path.join(case_path, 'mcmc_diagnostics', 'trace_plots'), exist_ok=True)
        os.makedirs(os.path.join(case_path, 'mcmc_diagnostics', 'histograms'), exist_ok=True)
        os.makedirs(os.path.join(case_path, 'model_predictions'), exist_ok=True)
        os.makedirs(os.path.join(case_path, 'treatment_effects'), exist_ok=True)
        os.makedirs(os.path.join(case_path, 'post_mcmc_analysis'), exist_ok=True)

    os.makedirs(os.path.join(base_path, 'least_squares'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'epidemiological_metrics'), exist_ok=True)

def load_data():
    data_fatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0],
        'virusload': [32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, 141253.8]
    })
    data_nonfatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'virusload': [165958.7, 52480.75, 2754229.0, 3548134.0, 1288250.0, 1584893.0, 199526.2, 371535.2, 107151.9, 14791.08, 31622.78, 70794.58, 7413.102, 9772.372, 5623.413]
    })
    time_fatal = data_fatal['time'].values
    time_nonfatal = data_nonfatal['time'].values
    observed_data_fatal = np.log10(data_fatal['virusload'].values).reshape(-1, 1)
    observed_data_nonfatal = np.log10(data_nonfatal['virusload'].values).reshape(-1, 1)
    return time_fatal, observed_data_fatal, time_nonfatal, observed_data_nonfatal

def calculate_parameter_statistics(chains, burn_in_period):
    latter_chains = chains[:, burn_in_period::100, :]  # Thinning: use every 100th set after burn-in
    flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
    
    medians = np.median(flattened_chains, axis=0)
    lower_ci = np.percentile(flattened_chains, 2.5, axis=0)
    upper_ci = np.percentile(flattened_chains, 97.5, axis=0)
    
    return medians, lower_ci, upper_ci

def estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf):
    def residuals(params, time, observed_data):
        model_data = VirusModel.solve(params, time)
        return (observed_data.flatten() - model_data[:, 2]).flatten()

    initial_guess = [1e-9, 1e-6, 2, 2000, 0.9953, 30000]
    bounds = ([PARAM_BOUNDS[param][0] for param in PARAM_NAMES],
              [PARAM_BOUNDS[param][1] for param in PARAM_NAMES])
    
    # Estimate for fatal cases
    result_f = least_squares(residuals, initial_guess, args=(time_f, observed_data_f), bounds=bounds)
    
    # Estimate for non-fatal cases
    result_nf = least_squares(residuals, initial_guess, args=(time_nf, observed_data_nf), bounds=bounds)
    
    return result_f.x, result_nf.x

def fit_gamma_to_median_iqr(median, iqr, offset=1):
    def objective(params):
        shape, scale = params
        q1, q3 = gamma.ppf([0.25, 0.75], shape, scale=scale) + offset
        model_median = gamma.ppf(0.5, shape, scale=scale) + offset
        return ((q3 - q1) - (iqr[1] - iqr[0]))**2 + (model_median - (median + offset))**2

    result = minimize(objective, [2, 2], method='Nelder-Mead')
    return result.x

def sample_t_star():
    median = 3.5
    iqr = (2, 6)
    offset = 1
    shape, scale = fit_gamma_to_median_iqr(median, iqr, offset)
    t_star = gamma.rvs(shape, scale=scale) + offset
    return np.clip(t_star, offset, 21)

def calculate_time_to_threshold(solution, time, threshold=4):
    log_V = solution[:, 2]
    threshold_crossed = np.where(log_V < threshold)[0]
    if len(threshold_crossed) > 0:
        return time[threshold_crossed[0]] - 21  # Subtract isolation period
    else:
        return 9  # 30 - 21, if threshold is never crossed