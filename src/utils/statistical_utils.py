import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize, least_squares
from src.model.virus_model import cached_solve
from config.config import Config

def fit_gamma_to_mean_iqr(mean, iqr):
    def objective(params):
        shape, scale = params
        model_mean = shape * scale
        model_iqr = gamma.ppf(0.75, shape, scale=scale) - gamma.ppf(0.25, shape, scale=scale)
        return (model_mean - mean)**2 + (model_iqr - (iqr[1] - iqr[0]))**2

    result = minimize(objective, [2, 2], method='Nelder-Mead')
    return result.x

def sample_t_star():
    mean = 4.8
    iqr = (2, 6)
    shape, scale = fit_gamma_to_mean_iqr(mean, iqr)
    t_star = gamma.rvs(shape, scale=scale)
    return max(t_star, 0.1)  # Ensure t_star is at least 0.1

def calculate_time_to_threshold(solution, time, threshold=4):
    log_V = solution[:, 2]
    threshold_crossed = np.where(log_V < threshold)[0]
    if len(threshold_crossed) > 0:
        return time[threshold_crossed[0]] - 21  # Subtract isolation period
    else:
        return 9  # 30 - 21, if threshold is never crossed

def estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf):
    config = Config()

    def residuals(params, time, observed_data):
        model_data = cached_solve(params, time)
        return (observed_data.flatten() - model_data[:, 2]).flatten()

    initial_guess = [1e-9, 1e-6, 2, 2000, 0.9953, 30000]
    bounds = ([config.PARAM_BOUNDS[param][0] for param in config.PARAM_NAMES],
              [config.PARAM_BOUNDS[param][1] for param in config.PARAM_NAMES])
    
    # Estimate for fatal cases
    result_f = least_squares(residuals, initial_guess, args=(time_f, observed_data_f), bounds=bounds)
    
    # Estimate for non-fatal cases
    result_nf = least_squares(residuals, initial_guess, args=(time_nf, observed_data_nf), bounds=bounds)
    
    return result_f.x, result_nf.x