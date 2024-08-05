import pytest
import numpy as np
from virus_model import VirusModel
from model_fitting import ModelFitting
from constants import FIXED_PARAMS, PARAM_BOUNDS
from scipy.integrate import odeint
import math

def test_virus_model_ode():
    y = [0.0047, 0.9953, 30000]
    t = 0
    params = [1e-9, 5e-7, 2.27, 2000, 0, 0]  # alpha_f, beta, delta_f, gamma, epsilon, t_star
    
    result = VirusModel.ode(y, t, *params)
    expected = [-0.000040641, -0.000029859, 213900]
    
    print(f"\nODE test - Result: {result}")
    print(f"ODE test - Expected: {expected}")
    print(f"Relative difference: {np.abs((np.array(result) - np.array(expected)) / np.array(expected))}")
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_calculate_time_to_threshold():
    parameters = [1e-9, 5e-7, 2.27, 2000, 0.0047, 0.9953, 30000]
    epsilon = 0
    t_star = 0
    threshold = 4
    
    result = VirusModel.calculate_time_to_threshold(parameters, epsilon, t_star, threshold)
    
    # The viral load starts above 4 log10, so it should take some time to drop below
    assert result > 0
    assert result < 30  # It should drop below threshold before 30 days
    
    # Calculate the exact time (this requires solving the ODE)
    time = np.linspace(0, 30, 301)
    solution = VirusModel.solve(parameters, time, epsilon, t_star)
    log_V = solution[:, 2]
    expected_time = time[np.where(log_V <= threshold)[0][0]]
    
    np.testing.assert_allclose(result, expected_time, rtol=1e-3)

def test_calculate_time_to_threshold_never_below():
    # Use parameters that keep viral load high
    parameters = [1e-9, 5e-7, 0.1, 2000, 0.0047, 0.9953, 30000]
    epsilon = 0
    t_star = 0
    
    result = VirusModel.calculate_time_to_threshold(parameters, epsilon, t_star)
    assert result == 30, f"Expected 30, but got {result}"

def test_calculate_log_likelihood():
    parameters = [1e-9, 5e-7, 2.27, 2000]
    time = np.array([0, 1, 2, 3, 4, 5])
    observed_data = np.array([[4.51], [5.18], [5.83], [6.36], [6.39], [6.60]])
    sigma = 0.1
    
    log_likelihood = ModelFitting.calculate_log_likelihood(parameters, time, observed_data, sigma)
    
    # Calculate expected log-likelihood
    full_params = np.concatenate([parameters, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
    model_data = VirusModel.solve(full_params, time)
    model_log_virusload = model_data[:, 2].reshape(-1, 1)
    residuals = observed_data - model_log_virusload
    n = len(observed_data)
    expected_log_likelihood = -0.5 * (n * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / sigma**2)
    
    np.testing.assert_allclose(log_likelihood, expected_log_likelihood, rtol=1e-6)

def test_propose_new_parameters():
    np.random.seed(42)  # For reproducibility
    current_parameters = np.array([1e-9, 5e-7, 2.27, 2000])
    
    for _ in range(1000):  # Test many times to ensure bounds are always respected
        proposed_parameters = ModelFitting.propose_new_parameters(current_parameters)
        assert len(proposed_parameters) == len(current_parameters)
        for prop, curr, param_name in zip(proposed_parameters, current_parameters, ['alpha_f', 'beta', 'delta_f', 'gamma']):
            assert PARAM_BOUNDS[param_name][0] <= prop <= PARAM_BOUNDS[param_name][1], f"{param_name} out of bounds"

def test_calculate_rhat():
    chains = np.array([
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]
    ])
    n = chains.shape[1]  # Length of each chain
    m = chains.shape[0]  # Number of chains

    # Mean of each chain
    chain_means = np.mean(chains, axis=1)

    # Variance within each chain
    chain_variances = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_variances)

    # Mean of means
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)

    # Estimate of target distribution variance
    V_hat = (n - 1) / n * W + B / n

    # Calculate R-hat
    R_hat = np.sqrt(V_hat / W)

    # Use the implemented function
    r_hat = ModelFitting.calculate_rhat(chains)

    print(f"R-hat calculated manually: {R_hat}")
    print(f"R-hat calculated by function: {r_hat}")
    print(f"Chain means: {chain_means}")
    print(f"Chain variances: {chain_variances}")

    np.testing.assert_allclose(r_hat, R_hat, rtol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__])