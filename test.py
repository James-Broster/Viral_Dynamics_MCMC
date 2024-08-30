import pytest
import numpy as np
from virus_model import VirusModel
from model_fitting import ModelFitting
from config import PARAM_BOUNDS, PARAM_NAMES, PARAM_STDS
from utils import estimate_params_least_squares, calculate_parameter_statistics
from visualization import Visualization

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

def test_virus_model_solve():
    parameters = [1e-9, 5e-7, 2.27, 2000, 0.9953, 30000]
    time = np.linspace(0, 10, 11)
    
    result = VirusModel.solve(parameters, time)
    
    assert result.shape == (11, 3)
    assert np.all(result >= 0)
    assert np.all(result[:, 0] + result[:, 1] <= 1 + 1e-10)  # f1 + f2 should always be <= 1 (with small tolerance for numerical errors)

def test_calculate_time_to_threshold():
    parameters = [1e-9, 5e-7, 2.27, 2000, 0.9953, 30000]
    epsilon = 0
    t_star = 0
    threshold = 4
    
    result = VirusModel.calculate_time_to_threshold(parameters, epsilon, t_star, threshold)
    
    assert 0 <= result <= 30

def test_estimate_params_least_squares():
    # Generate some mock data
    time_f = np.linspace(0, 14, 15)
    time_nf = np.linspace(0, 14, 15)
    observed_data_f = np.random.rand(15, 1) * 5 + 4  # Random data between 4 and 9
    observed_data_nf = np.random.rand(15, 1) * 5 + 2  # Random data between 2 and 7
    
    result_f, result_nf = estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf)
    
    assert len(result_f) == len(PARAM_NAMES)
    assert len(result_nf) == len(PARAM_NAMES)
    
    for param_name, value_f, value_nf in zip(PARAM_NAMES, result_f, result_nf):
        assert PARAM_BOUNDS[param_name][0] <= value_f <= PARAM_BOUNDS[param_name][1]
        assert PARAM_BOUNDS[param_name][0] <= value_nf <= PARAM_BOUNDS[param_name][1]

def test_calculate_log_likelihood():
    parameters = [1e-9, 5e-7, 2.27, 2000, 0.9953, 30000]
    time = np.array([0, 1, 2, 3, 4, 5])
    observed_data = np.array([[4.51], [5.18], [5.83], [6.36], [6.39], [6.60]])
    sigma = 0.1
    
    log_likelihood = ModelFitting.calculate_log_likelihood(parameters, time, observed_data, sigma)
    
    assert isinstance(log_likelihood, float)
    assert log_likelihood <= 0  # Log-likelihood should be non-positive

def test_propose_new_parameters():
    np.random.seed(42)  # For reproducibility
    current_parameters = np.array([1e-9, 5e-7, 2.27, 2000, 0.9953, 30000])
    
    for _ in range(1000):  # Test many times to ensure bounds are always respected
        proposed_parameters = ModelFitting.propose_new_parameters(current_parameters)
        assert len(proposed_parameters) == len(current_parameters)
        for prop, curr, param_name in zip(proposed_parameters, current_parameters, PARAM_NAMES):
            assert PARAM_BOUNDS[param_name][0] <= prop <= PARAM_BOUNDS[param_name][1], f"{param_name} out of bounds"

def test_calculate_rhat():
    chains = np.array([
        [[0, 1, 2, 3, 4, 5]] * 6,
        [[0, 1, 2, 3, 4, 5]] * 6
    ])
    
    r_hat = ModelFitting.calculate_rhat(chains)
    
    assert len(r_hat) == 6
    assert np.all(r_hat >= 1)

def test_calculate_parameter_statistics():
    chains = np.random.rand(4, 1000, 6)  # 4 chains, 1000 iterations, 6 parameters
    burn_in_period = 200
    
    medians, lower_ci, upper_ci = calculate_parameter_statistics(chains, burn_in_period)
    
    assert len(medians) == 6
    assert len(lower_ci) == 6
    assert len(upper_ci) == 6
    assert np.all(lower_ci <= medians)
    assert np.all(medians <= upper_ci)

def test_visualization_methods():
    # This is more of a smoke test to ensure the visualization methods don't raise exceptions
    chains = np.random.rand(4, 1000, 6)
    burn_in_period = 200
    output_dir = "test_output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        Visualization.plot_trace(chains, 0, "test_param", burn_in_period, output_dir, "test_case")
        Visualization.plot_rhat(np.random.rand(6), PARAM_NAMES, output_dir, "test_case")
        Visualization.plot_correlation_heatmap(np.random.rand(6, 6), PARAM_NAMES, output_dir, "test_case")
        Visualization.plot_parameter_histograms(chains, burn_in_period, output_dir, PARAM_NAMES[0], "test_case", np.random.rand(6))
        Visualization.plot_metric_distribution(np.random.rand(1000), 0.5, output_dir, "test_metric")
        Visualization.plot_viral_load_curves(chains, chains, 0.5, burn_in_period, output_dir)
        Visualization.plot_least_squares_fit(np.linspace(0, 14, 15), np.random.rand(15, 1), np.random.rand(6), output_dir, "test_case")
    except Exception as e:
        pytest.fail(f"Visualization method raised an exception: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])