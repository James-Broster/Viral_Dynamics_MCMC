import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from multiprocessing import Pool

from virus_model import VirusModel
from model_fitting import ModelFitting
from utils import load_data, calculate_parameter_statistics, estimate_params_least_squares, analyze_epidemiological_metrics, post_mcmc_analysis, plot_treatment_effects
from constants import PARAM_NAMES, PARAM_STDS
from visualization import Visualization

def create_directory_structure(base_path):
    cases = ['fatal', 'non_fatal', 'least_squares']
    for case in cases:
        case_path = os.path.join(base_path, case)
        os.makedirs(case_path, exist_ok=True)
        
        if case != 'least_squares':
            os.makedirs(os.path.join(case_path, 'trace_plots'), exist_ok=True)
            os.makedirs(os.path.join(case_path, 'histograms'), exist_ok=True)
        
        os.makedirs(os.path.join(case_path, 'model_predictions'), exist_ok=True)

def main():
    base_output_dir = '/Users/james/Desktop/VD_MCMC/output'
    create_directory_structure(base_output_dir)

    time_f, observed_data_f, time_nf, observed_data_nf = load_data()
    
    # Estimate parameters using least squares
    estimated_params_f, estimated_params_nf = estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf)
    print("Estimated parameters from least squares (Fatal):", estimated_params_f)
    print("Estimated parameters from least squares (Non-Fatal):", estimated_params_nf)

    # Use these estimates to set the means for the MCMC
    PARAM_MEANS_F = estimated_params_f
    PARAM_MEANS_NF = estimated_params_nf

    num_iterations = 10000
    burn_in_period = int(num_iterations * 0.2)
    transition_period = int(num_iterations * 0.3)

    # Fit model for fatal cases
    chains_f, acceptance_rates_f, r_hat_f, acceptance_rates_over_time_f = ModelFitting.execute_parallel_mcmc(
        observed_data_f, time_f, 4, num_iterations, 
        burn_in_period, transition_period,
        PARAM_MEANS_F, PARAM_STDS, is_fatal=True
    )

    # Fit model for non-fatal cases
    chains_nf, acceptance_rates_nf, r_hat_nf, acceptance_rates_over_time_nf = ModelFitting.execute_parallel_mcmc(
        observed_data_nf, time_nf, 4, num_iterations, 
        burn_in_period, transition_period,
        PARAM_MEANS_NF, PARAM_STDS, is_fatal=False
    )

    # Calculate parameter statistics and plot MCMC diagnostics
    for chains, case, r_hat, acceptance_rates, acceptance_rates_over_time in [
        (chains_f, 'fatal', r_hat_f, acceptance_rates_f, acceptance_rates_over_time_f),
        (chains_nf, 'non_fatal', r_hat_nf, acceptance_rates_nf, acceptance_rates_over_time_nf)
    ]:
        case_dir = os.path.join(base_output_dir, case)
        medians, lower_ci, upper_ci = calculate_parameter_statistics(chains, burn_in_period)
        
        with open(os.path.join(case_dir, f'estimated_parameters.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Median', '2.5% CI', '97.5% CI'])
            for i, param_name in enumerate(PARAM_NAMES):
                writer.writerow([param_name, medians[i], lower_ci[i], upper_ci[i]])
        
        # Plot MCMC diagnostics
        for param_index, param_name in enumerate(PARAM_NAMES):
            Visualization.plot_trace(chains, param_index, param_name, burn_in_period, os.path.join(case_dir, 'trace_plots'), case)
            param_mean = PARAM_MEANS_F[param_index] if case == 'fatal' else PARAM_MEANS_NF[param_index]
            Visualization.plot_parameter_histograms(chains, burn_in_period, os.path.join(case_dir, 'histograms'), param_name, case, param_mean)
        
        Visualization.plot_rhat(r_hat, PARAM_NAMES, case_dir, case)
        Visualization.plot_acceptance_rates(acceptance_rates_over_time, PARAM_NAMES, case_dir, case)

        # Calculate and plot correlation heatmap
        correlations = ModelFitting.calculate_correlations(chains[:, burn_in_period:, :])
        Visualization.plot_correlation_heatmap(correlations, PARAM_NAMES, case_dir, case)

    # Plot model predictions
    Visualization.plot_model_predictions(time_f, time_nf, observed_data_f, observed_data_nf, 
                                        chains_f, chains_nf, burn_in_period, 
                                        os.path.join(base_output_dir, 'fatal', 'model_predictions'), 
                                        os.path.join(base_output_dir, 'non_fatal', 'model_predictions'))

    # Plot least squares fit
    Visualization.plot_least_squares_fit(time_f, observed_data_f, estimated_params_f, os.path.join(base_output_dir, 'least_squares'), 'fatal')
    Visualization.plot_least_squares_fit(time_nf, observed_data_nf, estimated_params_nf, os.path.join(base_output_dir, 'least_squares'), 'non_fatal')

    # Analyze epidemiological metrics
    p_fatal_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    analyze_epidemiological_metrics(chains_f, chains_nf, p_fatal_values, burn_in_period, base_output_dir)

    # Plot viral load curves
    for p_fatal in p_fatal_values:
        Visualization.plot_viral_load_curves(chains_f, chains_nf, p_fatal, burn_in_period, base_output_dir)

    # Perform post-MCMC analysis
    time_extended = np.linspace(0, 30, 301)  # 30 days, 301 points
    epsilon_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for chains, case in [
        (chains_f, 'fatal'),
        (chains_nf, 'non_fatal')
    ]:
        case_dir = os.path.join(base_output_dir, case)
        
        # Use all parameter sets after burn-in period and thinning
        parameter_samples = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])

        results = post_mcmc_analysis(parameter_samples, time_extended, num_t_star_samples=100, epsilon_values=epsilon_values)
        
        # Analyze and visualize results
        for epsilon in epsilon_values:
            epsilon_results = [r for r in results if r['epsilon'] == epsilon]
            t_stars = [r['t_star'] for r in epsilon_results]
            times_to_threshold = [r['time_to_threshold'] + 21 for r in epsilon_results]  # Add back the 21 days
            
            plt.figure(figsize=(10, 6))
            plt.scatter(t_stars, times_to_threshold, alpha=0.1)
            plt.xlabel('Treatment Start Time (days)')
            plt.ylabel('Time to Threshold (days)')
            plt.title(f'Time to Threshold vs Treatment Start Time ({case}, ε={epsilon})')
            plt.xlim(0, 25)  # Set x-axis from 0 to 25 days
            plt.ylim(0, 30)  # Set y-axis from 0 to 30 days
            plt.xticks(range(0, 26, 5))  # Set x-axis ticks every 5 days
            plt.yticks(range(0, 31, 5))  # Set y-axis ticks every 5 days
            plt.plot([0, 25], [0, 25], 'r--', alpha=0.5, label='y=x')  # Add diagonal line
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f'time_to_threshold_vs_treatment_start_{case}_epsilon_{epsilon}.png'))
            plt.close()
        
        # Save results to CSV
        with open(os.path.join(case_dir, f'post_mcmc_analysis_results_{case}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Treatment Start Time', 'Time to Threshold', 'Epsilon'])
            for result in results:
                writer.writerow([result['t_star'], result['time_to_threshold'] + 21, result['epsilon']])  # Add back the 21 days

    print("Post-MCMC analysis completed.")


    # Plot treatment effects
    t_star_values = [1, 3, 5, 7]  # Changed as requested
    epsilon_values = [0, 0.3, 0.6, 0.9]  # Same as used in post_mcmc_analysis
    for chains, case in [
        (chains_f, 'fatal'),
        (chains_nf, 'non_fatal')
    ]:
        case_dir = os.path.join(base_output_dir, case, 'treatment_effects')
        os.makedirs(case_dir, exist_ok=True)
        
        parameter_samples = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])
        plot_treatment_effects(parameter_samples, time_extended, t_star_values, epsilon_values, case_dir)

    print("Treatment effect plots completed.")

if __name__ == "__main__":
    main()