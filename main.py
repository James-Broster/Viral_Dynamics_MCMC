import os
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool

from virus_model import VirusModel
from model_fitting import ModelFitting
from visualization import Visualization
from utils import load_data, calculate_parameter_statistics, estimate_params_least_squares
from constants import PARAM_NAMES, PARAM_MEANS, PARAM_STDS, FIXED_PARAMS

def process_therapy_combination(args):
    chains, burn_in_period, epsilon, t_star, time, observed_data, output_dir = args
    median, lower_ci, upper_ci = ModelFitting.calculate_time_to_threshold_stats(
        chains, burn_in_period, epsilon, t_star
    )
    Visualization.plot_therapy_comparison(
        time, observed_data, chains, burn_in_period, epsilon, t_star, output_dir
    )
    return epsilon, t_star, median, lower_ci, upper_ci

def main():
    output_dir = '/Users/james/Desktop/VD_MCMC/output'
    os.makedirs(output_dir, exist_ok=True)

    time, observed_data_fatal = load_data()
    
    # Estimate parameters using least squares
    estimated_params = estimate_params_least_squares(time, observed_data_fatal)
    print("Estimated parameters from least squares:", estimated_params)
    
    # Update PARAM_MEANS with estimated values
    for i, param_name in enumerate(PARAM_NAMES):
        PARAM_MEANS[i] = estimated_params[i]

    num_iterations = 100000
    burn_in_period = int(num_iterations * 0.2)
    transition_period = int(num_iterations * 0.3)

    # Prepare initial conditions
    initial_conditions = [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]

    try:
        chains, viral_loads, acceptance_rates, r_hat = ModelFitting.execute_parallel_mcmc(
            observed_data_fatal, 4, num_iterations, 
            time, burn_in_period, transition_period,
            PARAM_MEANS, PARAM_STDS, initial_conditions
        )

        for i, param_name in enumerate(PARAM_NAMES):
            Visualization.plot_trace(chains, i, param_name, burn_in_period, output_dir)

        Visualization.plot_model_prediction(
            time, observed_data_fatal, chains, viral_loads, np.concatenate([estimated_params, initial_conditions]), burn_in_period, output_dir, y_min=0, y_max=10
        )

        Visualization.plot_rhat(r_hat, PARAM_NAMES, output_dir)

        correlations = ModelFitting.calculate_correlations(chains[:, burn_in_period:, :])
        
        Visualization.plot_correlation_heatmap(correlations, PARAM_NAMES, output_dir)

        for param_name in PARAM_NAMES:
            Visualization.plot_parameter_histograms(chains, burn_in_period, output_dir, param_name)

        ess_values = ModelFitting.calculate_multichain_ess(chains[:, burn_in_period:, :])

        with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
            f.write("R-hat values:\n")
            for param_name, r in zip(PARAM_NAMES, r_hat):
                f.write(f"{param_name}: {r:.4f}\n")
            
            f.write("\nAcceptance rates:\n")
            for param_name, rate in zip(PARAM_NAMES, np.mean(acceptance_rates, axis=0)):
                f.write(f"{param_name}: {rate:.4f}\n")
            
            f.write("\nEffective Sample Sizes:\n")
            for param_name, ess in zip(PARAM_NAMES, ess_values):
                f.write(f"{param_name}: {ess:.2f}\n")

        medians, lower_ci, upper_ci = calculate_parameter_statistics(chains, burn_in_period)
        
        with open(os.path.join(output_dir, 'parameter_statistics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Median', '2.5% CI', '97.5% CI'])
            for i, param_name in enumerate(PARAM_NAMES):
                writer.writerow([param_name, medians[i], lower_ci[i], upper_ci[i]])

        # Therapy analysis
        epsilon_values = np.round(np.arange(0.0, 1.1, 0.1), 2)  # 0.0 to 1.0 with 0.1 increments
        t_star_values = list(range(0, 16))  # 0 to 15 with 1 day increments

        # Prepare arguments for parallel processing
        args_list = [
            (chains, burn_in_period, epsilon, t_star, time, observed_data_fatal, output_dir)
            for epsilon in epsilon_values
            for t_star in t_star_values
        ]

        # Use multiprocessing to parallelize the calculations
        with Pool(processes=8) as pool:
            results = list(tqdm(pool.imap(process_therapy_combination, args_list), 
                                total=len(args_list), desc="Therapy Combinations"))

        # Process results
        median_results = np.zeros((len(epsilon_values), len(t_star_values)))
        lower_ci_results = np.zeros((len(epsilon_values), len(t_star_values)))
        upper_ci_results = np.zeros((len(epsilon_values), len(t_star_values)))

        for result in results:
            epsilon, t_star, median, lower_ci, upper_ci = result
            i = np.where(epsilon_values == epsilon)[0][0]
            j = t_star_values.index(t_star)
            median_results[i, j] = median
            lower_ci_results[i, j] = lower_ci
            upper_ci_results[i, j] = upper_ci

        # Visualize heatmap results
        Visualization.plot_time_to_threshold_heatmap(
            median_results, lower_ci_results, upper_ci_results, 
            epsilon_values, t_star_values, output_dir
        )

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()