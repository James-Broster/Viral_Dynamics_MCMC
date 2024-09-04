import numpy as np
from utils import load_data, estimate_params_least_squares, create_directory_structure
from model_fitting import ModelFitting
from visualization import Visualization
from analysis import analyze_epidemiological_metrics, calculate_risk_and_burden, post_mcmc_analysis, calculate_risk_burden_sampled_tstar
import config
import os
import logging
import json
import time


def calculate_risk_burden_for_epsilon_tstar(chains, time_extended, base_output_dir):
    results = {}
    for epsilon in config.EPSILON_VALUES:
        for t_star in config.T_STAR_VALUES:
            logging.info(f"Calculating risk and burden for epsilon={epsilon}, t_star={t_star}")
            
            risk_burden = calculate_risk_and_burden(
                chains, 
                config.ISOLATION_PERIODS, 
                time_extended, 
                config.VIRAL_LOAD_THRESHOLDS, 
                {}, 
                base_output_dir,
                epsilon=epsilon,
                t_star=t_star
            )
            
            results[(epsilon, t_star)] = risk_burden
    
    return results

def main():
    create_directory_structure(config.BASE_OUTPUT_DIR)

    time_f, observed_data_f, time_nf, observed_data_nf = load_data()

    estimated_params_f, estimated_params_nf = estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf)
    print("Estimated parameters from least squares (Fatal):", estimated_params_f)
    print("Estimated parameters from least squares (Non-Fatal):", estimated_params_nf)

    burn_in_period = int(config.NUM_ITERATIONS * config.BURN_IN_FRACTION)
    transition_period = int(config.NUM_ITERATIONS * config.TRANSITION_FRACTION)

    chains_f, acceptance_rates_f, r_hat_f, acceptance_rates_over_time_f = ModelFitting.execute_parallel_mcmc(
        observed_data_f, time_f, config.NUM_CHAINS, config.NUM_ITERATIONS, 
        burn_in_period, transition_period,
        estimated_params_f, config.PARAM_STDS, is_fatal=True
    )

    chains_nf, acceptance_rates_nf, r_hat_nf, acceptance_rates_over_time_nf = ModelFitting.execute_parallel_mcmc(
        observed_data_nf, time_nf, config.NUM_CHAINS, config.NUM_ITERATIONS, 
        burn_in_period, transition_period,
        estimated_params_nf, config.PARAM_STDS, is_fatal=False
    )

    Visualization.plot_all_diagnostics(chains_f, chains_nf, burn_in_period, r_hat_f, r_hat_nf, 
                                       acceptance_rates_over_time_f, acceptance_rates_over_time_nf, 
                                       estimated_params_f, estimated_params_nf)

    Visualization.plot_model_predictions(time_f, time_nf, observed_data_f, observed_data_nf, 
                                         chains_f, chains_nf, burn_in_period, 
                                         config.BASE_OUTPUT_DIR)

    Visualization.plot_least_squares_fit(time_f, observed_data_f, estimated_params_f, config.BASE_OUTPUT_DIR, 'fatal')
    Visualization.plot_least_squares_fit(time_nf, observed_data_nf, estimated_params_nf, config.BASE_OUTPUT_DIR, 'non_fatal')

    time_extended = np.linspace(0, 30, 301)

    # Analyze epidemiological metrics
    analyze_epidemiological_metrics(chains_f, chains_nf, config.P_FATAL_VALUES, burn_in_period, config.BASE_OUTPUT_DIR)

    # Calculate risk and burden
    debug_shapes = {"initial": {"chains_f": chains_f.shape, "chains_nf": chains_nf.shape}}

    for chains, case in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
        start_time = time.time()
        processed_chains = chains[:, burn_in_period:, :]
        debug_shapes[f"{case}_processed"] = processed_chains.shape

        # Debug information
        logging.info(f"processed_chains type: {type(processed_chains)}")
        logging.info(f"processed_chains shape: {processed_chains.shape}")
        logging.info(f"processed_chains dtype: {processed_chains.dtype}")
        logging.info(f"First element of processed_chains: {processed_chains[0]}")

        # We'll keep this for other purposes if needed
        case_results = post_mcmc_analysis(processed_chains, time_extended, epsilon_values=config.EPSILON_VALUES)
        debug_shapes[f"{case}_post_mcmc"] = np.array(case_results).shape

        # Pass processed_chains instead of case_results
        risk_burden = calculate_risk_and_burden(processed_chains, config.ISOLATION_PERIODS, time_extended, 
                                                config.VIRAL_LOAD_THRESHOLDS, debug_shapes, 
                                                os.path.join(config.BASE_OUTPUT_DIR, case))
        
        Visualization.plot_risk_burden(risk_burden, case, os.path.join(config.BASE_OUTPUT_DIR, case))
        logging.info(f"Time taken for calculate_risk_and_burden and Visualization.plot_risk_burden for {case}: {time.time() - start_time:.2f} seconds")

    for chains, case in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
        processed_chains = chains[:, burn_in_period:, :]
        results = calculate_risk_burden_for_epsilon_tstar(processed_chains, time_extended, os.path.join(config.BASE_OUTPUT_DIR, case))
        Visualization.plot_risk_burden_epsilon_tstar(results, case, os.path.join(config.BASE_OUTPUT_DIR, case))


    for chains, case in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
        processed_chains = chains[:, burn_in_period:, :]
        
        # Calculate risk burden with sampled T_STAR
        results_sampled_tstar, t_star_samples = calculate_risk_burden_sampled_tstar(
            processed_chains, 
            config.ISOLATION_PERIODS, 
            time_extended, 
            config.VIRAL_LOAD_THRESHOLDS,
            debug_shapes,
            os.path.join(config.BASE_OUTPUT_DIR, case)
        )
        
        # Plot results with sampled T_STAR
        Visualization.plot_risk_burden_sampled_tstar(
            results_sampled_tstar, 
            t_star_samples,
            case, 
            os.path.join(config.BASE_OUTPUT_DIR, case)
        )

    # Plot treatment effects
    Visualization.plot_treatment_effects(chains_f, chains_nf, burn_in_period, time_extended, 
                                         config.T_STAR_VALUES, config.EPSILON_VALUES, config.BASE_OUTPUT_DIR)

    Visualization.plot_all_viral_load_curves(chains_f, burn_in_period, 'fatal', os.path.join(config.BASE_OUTPUT_DIR, 'fatal', 'model_predictions'))
    Visualization.plot_all_viral_load_curves(chains_nf, burn_in_period, 'non_fatal', os.path.join(config.BASE_OUTPUT_DIR, 'non_fatal', 'model_predictions'))

    # Save debug shapes to a file
    debug_file_path = os.path.join(config.BASE_OUTPUT_DIR, 'shape_debug.json')
    with open(debug_file_path, 'w') as f:
        json.dump(debug_shapes, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    logging.info(f"Shape debug information saved to {debug_file_path}")

if __name__ == "__main__":
    main()