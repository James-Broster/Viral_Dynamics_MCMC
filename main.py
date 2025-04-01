import numpy as np
import pandas as pd
import os
import logging

from src.utils.data_loading import load_data
from src.utils.statistical_utils import estimate_params_least_squares
from src.utils.directory_setup import setup_directories
from src.model.mcmc_fitting import ModelFitting
from src.visualization.mcmc_diagnostics import MCMCDiagnostics
from src.visualization.model_predictions import ModelPredictions
from src.analysis.epidemiological_metrics import analyze_epidemiological_metrics
from src.analysis.risk_burden import (
    analyze_chains,
    calculate_and_plot_risk_burden,
    calculate_risk_burden_for_epsilon_tstar,
    calculate_risk_burden_fixed_tstar
)
from src.visualization.risk_burden_plots import RiskBurdenPlots
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_parameter_statistics(chains, burn_in_period):
    """
    Calculate the median and confidence intervals for each parameter 
    after discarding the burn-in period.
    """
    processed_chains = chains[:, burn_in_period:, :]
    flattened_chains = processed_chains.reshape(-1, processed_chains.shape[-1])
    
    medians = np.median(flattened_chains, axis=0)
    ci_lower = np.percentile(flattened_chains, 2.5, axis=0)
    ci_upper = np.percentile(flattened_chains, 97.5, axis=0)
    
    return medians, ci_lower, ci_upper


def save_parameter_statistics(medians, ci_lower, ci_upper, param_names, output_path):
    """
    Save parameter statistics (median and confidence intervals) to a CSV file.
    """
    data = {
        'Parameter': param_names,
        'Median': medians,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Parameter statistics saved to {output_path}")


def analyze_treatment_effects(chains, case_type, config, directories, time_extended, debug_shapes):
    """
    Analyze treatment effects for a given case type.
    
    This function computes the risk burden over a grid of treatment parameters (epsilon and t_star),
    computes risk burden with fixed t_star values, and then plots the corresponding risk burden and 
    viral load curves.
    
    Parameters:
        chains: MCMC chains for the case (fatal or non_fatal).
        case_type: String 'fatal' or 'non_fatal'.
        config: Configuration object with model parameters.
        directories: Dictionary of output directories.
        time_extended: Extended time array for simulation.
        debug_shapes: Debug information from previous analyses.
    """
    logger.info(f"Starting treatment effects analysis for {case_type} case")
    case_dir = directories[case_type]
    treatment_dir = directories[f'{case_type}_treatment_effects']

    # Calculate and plot risk burden over a grid of epsilon and t_star values.
    risk_burden_grid, no_treatment_results = calculate_risk_burden_for_epsilon_tstar(chains, time_extended, case_dir)
    RiskBurdenPlots.plot_risk_burden_epsilon_tstar(
        risk_burden_grid,
        no_treatment_results,
        case_type,
        treatment_dir
    )
    logger.info(f"Risk burden grid plotted for {case_type}")

    # Calculate and plot risk burden with fixed t_star values.
    risk_fixed, t_star_values, weights = calculate_risk_burden_fixed_tstar(
        chains,
        config.ISOLATION_PERIODS,
        time_extended,
        config.VIRAL_LOAD_THRESHOLDS,
        debug_shapes,
        case_dir
    )
    RiskBurdenPlots.plot_risk_burden_fixed_tstar(
        risk_fixed,
        t_star_values,
        weights,
        no_treatment_results,
        case_type,
        treatment_dir
    )
    logger.info(f"Fixed t_star risk burden plotted for {case_type}")

    # Plot viral load curves.
    ModelPredictions.plot_viral_load_curves(
        chains,
        config.BURN_IN_PERIOD,
        treatment_dir,
        case_type,
        time_extended
    )
    logger.info(f"Viral load curves plotted for {case_type}")


def main():
    config = Config()
    time_extended = np.linspace(0, 30, 301)
    
    # Set up output directories
    directories = setup_directories(config.BASE_OUTPUT_DIR)

    # Load observed data for fatal and non-fatal cases
    time_f, observed_data_f, time_nf, observed_data_nf = load_data()

    # Estimate initial parameters using least squares
    estimated_params_f, estimated_params_nf = estimate_params_least_squares(
        time_f, time_nf, observed_data_f, observed_data_nf
    )
    
    # Execute parallel MCMC for fatal cases
    chains_f, acceptance_rates_f, r_hat_f, acceptance_rates_over_time_f = ModelFitting.execute_parallel_mcmc(
        observed_data_f, time_f, config.NUM_CHAINS, config.NUM_ITERATIONS,
        config.BURN_IN_PERIOD, config.TRANSITION_PERIOD,
        estimated_params_f, config.PARAM_STDS, is_fatal=True
    )
    medians_f, ci_lower_f, ci_upper_f = calculate_parameter_statistics(chains_f, config.BURN_IN_PERIOD)
    save_parameter_statistics(
        medians_f, ci_lower_f, ci_upper_f,
        config.PARAM_NAMES,
        os.path.join(directories['fatal_model_predictions'], 'parameters.csv')
    )

    # Execute parallel MCMC for non-fatal cases
    chains_nf, acceptance_rates_nf, r_hat_nf, acceptance_rates_over_time_nf = ModelFitting.execute_parallel_mcmc(
        observed_data_nf, time_nf, config.NUM_CHAINS, config.NUM_ITERATIONS,
        config.BURN_IN_PERIOD, config.TRANSITION_PERIOD,
        estimated_params_nf, config.PARAM_STDS, is_fatal=False
    )
    medians_nf, ci_lower_nf, ci_upper_nf = calculate_parameter_statistics(chains_nf, config.BURN_IN_PERIOD)
    save_parameter_statistics(
        medians_nf, ci_lower_nf, ci_upper_nf,
        config.PARAM_NAMES,
        os.path.join(directories['non_fatal_model_predictions'], 'parameters.csv')
    )

    # Plot diagnostics and model predictions
    MCMCDiagnostics.plot_all_diagnostics(
        chains_f, chains_nf, config.BURN_IN_PERIOD, r_hat_f, r_hat_nf,
        acceptance_rates_over_time_f, acceptance_rates_over_time_nf,
        estimated_params_f, estimated_params_nf,
        directories['fatal_mcmc_diagnostics'], directories['non_fatal_mcmc_diagnostics']
    )
    ModelPredictions.plot_model_predictions(
        time_f, time_nf, observed_data_f, observed_data_nf,
        chains_f, chains_nf, config.BURN_IN_PERIOD,
        directories['fatal_model_predictions'], directories['non_fatal_model_predictions']
    )
    ModelPredictions.plot_least_squares_fit(
        time_f, observed_data_f, estimated_params_f,
        directories['fatal_model_predictions'], 'fatal'
    )
    ModelPredictions.plot_least_squares_fit(
        time_nf, observed_data_nf, estimated_params_nf,
        directories['non_fatal_model_predictions'], 'non_fatal'
    )

    # Analyze epidemiological metrics
    analyze_epidemiological_metrics(chains_f, chains_nf, config.P_FATAL_VALUES, config.BURN_IN_PERIOD, directories['base'])

    # Analyze chains and calculate risk burden across scenarios
    results, debug_shapes = analyze_chains(chains_f, chains_nf, config.BURN_IN_PERIOD, time_extended)
    risk_burdens = calculate_and_plot_risk_burden(
        chains_f, chains_nf, config.BURN_IN_PERIOD, time_extended,
        directories['fatal'], directories['non_fatal']
    )

    # Treatment effect analysis for each case (fatal and non-fatal)
    for chains, case_type in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
        analyze_treatment_effects(chains, case_type, config, directories, time_extended, debug_shapes)

    logger.info("Analysis and visualization completed.")


if __name__ == "__main__":
    main()
