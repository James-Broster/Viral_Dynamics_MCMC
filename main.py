import numpy as np
import os
import logging
from src.utils.data_loading import load_data
from src.utils.statistical_utils import estimate_params_least_squares
from src.utils.directory_setup import setup_directories
from src.model.mcmc_fitting import ModelFitting
from src.visualization.mcmc_diagnostics import MCMCDiagnostics
from src.visualization.model_predictions import ModelPredictions
from src.analysis.epidemiological_metrics import analyze_epidemiological_metrics
from src.analysis.risk_burden import analyze_chains, calculate_and_plot_risk_burden, calculate_risk_burden_for_epsilon_tstar, calculate_risk_burden_sampled_tstar
from src.visualization.risk_burden_plots import RiskBurdenPlots
from config.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = Config()
    time_extended = np.linspace(0, 30, 301)
    # Set up directory structure
    directories = setup_directories(config.BASE_OUTPUT_DIR)

    # Load data
    time_f, observed_data_f, time_nf, observed_data_nf = load_data()

    # Estimate initial parameters
    estimated_params_f, estimated_params_nf = estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf)
    
    # Execute MCMC
    chains_f, acceptance_rates_f, r_hat_f, acceptance_rates_over_time_f = ModelFitting.execute_parallel_mcmc(
        observed_data_f, time_f, config.NUM_CHAINS, config.NUM_ITERATIONS, 
        config.BURN_IN_PERIOD, config.TRANSITION_PERIOD,
        estimated_params_f, config.PARAM_STDS, is_fatal=True
    )

    chains_nf, acceptance_rates_nf, r_hat_nf, acceptance_rates_over_time_nf = ModelFitting.execute_parallel_mcmc(
        observed_data_nf, time_nf, config.NUM_CHAINS, config.NUM_ITERATIONS, 
        config.BURN_IN_PERIOD, config.TRANSITION_PERIOD,
        estimated_params_nf, config.PARAM_STDS, is_fatal=False
    )

    # Plotting and analysis
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

    ModelPredictions.plot_least_squares_fit(time_f, observed_data_f, estimated_params_f, directories['fatal_model_predictions'], 'fatal')
    ModelPredictions.plot_least_squares_fit(time_nf, observed_data_nf, estimated_params_nf, directories['non_fatal_model_predictions'], 'non_fatal')

    analyze_epidemiological_metrics(chains_f, chains_nf, config.P_FATAL_VALUES, config.BURN_IN_PERIOD, directories['base'])

    results, debug_shapes = analyze_chains(chains_f, chains_nf, config.BURN_IN_PERIOD, time_extended)
    risk_burdens = calculate_and_plot_risk_burden(
        chains_f, chains_nf, config.BURN_IN_PERIOD, time_extended,
        directories['fatal'], directories['non_fatal']
    )

    # Treatment effect analysis
    for chains, case_type in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
        print(f"[DEBUG] Processing {case_type} case")
        case_dir = directories[case_type]
        
        # Risk burden for different epsilon and t_star values
        print(f"[DEBUG] Calculating risk burden for different epsilon and t_star values ({case_type})")
        risk_burden_epsilon_tstar, no_treatment_results = calculate_risk_burden_for_epsilon_tstar(chains, time_extended, case_dir)
        
        print(f"[DEBUG] Plotting risk burden for different epsilon and t_star values ({case_type})")
        RiskBurdenPlots.plot_risk_burden_epsilon_tstar(
            risk_burden_epsilon_tstar, 
            no_treatment_results, 
            case_type, 
            directories[f'{case_type}_treatment_effects']
        )

        # Risk burden with sampled t_star
        print(f"[DEBUG] Calculating risk burden with sampled t_star ({case_type})")
        results_sampled_tstar, t_star_samples = calculate_risk_burden_sampled_tstar(
            chains, 
            config.ISOLATION_PERIODS, 
            time_extended, 
            config.VIRAL_LOAD_THRESHOLDS,
            debug_shapes,
            case_dir
        )
        
        print(f"[DEBUG] Plotting risk burden with sampled t_star ({case_type})")
        RiskBurdenPlots.plot_risk_burden_sampled_tstar(
            results_sampled_tstar, 
            t_star_samples,
            no_treatment_results,  # Add this line
            case_type, 
            directories[f'{case_type}_treatment_effects']
        )
        
        # Add this new section
        print(f"[DEBUG] Plotting viral load curves ({case_type})")
        ModelPredictions.plot_viral_load_curves(
            chains, 
            config.BURN_IN_PERIOD,
            directories[f'{case_type}_treatment_effects'],
            case_type,
            time_extended
        )

    print("[DEBUG] Analysis and visualization completed.")

if __name__ == "__main__":
    main()