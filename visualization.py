import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config
from tqdm import tqdm

from config import PARAM_NAMES, PARAM_BOUNDS, PARAM_STDS, EPSILON_VALUES, T_STAR_VALUES
from virus_model import cached_solve
from scipy.stats import norm, gamma, truncnorm
from utils import fit_gamma_to_median_iqr
from model_fitting import ModelFitting

class Visualization:
    @staticmethod
    def plot_all_diagnostics(chains_f, chains_nf, burn_in_period, r_hat_f, r_hat_nf, 
                             acceptance_rates_over_time_f, acceptance_rates_over_time_nf, 
                             param_means_f, param_means_nf):
        for chains, case, r_hat, acceptance_rates_over_time, param_means in [
            (chains_f, 'fatal', r_hat_f, acceptance_rates_over_time_f, param_means_f),
            (chains_nf, 'non_fatal', r_hat_nf, acceptance_rates_over_time_nf, param_means_nf)
        ]:
            case_dir = os.path.join(config.BASE_OUTPUT_DIR, case, 'mcmc_diagnostics')
            for param_index, param_name in enumerate(PARAM_NAMES):
                Visualization.plot_trace(chains, param_index, param_name, burn_in_period, 
                                         os.path.join(case_dir, 'trace_plots'), case)
                Visualization.plot_parameter_histograms(chains, burn_in_period, 
                                                        os.path.join(case_dir, 'histograms'), 
                                                        param_name, case, param_means[param_index])
            
            Visualization.plot_rhat(r_hat, PARAM_NAMES, case_dir, case)
            Visualization.plot_acceptance_rates(acceptance_rates_over_time, PARAM_NAMES, case_dir, case)

            correlations = ModelFitting.calculate_correlations(chains[:, burn_in_period:, :])
            Visualization.plot_correlation_heatmap(correlations, PARAM_NAMES, case_dir, case)

    @staticmethod
    def plot_post_mcmc_results(analysis_results, base_output_dir):
        for case, results in analysis_results.items():
            case_dir = os.path.join(base_output_dir, case, 'post_mcmc_analysis')
            os.makedirs(case_dir, exist_ok=True)

            # Flatten the results from all chains
            flattened_results = [item for chain in results for item in chain]

            unique_epsilons = set([r['epsilon'] for r in flattened_results])

            for epsilon in tqdm(unique_epsilons, desc=f"Processing {case} case"):
                epsilon_results = [r for r in flattened_results if r['epsilon'] == epsilon]
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
                plt.savefig(os.path.join(case_dir, f'time_to_threshold_vs_treatment_start_epsilon_{epsilon}.png'))
                plt.close()

    @staticmethod
    def plot_trace(chains, param_index, param_name, burn_in_period, output_dir, case):
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'green', 'red', 'purple']
        for i, chain in enumerate(chains):
            plt.plot([params[param_index] for params in chain], color=colors[i], alpha=0.6, label=f'Chain {i + 1}')
        plt.axvline(x=burn_in_period, color='black', linestyle='--', label='End of Burn-in Period')
        plt.xlabel('Iteration')
        plt.ylabel(param_name)
        plt.title(f'Trace of {param_name} over Iterations ({case})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'trace_plot_{param_name}_{case}.png'))
        plt.close()
    
    @staticmethod
    def plot_acceptance_rates(acceptance_rates, param_names, output_dir, case):
        num_params = len(param_names)
        num_iterations = len(acceptance_rates)
        
        plt.figure(figsize=(12, 8))
        for i, param_name in enumerate(param_names):
            plt.plot(range(num_iterations), acceptance_rates[:, i], label=param_name, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance Rate')
        plt.title(f'Acceptance Rates Over Time ({case})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'acceptance_rates_over_time_{case}.png'))
        plt.close()

    @staticmethod
    def plot_model_predictions(time_f, time_nf, observed_data_f, observed_data_nf, chains_f, chains_nf, burn_in_period, base_output_dir, y_min=0, y_max=10):
        print("Starting plot_model_predictions")
        extended_time = np.linspace(0, 30, 300)
        
        # Set the desired median and IQR for t_star distribution
        median = 3.5
        iqr = (2, 6)
        offset = 1

        # Calculate shape and scale parameters for the gamma distribution
        shape, scale = fit_gamma_to_median_iqr(median, iqr, offset)
        
        for chains, label, color, time, observed_data, output_dir in [
            (chains_f, 'Fatal', 'red', time_f, observed_data_f, os.path.join(base_output_dir, 'fatal', 'model_predictions')),
            (chains_nf, 'Non-Fatal', 'blue', time_nf, observed_data_nf, os.path.join(base_output_dir, 'non_fatal', 'model_predictions'))
        ]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot observed data and model predictions (RNA copies)
            ax1.plot(time, observed_data, 'o', label=f'Observed Data ({label})', color=color, alpha=0.7)
            
            latter_chains = chains[:, burn_in_period::100, :]
            flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
            
            predictions = []
            for params in flattened_chains:
                predictions.append(cached_solve(params, extended_time)[:, 2])
            
            predictions = np.array(predictions)
            
            median = np.median(predictions, axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
            
            ax1.plot(extended_time, median, '-', label=f'RNA copies/ml ({label})', color=color)
            ax1.fill_between(extended_time, lower_ci, upper_ci, alpha=0.2, color=color)
        
            # Plot estimated infectious titer range
            estimated_pfu_median_lower = 10**(median - 4)
            estimated_pfu_median_upper = 10**(median - 3)
            
            ax1.plot(extended_time, np.log10(estimated_pfu_median_lower), '--', label=f'Estimated infectious titer range ({label})', color=color)
            ax1.plot(extended_time, np.log10(estimated_pfu_median_upper), '--', color=color)
            ax1.fill_between(extended_time, np.log10(estimated_pfu_median_lower), np.log10(estimated_pfu_median_upper), alpha=0.1, color=color)
            
            ax1.axvline(x=21, color='gray', linestyle='--', label='Day 21')
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('log10(Viral Load)')
            ax1.set_title(f'Model Predictions for Viral Load ({label})')
            ax1.legend()
            ax1.set_xlim(0, 30)
            ax1.set_ylim(y_min, y_max)
            ax1.grid(True, which='both', linestyle=':', alpha=0.5)
            
            # Plot therapy initiation time distribution
            x = np.linspace(offset, 21, 200)
            pdf = gamma.pdf(x - offset, a=shape, scale=scale)
            
            ax2.plot(x, pdf, 'k-', lw=2)
            ax2.fill_between(x, pdf, alpha=0.3)
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Treatment Initiation Time Distribution')
            ax2.set_xlim(0, 30)
            ax2.set_ylim(0, 0.175)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'model_predictions_viral_load_with_infectiousness_{label.lower()}.png'))
            plt.close()

        print("plot_model_predictions completed")

    @staticmethod
    def plot_rhat(r_hat, param_names, output_dir, case):
        plt.figure(figsize=(10, 6))
        plt.bar(param_names, r_hat)
        plt.axhline(y=1.1, color='r', linestyle='--')
        plt.ylabel('R-hat')
        plt.title(f'R-hat Values for Each Parameter ({case})')
        plt.savefig(os.path.join(output_dir, f'rhat_plot_{case}.png'))
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(correlations, param_names, output_dir, case):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    xticklabels=param_names, yticklabels=param_names)
        plt.title(f'Parameter Correlations ({case})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_correlations_heatmap_{case}.png'))
        plt.close()

    @staticmethod
    def plot_parameter_histograms(chains, burn_in_period, output_dir, param_name, case, param_mean):
        flattened_chains = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])
        
        i = PARAM_NAMES.index(param_name)
        plt.figure(figsize=(10, 6))
        
        # Get the data for the specific parameter
        param_data = flattened_chains[:, i]
        
        # Get parameter bounds
        lower_bound, upper_bound = PARAM_BOUNDS[param_name]
        param_std = PARAM_STDS[i]
        
        # Calculate the range to plot
        plot_min = max(lower_bound, param_mean - 4*param_std)
        plot_max = min(upper_bound, param_mean + 4*param_std)
        
        # Plot posterior
        sns.histplot(param_data, kde=True, stat="density", label="Posterior", color='blue', alpha=0.6)
        
        # Plot truncated normal prior
        x = np.linspace(plot_min, plot_max, 1000)
        a, b = (lower_bound - param_mean) / param_std, (upper_bound - param_mean) / param_std
        prior = truncnorm.pdf(x, a, b, loc=param_mean, scale=param_std)
        plt.plot(x, prior, 'r--', label="Prior", linewidth=2)
        
        # Fill the area under the prior curve
        plt.fill_between(x, prior, alpha=0.2, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel("Density")
        plt.title(f"Posterior and Prior Distribution of {param_name} ({case})")
        plt.legend()
        
        # Set x-axis limits
        plt.xlim(plot_min, plot_max)
        
        # Adjust y-axis to fit both distributions
        y_max = max(plt.gca().get_ylim()[1], np.max(prior))
        plt.ylim(0, y_max * 1.1)  # Add 10% padding on top
        
        plt.savefig(os.path.join(output_dir, f'histogram_{param_name}_{case}.png'))
        plt.close()

    @staticmethod
    def plot_least_squares_fit(time, observed_data, params, base_output_dir, case):
        output_dir = os.path.join(base_output_dir, 'least_squares')
        plt.figure(figsize=(10, 6))
        plt.plot(time, observed_data, 'o', label='Observed Data', alpha=0.7)
        
        extended_time = np.linspace(0, 30, 300)
        prediction = cached_solve(params, extended_time)[:, 2]
        
        plt.plot(extended_time, prediction, '-', label='Least Squares Fit')
        plt.xlabel('Time (days)')
        plt.ylabel('log10(Viral Load)')
        plt.title(f'Least Squares Fit for {case.capitalize()} Case')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'least_squares_fit_{case}.png'))
        plt.close()

    @staticmethod
    def plot_treatment_effects(chains_f, chains_nf, burn_in_period, time, t_star_values, epsilon_values, base_output_dir):

        def solve_for_params(params, epsilon, t_star):
            return cached_solve(params, time, epsilon=epsilon, t_star=t_star)[:, 2]

        for chains, case in [(chains_f, 'fatal'), (chains_nf, 'non_fatal')]:
            output_dir = os.path.join(base_output_dir, case, 'treatment_effects')
            os.makedirs(output_dir, exist_ok=True)

            latter_chains = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])
            
            # Adjust sample size based on available data
            sample_size = min(100, latter_chains.shape[0])
            if sample_size < latter_chains.shape[0]:
                parameter_samples = latter_chains[np.random.choice(latter_chains.shape[0], sample_size, replace=False)]
            else:
                parameter_samples = latter_chains  # Use all available samples if less than 100

            fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(5*len(epsilon_values), 5*len(t_star_values)), squeeze=False)
            
            # Solve for no treatment
            print(f"Calculating no treatment results for {case} case...")
            no_treatment_results = np.array([solve_for_params(params, 0, np.inf) for params in tqdm(parameter_samples, desc="No treatment")])
            no_treatment_median = np.median(no_treatment_results, axis=0)
            no_treatment_ci = np.percentile(no_treatment_results, [2.5, 97.5], axis=0)

            total_iterations = len(t_star_values) * len(epsilon_values)
            with tqdm(total=total_iterations, desc=f"Processing {case} case") as pbar:
                for i, t_star in enumerate(t_star_values):
                    for j, epsilon in enumerate(epsilon_values):
                        ax = axes[i, j]
                        
                        # Plot no treatment
                        ax.plot(time, no_treatment_median, 'b-', label='No Treatment')
                        ax.fill_between(time, no_treatment_ci[0], no_treatment_ci[1], color='b', alpha=0.2)

                        # Plot treatment
                        treatment_results = np.array([solve_for_params(params, epsilon, t_star) for params in parameter_samples])
                        treatment_median = np.median(treatment_results, axis=0)
                        treatment_ci = np.percentile(treatment_results, [2.5, 97.5], axis=0)

                        ax.plot(time, treatment_median, 'r--', label=f'Treatment (ε = {epsilon})')
                        ax.fill_between(time, treatment_ci[0], treatment_ci[1], color='r', alpha=0.2)

                        ax.axvline(x=t_star, color='g', linestyle=':', label=f't* = {t_star}')
                        ax.axhline(y=4, color='k', linestyle='--', label='Threshold')
                        
                        ax.set_xlabel('Time (days)')
                        ax.set_ylabel('log10(Viral Load)')
                        ax.set_title(f't* = {t_star}, ε = {epsilon}')
                        ax.legend()
                        ax.grid(True, which='both', linestyle='--', alpha=0.5)
                        ax.set_ylim(0, 10)  # Set y-axis range from 0 to 10

                        pbar.update(1)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'treatment_effects_{case}.png'))
            plt.close()

        print(f"Treatment effect plots saved in {base_output_dir}")

    @staticmethod
    def plot_risk_burden(risk_burden, case, output_dir):
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'proportion_below_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days Unnecessarily Isolated',
            'Days Above Threshold Post-Release',
            'Proportion Above Threshold at Release',
            'Proportion Below Threshold at Release',
            'Cumulative Risk Score'
        ]

        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(12, 8))

            for threshold in risk_burden.keys():
                x = sorted(risk_burden[threshold].keys())  # isolation periods
                y = [risk_burden[threshold][period][metric]['avg'] for period in x]

                plt.plot(x, y, marker='o', label=f'{threshold} log10 copies/mL')

            plt.xlabel('Isolation Period (days)')
            plt.ylabel(title)
            plt.title(f'{title} ({case})')
            plt.legend(title='Viral Load Threshold')
            plt.grid(True, linestyle='--', alpha=0.7)

            if 'proportion' in metric:
                plt.ylim(0, 1)  # Set y-axis from 0 to 1 for proportions
            elif metric == 'risk_score':
                plt.yscale('linear')  # Use linear scale for risk score

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}{case}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Risk and burden plots saved in {output_dir}")


    @staticmethod
    def plot_all_viral_load_curves(chains, burn_in_period, case, output_dir):
        latter_chains = chains[:, burn_in_period:, :]
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])

        plt.figure(figsize=(12, 8))
        extended_time = np.linspace(0, 30, 301)

        for params in flattened_chains:
            prediction = cached_solve(params, extended_time)[:, 2]  # log10 of viral load
            plt.plot(extended_time, prediction, alpha=0.1, color='blue')

        plt.xlabel('Time (days)')
        plt.ylabel('log10(Viral Load)')
        plt.title(f'All Viral Load Curves ({case.capitalize()} Case)')
        plt.ylim(0, 10)  # Adjust if needed
        plt.xlim(0, 30)
        plt.grid(True, which='both', linestyle=':', alpha=0.5)

        output_file = os.path.join(output_dir, f'all_viral_load_curves_{case}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"All viral load curves plot saved to {output_file}")


    @staticmethod
    def plot_risk_burden_epsilon_tstar(results, case, base_output_dir):
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'proportion_below_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days Unnecessarily Isolated',
            'Days Above Threshold Post-Release',
            'Proportion Above Threshold at Release',
            'Proportion Below Threshold at Release',
            'Cumulative Risk Score'
        ]

        epsilon_values = sorted(set(e for e, _ in results.keys()))
        t_star_values = sorted(set(t for _, t in results.keys()))

        # Get the baseline data (T_STAR = 0 and epsilon = 0)
        baseline_data = results.get((0.0, 0), results[min(results.keys())])

        # Define colors for each threshold
        threshold_colors = {
            3: 'blue',
            4: 'orange',
            5: 'green'
        }

        for metric, title in zip(metrics, titles):
            fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(5*len(epsilon_values), 5*len(t_star_values)), squeeze=False)
            
            total_iterations = len(t_star_values) * len(epsilon_values)
            with tqdm(total=total_iterations, desc=f"Processing {case} case for {title}") as pbar:
                for i, t_star in enumerate(t_star_values):
                    for j, epsilon in enumerate(epsilon_values):
                        ax = axes[i, j]
                        risk_burden = results[(epsilon, t_star)]

                        for threshold in risk_burden.keys():
                            color = threshold_colors.get(threshold, 'gray')  # Default to gray if threshold not in our color map
                            
                            # Plot baseline for this threshold
                            x_baseline = sorted(baseline_data[threshold].keys())
                            y_baseline = [baseline_data[threshold][period][metric]['avg'] for period in x_baseline]
                            ax.plot(x_baseline, y_baseline, ':', color=color, label=f'Baseline {threshold} log10' if threshold == list(risk_burden.keys())[0] else "")

                            # Plot data for this epsilon and t_star
                            x = sorted(risk_burden[threshold].keys())  # isolation periods
                            y = [risk_burden[threshold][period][metric]['avg'] for period in x]
                            ax.plot(x, y, marker='o', color=color, label=f'{threshold} log10 copies/mL')

                        ax.set_xlabel('Isolation Period (days)')
                        ax.set_ylabel(title)
                        ax.set_title(f't* = {t_star}, ε = {epsilon}')
                        ax.legend(title='Viral Load Threshold', fontsize='x-small')
                        ax.grid(True, which='both', linestyle='--', alpha=0.5)

                        if 'proportion' in metric:
                            ax.set_ylim(0, 1)
                        elif metric == 'risk_score':
                            ax.set_yscale('linear')

                        pbar.update(1)

            plt.tight_layout()
            output_dir = os.path.join(base_output_dir, 'treatment_effects')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{metric}_{case}.png')
            plt.savefig(output_file)
            plt.close(fig)

        print(f"Risk and burden plots for different epsilon and t_star values saved in {os.path.join(base_output_dir, 'treatment_effects')}")