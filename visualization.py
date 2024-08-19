import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from constants import PARAM_NAMES, PARAM_BOUNDS, PARAM_STDS
from virus_model import VirusModel
from scipy.stats import norm, gamma, truncnorm
from utils import fit_gamma_to_median_iqr
from scipy.optimize import minimize

class Visualization:
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
    def plot_model_predictions(time_f, time_nf, observed_data_f, observed_data_nf, chains_f, chains_nf, burn_in_period, output_dir_f, output_dir_nf, y_min=0, y_max=10):
        print("Starting plot_model_predictions")
        extended_time = np.linspace(0, 30, 300)
        
        # Set the desired median and IQR for t_star distribution
        median = 3.5
        iqr = (2, 6)
        offset = 1

        # Calculate shape and scale parameters for the gamma distribution
        shape, scale = fit_gamma_to_median_iqr(median, iqr, offset)
        
        for chains, label, color, time, observed_data, output_dir in [
            (chains_f, 'Fatal', 'red', time_f, observed_data_f, output_dir_f),
            (chains_nf, 'Non-Fatal', 'blue', time_nf, observed_data_nf, output_dir_nf)
        ]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot observed data and model predictions
            ax1.plot(time, observed_data, 'o', label=f'Observed Data ({label})', color=color, alpha=0.7)
            
            latter_chains = chains[:, burn_in_period::100, :]
            flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
            
            predictions = []
            for params in flattened_chains:
                predictions.append(VirusModel.solve(params, extended_time)[:, 2])
            
            predictions = np.array(predictions)
            
            median = np.median(predictions, axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
            
            ax1.plot(extended_time, median, '-', label=f'Model Prediction ({label})', color=color)
            ax1.fill_between(extended_time, lower_ci, upper_ci, alpha=0.2, color=color)
        
            ax1.axvline(x=21, color='gray', linestyle='--', label='Day 21')
            ax1.axhline(y=4, color='green', linestyle='--', label='4 Log10 Viral Load')
            
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('log10(V)')
            ax1.set_title(f'Model Predictions for Viral Load ({label})')
            ax1.legend()
            ax1.set_xlim(0, 30)
            ax1.set_ylim(y_min, y_max)
            
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
            plt.savefig(os.path.join(output_dir, f'model_predictions_viral_load_with_treatment_dist_{label.lower()}.png'))
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
    def plot_metric_distribution(metrics, p_fatal, output_dir, metric_name):
        plt.figure(figsize=(10, 6))
        plt.hist(metrics, bins=30, density=True, alpha=0.7)
        plt.axvline(np.median(metrics), color='r', linestyle='dashed', linewidth=2, label='Median')
        plt.axvline(np.percentile(metrics, 2.5), color='g', linestyle='dashed', linewidth=2, label='95% CI')
        plt.axvline(np.percentile(metrics, 97.5), color='g', linestyle='dashed', linewidth=2)
        plt.xlabel('Metric Value')
        plt.ylabel('Density')
        plt.title(f'Distribution of {metric_name} (p_fatal = {p_fatal:.2f})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric_name.replace(" ", "_")}_distribution_p{p_fatal:.2f}.png'))
        plt.close()

    @staticmethod
    def plot_viral_load_curves(chains_f, chains_nf, p_fatal, burn_in_period, output_dir):
        latter_chains_f = chains_f[:, burn_in_period::100, :].reshape(-1, chains_f.shape[-1])
        latter_chains_nf = chains_nf[:, burn_in_period::100, :].reshape(-1, chains_nf.shape[-1])

        time = np.linspace(0, 30, 301)
        
        plt.figure(figsize=(12, 6))
        
        # Plot fatal curves
        for params in latter_chains_f[:100]:  # Plot first 100 curves
            solution = VirusModel.solve(params, time)
            plt.plot(time, solution[:, 2], 'r-', alpha=0.1)
        
        # Plot non-fatal curves
        for params in latter_chains_nf[:100]:  # Plot first 100 curves
            solution = VirusModel.solve(params, time)
            plt.plot(time, solution[:, 2], 'b-', alpha=0.1)
        
        plt.xlabel('Time (days)')
        plt.ylabel('log10(Viral Load)')
        plt.title(f'Viral Load Curves (p_fatal = {p_fatal:.2f})')
        plt.savefig(os.path.join(output_dir, f'viral_load_curves_p{p_fatal:.2f}.png'))
        plt.close()

    @staticmethod
    def plot_least_squares_fit(time, observed_data, params, output_dir, case):
        plt.figure(figsize=(10, 6))
        plt.plot(time, observed_data, 'o', label='Observed Data', alpha=0.7)
        
        extended_time = np.linspace(0, 30, 300)
        prediction = VirusModel.solve(params, extended_time)[:, 2]
        
        plt.plot(extended_time, prediction, '-', label='Least Squares Fit')
        plt.xlabel('Time (days)')
        plt.ylabel('log10(Viral Load)')
        plt.title(f'Least Squares Fit for {case.capitalize()} Case')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'least_squares_fit_{case}.png'))
        plt.close()