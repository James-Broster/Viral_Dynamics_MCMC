import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import truncnorm
from config.config import Config
import pandas as pd

class MCMCDiagnostics:

    @staticmethod
    def plot_all_diagnostics(chains_f, chains_nf, burn_in_period, r_hat_f, r_hat_nf, 
                             acceptance_rates_over_time_f, acceptance_rates_over_time_nf, 
                             param_means_f, param_means_nf, fatal_dir, non_fatal_dir):
        config = Config()
        for chains, case, r_hat, acceptance_rates_over_time, param_means, case_dir in [
            (chains_f, 'fatal', r_hat_f, acceptance_rates_over_time_f, param_means_f, fatal_dir),
            (chains_nf, 'non_fatal', r_hat_nf, acceptance_rates_over_time_nf, param_means_nf, non_fatal_dir)
        ]:
            trace_dir = os.path.join(case_dir, 'trace_plots')
            hist_dir = os.path.join(case_dir, 'histograms')
            
            for param_index, param_name in enumerate(config.PARAM_NAMES):
                MCMCDiagnostics.plot_trace(chains, param_index, param_name, burn_in_period, 
                                           trace_dir, case)
                MCMCDiagnostics.plot_parameter_histograms(chains, burn_in_period, 
                                                          hist_dir, 
                                                          param_name, case, param_means[param_index])
            
            MCMCDiagnostics.plot_rhat(r_hat, config.PARAM_NAMES, case_dir, case)
            MCMCDiagnostics.plot_acceptance_rates(acceptance_rates_over_time, config.PARAM_NAMES, case_dir, case)

            correlations = np.corrcoef(chains[:, burn_in_period:, :].reshape(-1, chains.shape[-1]).T)
            MCMCDiagnostics.plot_correlation_heatmap(correlations, config.PARAM_NAMES, case_dir, case)

    @staticmethod
    def plot_trace(chains, param_index, param_name, burn_in_period, output_dir, case):
        plt.figure(figsize=(12, 6))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        colors = ['blue', 'green', 'red', 'purple']
        for i, chain in enumerate(chains):
            plt.plot([params[param_index] for params in chain], color=colors[i], alpha=0.6, label=f'Chain {i + 1}')
        plt.axvline(x=burn_in_period, color='black', linestyle='--', label='End of Burn-in Period')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel(param_name, fontsize=14)
        plt.title(f'Trace of {param_name} over Iterations ({case})', fontsize=16)
        plt.legend(fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trace_plot_{param_name}_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_acceptance_rates(acceptance_rates, param_names, output_dir, case):
        num_params = len(param_names)
        num_iterations = len(acceptance_rates)
        
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        for i, param_name in enumerate(param_names):
            plt.plot(range(num_iterations), acceptance_rates[:, i], label=param_name, alpha=0.7)
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Acceptance Rate', fontsize=14)
        plt.title(f'Acceptance Rates Over Time ({case})', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'acceptance_rates_over_time_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_rhat(r_hat, param_names, output_dir, case):
        # Existing plotting code
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        plt.bar(param_names, r_hat)
        plt.axhline(y=1.1, color='r', linestyle='--')
        plt.ylabel('R-hat', fontsize=14)
        plt.title(f'R-hat Values for Each Parameter ({case})', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylim(bottom=0)  # Ensure y-axis starts at 0
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rhat_plot_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save R-hat values to CSV
        rhat_df = pd.DataFrame({'Parameter': param_names, 'R-hat': r_hat})
        csv_path = os.path.join(output_dir, f'rhat_values_{case}.csv')
        rhat_df.to_csv(csv_path, index=False)
        print(f"R-hat values saved to {csv_path}")

    @staticmethod
    def plot_correlation_heatmap(correlations, param_names, output_dir, case):
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    xticklabels=param_names, yticklabels=param_names)
        plt.title(f'Parameter Correlations ({case})', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_correlations_heatmap_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_parameter_histograms(chains, burn_in_period, output_dir, param_name, case, param_mean):
        config = Config()
        flattened_chains = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])
        
        i = config.PARAM_NAMES.index(param_name)
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        
        param_data = flattened_chains[:, i]
        
        lower_bound, upper_bound = config.PARAM_BOUNDS[param_name]
        param_std = config.PARAM_STDS[i]
        
        plot_min = max(lower_bound, param_mean - 4*param_std)
        plot_max = min(upper_bound, param_mean + 4*param_std)
        
        sns.histplot(param_data, kde=True, stat="density", label="Posterior", color='blue', alpha=0.6)
        
        x = np.linspace(plot_min, plot_max, 1000)
        a, b = (lower_bound - param_mean) / param_std, (upper_bound - param_mean) / param_std
        prior = truncnorm.pdf(x, a, b, loc=param_mean, scale=param_std)
        plt.plot(x, prior, 'r--', label="Prior", linewidth=2)
        
        plt.fill_between(x, prior, alpha=0.2, color='red')
        
        plt.xlabel(param_name, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.title(f"Posterior and Prior Distribution of {param_name} ({case})", fontsize=16)
        plt.legend(fontsize=10)
        
        plt.xlim(plot_min, plot_max)
        
        y_max = max(plt.gca().get_ylim()[1], np.max(prior))
        plt.ylim(0, y_max * 1.1)
        
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'histogram_{param_name}_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()