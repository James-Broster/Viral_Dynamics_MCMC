import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from src.model.virus_model import cached_solve
from src.utils.statistical_utils import fit_gamma_to_median_iqr
from config.config import Config
from scipy.stats import gamma

class ModelPredictions:
    
    @staticmethod
    def calculate_and_save_parameter_stats(chains, burn_in_period, output_dir, case):
        latter_chains = chains[:, burn_in_period:, :]
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
        
        param_names = ['alpha_f', 'beta', 'delta_f', 'gamma', 'f2_0', 'V_0']
        stats = []
        
        for i, param_name in enumerate(param_names):
            param_values = flattened_chains[:, i]
            mean = np.mean(param_values)
            lower_ci = np.percentile(param_values, 2.5)
            upper_ci = np.percentile(param_values, 97.5)
            
            stats.append({
                'parameter': param_name,
                'mean_ci': f"{mean:.2e} ({lower_ci:.2e}, {upper_ci:.2e})"
            })
        
        # Save the statistics to a CSV file
        output_file = os.path.join(output_dir, f'parameter_stats_{case}.csv')
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['parameter', 'Mean (Confidence Intervals)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for stat in stats:
                writer.writerow({'parameter': stat['parameter'], 'Mean (Confidence Intervals)': stat['mean_ci']})
        
        print(f"Parameter statistics saved to {output_file}")
        return stats

    @staticmethod
    def plot_model_predictions(time_f, time_nf, observed_data_f, observed_data_nf, chains_f, chains_nf, burn_in_period, fatal_dir, non_fatal_dir, y_min=0, y_max=10):
        print("Starting plot_model_predictions")
        extended_time = np.linspace(0, 30, 300)

        for chains, label, color, time, observed_data, output_dir in [
            (chains_f, 'Fatal', 'red', time_f, observed_data_f, fatal_dir),
            (chains_nf, 'Non-Fatal', 'blue', time_nf, observed_data_nf, non_fatal_dir)
        ]:
            # Calculate and save parameter statistics
            ModelPredictions.calculate_and_save_parameter_stats(chains, burn_in_period, output_dir, label.lower())
            
            plt.figure(figsize=(12, 8))
            
            plt.plot(time, observed_data, 'o', label=f'Observed Data ({label})', color=color, alpha=0.7)
            
            latter_chains = chains[:, burn_in_period::100, :]
            flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
            
            predictions = []
            for params in flattened_chains:
                predictions.append(cached_solve(params, extended_time)[:, 2])
            
            predictions = np.array(predictions)
            
            median = np.median(predictions, axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
            
            plt.plot(extended_time, median, '-', label=f'RNA copies/ml ({label})', color=color)
            plt.fill_between(extended_time, lower_ci, upper_ci, alpha=0.2, color=color)
        
            plt.xlabel('Time (days)')
            plt.ylabel('log10(Viral Load)')
            plt.title(f'Model Predictions for Viral Load ({label})')
            plt.legend()
            plt.xlim(0, 30)
            plt.ylim(y_min, y_max)
            plt.grid(True, which='both', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'model_predictions_viral_load_{label.lower()}.png'), dpi=600, bbox_inches='tight')
            plt.close()

        print("plot_model_predictions completed")

    @staticmethod
    def plot_viral_load_curves(chains, burn_in_period, output_dir, case, time_extended):
        config = Config()
        print(f"Starting plot_viral_load_curves for {case} case")
        
        extended_time = time_extended
        epsilon_values = config.EPSILON_VALUES
        t_star_values = config.T_STAR_VALUES

        latter_chains = chains[:, burn_in_period::100, :]
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])

        fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(5*len(epsilon_values), 5*len(t_star_values)), squeeze=False)

        for i, t_star in enumerate(t_star_values):
            for j, epsilon in enumerate(epsilon_values):
                ax = axes[i, j]
                
                # Calculate treatment predictions
                treatment_predictions = []
                for params in flattened_chains:
                    treatment_predictions.append(cached_solve(params, extended_time, epsilon=epsilon, t_star=t_star)[:, 2])
                treatment_predictions = np.array(treatment_predictions)
                
                # Calculate no-treatment predictions
                no_treatment_predictions = []
                for params in flattened_chains:
                    no_treatment_predictions.append(cached_solve(params, extended_time, epsilon=0, t_star=0)[:, 2])
                no_treatment_predictions = np.array(no_treatment_predictions)
                
                # Plot treatment curves
                treatment_median = np.median(treatment_predictions, axis=0)
                treatment_lower_ci = np.percentile(treatment_predictions, 2.5, axis=0)
                treatment_upper_ci = np.percentile(treatment_predictions, 97.5, axis=0)
                
                ax.plot(extended_time, treatment_median, '-', label='Treatment', color='blue')
                ax.fill_between(extended_time, treatment_lower_ci, treatment_upper_ci, alpha=0.2, color='blue')
                
                # Plot no-treatment curves
                no_treatment_median = np.median(no_treatment_predictions, axis=0)
                no_treatment_lower_ci = np.percentile(no_treatment_predictions, 2.5, axis=0)
                no_treatment_upper_ci = np.percentile(no_treatment_predictions, 97.5, axis=0)
                
                ax.plot(extended_time, no_treatment_median, '-', label='No Treatment', color='red')
                ax.fill_between(extended_time, no_treatment_lower_ci, no_treatment_upper_ci, alpha=0.2, color='red')
                
                ax.axvline(x=21, color='gray', linestyle='--', label='Day 21')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('log10(Viral Load)')
                ax.set_title(f't* = {t_star}, ε = {epsilon}')
                ax.set_xlim(0, 30)
                ax.set_ylim(0, 10)  # Adjust if needed
                ax.grid(True, which='both', linestyle=':', alpha=0.5)
                
                if i == 0 and j == 0:
                    ax.legend(fontsize='x-small')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'viral_load_curves_{case}.png')
        plt.savefig(output_file)
        plt.close()

        print(f"Viral load curves plot saved to {output_file}")

    @staticmethod
    def plot_least_squares_fit(time, observed_data, params, output_dir, case):
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