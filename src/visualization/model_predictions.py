import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from src.model.virus_model import cached_solve
from config.config import Config
from scipy.stats import gamma

class ModelPredictions:
    
    @staticmethod
    def calculate_and_save_parameter_stats(chains, burn_in_period, output_dir, case):
        # ... (Keep the existing implementation)
        pass

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
            plt.rcParams.update({'font.size': 14})  # Increase default font size
            
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
        
            plt.xlabel('Time (days)', fontsize=16)
            plt.ylabel('log10(Viral Load)', fontsize=16)
            plt.title(f'Model Predictions for Viral Load ({label})', fontsize=18)
            plt.legend(fontsize=12)
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

        fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(20, 25), squeeze=False)
        plt.rcParams.update({'font.size': 18})  # Increase default font size

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
                
                ax.plot(extended_time, treatment_median, '-', label='Treatment', color='blue', linewidth=2)
                ax.fill_between(extended_time, treatment_lower_ci, treatment_upper_ci, alpha=0.2, color='blue')
                
                # Plot no-treatment curves
                no_treatment_median = np.median(no_treatment_predictions, axis=0)
                no_treatment_lower_ci = np.percentile(no_treatment_predictions, 2.5, axis=0)
                no_treatment_upper_ci = np.percentile(no_treatment_predictions, 97.5, axis=0)
                
                ax.plot(extended_time, no_treatment_median, '-', label='No Treatment', color='red', linewidth=2)
                ax.fill_between(extended_time, no_treatment_lower_ci, no_treatment_upper_ci, alpha=0.2, color='red')
                
                ax.axvline(x=21, color='gray', linestyle='--', label='Day 21', linewidth=2)
                ax.set_xlabel('Time (days)' if i == len(t_star_values) - 1 else '', fontsize=22)
                ax.set_ylabel('log10(Viral Load)' if j == 0 else '', fontsize=22)
                ax.set_title(f't* = {t_star}, ε = {epsilon}', fontsize=24)
                ax.set_xlim(0, 30)
                ax.set_ylim(0, 10)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.tick_params(axis='both', which='major', labelsize=20)
                
                if i == 0 and j == 0:
                    ax.legend(fontsize=16, loc='upper right')

        # Add a common legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=18)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'viral_load_curves_{case}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Viral load curves plot saved to {output_file}")

    @staticmethod
    def plot_least_squares_fit(time, observed_data, params, output_dir, case):
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        
        plt.plot(time, observed_data, 'o', label='Observed Data', alpha=0.7)
        
        extended_time = np.linspace(0, 30, 300)
        prediction = cached_solve(params, extended_time)[:, 2]
        
        plt.plot(extended_time, prediction, '-', label='Least Squares Fit')
        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('log10(Viral Load)', fontsize=14)
        plt.title(f'Least Squares Fit for {case.capitalize()} Case', fontsize=16)
        plt.legend(fontsize=12)
        plt.xlim(0, 30)
        plt.ylim(bottom=0)  # Ensure y-axis starts at 0
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'least_squares_fit_{case}.png'), dpi=300, bbox_inches='tight')
        plt.close()