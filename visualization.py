import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from constants import PARAM_NAMES, PARAM_BOUNDS, PARAM_MEANS, PARAM_STDS, FIXED_PARAMS
from virus_model import VirusModel
from scipy.stats import norm

class Visualization:
    @staticmethod
    def plot_trace(chains, param_index, param_name, burn_in_period, output_dir):
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'green', 'red', 'purple']
        for i, chain in enumerate(chains):
            plt.plot([params[param_index] for params in chain], color=colors[i], alpha=0.6, label=f'Chain {i + 1}')
        plt.axvline(x=burn_in_period, color='black', linestyle='--', label='End of Burn-in Period')
        plt.xlabel('Iteration')
        plt.ylabel(param_name)
        plt.title(f'Trace of {param_name} over Iterations')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'trace_plot_{param_name}.png'))
        plt.close()

    @staticmethod
    def plot_model_prediction(time, observed_data, chains, viral_loads, true_parameters, burn_in_period, output_dir, y_min=0, y_max=10):
        print("Starting plot_model_prediction")
        latter_chains = viral_loads[:, burn_in_period::100, :, :]  # Thinning: use every 100th set after burn-in
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-2], latter_chains.shape[-1])
        
        print("Calculating statistics:")
        median_prediction = np.median(flattened_chains, axis=0)
        print("Median calculated")
        lower_ci = np.percentile(flattened_chains, 2.5, axis=0)
        print("Lower CI calculated")
        upper_ci = np.percentile(flattened_chains, 97.5, axis=0)
        print("Upper CI calculated")
        
        # Extend time to 30 days
        extended_time = np.linspace(0, 30, 300)
        true_model_prediction = VirusModel.solve(true_parameters, extended_time)
        print("True model prediction calculated")

        # Recalculate predictions for extended time
        extended_predictions = []
        for chain in chains[:, burn_in_period::100, :]:  # Thinning: use every 100th set after burn-in
            for params in chain:
                full_params = np.concatenate([params, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
                extended_predictions.append(VirusModel.solve(full_params, extended_time))
        extended_predictions = np.array(extended_predictions)
        
        extended_median = np.median(extended_predictions, axis=0)
        extended_lower_ci = np.percentile(extended_predictions, 2.5, axis=0)
        extended_upper_ci = np.percentile(extended_predictions, 97.5, axis=0)

        print(f"Time shape: {time.shape}")
        print(f"Viral loads shape: {viral_loads.shape}")
        print(f"Flattened chains shape: {flattened_chains.shape}")
        print(f"Extended predictions shape: {extended_predictions.shape}")
        print(f"Extended median shape: {extended_median.shape}")

        # Plot for viral load (log10(V))
        plt.figure(figsize=(12, 6))
        plt.plot(time, observed_data[:, 0], 'o', label='Observed Data')
        plt.plot(extended_time, extended_median[:, 2], '-', label='Model Prediction (Median)')
        plt.fill_between(extended_time, extended_lower_ci[:, 2], extended_upper_ci[:, 2], alpha=0.2, label='95% CI')
        plt.plot(extended_time, true_model_prediction[:, 2], '--', label='Model Prediction (True Parameters)')
        
        # Add vertical line at day 21
        plt.axvline(x=21, color='r', linestyle='--', label='Day 21')
        
        # Add horizontal line at 4 Log10 viral load
        plt.axhline(y=4, color='g', linestyle='--', label='4 Log10 Viral Load')
        
        plt.xlabel('Time (days)')
        plt.ylabel('log10(V)')
        plt.title('Observed Data and Model Prediction for Viral Load (Log Scale)')
        plt.legend()
        plt.xlim(0, 30)  # Set x-axis limit to 30 days
        plt.ylim(y_min, y_max)  # Set y-axis limits
        print(f"Saving viral load figure to {output_dir}")
        plt.savefig(os.path.join(output_dir, 'model_prediction_viral_load.png'))
        plt.close()

        # Plot for f1
        plt.figure(figsize=(12, 6))
        plt.plot(extended_time, extended_median[:, 0], '-', label='Model Prediction (Median)')
        plt.fill_between(extended_time, extended_lower_ci[:, 0], extended_upper_ci[:, 0], alpha=0.2, label='95% CI')
        plt.plot(extended_time, true_model_prediction[:, 0], '--', label='Model Prediction (True Parameters)')
        plt.xlabel('Time (days)')
        plt.ylabel('f1')
        plt.title('Model Prediction for f1')
        plt.legend()
        plt.xlim(0, 30)  # Set x-axis limit to 30 days
        print(f"Saving f1 figure to {output_dir}")
        plt.savefig(os.path.join(output_dir, 'model_prediction_f1.png'))
        plt.close()

        # Plot for f2
        plt.figure(figsize=(12, 6))
        plt.plot(extended_time, extended_median[:, 1], '-', label='Model Prediction (Median)')
        plt.fill_between(extended_time, extended_lower_ci[:, 1], extended_upper_ci[:, 1], alpha=0.2, label='95% CI')
        plt.plot(extended_time, true_model_prediction[:, 1], '--', label='Model Prediction (True Parameters)')
        plt.xlabel('Time (days)')
        plt.ylabel('f2')
        plt.title('Model Prediction for f2')
        plt.legend()
        plt.xlim(0, 30)  # Set x-axis limit to 30 days
        print(f"Saving f2 figure to {output_dir}")
        plt.savefig(os.path.join(output_dir, 'model_prediction_f2.png'))
        plt.close()

        print("plot_model_prediction completed")

    @staticmethod
    def plot_rhat(r_hat, param_names, output_dir):
        plt.figure(figsize=(10, 6))
        plt.bar(param_names, r_hat)
        plt.axhline(y=1.1, color='r', linestyle='--')
        plt.ylabel('R-hat')
        plt.title('R-hat Values for Each Parameter')
        plt.savefig(os.path.join(output_dir, 'rhat_plot.png'))
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(correlations, param_names, output_dir):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    xticklabels=param_names, yticklabels=param_names)
        plt.title('Parameter Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_correlations_heatmap.png'))
        plt.close()

    @staticmethod
    def plot_parameter_histograms(chains, burn_in_period, output_dir, param_name):
        flattened_chains = chains[:, burn_in_period::100, :].reshape(-1, chains.shape[-1])  # Thinning: use every 100th set after burn-in
        
        i = PARAM_NAMES.index(param_name)
        plt.figure(figsize=(10, 6))
        
        # Plot posterior
        sns.histplot(flattened_chains[:, i], kde=True, stat="density", label="Posterior")
        
        # Plot prior (normal distribution)
        x = np.linspace(PARAM_BOUNDS[param_name][0], PARAM_BOUNDS[param_name][1], 100)
        prior = norm.pdf(x, loc=PARAM_MEANS[i], scale=PARAM_STDS[i])
        plt.plot(x, prior, 'r--', label="Prior")
        
        plt.xlabel(param_name)
        plt.ylabel("Density")
        plt.title(f"Posterior and Prior Distribution of {param_name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'histogram_{param_name}.png'))
        plt.close()

    @staticmethod
    def plot_therapy_comparison(time, observed_data, chains, burn_in_period, epsilon, t_star, output_dir):
        latter_chains = chains[:, burn_in_period::100, :]  # Thinning: use every 100th set after burn-in
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
        
        extended_time = np.linspace(0, 30, 300)
        
        # Base model predictions
        base_predictions = []
        for params in flattened_chains:
            full_params = np.concatenate([params, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
            base_predictions.append(VirusModel.solve(full_params, extended_time, 0, 0))
        base_predictions = np.array(base_predictions)
        
        base_median = np.median(base_predictions, axis=0)
        base_lower_ci = np.percentile(base_predictions, 2.5, axis=0)
        base_upper_ci = np.percentile(base_predictions, 97.5, axis=0)
        
        # Therapy model predictions
        therapy_predictions = []
        for params in flattened_chains:
            full_params = np.concatenate([params, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
            therapy_predictions.append(VirusModel.solve(full_params, extended_time, epsilon, t_star))
        therapy_predictions = np.array(therapy_predictions)
        
        therapy_median = np.median(therapy_predictions, axis=0)
        therapy_lower_ci = np.percentile(therapy_predictions, 2.5, axis=0)
        therapy_upper_ci = np.percentile(therapy_predictions, 97.5, axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, observed_data[:, 0], 'o', label='Observed Data')
        plt.plot(extended_time, base_median[:, 2], '-', color='blue', label='Base Model (Median)')
        plt.fill_between(extended_time, base_lower_ci[:, 2], base_upper_ci[:, 2], alpha=0.2, color='blue')
        plt.plot(extended_time, therapy_median[:, 2], '-', color='red', label=f'Therapy Model (ε={epsilon}, t*={t_star})')
        plt.fill_between(extended_time, therapy_lower_ci[:, 2], therapy_upper_ci[:, 2], alpha=0.2, color='red')
        
        plt.axvline(x=21, color='gray', linestyle='--', label='Day 21')
        plt.axhline(y=4, color='green', linestyle='--', label='4 Log10 Viral Load')
        plt.axvline(x=t_star, color='red', linestyle=':', label='Therapy Start')
        
        plt.xlabel('Time (days)')
        plt.ylabel('log10(V)')
        plt.title(f'Viral Load Comparison: Base vs Therapy (ε={epsilon}, t*={t_star})')
        plt.legend()
        plt.xlim(0, 30)
        plt.ylim(0, 10)  # Set y-axis limits between 0 and 10 log10(V)
        therapy_plots_dir = os.path.join(output_dir, 'therapy_plots')
        os.makedirs(therapy_plots_dir, exist_ok=True)
        plt.savefig(os.path.join(therapy_plots_dir, f'therapy_comparison_e{epsilon}_t{t_star}.png'))
        plt.close()

    @staticmethod
    def plot_time_to_threshold_heatmap(median_results, lower_ci_results, upper_ci_results, epsilon_values, t_star_values, output_dir):
        plt.figure(figsize=(16, 12))
        
        # Create a new array to store the formatted text for each cell
        cell_text = np.empty_like(median_results, dtype=object)
        
        for i in range(median_results.shape[0]):
            for j in range(median_results.shape[1]):
                median = median_results[i, j]
                lower = lower_ci_results[i, j]
                upper = upper_ci_results[i, j]
                cell_text[i, j] = f"{median:.1f}\n({lower:.1f}-{upper:.1f})"
        
        # Create the heatmap using the median values
        ax = sns.heatmap(median_results, annot=cell_text, fmt="", cmap='YlGnBu',
                         xticklabels=t_star_values, yticklabels=epsilon_values,
                         cbar_kws={'label': 'Days to 4log10 Threshold'})
        
        # Adjust the alignment and font size of the annotations
        for t in ax.texts:
            t.set_verticalalignment('center')
            t.set_fontsize(6)  # Adjust this value if needed
        
        plt.xlabel('Therapy Start Day (t_star)')
        plt.ylabel('Therapy Efficacy (epsilon)')
        plt.title('Days to 4log10 Threshold: Median (95% CI)')
        plt.tight_layout()
        therapy_plots_dir = os.path.join(output_dir, 'therapy_plots')
        os.makedirs(therapy_plots_dir, exist_ok=True)
        plt.savefig(os.path.join(therapy_plots_dir, 'time_to_threshold_heatmap.png'), dpi=300)
        plt.close()