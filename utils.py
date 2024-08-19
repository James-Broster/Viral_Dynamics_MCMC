import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares, minimize
from virus_model import VirusModel
from constants import PARAM_BOUNDS, PARAM_NAMES, PARAM_STDS
from tqdm import tqdm
from scipy import stats
from scipy.stats import gamma
import csv


def load_data():
    data_fatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0],
        'virusload': [32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, 141253.8]
    })
    data_nonfatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'virusload': [165958.7, 52480.75, 2754229.0, 3548134.0, 1288250.0, 1584893.0, 199526.2, 371535.2, 107151.9, 14791.08, 31622.78, 70794.58, 7413.102, 9772.372, 5623.413]
    })
    time_fatal = data_fatal['time'].values
    time_nonfatal = data_nonfatal['time'].values
    observed_data_fatal = np.log10(data_fatal['virusload'].values).reshape(-1, 1)
    observed_data_nonfatal = np.log10(data_nonfatal['virusload'].values).reshape(-1, 1)
    print(observed_data_fatal)
    print(observed_data_nonfatal)
    return time_fatal, observed_data_fatal, time_nonfatal, observed_data_nonfatal

def calculate_parameter_statistics(chains, burn_in_period):
    latter_chains = chains[:, burn_in_period::100, :]  # Thinning: use every 100th set after burn-in
    flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
    
    medians = np.median(flattened_chains, axis=0)
    lower_ci = np.percentile(flattened_chains, 2.5, axis=0)
    upper_ci = np.percentile(flattened_chains, 97.5, axis=0)
    
    return medians, lower_ci, upper_ci

def estimate_params_least_squares(time_f, time_nf, observed_data_f, observed_data_nf):
    def residuals(params, time, observed_data):
        model_data = VirusModel.solve(params, time)
        return (observed_data.flatten() - model_data[:, 2]).flatten()

    initial_guess = [1e-9, 1e-6, 2, 2000, 0.9953, 30000]
    bounds = ([PARAM_BOUNDS[param][0] for param in PARAM_NAMES],
              [PARAM_BOUNDS[param][1] for param in PARAM_NAMES])
    
    # Estimate for fatal cases
    result_f = least_squares(residuals, initial_guess, args=(time_f, observed_data_f), bounds=bounds)
    
    # Estimate for non-fatal cases
    result_nf = least_squares(residuals, initial_guess, args=(time_nf, observed_data_nf), bounds=bounds)
    
    return result_f.x, result_nf.x

def fit_gamma_to_median_iqr(median, iqr, offset=1):
    def objective(params):
        shape, scale = params
        q1, q3 = gamma.ppf([0.25, 0.75], shape, scale=scale) + offset
        model_median = gamma.ppf(0.5, shape, scale=scale) + offset
        return ((q3 - q1) - (iqr[1] - iqr[0]))**2 + (model_median - (median + offset))**2

    result = minimize(objective, [2, 2], method='Nelder-Mead')
    return result.x

def sample_t_star():
    median = 3.5
    iqr = (2, 6)
    offset = 1
    shape, scale = fit_gamma_to_median_iqr(median, iqr, offset)
    t_star = gamma.rvs(shape, scale=scale) + offset
    return np.clip(t_star, offset, 21)

def post_mcmc_analysis(parameter_samples, time, num_t_star_samples=100, epsilon_values=[0, 0.3, 0.6, 0.9]):
    results = []
    
    for params in parameter_samples:
        for _ in range(num_t_star_samples):
            t_star = sample_t_star()
            
            for epsilon in epsilon_values:
                solution = VirusModel.solve(params, time, epsilon=epsilon, t_star=t_star)
                time_to_threshold = calculate_time_to_threshold(solution, time)
                
                results.append({
                    'parameters': params,
                    't_star': t_star,
                    'epsilon': epsilon,
                    'time_to_threshold': time_to_threshold
                })
    
    return results

def calculate_time_to_threshold(solution, time, threshold=4):
    log_V = solution[:, 2]
    threshold_crossed = np.where(log_V < threshold)[0]
    if len(threshold_crossed) > 0:
        return time[threshold_crossed[0]] - 21  # Subtract isolation period
    else:
        return 9  # 30 - 21, if threshold is never crossed

def plot_treatment_effects(parameter_samples, time, t_star_values, epsilon_values, output_dir):
    def solve_for_params(params, epsilon, t_star):
        return VirusModel.solve(params, time, epsilon=epsilon, t_star=t_star)[:, 2]

    fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(5*len(epsilon_values), 5*len(t_star_values)), squeeze=False)
    
    # Solve for no treatment
    no_treatment_results = np.array([solve_for_params(params, 0, np.inf) for params in parameter_samples])
    no_treatment_median = np.median(no_treatment_results, axis=0)
    no_treatment_ci = np.percentile(no_treatment_results, [2.5, 97.5], axis=0)

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'treatment_effects.png'))
    plt.close()

    print(f"Treatment effect plot saved in {output_dir}")

def analyze_epidemiological_metrics(chains_f, chains_nf, p_fatal_values, burn_in_period, base_output_dir):
    output_dir = os.path.join(base_output_dir, 'epidemiological_metrics')
    os.makedirs(output_dir, exist_ok=True)

    def calculate_metric(params, epsilon):
        threshold = 4  # log10 VL threshold
        isolation_period = 21  # days of isolation
        time = np.linspace(0, 30, 301)  # 30 days, 301 points
        t_star = sample_t_star()
        solution = VirusModel.solve(params, time, epsilon=epsilon, t_star=t_star)
        return calculate_time_to_threshold(solution, time)

    latter_chains_f = chains_f[:, burn_in_period::100, :].reshape(-1, chains_f.shape[-1])
    latter_chains_nf = chains_nf[:, burn_in_period::100, :].reshape(-1, chains_nf.shape[-1])

    epsilon_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_simulations = 100  # Number of simulations for each parameter set

    print("Calculating metrics for different p_fatal values...")
    for p_fatal in tqdm(p_fatal_values, desc="p_fatal"):
        p_fatal_dir = os.path.join(output_dir, f'p_fatal_{p_fatal:.2f}')
        os.makedirs(p_fatal_dir, exist_ok=True)
        
        all_results = []
        summary_stats = []
        
        for epsilon in tqdm(epsilon_values, desc="epsilon", leave=False):
            metrics_f = []
            metrics_nf = []
            
            for _ in range(n_simulations):
                params_f = latter_chains_f[np.random.randint(latter_chains_f.shape[0])]
                params_nf = latter_chains_nf[np.random.randint(latter_chains_nf.shape[0])]
                
                metrics_f.append(calculate_metric(params_f, epsilon))
                metrics_nf.append(calculate_metric(params_nf, epsilon))

            combined_metrics = np.random.choice(metrics_f + metrics_nf, size=10000, 
                                                p=[p_fatal/len(metrics_f)]*len(metrics_f) + 
                                                  [(1-p_fatal)/len(metrics_nf)]*len(metrics_nf))
            
            all_results.append(combined_metrics)
            
            summary_stats.append({
                'epsilon': epsilon,
                'median': np.median(combined_metrics),
                'mean': np.mean(combined_metrics),
                'std': np.std(combined_metrics),
                'q25': np.percentile(combined_metrics, 25),
                'q75': np.percentile(combined_metrics, 75),
                'ci_lower': np.percentile(combined_metrics, 2.5),
                'ci_upper': np.percentile(combined_metrics, 97.5)
            })

        # Create violin plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=all_results)
        plt.axhline(y=0, color='r', linestyle='--', label='Isolation period end')
        plt.xticks(range(len(epsilon_values)), epsilon_values)
        plt.xlabel('Efficacy (ε)')
        plt.ylabel('Days Relative to Isolation Period End')
        plt.title(f'Time to Threshold Relative to Isolation Period End (p_fatal = {p_fatal:.2f})')
        plt.legend()
        plt.savefig(os.path.join(p_fatal_dir, 'time_to_threshold_relative_to_isolation_violin.png'))
        plt.close()
        

        # Create KDE plot
        plt.figure(figsize=(12, 8))
        for i, epsilon in enumerate(epsilon_values):
            if np.var(all_results[i]) > 0:  # Only plot if there's variance
                sns.kdeplot(all_results[i], label=f'ε = {epsilon}', fill=True)
            else:
                plt.axvline(all_results[i][0], label=f'ε = {epsilon}', color=f'C{i}')
                print(f"Warning: Zero variance for ε = {epsilon}, p_fatal = {p_fatal}")
        
        plt.xlabel('Days Above Threshold After Isolation')
        plt.ylabel('Density')
        plt.title(f'Kernel Density Estimation (p_fatal = {p_fatal:.2f})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(p_fatal_dir, 'days_above_threshold_after_isolation_kde.png'))
        plt.close()

        # Create histogram plot
        plt.figure(figsize=(12, 8))
        for i, epsilon in enumerate(epsilon_values):
            plt.hist(all_results[i], bins=30, alpha=0.7, label=f'ε = {epsilon}', density=True)
        
        plt.xlabel('Days Above Threshold After Isolation')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Days Above Threshold After Isolation (p_fatal = {p_fatal:.2f})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add 95% CI lines for each epsilon
        for i, epsilon in enumerate(epsilon_values):
            lower_ci = np.percentile(all_results[i], 2.5)
            upper_ci = np.percentile(all_results[i], 97.5)
            plt.axvline(lower_ci, color=f'C{i}', linestyle='--')
            plt.axvline(upper_ci, color=f'C{i}', linestyle='--')
            print(f"95% CI for ε = {epsilon}: ({lower_ci:.2f}, {upper_ci:.2f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(p_fatal_dir, 'days_above_threshold_after_isolation_histogram.png'))
        plt.close()
        
        
        # Create box plot
        plt.figure(figsize=(12, 8))
        plt.boxplot(all_results, labels=epsilon_values)
        plt.axhline(y=0, color='r', linestyle='--', label='Isolation period end')
        plt.xlabel('Efficacy (ε)')
        plt.ylabel('Days Relative to Isolation Period End')
        plt.title(f'Time to Threshold Relative to Isolation Period End (p_fatal = {p_fatal:.2f})')
        plt.legend()
        plt.savefig(os.path.join(p_fatal_dir, 'time_to_threshold_relative_to_isolation_boxplot.png'))
        plt.close()

        # Save summary statistics as CSV
        csv_path = os.path.join(p_fatal_dir, 'summary_statistics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epsilon', 'Median', 'Mean', 'Std Dev', '25th Perc', '75th Perc', '2.5th Perc', '97.5th Perc'])
            for stat in summary_stats:
                writer.writerow([f"{stat['epsilon']:.1f}",
                                 f"{stat['median']:.2f}",
                                 f"{stat['mean']:.2f}",
                                 f"{stat['std']:.2f}",
                                 f"{stat['q25']:.2f}",
                                 f"{stat['q75']:.2f}",
                                 f"{stat['ci_lower']:.2f}",
                                 f"{stat['ci_upper']:.2f}"])

        print(f"Summary statistics saved to {csv_path}")

    print("Epidemiological metrics analysis completed.")