import numpy as np
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.model.virus_model import cached_solve
from src.utils.statistical_utils import sample_t_star, calculate_time_to_threshold
from config.config import Config
import logging

logger = logging.getLogger(__name__)


def analyze_epidemiological_metrics(chains_f, chains_nf, p_fatal_values, burn_in_period, base_output_dir):
    """
    Analyze epidemiological metrics over different p_fatal values.
    
    For each p_fatal value, the function processes fatal and non-fatal chains,
    performs simulations to compute time-to-threshold metrics, creates violin and KDE plots,
    and saves summary statistics.
    """
    output_dir = os.path.join(base_output_dir, 'epidemiological_metrics')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for p_fatal in tqdm(p_fatal_values, desc="Processing p_fatal values"):
        result = process_single_p_fatal(p_fatal, chains_f, chains_nf, burn_in_period, output_dir)
        results.append(result)
        logger.info(result)

    logger.info("Epidemiological metrics analysis completed.")
    return results


def process_single_p_fatal(p_fatal, chains_f, chains_nf, burn_in_period, output_dir):
    """
    Process a single p_fatal value by simulating metrics for both fatal and non-fatal cases,
    combining the results using the specified weights, and generating plots and CSV summaries.
    """
    config = Config()
    p_fatal_dir = os.path.join(output_dir, f'p_fatal_{p_fatal:.2f}')
    os.makedirs(p_fatal_dir, exist_ok=True)
    
    all_results = []
    summary_stats = []
    n_simulations = 100  # Number of simulations for each parameter set
    
    for epsilon in config.EPSILON_VALUES:
        metrics_f = simulate_metrics(chains_f, burn_in_period, n_simulations, epsilon, desc="Fatal cases")
        metrics_nf = simulate_metrics(chains_nf, burn_in_period, n_simulations, epsilon, desc="Non-fatal cases")

        # Combine fatal and non-fatal metrics according to p_fatal weight
        combined_metrics = np.random.choice(
            metrics_f + metrics_nf, size=10000, 
            p=[p_fatal/len(metrics_f)]*len(metrics_f) + [(1-p_fatal)/len(metrics_nf)]*len(metrics_nf)
        )
        all_results.append(combined_metrics)
        summary_stats.append({'epsilon': epsilon, **calculate_summary_stats(combined_metrics)})

    plot_violin(all_results, config.EPSILON_VALUES, p_fatal, p_fatal_dir)
    plot_kde(all_results, config.EPSILON_VALUES, p_fatal, p_fatal_dir)
    
    # Save summary statistics to CSV
    csv_path = os.path.join(p_fatal_dir, 'summary_statistics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epsilon', 'Median', 'Mean', 'Std Dev', '25th Perc', '75th Perc', '2.5th Perc', '97.5th Perc'])
        for stat in summary_stats:
            writer.writerow([
                f"{stat['epsilon']:.1f}",
                f"{stat['median']:.2f}",
                f"{stat['mean']:.2f}",
                f"{stat['std']:.2f}",
                f"{stat['q25']:.2f}",
                f"{stat['q75']:.2f}",
                f"{stat['ci_lower']:.2f}",
                f"{stat['ci_upper']:.2f}"
            ])

    return f"Summary statistics saved to {csv_path}"


def simulate_metrics(chains, burn_in_period, n_simulations, epsilon, desc=""):
    """
    Simulate the time-to-threshold metric n_simulations times for a given epsilon value.
    """
    latter_chains = chains[:, burn_in_period:, :]
    flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
    return [
        calculate_metric(flattened_chains[np.random.randint(flattened_chains.shape[0])], epsilon)
        for _ in tqdm(range(n_simulations), desc=desc, leave=False)
    ]


def calculate_metric(params, epsilon):
    """
    Calculate the time-to-threshold metric for given model parameters and epsilon.
    """
    threshold = 4
    isolation_period = 21  # days of isolation
    time = np.linspace(0, 30, 301)  # 30 days, 301 points
    t_star = sample_t_star()
    solution = cached_solve(params, time, epsilon=epsilon, t_star=t_star)
    return calculate_time_to_threshold(solution, time)


def calculate_summary_stats(metrics):
    """
    Calculate summary statistics for an array of metrics.
    """
    return {
        'median': np.median(metrics),
        'mean': np.mean(metrics),
        'std': np.std(metrics),
        'q25': np.percentile(metrics, 25),
        'q75': np.percentile(metrics, 75),
        'ci_lower': np.percentile(metrics, 2.5),
        'ci_upper': np.percentile(metrics, 97.5)
    }


def plot_violin(all_results, epsilon_values, p_fatal, output_dir):
    """
    Generate and save a violin plot for the simulated metrics.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=all_results)
    plt.axhline(y=0, color='r', linestyle='--', label='Isolation period end')
    plt.xticks(range(len(epsilon_values)), epsilon_values)
    plt.xlabel('Efficacy (ε)')
    plt.ylabel('Days Relative to Isolation Period End')
    plt.title(f'Time to Threshold Relative to Isolation Period End (p_fatal = {p_fatal:.2f})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_to_threshold_relative_to_isolation_violin.png'))
    plt.close()


def plot_kde(all_results, epsilon_values, p_fatal, output_dir):
    """
    Generate and save a kernel density estimation plot for the simulated metrics.
    """
    plt.figure(figsize=(12, 8))
    for i, epsilon in enumerate(epsilon_values):
        if np.var(all_results[i]) > 0:
            sns.kdeplot(all_results[i], label=f'ε = {epsilon}', fill=True)
        else:
            plt.axvline(all_results[i][0], label=f'ε = {epsilon}', color=f'C{i}')
            logger.warning(f"Zero variance for ε = {epsilon}, p_fatal = {p_fatal}")
    plt.xlabel('Days Above Threshold After Isolation')
    plt.ylabel('Density')
    plt.title(f'Kernel Density Estimation (p_fatal = {p_fatal:.2f})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'days_above_threshold_after_isolation_kde.png'))
    plt.close()
