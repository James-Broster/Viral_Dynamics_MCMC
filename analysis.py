import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv
import logging
from utils import sample_t_star, calculate_time_to_threshold
from virus_model import cached_solve
from visualization import Visualization
import config
from multiprocessing import Pool
from functools import partial
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_chains(chains_f, chains_nf, burn_in_period, time_extended):
    p_fatal_values = config.P_FATAL_VALUES
    epsilon_values = config.EPSILON_VALUES
    base_output_dir = config.BASE_OUTPUT_DIR

    logging.basicConfig(filename=os.path.join(base_output_dir, 'debug.log'), level=logging.INFO)

    initial_shapes = {
        'chains_f': chains_f.shape,
        'chains_nf': chains_nf.shape
    }
    debug_shapes = {"initial": initial_shapes}

    start_time = time.time()
    analyze_epidemiological_metrics(chains_f, chains_nf, p_fatal_values, burn_in_period, base_output_dir)
    logging.info(f"Time taken for analyze_epidemiological_metrics: {time.time() - start_time} seconds")

    results = {}
    for chains, case in [
        (chains_f, 'fatal'),
        (chains_nf, 'non_fatal')
    ]:
        start_time = time.time()
        processed_chains = chains[:, burn_in_period:, :]
        debug_shapes[f"{case}_processed"] = processed_chains.shape

        case_results = post_mcmc_analysis(processed_chains, time_extended, epsilon_values=epsilon_values)
        results[case] = case_results
        logging.info(f"Time taken for post_mcmc_analysis for {case}: {time.time() - start_time} seconds")

        debug_shapes[f"{case}_post_mcmc"] = np.array(case_results).shape

    # Save debug shapes to a file
    debug_file_path = os.path.join(base_output_dir, 'shape_debug.json')
    with open(debug_file_path, 'w') as f:
        json.dump(debug_shapes, f, indent=2, default=numpy_to_python)

    logging.info(f"Shape debug information saved to {debug_file_path}")

    return results, debug_shapes

def calculate_and_plot_risk_burden(chains_f, chains_nf, burn_in_period, time_extended):
    isolation_periods = config.ISOLATION_PERIODS
    thresholds = config.VIRAL_LOAD_THRESHOLDS
    base_output_dir = config.BASE_OUTPUT_DIR

    risk_burdens = {}
    for chains, case in [
        (chains_f, 'fatal'),
        (chains_nf, 'non_fatal')
    ]:
        processed_chains = chains[:, burn_in_period:, :]
        start_time = time.time()
        risk_burden = calculate_risk_and_burden(processed_chains, isolation_periods, time_extended, thresholds, {}, base_output_dir)
        Visualization.plot_risk_burden(risk_burden, case, os.path.join(base_output_dir, case))
        logging.info(f"Time taken for calculate_risk_and_burden and Visualization.plot_risk_burden for {case}: {time.time() - start_time} seconds")
        risk_burdens[case] = risk_burden

    return risk_burdens

def process_single_chain(chain, time, epsilon_values):
    results = []
    for params in chain:
        for epsilon in epsilon_values:
            t_star = sample_t_star()
            solution = cached_solve(params, time, epsilon=epsilon, t_star=t_star)
            time_to_threshold = calculate_time_to_threshold(solution, time)
            results.append({
                'parameters': params,
                't_star': t_star,
                'epsilon': epsilon,
                'time_to_threshold': time_to_threshold
            })
    return results

def post_mcmc_analysis(chains, time_extended, epsilon_values=[0, 0.3, 0.6, 0.9]):
    logging.info(f"Starting post_mcmc_analysis with chains shape: {chains.shape}")
    num_cores = 8
    pool = Pool(processes=num_cores)

    process_func = partial(process_single_chain, time=time_extended, 
                           epsilon_values=epsilon_values)

    all_results = pool.map(process_func, chains)

    pool.close()
    pool.join()

    logging.info(f"post_mcmc_analysis completed with results for {len(all_results)} chains")
    return all_results

def calculate_risk_and_burden(chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, epsilon=0, t_star=0):
    logging.info(f"Starting calculate_risk_and_burden for epsilon={epsilon}, t_star={t_star}...")
    start_time = time.time()

    risk_burden = process_all_scenarios(chains, thresholds, isolation_periods, time_extended, debug_shapes, base_output_dir, epsilon, t_star)

    end_time = time.time()
    logging.info(f"Completed calculate_risk_and_burden for epsilon={epsilon}, t_star={t_star}. Total time taken: {end_time - start_time:.2f} seconds")

    return risk_burden

def process_all_scenarios(chains, thresholds, periods, time_extended, debug_shapes, base_output_dir, epsilon, t_star):
    logging.info(f"Starting to solve ODEs for all parameter sets with epsilon={epsilon}, t_star={t_star}...")
    start_time = time.time()

    # Process each chain separately
    all_viral_loads = []
    for chain in chains:
        chain_viral_loads = []
        for params in chain:
            try:
                solution = cached_solve(params, time_extended, epsilon=epsilon, t_star=t_star)
                chain_viral_loads.append(solution[:, 2])  # log10 of viral load is the third column
            except Exception as e:
                logging.error(f"Error solving ODE: {str(e)}")
                logging.error(f"params causing error: {params}")
        all_viral_loads.append(np.array(chain_viral_loads))

    debug_shapes["all_viral_loads"] = [vl.shape for vl in all_viral_loads]

    logging.info(f"Completed solving ODEs for all parameter sets. Time taken: {time.time() - start_time:.2f} seconds")

    risk_burden = {threshold: {} for threshold in thresholds}
    debug_summary = {
        "shapes": debug_shapes,
        "structure": {},
        "sample_data": {}
    }

    def find_crossing_day(viral_load, threshold):
        # Find all crossing points
        crossings = np.where(np.diff(viral_load > threshold))[0]

        if len(crossings) == 0:
            # Never crosses threshold
            return 30 if viral_load[0] >= threshold else 0

        # Find the last crossing with a negative gradient
        for crossing in reversed(crossings):
            if viral_load[crossing] > viral_load[crossing + 1]:
                return crossing // 10  # Convert to days (301 points / 30 days ≈ 10 points per day)

        # If no crossing with negative gradient is found, return the last crossing
        return crossings[-1] // 10

    def calculate_days_above_threshold(viral_load, threshold, period):
        crossing_day = find_crossing_day(viral_load, threshold)
        return max(0, crossing_day - period)

    def calculate_days_unnecessarily_isolated(viral_load, threshold, period):
        crossing_day = find_crossing_day(viral_load, threshold)
        return max(0, period - crossing_day)

    def calculate_risk_score(viral_load, threshold, period):
        daily_vl = viral_load[::10]  # Take every 10th point to get daily values
        risk_contributions = np.maximum(0, daily_vl[period:] - threshold)
        return np.sum(risk_contributions)

    for threshold in thresholds:
        for period in periods:
            logging.info(f"Processing threshold {threshold}, period {period}")
            scenario_start_time = time.time()

            days_unnecessarily_isolated = [[calculate_days_unnecessarily_isolated(vl, threshold, period) for vl in chain] for chain in all_viral_loads]
            days_above_threshold_post_release = [[calculate_days_above_threshold(vl, threshold, period) for vl in chain] for chain in all_viral_loads]

            proportion_above_threshold_at_release = [[int(vl[period*10] > threshold) for vl in chain] for chain in all_viral_loads]
            proportion_below_threshold_at_release = [[1 - above for above in chain] for chain in proportion_above_threshold_at_release]

            risk_scores = [[calculate_risk_score(vl, threshold, period) for vl in chain] for chain in all_viral_loads]

            # Flatten for overall statistics
            flat_days_isolated = [day for chain in days_unnecessarily_isolated for day in chain]
            flat_days_released = [day for chain in days_above_threshold_post_release for day in chain]
            flat_released_above = [rel for chain in proportion_above_threshold_at_release for rel in chain]
            flat_released_below = [rel for chain in proportion_below_threshold_at_release for rel in chain]
            flat_risk_scores = [score for chain in risk_scores for score in chain]

            risk_burden[threshold][period] = {
                'days_unnecessarily_isolated': {
                    'avg': np.mean(flat_days_isolated),
                    'ci_lower': np.percentile(flat_days_isolated, 2.5),
                    'ci_upper': np.percentile(flat_days_isolated, 97.5)
                },
                'days_above_threshold_post_release': {
                    'avg': np.mean(flat_days_released),
                    'ci_lower': np.percentile(flat_days_released, 2.5),
                    'ci_upper': np.percentile(flat_days_released, 97.5)
                },
                'proportion_above_threshold_at_release': {
                    'avg': np.mean(flat_released_above),
                    'ci_lower': np.percentile(flat_released_above, 2.5),
                    'ci_upper': np.percentile(flat_released_above, 97.5)
                },
                'proportion_below_threshold_at_release': {
                    'avg': np.mean(flat_released_below),
                    'ci_lower': np.percentile(flat_released_below, 2.5),
                    'ci_upper': np.percentile(flat_released_below, 97.5)
                },
                'risk_score': {
                    'avg': np.mean(flat_risk_scores),
                    'ci_lower': np.percentile(flat_risk_scores, 2.5),
                    'ci_upper': np.percentile(flat_risk_scores, 97.5)
                }
            }

            # Add structure to debug summary
            if threshold not in debug_summary["structure"]:
                debug_summary["structure"][threshold] = {}
            debug_summary["structure"][threshold][period] = list(risk_burden[threshold][period].keys())

            # Add sample data for specific points
            if threshold == thresholds[0] and period in [0, 15, 30]:
                debug_summary["sample_data"][f"threshold_{threshold}_period_{period}"] = risk_burden[threshold][period]

            logging.info(f"Completed processing for threshold {threshold}, period {period}. Time taken: {time.time() - scenario_start_time:.2f} seconds")

    # Save debug summary to a file
    debug_file_path = os.path.join(base_output_dir, 'debug_summary.json')
    with open(debug_file_path, 'w') as f:
        json.dump(debug_summary, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    logging.info(f"Debug summary saved to {debug_file_path}")

    return risk_burden


def analyze_epidemiological_metrics(chains_f, chains_nf, p_fatal_values, burn_in_period, base_output_dir):
    output_dir = os.path.join(base_output_dir, 'epidemiological_metrics')
    os.makedirs(output_dir, exist_ok=True)

    num_cores = 8
    pool = Pool(processes=num_cores)
    
    process_func = partial(process_single_p_fatal, chains_f=chains_f, chains_nf=chains_nf, 
                           burn_in_period=burn_in_period, output_dir=output_dir)
    
    results = list(tqdm(pool.imap(process_func, p_fatal_values), total=len(p_fatal_values), desc="Processing p_fatal values"))
    
    pool.close()
    pool.join()
    
    for result in results:
        print(result)

    print("Epidemiological metrics analysis completed.")

def process_single_p_fatal(p_fatal, chains_f, chains_nf, burn_in_period, output_dir):
    p_fatal_dir = os.path.join(output_dir, f'p_fatal_{p_fatal:.2f}')
    os.makedirs(p_fatal_dir, exist_ok=True)
    
    all_results = []
    summary_stats = []
    
    n_simulations = 100  # Number of simulations for each parameter set
    
    for epsilon in config.EPSILON_VALUES:
        metrics_f = simulate_metrics(chains_f, burn_in_period, n_simulations, epsilon, desc="Fatal cases")
        metrics_nf = simulate_metrics(chains_nf, burn_in_period, n_simulations, epsilon, desc="Non-fatal cases")

        combined_metrics = np.random.choice(metrics_f + metrics_nf, size=10000, 
                                            p=[p_fatal/len(metrics_f)]*len(metrics_f) + 
                                              [(1-p_fatal)/len(metrics_nf)]*len(metrics_nf))
        
        all_results.append(combined_metrics)
        summary_stats.append({'epsilon': epsilon, **calculate_summary_stats(combined_metrics)})

    plot_violin(all_results, config.EPSILON_VALUES, p_fatal, p_fatal_dir)
    plot_kde(all_results, config.EPSILON_VALUES, p_fatal, p_fatal_dir)
    
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

    return f"Summary statistics saved to {csv_path}"

def simulate_metrics(chains, burn_in_period, n_simulations, epsilon, desc=""):
    latter_chains = chains[:, burn_in_period:, :]
    flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
    return [calculate_metric(flattened_chains[np.random.randint(flattened_chains.shape[0])], epsilon) 
            for _ in tqdm(range(n_simulations), desc=desc, leave=False)]

def calculate_metric(params, epsilon):
    threshold = 4  # log10 VL threshold
    isolation_period = 21  # days of isolation
    time = np.linspace(0, 30, 301)  # 30 days, 301 points
    t_star = sample_t_star()
    solution = cached_solve(params, time, epsilon=epsilon, t_star=t_star)
    return calculate_time_to_threshold(solution, time)

def calculate_summary_stats(metrics):
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
    plt.figure(figsize=(12, 8))
    for i, epsilon in enumerate(epsilon_values):
        if np.var(all_results[i]) > 0:
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
    plt.savefig(os.path.join(output_dir, 'days_above_threshold_after_isolation_kde.png'))
    plt.close()