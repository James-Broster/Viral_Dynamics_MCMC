import numpy as np
import os
import logging
from tqdm import tqdm
import json
from multiprocessing import Pool
from functools import partial
import time
import csv
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from src.model.virus_model import cached_solve
from src.utils.statistical_utils import sample_t_star, fit_gamma_to_mean_iqr
from config.config import Config
from src.visualization.risk_burden_plots import RiskBurdenPlots
from scipy.stats import gamma

def analyze_chains(chains_f, chains_nf, burn_in_period, time_extended):
    config = Config()
    logging.basicConfig(filename=os.path.join(config.BASE_OUTPUT_DIR, 'debug.log'), level=logging.INFO)

    initial_shapes = {
        'chains_f': chains_f.shape,
        'chains_nf': chains_nf.shape
    }
    debug_shapes = {"initial": initial_shapes}

    results = {}
    for chains, case in [
        (chains_f, 'fatal'),
        (chains_nf, 'non_fatal')
    ]:
        processed_chains = chains[:, burn_in_period:, :]
        debug_shapes[f"{case}_processed"] = processed_chains.shape

        case_results = post_mcmc_analysis(processed_chains, time_extended, epsilon_values=config.EPSILON_VALUES)
        results[case] = case_results
        debug_shapes[f"{case}_post_mcmc"] = np.array(case_results).shape

    debug_file_path = os.path.join(config.BASE_OUTPUT_DIR, 'shape_debug.json')
    with open(debug_file_path, 'w') as f:
        json.dump(debug_shapes, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    logging.info(f"Shape debug information saved to {debug_file_path}")

    return results, debug_shapes

def calculate_and_plot_risk_burden(chains_f, chains_nf, burn_in_period, time_extended, fatal_dir, non_fatal_dir):
    config = Config()
    isolation_periods = config.ISOLATION_PERIODS
    thresholds = config.VIRAL_LOAD_THRESHOLDS

    risk_burdens = {}
    for chains, case, output_dir in [
        (chains_f, 'fatal', fatal_dir),
        (chains_nf, 'non_fatal', non_fatal_dir)
    ]:
        processed_chains = chains[:, burn_in_period:, :]
        start_time = time.time()
        risk_burden = calculate_risk_and_burden(processed_chains, isolation_periods, time_extended, thresholds, {}, output_dir)
        RiskBurdenPlots.plot_risk_burden(risk_burden, case, os.path.join(output_dir, 'risk_burden'))
        logging.info(f"Time taken for calculate_risk_and_burden and RiskBurdenPlots.plot_risk_burden for {case}: {time.time() - start_time:.2f} seconds")
        risk_burdens[case] = risk_burden

    return risk_burdens

def post_mcmc_analysis(chains, time_extended, epsilon_values=[0, 0.3, 0.6, 0.9]):
    logging.info(f"Starting post_mcmc_analysis with chains shape: {chains.shape}")
    
    all_results = []
    for chain in chains:
        chain_results = []
        for params in chain:
            for epsilon in epsilon_values:
                t_star = sample_t_star()
                solution = cached_solve(params, time_extended, epsilon=epsilon, t_star=t_star)
                time_to_threshold = calculate_time_to_threshold(solution, time_extended)
                chain_results.append({
                    'parameters': params,
                    't_star': t_star,
                    'epsilon': epsilon,
                    'time_to_threshold': time_to_threshold
                })
        all_results.append(chain_results)

    logging.info(f"post_mcmc_analysis completed with results for {len(all_results)} chains")
    return all_results

def calculate_risk_and_burden(chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, epsilon=0, t_star=0, save_to_csv=True):
    logging.info(f"Starting calculate_risk_and_burden for epsilon={epsilon}, t_star={t_star}...")
    start_time = time.time()

    all_viral_loads = []
    for chain in chains:
        chain_viral_loads = []
        for params in chain:
            try:
                solution = cached_solve(params, time_extended, epsilon=epsilon, t_star=t_star)
                chain_viral_loads.append(solution[:, 2])  # VL third
            except Exception as e:
                logging.error(f"Error solving ODE: {str(e)}")
                logging.error(f"params causing error: {params}")
        all_viral_loads.append(np.array(chain_viral_loads))

    debug_shapes["all_viral_loads"] = [vl.shape for vl in all_viral_loads]

    risk_burden = {threshold: {} for threshold in thresholds}
    for threshold in thresholds:
        for period in isolation_periods:
            risk_burden[threshold][period] = calculate_metrics(all_viral_loads, threshold, period)

    # Determine the case (fatal or non-fatal) based on the base_output_dir
    case = 'fatal' if 'fatal' in base_output_dir else 'non_fatal'
    
    # Create the csv_output_dir with the correct structure
    csv_output_dir = os.path.join(base_output_dir, 'treatment_effects', 'csv_outputs')
    
    # Write results to CSV if save_to_csv is True
    if save_to_csv:
        write_results_to_csv(risk_burden, csv_output_dir, case, epsilon, t_star)

    end_time = time.time()
    logging.info(f"Completed calculate_risk_and_burden for epsilon={epsilon}, t_star={t_star}. Total time taken: {end_time - start_time:.2f} seconds")

    return risk_burden

def calculate_metrics(all_viral_loads, threshold, period):
    days_unnecessarily_isolated = [[calculate_days_unnecessarily_isolated(vl, threshold, period) for vl in chain] for chain in all_viral_loads]
    days_above_threshold_post_release = [[calculate_days_above_threshold(vl, threshold, period) for vl in chain] for chain in all_viral_loads]
    proportion_above_threshold_at_release = [[int(is_above_threshold_at_release(vl, threshold, period)) for vl in chain] for chain in all_viral_loads]
    risk_scores = [[calculate_risk_score(vl, threshold, period) for vl in chain] for chain in all_viral_loads]

    flat_days_isolated = [day for chain in days_unnecessarily_isolated for day in chain]
    flat_days_released = [day for chain in days_above_threshold_post_release for day in chain]
    flat_released_above = [rel for chain in proportion_above_threshold_at_release for rel in chain]
    flat_risk_scores = [score for chain in risk_scores for score in chain]

    return {
        'days_unnecessarily_isolated': calculate_stats(flat_days_isolated),
        'days_above_threshold_post_release': calculate_stats(flat_days_released),
        'proportion_above_threshold_at_release': calculate_stats(flat_released_above, is_proportion=True),
        'risk_score': calculate_stats(flat_risk_scores)
    }

def calculate_stats(data, is_proportion=False, decimal_places=4):
    if is_proportion:
        n = len(data)
        successes = sum(data)
        proportion = successes / n
        ci_lower, ci_upper = proportion_confint(successes, n, method='wilson')
        return {
            'avg': round(proportion, decimal_places),
            'ci_lower': round(ci_lower, decimal_places),
            'ci_upper': round(ci_upper, decimal_places)
        }
    else:
        return {
            'avg': np.median(data),
            'ci_lower': np.percentile(data, 2.5),
            'ci_upper': np.percentile(data, 97.5)
        }

def is_above_threshold_at_release(viral_load, threshold, period):
    release_index = period * 10  # Assuming 10 points per day
    future_indices = range(release_index, len(viral_load))
    
    # Check if viral load is above threshold at release or will be above in the future
    for i in future_indices:
        if viral_load[i] > threshold:
            # Check if this is a peak (i.e., viral load decreases after this point)
            if i == len(viral_load) - 1 or viral_load[i] > viral_load[i+1]:
                return True
    return False

def calculate_days_unnecessarily_isolated(viral_load, threshold, period):
    crossing_day = find_crossing_day(viral_load, threshold)
    return max(0, period - crossing_day)

def calculate_days_above_threshold(viral_load, threshold, period):
    crossing_day = find_crossing_day(viral_load, threshold)
    return max(0, crossing_day - period)

def calculate_risk_score(viral_load, threshold, period):
    daily_vl = viral_load[::10]  # daily instead of finer grained 
    risk_contributions = np.maximum(0, daily_vl[period:] - threshold)
    return np.sum(risk_contributions)

def find_crossing_day(viral_load, threshold):
    crossings = np.where(np.diff(viral_load > threshold))[0]
    if len(crossings) == 0:
        return 30 if viral_load[0] >= threshold else 0
    for crossing in reversed(crossings):
        if viral_load[crossing] > viral_load[crossing + 1]:
            return crossing // 10
    return crossings[-1] // 10

def calculate_time_to_threshold(solution, time, threshold=4):
    log_V = solution[:, 2]
    threshold_crossed = np.where(log_V < threshold)[0]
    if len(threshold_crossed) > 0:
        return time[threshold_crossed[0]] - 21  # Subtract isolation period
    else:
        return 9  # 30 - 21, if threshold is never crossed

def calculate_risk_burden_for_epsilon_tstar(chains, time_extended, base_output_dir):
    config = Config()
    results = {}
    no_treatment_results = {}
    
    print(f"Starting calculate_risk_burden_for_epsilon_tstar with chains shape: {chains.shape}")
    logging.info("Calculating no-treatment results...")
    # Calculate no-treatment results first
    no_treatment_risk_burden = calculate_risk_and_burden(
        chains, 
        config.ISOLATION_PERIODS, 
        time_extended, 
        config.VIRAL_LOAD_THRESHOLDS, 
        {}, 
        base_output_dir,
        epsilon=0,
        t_star=0
    )
    
    print("No-treatment risk burden calculated. Processing results...")
    logging.info("Processing no-treatment results...")
    no_treatment_results = {threshold: {period: {metric: risk_burden[metric] 
                                                 for metric in METRICS}
                                        for period, risk_burden in periods.items()}
                            for threshold, periods in no_treatment_risk_burden.items()}

    print(f"No-treatment results processed. Keys: {list(no_treatment_results.keys())}")
    logging.info("Calculating treatment results for different epsilon and t_star values...")
    for epsilon in config.EPSILON_VALUES:
        for t_star in config.T_STAR_VALUES:
            print(f"Calculating risk and burden for epsilon={epsilon}, t_star={t_star}")
            logging.info(f"Calculating risk and burden for epsilon={epsilon}, t_star={t_star}")
            
            risk_burden = calculate_risk_and_burden(
                chains, 
                config.ISOLATION_PERIODS, 
                time_extended, 
                config.VIRAL_LOAD_THRESHOLDS, 
                {}, 
                base_output_dir,
                epsilon=epsilon,
                t_star=t_star
            )
            
            results[(epsilon, t_star)] = risk_burden
            print(f"Completed calculation for epsilon={epsilon}, t_star={t_star}")

    print(f"Completed calculating risk burden for all scenarios. Results keys: {list(results.keys())}")
    logging.info("Completed calculating risk burden for all scenarios.")
    return results, no_treatment_results


def calculate_risk_burden_fixed_tstar(chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, save_to_csv=False):
    config = Config()
    num_cores = 30  # Or however many cores you want to use
    logging.info(f"Starting calculate_risk_burden_fixed_tstar using {num_cores} cores...")
    start_time = time.time()

    # Generate fixed t_star values
    t_star_values = np.arange(0, 30.5, 0.5)

    # Calculate weights for each t_star value
    mean = 4.8
    iqr = (2, 6)
    shape, scale = fit_gamma_to_mean_iqr(mean, iqr)
    weights = gamma.pdf(t_star_values, shape, scale=scale)
    weights /= np.sum(weights)  # Normalize weights

    all_tasks = [(epsilon, t_star) 
                 for epsilon in config.EPSILON_VALUES 
                 for t_star in t_star_values]

    process_task = partial(process_single_task, 
                           chains=chains, 
                           isolation_periods=isolation_periods, 
                           time_extended=time_extended, 
                           thresholds=thresholds, 
                           debug_shapes=debug_shapes, 
                           base_output_dir=base_output_dir,
                           save_to_csv=save_to_csv)

    with Pool(processes=num_cores) as pool:
        all_results = list(tqdm(pool.imap(process_task, all_tasks), 
                                total=len(all_tasks),
                                desc="Processing epsilon-tstar pairs"))

    expected_results = len(config.EPSILON_VALUES) * len(t_star_values)
    assert len(all_results) == expected_results, f"Expected {expected_results} results, but got {len(all_results)}"

    results = {epsilon: {threshold: {period: {metric: {'avg': 0, 'ci_lower': 0, 'ci_upper': 0} for metric in METRICS} 
                                     for period in isolation_periods} 
                         for threshold in thresholds} 
               for epsilon in config.EPSILON_VALUES}

    for epsilon, t_star, epsilon_result in all_results:
        if epsilon_result is not None:
            weight = weights[np.where(t_star_values == t_star)[0][0]]
            for threshold in thresholds:
                for period in isolation_periods:
                    for metric in METRICS:
                        results[epsilon][threshold][period][metric]['avg'] += epsilon_result[threshold][period][metric]['avg'] * weight
                        results[epsilon][threshold][period][metric]['ci_lower'] += epsilon_result[threshold][period][metric]['ci_lower'] * weight
                        results[epsilon][threshold][period][metric]['ci_upper'] += epsilon_result[threshold][period][metric]['ci_upper'] * weight
        else:
            logging.warning(f"Skipping failed result for epsilon={epsilon}, t_star={t_star}")

    end_time = time.time()
    logging.info(f"Completed calculate_risk_burden_fixed_tstar. Total time taken: {end_time - start_time:.2f} seconds")

    return results, t_star_values, weights



def process_single_task(task, chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, save_to_csv):
    epsilon, t_star = task
    
    try:
        risk_burden = calculate_risk_and_burden(chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, epsilon, t_star, save_to_csv=save_to_csv)
        
        epsilon_result = {threshold: {period: {metric: risk_burden[threshold][period][metric] for metric in METRICS} 
                                      for period in isolation_periods} 
                          for threshold in thresholds}
        
        return epsilon, t_star, epsilon_result
    except Exception as e:
        logging.error(f"Error processing task (epsilon={epsilon}, t_star={t_star}): {str(e)}")
        return epsilon, t_star, None


def write_results_to_csv(risk_burden, output_dir, case, epsilon, t_star):
    metrics = [
        'days_unnecessarily_isolated',
        'days_above_threshold_post_release',
        'proportion_above_threshold_at_release',
        'risk_score'
    ]

    for threshold in risk_burden.keys():
        for metric in metrics:
            # Create a directory for each threshold and metric
            threshold_metric_dir = os.path.join(output_dir, str(threshold), metric)
            os.makedirs(threshold_metric_dir, exist_ok=True)

            filename = f"{case}_epsilon_{epsilon}_tstar_{t_star}.csv"
            filepath = os.path.join(threshold_metric_dir, filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Day of isolation', 'Mean', 'Lower_CI', 'Upper_CI'])
                
                for isolation_day in sorted(risk_burden[threshold].keys()):
                    result = risk_burden[threshold][isolation_day][metric]
                    writer.writerow([
                        isolation_day,
                        f"{result['avg']}",
                        f"{result['ci_lower']}",
                        f"{result['ci_upper']}"
                    ])
            
            print(f"CSV file for threshold {threshold}, {metric} saved to {filepath}")

METRICS = ['days_unnecessarily_isolated', 'days_above_threshold_post_release', 
           'proportion_above_threshold_at_release', 'risk_score']