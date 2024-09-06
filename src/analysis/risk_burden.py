import numpy as np
import os
import logging
from tqdm import tqdm
import json
from multiprocessing import Pool
from functools import partial
import time
from src.model.virus_model import cached_solve
from src.utils.statistical_utils import sample_t_star
from config.config import Config
from src.visualization.risk_burden_plots import RiskBurdenPlots

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

    risk_burden = calculate_risk_burden_metrics(all_viral_loads, thresholds, periods)

    return risk_burden

def calculate_risk_burden_metrics(all_viral_loads, thresholds, periods):
    risk_burden = {threshold: {} for threshold in thresholds}

    for threshold in thresholds:
        for period in periods:
            metrics = calculate_metrics(all_viral_loads, threshold, period)
            risk_burden[threshold][period] = metrics

    return risk_burden

def calculate_metrics(all_viral_loads, threshold, period):
    days_unnecessarily_isolated = [[calculate_days_unnecessarily_isolated(vl, threshold, period) for vl in chain] for chain in all_viral_loads]
    days_above_threshold_post_release = [[calculate_days_above_threshold(vl, threshold, period) for vl in chain] for chain in all_viral_loads]
    proportion_above_threshold_at_release = [[int(vl[period*10] > threshold) for vl in chain] for chain in all_viral_loads]
    proportion_below_threshold_at_release = [[1 - above for above in chain] for chain in proportion_above_threshold_at_release]
    risk_scores = [[calculate_risk_score(vl, threshold, period) for vl in chain] for chain in all_viral_loads]

    flat_days_isolated = [day for chain in days_unnecessarily_isolated for day in chain]
    flat_days_released = [day for chain in days_above_threshold_post_release for day in chain]
    flat_released_above = [rel for chain in proportion_above_threshold_at_release for rel in chain]
    flat_released_below = [rel for chain in proportion_below_threshold_at_release for rel in chain]
    flat_risk_scores = [score for chain in risk_scores for score in chain]

    return {
        'days_unnecessarily_isolated': calculate_stats(flat_days_isolated),
        'days_above_threshold_post_release': calculate_stats(flat_days_released),
        'proportion_above_threshold_at_release': calculate_stats(flat_released_above),
        'proportion_below_threshold_at_release': calculate_stats(flat_released_below),
        'risk_score': calculate_stats(flat_risk_scores)
    }

def calculate_stats(data):
    return {
        'avg': np.mean(data),
        'ci_lower': np.percentile(data, 2.5),
        'ci_upper': np.percentile(data, 97.5)
    }

def calculate_days_unnecessarily_isolated(viral_load, threshold, period):
    crossing_day = find_crossing_day(viral_load, threshold)
    return max(0, period - crossing_day)

def calculate_days_above_threshold(viral_load, threshold, period):
    crossing_day = find_crossing_day(viral_load, threshold)
    return max(0, crossing_day - period)

def calculate_risk_score(viral_load, threshold, period):
    daily_vl = viral_load[::10]  # Take every 10th point to get daily values
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
    for epsilon in config.EPSILON_VALUES:
        for t_star in config.T_STAR_VALUES:
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
    
    return results

def calculate_risk_burden_sampled_tstar(chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir, num_samples=10):
    config = Config()
    num_cores = 6  # Or however many cores you want to use
    logging.info(f"Starting calculate_risk_burden_sampled_tstar with {num_samples} samples per epsilon using {num_cores} cores...")
    start_time = time.time()

    all_tasks = [(epsilon, sample_index) 
                 for epsilon in config.EPSILON_VALUES 
                 for sample_index in range(num_samples)]

    process_task = partial(process_single_task, 
                           chains=chains, 
                           isolation_periods=isolation_periods, 
                           time_extended=time_extended, 
                           thresholds=thresholds, 
                           debug_shapes=debug_shapes, 
                           base_output_dir=base_output_dir)

    with Pool(processes=num_cores) as pool:
        all_results = list(tqdm(pool.imap(process_task, all_tasks), 
                                total=len(all_tasks),
                                desc="Processing epsilon-sample pairs"))

    expected_results = len(config.EPSILON_VALUES) * num_samples
    assert len(all_results) == expected_results, f"Expected {expected_results} results, but got {len(all_results)}"

    results = {epsilon: {threshold: {period: {metric: [] for metric in METRICS} 
                                     for period in isolation_periods} 
                         for threshold in thresholds} 
               for epsilon in config.EPSILON_VALUES}
    t_star_samples = {epsilon: [] for epsilon in config.EPSILON_VALUES}

    for epsilon, sample_index, epsilon_result, t_star in all_results:
        if epsilon_result is not None and t_star is not None:
            t_star_samples[epsilon].append(t_star)
            for threshold in thresholds:
                for period in isolation_periods:
                    for metric in METRICS:
                        results[epsilon][threshold][period][metric].append(epsilon_result[threshold][period][metric])
        else:
            logging.warning(f"Skipping failed result for epsilon={epsilon}, sample={sample_index}")

    for epsilon in config.EPSILON_VALUES:
        for threshold in thresholds:
            for period in isolation_periods:
                for metric in METRICS:
                    values = results[epsilon][threshold][period][metric]
                    if values:
                        results[epsilon][threshold][period][metric] = {
                            'avg': np.mean(values),
                            'ci_lower': np.percentile(values, 2.5),
                            'ci_upper': np.percentile(values, 97.5)
                        }
                    else:
                        logging.warning(f"No valid results for epsilon={epsilon}, threshold={threshold}, period={period}, metric={metric}")
                        results[epsilon][threshold][period][metric] = {
                            'avg': np.nan,
                            'ci_lower': np.nan,
                            'ci_upper': np.nan
                        }

    end_time = time.time()
    logging.info(f"Completed calculate_risk_burden_sampled_tstar. Total time taken: {end_time - start_time:.2f} seconds")

    return results, t_star_samples

def process_single_task(task, chains, isolation_periods, time_extended, thresholds, debug_shapes, base_output_dir):
    epsilon, sample_index = task
    np.random.seed()  # Ensure different random seeds for each process
    
    try:
        t_star = sample_t_star()
        risk_burden = process_all_scenarios(chains, thresholds, isolation_periods, time_extended, debug_shapes, base_output_dir, epsilon, t_star)
        
        epsilon_result = {threshold: {period: {metric: risk_burden[threshold][period][metric]['avg'] for metric in METRICS} 
                                      for period in isolation_periods} 
                          for threshold in thresholds}
        
        return epsilon, sample_index, epsilon_result, t_star
    except Exception as e:
        logging.error(f"Error processing task (epsilon={epsilon}, sample={sample_index}): {str(e)}")
        return epsilon, sample_index, None, None

METRICS = ['days_unnecessarily_isolated', 'days_above_threshold_post_release', 
           'proportion_above_threshold_at_release', 'proportion_below_threshold_at_release', 'risk_score']