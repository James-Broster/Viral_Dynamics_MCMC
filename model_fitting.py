import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from multiprocessing import Pool
from tqdm import tqdm

from virus_model import VirusModel
from constants import PARAM_NAMES, PARAM_BOUNDS, STEP_SIZES, FIXED_PARAMS

class ModelFitting:
    @staticmethod
    def calculate_log_likelihood(parameters, time, observed_data, sigma):
        full_params = np.concatenate([parameters, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
        model_data = VirusModel.solve(full_params, time, 0, 0)  # No therapy for initial fitting
        model_log_virusload = model_data[:, 2].reshape(-1, 1)
        residuals = observed_data - model_log_virusload
        n = len(observed_data)
        log_likelihood = -0.5 * (n * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / sigma**2)
        if np.isnan(log_likelihood):
            print(f"NaN log-likelihood detected. Parameters: {parameters}, Sigma: {sigma}")
            print(f"Model log virusload: {model_log_virusload}")
            print(f"Observed data: {observed_data}")
            log_likelihood = -np.inf  # Set to negative infinity only if NaN
        return log_likelihood

    @staticmethod
    def calculate_model_error(parameters, time, observed_data):
        full_params = np.concatenate([parameters, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
        model_data = VirusModel.solve(full_params, time, 0, 0)  # No therapy for initial fitting
        model_log_virusload = model_data[:, 2].reshape(-1, 1)
        residuals = observed_data - model_log_virusload
        error = np.sqrt(np.mean(residuals**2))
        if np.isnan(error):
            print(f"NaN error detected. Parameters: {parameters}")
            print(f"Model log virusload: {model_log_virusload}")
            print(f"Observed data: {observed_data}")
            error = 1e10  # Set a large error only if NaN
        return error

    @staticmethod
    def propose_new_parameters(current_parameters):
        proposed_parameters = current_parameters.copy()
        for j, param_name in enumerate(PARAM_NAMES):
            lower, upper = PARAM_BOUNDS[param_name]
            step_size = STEP_SIZES[param_name]
            while True:
                proposal = np.random.normal(current_parameters[j], step_size)
                if lower <= proposal <= upper:
                    proposed_parameters[j] = proposal
                    break
        return proposed_parameters

    @staticmethod
    def calculate_log_prior(parameters, param_means, param_stds):
        log_prior = 0
        for j, param_name in enumerate(PARAM_NAMES):
            log_prior += norm.logpdf(parameters[j], loc=param_means[j], scale=param_stds[j])
        return log_prior

    @staticmethod
    def perform_mcmc_iteration(args):
        data, initial_parameters, num_iterations, time, burn_in_period, transition_period, chain_id, param_means, param_stds, initial_conditions = args
        parameter_values = [initial_parameters]
        viral_load_predictions = []
        current_parameters = np.array(initial_parameters)
        acceptance_counts = np.zeros(len(initial_parameters))
        total_proposals = np.zeros(len(initial_parameters))

        current_sigma = ModelFitting.calculate_model_error(current_parameters, time, data)
        current_log_prior = ModelFitting.calculate_log_prior(current_parameters, param_means, param_stds)
        current_prediction = VirusModel.solve(np.concatenate([current_parameters, initial_conditions]), time, 0, 0)  # No therapy for initial fitting
        print(f"Chain {chain_id}: Initial sigma: {current_sigma}")

        for i in range(num_iterations):
            proposed_parameters = ModelFitting.propose_new_parameters(current_parameters)
            proposed_prediction = VirusModel.solve(np.concatenate([proposed_parameters, initial_conditions]), time, 0, 0)  # No therapy for initial fitting
            proposed_sigma = ModelFitting.calculate_model_error(proposed_parameters, time, data)
            proposed_log_prior = ModelFitting.calculate_log_prior(proposed_parameters, param_means, param_stds)
            
            current_ll = ModelFitting.calculate_log_likelihood(current_parameters, time, data, current_sigma)
            proposed_ll = ModelFitting.calculate_log_likelihood(proposed_parameters, time, data, proposed_sigma)
            
            log_acceptance_ratio = float('-inf')  # Default to -inf
            if np.isfinite(current_ll) and np.isfinite(proposed_ll):
                log_acceptance_ratio = (proposed_ll + proposed_log_prior) - (current_ll + current_log_prior)
                if log_acceptance_ratio > 0 or np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
                    current_parameters = proposed_parameters
                    current_sigma = proposed_sigma
                    current_log_prior = proposed_log_prior
                    current_prediction = proposed_prediction
                    acceptance_counts += 1
            else:
                print(f"Chain {chain_id}: Non-finite log-likelihood detected. Current LL: {current_ll}, Proposed LL: {proposed_ll}")
                print(f"Current parameters: {current_parameters}, Proposed parameters: {proposed_parameters}")
                print(f"Current sigma: {current_sigma}, Proposed sigma: {proposed_sigma}")
            
            total_proposals += 1
            parameter_values.append(current_parameters.copy())
            viral_load_predictions.append(current_prediction)

            if (i + 1) % 100 == 0:
                ModelFitting.print_iteration_info(chain_id, i, num_iterations, log_acceptance_ratio, 
                                                  acceptance_counts / total_proposals, current_parameters, 
                                                  proposed_parameters, current_sigma)

        final_acceptance_rates = acceptance_counts / total_proposals
        return parameter_values, viral_load_predictions, final_acceptance_rates

    @staticmethod
    def print_iteration_info(chain_id, iteration, num_iterations, log_acceptance_ratio, acceptance_rates, current_parameters, proposed_parameters, current_sigma):
        print(f"Chain {chain_id}: Iteration {iteration + 1}/{num_iterations}, "
              f"Log Acceptance Ratio: {log_acceptance_ratio:.4f}, "
              f"Mean Acceptance: {np.mean(acceptance_rates):.4f}, "
              f"Current Sigma: {current_sigma:.4e}")
        print(f"Current Parameters: {current_parameters}")
        print(f"Proposed Parameters: {proposed_parameters}")
        print(f"Acceptance Rates: {acceptance_rates}")

    @staticmethod
    def calculate_rhat(chains):
        n = chains.shape[1]  # number of iterations
        m = chains.shape[0]  # number of chains

        # Mean of each chain
        chain_means = np.mean(chains, axis=1)

        # Variance within each chain
        chain_variances = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_variances)

        # Mean of means
        overall_mean = np.mean(chain_means)

        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)

        # Estimate of target distribution variance
        V_hat = (n - 1) / n * W + B / n

        # Calculate R-hat
        R_hat = np.sqrt(V_hat / W)

        return R_hat

    @staticmethod
    def execute_parallel_mcmc(data, num_chains, num_iterations, time, burn_in_period, transition_period, param_means, param_stds, initial_conditions):
        args = []
        for chain_id in range(num_chains):
            initial_params = []
            for i, param_name in enumerate(PARAM_NAMES):
                lower, upper = PARAM_BOUNDS[param_name]
                initial_param = np.random.uniform(lower, upper)
                initial_params.append(initial_param)
            args.append((data, initial_params, num_iterations, time, burn_in_period, transition_period, chain_id, param_means, param_stds, initial_conditions))

        results = []
        try:
            with Pool(processes=num_chains) as pool:
                for result in tqdm(pool.imap_unordered(ModelFitting.perform_mcmc_iteration, args), total=num_chains, desc="MCMC Progress"):
                    results.append(result)
        except KeyboardInterrupt:
            print("MCMC interrupted. Terminating processes...")
            pool.terminate()
            pool.join()
            raise
        
        chains, viral_loads, acceptance_rates = zip(*results)
        chains = np.array(chains)
        viral_loads = np.array(viral_loads)
        
        r_hat = ModelFitting.calculate_rhat(chains[:, burn_in_period:, :])
        
        return chains, viral_loads, np.array(acceptance_rates), r_hat

    @staticmethod
    def calculate_correlations(chains):
        flattened_chains = chains.reshape(-1, chains.shape[-1])
        return np.corrcoef(flattened_chains.T)

    @staticmethod
    def calculate_ess(chain):
        """Calculate the Effective Sample Size for a single chain."""
        n = len(chain)
        if n <= 1:
            return 0
        acf_values = acf(chain, nlags=n//3, fft=False)  # Compute autocorrelations
        # Sum autocorrelations until they become negative
        positive_acf = np.where(acf_values > 0)[0]
        sum_rho = 2 * np.sum(acf_values[1:len(positive_acf)])
        ess = n / (1 + sum_rho)
        return ess

    @staticmethod
    def calculate_multichain_ess(chains):
        """Calculate ESS for multiple chains."""
        n_chains, n_samples, n_params = chains.shape
        ess_values = np.zeros(n_params)
        for i in range(n_params):
            combined_chain = chains[:, :, i].flatten()
            ess_values[i] = ModelFitting.calculate_ess(combined_chain)
        return ess_values

    @staticmethod
    def calculate_time_to_threshold_stats(chains, burn_in_period, epsilon, t_star):
        latter_chains = chains[:, burn_in_period::100, :]  # Thinning: use every 100th set after burn-in
        flattened_chains = latter_chains.reshape(-1, latter_chains.shape[-1])
        
        times = []
        for params in flattened_chains:
            full_params = np.concatenate([params, [FIXED_PARAMS['f1_0'], FIXED_PARAMS['f2_0'], FIXED_PARAMS['V_0']]])
            time = VirusModel.calculate_time_to_threshold(full_params, epsilon, t_star)
            times.append(time)
        
        times = np.array(times)
        median = np.median(times)
        lower_ci = np.percentile(times, 2.5)
        upper_ci = np.percentile(times, 97.5)
        
        return median, lower_ci, upper_ci