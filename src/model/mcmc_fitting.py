import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from multiprocessing import Pool
from tqdm import tqdm
from config.config import Config
from src.model.virus_model import cached_solve

class ModelFitting:
    @staticmethod
    def calculate_log_likelihood(parameters, time, observed_data, sigma):
        model_data = cached_solve(parameters, time)
        model_log_virusload = model_data[:, 2].reshape(-1, 1)
        residuals = observed_data - model_log_virusload
        n = len(observed_data)
        log_likelihood = -0.5 * (n * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / sigma**2)
        return log_likelihood

    @staticmethod
    def calculate_model_error(parameters, time, observed_data):
        model_data = cached_solve(parameters, time)
        model_log_virusload = model_data[:, 2].reshape(-1, 1)
        residuals = observed_data - model_log_virusload
        error = np.sqrt(np.mean(residuals**2))
        if np.isnan(error):
            print(f"NaN error detected. Parameters: {parameters}")
            error = 1e10  # Set a large error only if NaN
        return error

    @staticmethod
    def propose_new_parameters(current_parameters):
        config = Config()
        proposed_parameters = current_parameters.copy()
        for j, param_name in enumerate(config.PARAM_NAMES):
            lower, upper = config.PARAM_BOUNDS[param_name]
            step_size = config.STEP_SIZES[param_name]
            while True:
                proposal = np.random.normal(current_parameters[j], step_size)
                if lower <= proposal <= upper:
                    proposed_parameters[j] = proposal
                    break
        return proposed_parameters

    @staticmethod
    def calculate_log_prior(parameters, param_means, param_stds):
        log_prior = 0
        for j, param_name in enumerate(Config.PARAM_NAMES):
            log_prior += norm.logpdf(parameters[j], loc=param_means[j], scale=param_stds[j])
        return log_prior

    @staticmethod
    def perform_mcmc_iteration(args):
        data, time, initial_parameters, num_iterations, burn_in_period, transition_period, chain_id, param_means, param_stds = args
        parameter_values = [initial_parameters]
        current_parameters = np.array(initial_parameters)
        acceptance_counts = np.zeros(len(initial_parameters))
        total_proposals = np.zeros(len(initial_parameters))
        acceptance_rates_over_time = []

        current_sigma = ModelFitting.calculate_model_error(current_parameters, time, data)
        current_log_prior = ModelFitting.calculate_log_prior(current_parameters, param_means, param_stds)
        print(f"Chain {chain_id}: Initial sigma: {current_sigma}")

        for i in range(num_iterations):
            proposed_parameters = ModelFitting.propose_new_parameters(current_parameters)
            proposed_sigma = ModelFitting.calculate_model_error(proposed_parameters, time, data)
            proposed_log_prior = ModelFitting.calculate_log_prior(proposed_parameters, param_means, param_stds)
            
            current_ll = ModelFitting.calculate_log_likelihood(current_parameters, time, data, current_sigma)
            proposed_ll = ModelFitting.calculate_log_likelihood(proposed_parameters, time, data, proposed_sigma)
            
            log_acceptance_ratio = (proposed_ll + proposed_log_prior) - (current_ll + current_log_prior)
            
            if log_acceptance_ratio > 0 or np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
                current_parameters = proposed_parameters
                current_sigma = proposed_sigma
                current_log_prior = proposed_log_prior
                acceptance_counts += 1
            
            total_proposals += 1
            parameter_values.append(current_parameters.copy())

            acceptance_rates_over_time.append(acceptance_counts / (i + 1))

            if (i + 1) % 100 == 0:
                ModelFitting.print_iteration_info(chain_id, i, num_iterations, log_acceptance_ratio, 
                                                acceptance_counts / total_proposals, current_parameters, 
                                                proposed_parameters, current_sigma)

        final_acceptance_rates = acceptance_counts / total_proposals
        return parameter_values, final_acceptance_rates, np.array(acceptance_rates_over_time)

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
        
        B = n * np.var(np.mean(chains, axis=1), axis=0)
        W = np.mean(np.var(chains, axis=1), axis=0)
        V = (n - 1) / n * W + (m + 1) / (m * n) * B
        
        R_hat = np.sqrt(V / W)
        return R_hat

    @staticmethod
    def execute_parallel_mcmc(data, time, num_chains, num_iterations, burn_in_period, transition_period, param_means, param_stds, is_fatal):
        config = Config()
        args = []
        for chain_id in range(num_chains):
            initial_params = []
            for i, param_name in enumerate(config.PARAM_NAMES):
                lower, upper = config.PARAM_BOUNDS[param_name]
                initial_param = np.random.uniform(lower, upper)
                initial_params.append(initial_param)
            args.append((data, time, initial_params, num_iterations, burn_in_period, transition_period, chain_id, param_means, param_stds))

        results = []
        with Pool(processes=num_chains) as pool:
            for result in tqdm(pool.imap_unordered(ModelFitting.perform_mcmc_iteration, args), total=num_chains, desc="MCMC Progress"):
                results.append(result)
        
        chains, acceptance_rates, acceptance_rates_over_time = zip(*results)
        chains = np.array(chains)
        acceptance_rates_over_time = np.mean(np.array(acceptance_rates_over_time), axis=0)
        
        r_hat = ModelFitting.calculate_rhat(chains[:, burn_in_period:, :])
        
        return chains, np.array(acceptance_rates), r_hat, acceptance_rates_over_time

    @staticmethod
    def calculate_correlations(chains):
        flattened_chains = chains.reshape(-1, chains.shape[-1])
        return np.corrcoef(flattened_chains.T)

    @staticmethod
    def calculate_ess(chain):
        n = len(chain)
        if n <= 1:
            return n

        acf_values = acf(chain, nlags=min(n - 1, 1000), fft=False)
        
        rho_hat_t = np.ones(len(acf_values))
        for t in range(1, len(acf_values)):
            rho_hat_t[t] = min(acf_values[t], rho_hat_t[t-1])
        
        tau = np.where(rho_hat_t < 0)[0]
        if len(tau) > 0:
            tau = tau[0]
        else:
            tau = len(rho_hat_t)
        
        ess = n / (1 + 2 * np.sum(rho_hat_t[1:tau]))
        
        return ess

    @staticmethod
    def calculate_multichain_ess(chains):
        n_chains, n_samples, n_params = chains.shape
        ess_values = np.zeros(n_params)
        for i in tqdm(range(n_params), desc="Calculating ESS"):
            combined_chain = chains[:, :, i].flatten()
            ess_values[i] = ModelFitting.calculate_ess(combined_chain)
        return ess_values