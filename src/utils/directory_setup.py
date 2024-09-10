import os

def setup_directories(base_output_dir):

    directories = {
        'base': base_output_dir,
        'fatal': os.path.join(base_output_dir, 'fatal'),
        'non_fatal': os.path.join(base_output_dir, 'non_fatal'),
    }

    for case in ['fatal', 'non_fatal']:
        directories.update({
            f'{case}_model_predictions': os.path.join(directories[case], 'model_predictions'),
            f'{case}_mcmc_diagnostics': os.path.join(directories[case], 'mcmc_diagnostics'),
            f'{case}_mcmc_diagnostics_trace': os.path.join(directories[case], 'mcmc_diagnostics', 'trace_plots'),
            f'{case}_mcmc_diagnostics_histogram': os.path.join(directories[case], 'mcmc_diagnostics', 'histograms'),
            f'{case}_mcmc_diagnostics_rhat': os.path.join(directories[case], 'mcmc_diagnostics', 'rhat'),
            f'{case}_treatment_effects': os.path.join(directories[case], 'treatment_effects'),
            f'{case}_csv_outputs': os.path.join(directories[case], 'csv_outputs'),  # New CSV output directory
        })

    # Create all directories
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)

    return directories