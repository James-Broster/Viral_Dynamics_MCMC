import matplotlib.pyplot as plt
import os
import numpy as np
from config.config import Config

class RiskBurdenPlots:
    
    @staticmethod
    def plot_risk_burden(risk_burden, case, output_dir):
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days Unnecessarily Isolated',
            'Days Above Threshold Post-Release',
            'Proportion Above Threshold at Release',
            'Cumulative Risk Score'
        ]

        os.makedirs(output_dir, exist_ok=True)

        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(12, 8))
            plt.rcParams.update({'font.size': 14})  # Increase default font size

            for threshold in risk_burden.keys():
                x = sorted(risk_burden[threshold].keys())  # isolation periods
                y = [risk_burden[threshold][period][metric]['avg'] for period in x]
                ci_lower = [risk_burden[threshold][period][metric]['ci_lower'] for period in x]
                ci_upper = [risk_burden[threshold][period][metric]['ci_upper'] for period in x]

                plt.plot(x, y, marker='o', label=f'{threshold} log10 copies/mL')
                plt.fill_between(x, ci_lower, ci_upper, alpha=0.2)

            plt.xlabel('Isolation Period (days)', fontsize=16)
            plt.ylabel(title, fontsize=16)
            plt.title(f'{title} ({case})', fontsize=18)
            plt.legend(title='Viral Load Threshold', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.xlim(0, max(x))  # Ensure x-axis starts at 0
            if 'proportion' in metric:
                plt.ylim(0, 1)
            elif metric == 'risk_score':
                plt.yscale('linear')
                plt.ylim(bottom=0)  # Ensure y-axis starts at 0 for risk score
            else:
                plt.ylim(bottom=0)  # Ensure y-axis starts at 0 for other metrics

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_{case}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Risk and burden plots saved in {output_dir}")

        
    @staticmethod
    def plot_risk_burden_sampled_tstar(results, t_star_samples, no_treatment_results, case, output_dir):
        config = Config()
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days Unnecessarily Isolated',
            'Days Above Threshold Post-Release',
            'Proportion Above Threshold at Release',
            'Cumulative Risk Score'
        ]

        epsilon_values = config.EPSILON_VALUES
        thresholds = config.VIRAL_LOAD_THRESHOLDS

        fig, axes = plt.subplots(len(metrics) + 1, len(epsilon_values), figsize=(20, 25), squeeze=False)
        plt.rcParams.update({'font.size': 18})  # Increase default font size

        threshold_colors = {3: 'blue', 4: 'orange', 5: 'green'}

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            for j, epsilon in enumerate(epsilon_values):
                ax = axes[i, j]
                
                for threshold in thresholds:
                    color = threshold_colors.get(threshold, 'gray')
                    x = sorted(results[epsilon][threshold].keys())  # isolation periods
                    
                    # Plot treatment curve with confidence intervals
                    y = [results[epsilon][threshold][period][metric]['avg'] for period in x]
                    ci_lower = [results[epsilon][threshold][period][metric]['ci_lower'] for period in x]
                    ci_upper = [results[epsilon][threshold][period][metric]['ci_upper'] for period in x]
                    ax.plot(x, y, marker='o', color=color, label=f'{threshold} copies/mL (T)', markersize=4)
                    ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

                    # Plot no-treatment curve
                    y_no_treatment = [no_treatment_results[threshold][period][metric]['avg'] for period in x]
                    ax.plot(x, y_no_treatment, linestyle=':', color=color, label=f'{threshold} copies/mL (NT)')
                
                ax.set_xlabel('Isolation Period (days)' if i == len(metrics) - 1 else '', fontsize=22)
                ax.set_ylabel(title if j == 0 else '', fontsize=22)
                ax.set_title(f'ε = {epsilon}' if i == 0 else '', fontsize=24)
                
                if i == 0 and j == len(epsilon_values) - 1:
                    ax.legend(title='Viral Load Threshold', fontsize=16, title_fontsize=18, 
                              loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
                
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.tick_params(axis='both', which='major', labelsize=20)  # Increased tick label size
                
                ax.set_xlim(0, max(x))
                if 'proportion' in metric:
                    ax.set_ylim(0, 1)
                elif metric == 'risk_score':
                    ax.set_yscale('log')
                    ax.set_ylim(bottom=1)
                else:
                    ax.set_ylim(bottom=0)

        # Plot t_star samples
        for j, epsilon in enumerate(epsilon_values):
            ax = axes[-1, j]
            ax.hist(t_star_samples[epsilon], bins=20, edgecolor='black')
            ax.set_xlabel('T* Value', fontsize=22)
            ax.set_ylabel('Frequency' if j == 0 else '', fontsize=22)
            ax.set_title(f'T* Distribution (ε = {epsilon})', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'risk_burden_sampled_tstar_{case}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_risk_burden_epsilon_tstar(results, no_treatment_results, case, output_dir):
        config = Config()
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days Unnecessarily Isolated',
            'Days Above Threshold Post-Release',
            'Proportion Above Threshold at Release',
            'Cumulative Risk Score'
        ]

        epsilon_values = config.EPSILON_VALUES
        t_star_values = config.T_STAR_VALUES

        threshold_colors = {3: 'blue', 4: 'orange', 5: 'green'}

        for metric, title in zip(metrics, titles):
            fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(20, 25), squeeze=False)
            plt.rcParams.update({'font.size': 18})  # Increase default font size
            
            for i, t_star in enumerate(t_star_values):
                for j, epsilon in enumerate(epsilon_values):
                    ax = axes[i, j]
                    risk_burden = results[(epsilon, t_star)]

                    for threshold in risk_burden.keys():
                        color = threshold_colors.get(threshold, 'gray')
                        
                        x = sorted(risk_burden[threshold].keys())  # isolation periods
                        y = [risk_burden[threshold][period][metric]['avg'] for period in x]
                        ci_lower = [risk_burden[threshold][period][metric]['ci_lower'] for period in x]
                        ci_upper = [risk_burden[threshold][period][metric]['ci_upper'] for period in x]
                        
                        ax.plot(x, y, marker='o', color=color, label=f'{threshold} copies/mL (T)', markersize=4)
                        ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

                        # Plot no-treatment curve
                        y_no_treatment = [no_treatment_results[threshold][period][metric]['avg'] for period in x]
                        ax.plot(x, y_no_treatment, linestyle=':', color=color, label=f'{threshold} copies/mL (NT)')

                    ax.set_xlabel('Isolation Period (days)' if i == len(t_star_values) - 1 else '', fontsize=22)
                    ax.set_ylabel(title if j == 0 else '', fontsize=22)
                    ax.set_title(f't* = {t_star}, ε = {epsilon}', fontsize=24)
                    
                    if i == 0 and j == len(epsilon_values) - 1:
                        ax.legend(title='Viral Load Threshold', fontsize=16, title_fontsize=18, 
                                  loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
                    
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax.tick_params(axis='both', which='major', labelsize=20)  # Increased tick label size

                    ax.set_xlim(0, max(x))
                    if 'proportion' in metric:
                        ax.set_ylim(0, 1)
                    elif metric == 'risk_score':
                        ax.set_yscale('log')
                        ax.set_ylim(bottom=1)
                    else:
                        ax.set_ylim(bottom=0)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'{metric}_{case}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
