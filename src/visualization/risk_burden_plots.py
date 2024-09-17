import matplotlib.pyplot as plt
import os
import numpy as np
from config.config import Config

class RiskBurdenPlots:

    
    @staticmethod
    def plot_risk_burden(risk_burden, case, output_dir, legend_fontsize=20, xlabel_fontsize=20, ylabel_fontsize=20, 
                        title_fontsize=26, xtick_fontsize=20, ytick_fontsize=20):
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

            for threshold in risk_burden.keys():
                x = sorted(risk_burden[threshold].keys())  # isolation periods
                y = [risk_burden[threshold][period][metric]['avg'] for period in x]
                ci_lower = [risk_burden[threshold][period][metric]['ci_lower'] for period in x]
                ci_upper = [risk_burden[threshold][period][metric]['ci_upper'] for period in x]

                plt.plot(x, y, marker='o', label=f'{threshold} log10 copies/mL')
                plt.fill_between(x, ci_lower, ci_upper, alpha=0.2)

            plt.xlabel('Isolation Period (days)', fontsize=xlabel_fontsize)
            plt.ylabel(title, fontsize=ylabel_fontsize)
            plt.title(f'{title} ({case})', fontsize=title_fontsize)
            plt.legend(title='Viral Load Threshold', fontsize=legend_fontsize)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.xticks(fontsize=xtick_fontsize)
            plt.yticks(fontsize=ytick_fontsize)

            if 'proportion' in metric:
                plt.ylim(0, 1)
            elif metric == 'risk_score':
                plt.yscale('linear')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_{case}.png'), dpi=600, bbox_inches='tight')
            plt.close()

        print(f"Risk and burden plots saved in {output_dir}")


    @staticmethod
    def plot_risk_burden_epsilon_tstar(results, no_treatment_results, case, output_dir, 
                                    default_fontsize=27, xlabel_fontsize=36, ylabel_fontsize=36, 
                                    title_fontsize=38, tick_fontsize=33, legend_fontsize=35):
        config = Config()
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Days',
            'Days',
            'Proportion',
            'Cumulative Risk Score'
        ]

        epsilon_values = config.EPSILON_VALUES
        t_star_values = config.T_STAR_VALUES

        threshold_colors = {3: 'blue', 4: 'orange', 5: 'green'}

        plt.rcParams.update({'font.size': default_fontsize})  # Set default font size

        for metric, title in zip(metrics, titles):
            fig, axes = plt.subplots(len(t_star_values), len(epsilon_values), figsize=(24, 32), squeeze=False)
            
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
                        
                        ax.plot(x, y, marker='o', color=color, label=f'{threshold} log10 copies/mL (Treatment)', linewidth=2)
                        ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

                        # Plot no-treatment curve
                        y_no_treatment = [no_treatment_results[threshold][period][metric]['avg'] for period in x]
                        ax.plot(x, y_no_treatment, linestyle=':', color=color, label=f'{threshold} log10 copies/mL (No Treatment)', linewidth=2)

                    ax.set_xlabel('Isolation Period (days)' if i == len(t_star_values) - 1 else '', fontsize=xlabel_fontsize)
                    ax.set_ylabel(title if j == 0 else '', fontsize=ylabel_fontsize)
                    ax.set_title(f't* = {t_star}, ε = {epsilon}', fontsize=title_fontsize, pad=20)
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

                    if 'proportion' in metric:
                        ax.set_ylim(0, 1)
                    elif metric == 'risk_score':
                        ax.set_yscale('linear')

                    # Remove the legend from individual subplots
                    ax.get_legend().remove() if ax.get_legend() else None

            # Add a common legend for all subplots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=legend_fontsize)

            plt.tight_layout()
            plt.subplots_adjust(top=0.94)
            output_file = os.path.join(output_dir, f'{metric}_{case}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Risk burden epsilon-tstar plots saved to {output_dir}")

    @staticmethod
    def plot_risk_burden_fixed_tstar(results, t_star_values, weights, no_treatment_results, case, output_dir,
                                    default_fontsize=27, xlabel_fontsize=36, ylabel_fontsize=36, 
                                    title_fontsize=38, tick_fontsize=33, legend_fontsize=35):
        config = Config()
        metrics = [
            'days_unnecessarily_isolated',
            'days_above_threshold_post_release',
            'proportion_above_threshold_at_release',
            'risk_score'
        ]
        titles = [
            'Burden (Days)',
            'Premature Isolation (Days)',
            'Proportion',
            'Cumulative Risk Score'
        ]

        epsilon_values = config.EPSILON_VALUES
        thresholds = config.VIRAL_LOAD_THRESHOLDS

        plt.rcParams.update({'font.size': default_fontsize})  # Set default font size

        fig, axes = plt.subplots(len(metrics) + 1, len(epsilon_values), figsize=(24, 38), squeeze=False)
        
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
                    ax.plot(x, y, marker='o', color=color, label=f'{threshold} log10 copies/mL (Treatment)', linewidth=2)
                    ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

                    # Plot no-treatment curve
                    y_no_treatment = [no_treatment_results[threshold][period][metric]['avg'] for period in x]
                    ax.plot(x, y_no_treatment, linestyle=':', color=color, label=f'{threshold} log10 copies/mL (No Treatment)', linewidth=2)
                
                ax.set_xlabel('Isolation Period (days)' if i == len(metrics) - 1 else '', fontsize=xlabel_fontsize)
                ax.set_ylabel(title if j == 0 else '', fontsize=ylabel_fontsize)
                ax.set_title(f'ε = {epsilon}' if i == 0 else '', fontsize=title_fontsize, pad=20)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                if 'proportion' in metric:
                    ax.set_ylim(0, 1)
                elif metric == 'risk_score':
                    ax.set_yscale('linear')

                # Remove the legend from individual subplots
                ax.get_legend().remove() if ax.get_legend() else None

        # Plot t_star distribution
        for j, epsilon in enumerate(epsilon_values):
            ax = axes[-1, j]
            ax.bar(t_star_values, weights, width=0.4, edgecolor='black')
            ax.set_xlabel('T* Value', fontsize=xlabel_fontsize)
            ax.set_ylabel('Weight' if j == 0 else '', fontsize=ylabel_fontsize)
            ax.set_title(f'T* Distribution (ε = {epsilon})', fontsize=title_fontsize, pad=20)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Add a common legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=legend_fontsize)

        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        output_file = os.path.join(output_dir, f'risk_burden_fixed_tstar_{case}.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close(fig)

        print(f"Risk burden fixed t-star plot saved to {output_file}")