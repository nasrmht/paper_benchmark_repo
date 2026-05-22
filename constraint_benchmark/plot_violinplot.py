#!/usr/bin/env python3
"""
Visualization script to generate figures suitable for a poster.
This script generates, for each metric (Q2, RRMSE, interval_len), 
a plot containing 3 subplots (one for each output: f1, f2, f3).
Each subplot compares the violin plots of the 3 models (MOGP, Indep, LCM) 
based on the size of the training set N.
The figures are saved in SVG format with appropriate font and size.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import matplotlib.patches as mpatches

# Global configuration for poster
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 26,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

COLORS = {
    'mogp': '#2E86AB',      # Bleu profond
    'lcm': '#A23B72',       # Rose/Magenta
    'indep': '#F18F01'      # Orange
}

OUTPUT_NAMES = ['$f_1$ (Ishigami)', '$f_2$ (Branin)', '$f_3$ (Constraint)']
SCENARIOS = ['f1', 'f2', 'f3'] #
SCENARIO_LABELS_POSTER = ['deduced output $f_1$', 'deduced output $f_2$', 'deduced output $f_3$']
METRICS = ['q2', 'rrmse', 'interval_len', 'coverage_rate']
METRIC_LABELS = {
    'q2': 'Q²',
    'rrmse': 'RMSE', 
    'interval_len': 'Interval length',
    'coverage_rate': 'Coverage rate'
}


def load_and_aggregate_results(results_dir="results_c"):
    """Loads and aggregates benchmark results."""
    print(f"Loading results from {results_dir}...")
    pattern = os.path.join(results_dir, "benchmark_complet_seed=*.pkl")
    files = glob.glob(pattern)
    
    agg_file = os.path.join(results_dir, 'benchmark_complet_aggregated.pkl')
    if os.path.exists(agg_file):
        print(f"Found aggregated file: {agg_file}")
        with open(agg_file, 'rb') as f:
            return pickle.load(f)

    if not files:
        print("No pickle files found!")
        return {}
        
    print(f"{len(files)} individual seed files found.")
    
    with open(files[0], 'rb') as f:
        first_res = pickle.load(f)
    
    n_train_list = sorted(first_res.keys())
    scenarios = [
        'mogp_constrained',
        'lcm_deduced_f1', 'lcm_deduced_f2', 'lcm_deduced_f3',
        'indep_deduced_f1', 'indep_deduced_f2', 'indep_deduced_f3'
    ]
    
    final_results = {n: {s: {out: {m: [] for m in METRICS} for out in range(3)} 
                         for s in scenarios} for n in n_train_list}

    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                res = pickle.load(f)
            
            for n in n_train_list:
                if n in res:
                    for s in scenarios:
                        for out in range(3):
                            for met in METRICS:
                                if met in res[n][s][out]:
                                    val = res[n][s][out][met]
                                    final_results[n][s][out][met].extend(val)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return final_results


def plot_metric_for_poster(results, metric, poster_dir):
    """
    Generates 3 plots (one for each output f1, f2, f3) for a given metric.
    Each plot contains 3 subplots corresponding to the inference contexts 
    (deduction of f1, f2, f3).
    """
    n_list = sorted(results.keys())
    
    for output_idx, output_name in enumerate(OUTPUT_NAMES):
        # Prepare the figure: 1 row, 3 columns for deduction scenarios
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
       # fig.suptitle(f'Comparaison pour {metric.upper()} - Sortie: {output_name}', fontweight='bold', y=1.05)
        
        for scen_idx, ax in enumerate(axes):
            scenario = SCENARIOS[scen_idx]
            scenario_suffix = scenario.replace('deduced_', '') # f1, f2, f3
            
            scenarios_to_plot = [
                ('mogp_constrained', 'MOGP', COLORS['mogp']),
                (f'indep_deduced_{scenario_suffix}', f'Indep', COLORS['indep']),
                (f'lcm_deduced_{scenario_suffix}', f'LCM', COLORS['lcm']),
            ]
            
            all_positions = []
            all_data = []
            all_colors = []
            xtick_positions = []
            xtick_labels = []
            
            pos = 1
            for n in n_list:
                group_positions = []
                for scen_key, scen_label, color in scenarios_to_plot:
                    try:
                        data = results[n][scen_key][output_idx][metric]
                        if data:
                            all_positions.append(pos)
                            all_data.append(data)
                            all_colors.append(color)
                            group_positions.append(pos)
                    except Exception as e:
                        pass
                    pos += 1
                
                if group_positions:
                    xtick_positions.append(np.mean(group_positions))
                    xtick_labels.append(f'N={n}')
                
                pos += 1  # Space between groups
            
            if all_data:
                parts = ax.violinplot(all_data, positions=all_positions, widths=0.8,
                                     showmeans=True, showmedians=False, showextrema=False)
                
                for pc, color in zip(parts['bodies'], all_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.5)
                
                parts['cmeans'].set_color('black')
                parts['cmeans'].set_linewidth(2.0)
                
                for pos_val, data, color in zip(all_positions, all_data, all_colors):
                    ax.boxplot(data, positions=[pos_val], widths=0.2,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color='white', linewidth=2.5),
                               boxprops=dict(facecolor=color, alpha=0.9, edgecolor='black', linewidth=1.5),
                               capprops=dict(color='black', linewidth=1.5),
                               whiskerprops=dict(color='black', linewidth=1.5))
            
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, fontweight='bold')
            if metric == 'coverage_rate':
                ax.set_yticklabels([0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],fontsize=13, fontweight='bold')
            else:
                ax.set_yticklabels(ax.get_yticks(),fontsize=13, fontweight='bold')
            ax.set_title(f'{SCENARIO_LABELS_POSTER[scen_idx]}', fontweight='bold')
            
            if scen_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper())+f' {output_name}', fontweight='bold')
            
            if metric in ['rrmse', 'interval_len']:
                ax.set_yscale('log')
                
        # Legend
        handles = [mpatches.Patch(color=COLORS['mogp'], label='CMoGP'),
                   mpatches.Patch(color=COLORS['indep'], label='Indep. GP'),
                   mpatches.Patch(color=COLORS['lcm'], label='LCM')]
        
        fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=True, edgecolor='black')
        
        plt.tight_layout()
        output_name_clean = output_name.split()[0] # e.g. f1
        save_path = os.path.join(poster_dir, f'poster2_{metric}_{output_name_clean}.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {save_path}")


def main():
    # Determine results directory
    # Since we are in scripts/, the results are two levels up following the folder structure,
    # but let's handle both cases (running from scripts/ or from rapport_comparaison/).
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, '..', 'results', 'constraint_benchmark')

    results = load_and_aggregate_results(results_dir)

    if not results:
        print("No results found or loaded.")
        return

    # Create the folder for figures
    poster_dir = os.path.join(base_dir, 'figures')
    os.makedirs(poster_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING SVG FIGURES (PER OUTPUT) FOR THE POSTER")
    print("="*60)
    
    for metric in METRICS:
        print(f"\nGenerating plots for {metric}...")
        plot_metric_for_poster(results, metric, poster_dir)
        
    print("\n" + "="*60)
    print("All figures have been generated successfully!")
    print(f"Output directory: {poster_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
