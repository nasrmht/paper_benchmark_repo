import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

import sys
from pathlib import Path

# Add parent dir to sys.path to allow imports if ran directly
sys.path.insert(0, '..')

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer
from benchmark_pca_gp.postprocessing.ms_analyse_compare_n import (
    compute_pca_metrics, 
    _collect_records, 
    _pca_n_style, 
    _PREFIX_STYLE, 
    _font_sizes, 
    _apply_bold_ticks
)

def _band(ax, x, mean, std, color, label, **kw):
    line, = ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    return line

def extract_rrmse_stats(records, field_i):
    if not records:
        return None, None, None
    by_seed = {}
    for r in records:
        key = "cum_rrmse"
        if key in r and r[key] is not None:
            by_seed.setdefault(r["seed"], []).append(r[key][:, field_i])
    if not by_seed:
        return None, None, None
    
    # average within same seed if multiple (should be 1)
    seed_vals = np.array([np.mean(vs, axis=0) for vs in by_seed.values()])
    m_range = np.arange(1, seed_vals.shape[1] + 1)
    return m_range, seed_vals.mean(axis=0), seed_vals.std(axis=0)

def main():
    print("Loading data...")
    list_n30 = [f"results_lotka/results_lv10_seed{i}.zarr" for i in range(10)]
    list_n10 = [f"results_lotka/results_lv11_seed{i}.zarr" for i in range(10)]

    ana_n30 = MultiSeedAnalyzer(list_n30)
    ana_n10 = MultiSeedAnalyzer(list_n10)

    analyzers = [(ana_n30, "N=30"), (ana_n10, "N=10")]

    print("Computing PCA metrics...")
    agg_by_n = compute_pca_metrics(analyzers, max_m=10)

    print("Collecting RRMSE records...")
    # Get records for RC (p=-1) and CI (p=0) for both N
    rec_rc_30 = _collect_records(ana_n30, "N=30", ["RC"], [-1])
    rec_ci_30 = _collect_records(ana_n30, "N=30", ["CI"], [0])
    
    rec_rc_10 = _collect_records(ana_n10, "N=10", ["RC"], [-1])
    rec_ci_10 = _collect_records(ana_n10, "N=10", ["CI"], [0])

    Q = 4
    output_names = ['p', 'q', 'r', 's']
    fs = _font_sizes(1.0)
    
    # Setup Figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, Q, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Dominance (Full and Zoomed)
    ax_dom_full = fig.add_subplot(gs[0, 0:2])
    ax_dom_zoom = fig.add_subplot(gs[0, 2:4])
    
    C_n30 = "#1f77b4"
    C_n10 = "#d62728"
    colors = {"N=30": C_n30, "N=10": C_n10}
    
    for n_label, agg in agg_by_n.items():
        st = _pca_n_style(n_label)
        m_range = agg["m_range"]
        T = agg["T_mean"]
        
        diff_exact_n = agg["diff_exact_mean"] / T
        std_exact_n  = agg["diff_exact_std"] / T
        
        color = colors.get(n_label, "#2c2c2c")
        
        # Plot full
        _band(ax_dom_full, m_range, diff_exact_n, std_exact_n, color=color, 
              label=f"Dominance {n_label}", 
              linewidth=st["lw"]*1.2, linestyle=st["ls"], marker=st["marker"])
        
        # Plot zoomed
        _band(ax_dom_zoom, m_range, diff_exact_n, std_exact_n, color=color, 
              label=f"Dominance {n_label}", 
              linewidth=st["lw"]*1.5, linestyle=st["ls"], marker=st["marker"], markersize=6)

    for ax in [ax_dom_full, ax_dom_zoom]:
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("modes m", fontsize=fs["label"], fontweight="bold")
        _apply_bold_ticks(ax, fs)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=fs["legend"])
        
    ax_dom_full.set_title("Theorem 3 Dominance: (E$^{col}_m$ - E$^{row}_m$) / T", fontsize=fs["title"], fontweight="bold")
    ax_dom_full.set_ylabel("Dominance", fontsize=fs["label"], fontweight="bold")
    
    ax_dom_zoom.set_xlim(4.0, 10.0)
    ax_dom_zoom.set_title("Zoom on [m=4, 10]", fontsize=fs["title"], fontweight="bold")
    
    # Calculate tighter ylim for zoom based on m=4 to 10
    zoom_min, zoom_max = [], []
    for agg in agg_by_n.values():
        val = (agg["diff_exact_mean"] / agg["T_mean"])[3:10]
        std = (agg["diff_exact_std"] / agg["T_mean"])[3:10]
        zoom_min.append(np.min(val - std))
        zoom_max.append(np.max(val + std))
    
    y_zoom_abs = max(abs(np.min(zoom_min)), abs(np.max(zoom_max))) * 1.2
    ax_dom_zoom.set_ylim(-y_zoom_abs, y_zoom_abs)
    
    # Shading regions for dominance
    ax_dom_full.fill_between(agg_by_n["N=30"]["m_range"], 0, ax_dom_full.get_ylim()[1], color="#2166ac", alpha=0.04)
    ax_dom_full.fill_between(agg_by_n["N=30"]["m_range"], ax_dom_full.get_ylim()[0], 0, color="#d73027", alpha=0.04)
    
    ax_dom_zoom.fill_between(np.linspace(4, 10, 50), 0, y_zoom_abs, color="#2166ac", alpha=0.04)
    ax_dom_zoom.fill_between(np.linspace(4, 10, 50), -y_zoom_abs, 0, color="#d73027", alpha=0.04)

    # Row 2: RRMSE
    axes_rrmse = [fig.add_subplot(gs[1, i]) for i in range(Q)]
    
    for i in range(Q):
        ax = axes_rrmse[i]
        
        # N=30
        m_r30, mean_r30, std_r30 = extract_rrmse_stats(rec_rc_30, i)
        m_c30, mean_c30, std_c30 = extract_rrmse_stats(rec_ci_30, i)
        
        # N=10
        m_r10, mean_r10, std_r10 = extract_rrmse_stats(rec_rc_10, i)
        m_c10, mean_c10, std_c10 = extract_rrmse_stats(rec_ci_10, i)
        
        st_rc = _PREFIX_STYLE.get("RC", ("", "", 0, 0, "", False))
        st_ci = _PREFIX_STYLE.get("CI", ("", "", 0, 0, "", False))
        
        # N=30
        if m_r30 is not None:
            _band(ax, m_r30, mean_r30, std_r30, color=st_rc[0], label="RC (N=30)", 
                  linestyle="-", marker=st_rc[4], alpha=1.0)
        if m_c30 is not None:
            _band(ax, m_c30, mean_c30, std_c30, color=st_ci[0], label="CI (N=30)", 
                  linestyle="-", marker=st_ci[4], alpha=1.0)
                  
        # N=10
        if m_r10 is not None:
            _band(ax, m_r10, mean_r10, std_r10, color=st_rc[0], label="RC (N=10)", 
                  linestyle="--", marker=st_rc[4], alpha=0.7)
        if m_c10 is not None:
            _band(ax, m_c10, mean_c10, std_c10, color=st_ci[0], label="CI (N=10)", 
                  linestyle="--", marker=st_ci[4], alpha=0.7)
                  
        ax.set_title(f"Field: {output_names[i]}", fontsize=fs["title"], fontweight="bold")
        ax.set_xlabel("modes m", fontsize=fs["label"], fontweight="bold")
        if i == 0:
            ax.set_ylabel("RRMSE", fontsize=fs["label"], fontweight="bold")
        
        # Limit the view to focus on the comparison
        ax.set_xlim(3.5, 10.5)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle=":")
        _apply_bold_ticks(ax, fs)
        
        if i == Q - 1:
            ax.legend(fontsize=fs["legend"] - 1, loc="upper right")

    plt.tight_layout()
    out_path = "dominance_vs_rrmse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
