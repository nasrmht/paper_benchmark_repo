"""
CFD diffuser benchmark — Theorem 3 dominance + RRMSE figure.

Analogous to ``plot_proposal_A.py`` (Lotka-Volterra), but adapted for the CFD
benchmark: single n_train=50, three output fields (τ₁₁, τ₂₂, k).

Figure produced
---------------
``make_all_fields_figure_nozoom`` — 1 row × 4 cols::

    [Thm. 3 dominance | RRMSE τ₁₁ | RRMSE τ₂₂ | RRMSE k]

All style properties (colours, font sizes, legend layout, background colour)
are identical to ``plot_proposal_A.py`` for visual consistency.

Usage
-----
    python plot_cfd_proposal_A.py
    python plot_cfd_proposal_A.py --prefix results_n=50_cfd --seeds 51 52 53
    python plot_cfd_proposal_A.py --prefix results_n=50_cfd --seeds 51 52 53 \\
        --variant_filter CI:0,2 FI:0,2 FM:0,2 --outdir figures_cfd/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.legend import Legend
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer

# Default results directory — relative to this script's location
_RESULTS_CFD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'results', 'cfd')
from benchmark_pca_gp.postprocessing.ms_analyse_compare_n import (
    compute_pca_metrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# CFD-specific constants
# ──────────────────────────────────────────────────────────────────────────────

FIELD_NAMES_LATEX = [r"$\tau_{11}$", r"$\tau_{22}$", r"$k$"]
Q       = 3
K_STARS = [0, 1, 2]   # fixed_idx values for CFD (3 fields → indices 0, 1, 2)
M_XLIM  = (0.5, 8.5)


# ──────────────────────────────────────────────────────────────────────────────
# Palette  (identical to plot_proposal_A.py for visual consistency)
# ──────────────────────────────────────────────────────────────────────────────

# RC (row-wise / PCA-CGP) — bold red, always in front
C_RC  = "#D62728"
M_RC  = "o"
LS_RC = "-"
LW_RC = 2.6
MS_RC = 6
A_RC  = 1.00
Z_RC  = 10

# CI (col-wise indep.) — blue shades, l=0→2 lighten
C_CI  = {0: "#6baed6", 1: "#7cb7da", 2: "#8dc0df", 3: "#9ecae4"}
M_CI  = "s"
LS_CI = "--"
LW_CI = 1.3
MS_CI = 3
A_CI  = 0.82
Z_CI  = 6

# FI (field-wise indep.) — green shades
C_FI  = {0: "#74c476", 1: "#84ca85", 2: "#94d195", 3: "#a4d8a5"}
M_FI  = "^"
LS_FI = ":"
LW_FI = 1.3
MS_FI = 3
A_FI  = 0.82
Z_FI  = 5

# FM (field-wise MOGP-LCM) — muted purple shades
C_FM  = {0: "#9E9AC8", 1: "#ABA6D2", 2: "#B8B3DB", 3: "#C5C0E4"}
M_FM  = "D"
LS_FM = "-."
LW_FM = 1.3
MS_FM = 3
A_FM  = 0.82
Z_FM  = 4

# N-label palette (for dominance panel; single curve for CFD)
_N_PALETTE = [
    dict(color="#1f77b4", ls="-",  marker="o"),
    dict(color="#D2691E", ls="--", marker="s"),
    dict(color="#2ca02c", ls="-.", marker="^"),
    dict(color="#9467bd", ls=":",  marker="D"),
]

# Error-bar geometry
EB_LW      = 0.8
EB_CAPSIZE = 3.0
EB_CAPTHICK= 0.8
JITTER     = 0.0

BG_COLOR   = "#f8fafc"


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers  (identical to plot_proposal_A.py)
# ──────────────────────────────────────────────────────────────────────────────

def _poster_fs(scale: float = 1.0) -> dict:
    return dict(title=10*scale, label=9*scale, tick=9*scale,
                legend=8*scale, annot=8*scale)


def _n_styles(n_labels: List[str]) -> Dict[str, dict]:
    return {lbl: _N_PALETTE[i % len(_N_PALETTE)]
            for i, lbl in enumerate(n_labels)}


def _get_variants(prefix: str, variant_filter: Optional[dict]) -> List[int]:
    """Return the fixed_idx list to show for a given prefix."""
    if prefix == "RC":
        return [-1]
    if variant_filter is None:
        return K_STARS
    return variant_filter.get(prefix, K_STARS)


def _style_ax(
    ax,
    fs: dict,
    xlabel: str  = r"Latent dimension $m$",
    ylabel: str  = None,
    title:  str  = None,
    log_y:  bool = False,
    xlim:   tuple = M_XLIM,
) -> None:
    """Shared axis cosmetics (same as plot_proposal_A.py)."""
    if title:
        ax.set_title(title, fontsize=fs["title"], fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs["label"], fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs["label"], fontweight="bold")
    if log_y:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(labelsize=fs["tick"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(BG_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def _collect_records(
    ana:           MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
) -> List[dict]:
    """Collect per-model records with cumulative GP-prediction metrics."""
    records = []
    for seed, single_ana in zip(ana._seeds, ana._analyzers):
        for name in single_ana.list_models():
            res   = single_ana.storage.load_model_result_light(name)
            cfg   = res.get("config", {})
            inter = res.get("intermediate", {})

            cum_q2    = inter.get("cumulative_q2_pred_test")
            cum_rrmse = inter.get("cumulative_rrmse_pred_test")
            if cum_q2 is None or cum_rrmse is None:
                continue

            prefix    = name.split("_")[0]
            fixed_idx = int(cfg.get("fixed_idx", -1))

            if model_types   is not None and prefix    not in model_types:
                continue
            if fixed_indices is not None and fixed_idx not in fixed_indices:
                continue

            records.append({
                "seed":      seed,
                "prefix":    prefix,
                "n_modes":   cfg.get("n_modes", np.nan),
                "fixed_idx": fixed_idx,
                "cum_q2":    np.array(cum_q2),
                "cum_rrmse": np.array(cum_rrmse),
            })
    return records


def extract_rrmse_stats(
    records:  List[dict],
    field_i:  int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (m_range, mean, std) of cumulative RRMSE for one field."""
    if not records:
        return None, None, None
    by_seed: dict = {}
    for r in records:
        if "cum_rrmse" in r and r["cum_rrmse"] is not None:
            by_seed.setdefault(r["seed"], []).append(r["cum_rrmse"][:, field_i])
    if not by_seed:
        return None, None, None
    seed_vals = np.array([np.mean(vs, axis=0) for vs in by_seed.values()])
    m_range   = np.arange(1, seed_vals.shape[1] + 1)
    return m_range, seed_vals.mean(axis=0), seed_vals.std(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_data(
    zarr_specs: List[Tuple[List[str], str]],
) -> Tuple[dict, dict, str]:
    """Load zarr stores and compute dominance metrics + RRMSE records.

    Parameters
    ----------
    zarr_specs : list of (zarr_path_list, label).
        For CFD typically a single entry, e.g.::

            zarr_specs = [
                ([f"results_n=50_cfd_seed{s}.zarr" for s in [51, 52, 53]], "N=50"),
            ]

    Returns
    -------
    agg_by_n : dict {label → aggregated PCA metrics (mean ± std over seeds)}
    data     : dict {(label, prefix, k) → records list}
    n_label  : the label string of the first entry
    """
    print("Loading zarr stores …")
    analyzers = [(MultiSeedAnalyzer(paths), lbl) for paths, lbl in zarr_specs]
    n_label   = analyzers[0][1]

    print("Computing PCA dominance metrics …")
    agg_by_n = compute_pca_metrics(analyzers, max_m=8)

    print("Collecting RRMSE records …")
    data: dict = {}
    for ana, lbl in analyzers:
        data[(lbl, "RC", -1)] = _collect_records(ana, ["RC"], [-1])
        for k in K_STARS:
            data[(lbl, "CI", k)] = _collect_records(ana, ["CI"], [k])
            data[(lbl, "FI", k)] = _collect_records(ana, ["FI"], [k])
            data[(lbl, "FM", k)] = _collect_records(ana, ["FM"], [k])

    return agg_by_n, data, n_label


# ──────────────────────────────────────────────────────────────────────────────
# Panel: Dominance  (identical logic to plot_proposal_A.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_dominance(
    ax,
    agg_by_n: dict,
    fs:       dict,
    n_labels: List[str],
) -> None:
    """(E^col − E^row)/T — one curve per N label.

    For CFD there is a single N label, so one curve is drawn.
    The curve is positive when row-wise PCA reconstruction dominates col-wise.
    """
    n_styles = _n_styles(n_labels)
    for n_label in n_labels:
        if n_label not in agg_by_n:
            continue
        agg = agg_by_n[n_label]
        st  = n_styles[n_label]
        m   = agg["m_range"]
        T   = agg["T_mean"]
        mu  = -agg["diff_exact_mean"] / T
        sd  = agg["diff_exact_std"]   / T
        ax.errorbar(
            m, mu, yerr=sd,
            color=st["color"], linestyle=st["ls"],
            linewidth=2.0, marker=st["marker"], markersize=5,
            elinewidth=EB_LW, capsize=EB_CAPSIZE, capthick=EB_CAPTHICK,
            label=n_label, zorder=5,
        )

    ax.axhline(0, color="#888", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_yscale("log")
    _style_ax(
        ax, fs,
        ylabel=r"Normalised $\Delta \mathcal{E}^{\mathrm{row,col}}_m$",
        title=(r"Thm. 3:  "
               r"$\left(\mathcal{E}^{\mathrm{row}}_m "
               r"- \mathcal{E}^{\mathrm{col}}_m\right)$"),
    )
    if len(n_labels) > 1:
        ax.legend(fontsize=fs["legend"], loc="lower left",
                  framealpha=0.92, edgecolor="#ccc")


# ──────────────────────────────────────────────────────────────────────────────
# Panel: RRMSE  (adapted from plot_proposal_A.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_rrmse(
    ax,
    data:           dict,
    n_label:        str,
    field_i:        int,
    fs:             dict,
    show_ylabel:    bool            = True,
    show_errorbars: bool            = True,
    variant_filter: Optional[dict] = None,
    title:          Optional[str]  = None,
) -> None:
    """Draw RC + CI + FI + FM RRMSE curves on ``ax``."""
    # Build scenario list: FM, FI, CI back→front, RC on top
    scenarios = []
    for k in _get_variants("FM", variant_filter):
        scenarios.append(("FM", k))
    for k in _get_variants("FI", variant_filter):
        scenarios.append(("FI", k))
    for k in _get_variants("CI", variant_filter):
        scenarios.append(("CI", k))
    scenarios.append(("RC", -1))

    n_scen  = len(scenarios)
    offsets = (
        (np.arange(n_scen) - (n_scen - 1) / 2.0)
        * (JITTER / max(n_scen - 1, 1))
        if show_errorbars else np.zeros(n_scen)
    )

    for idx, (prefix, k) in enumerate(scenarios):
        key = (n_label, prefix, k)
        if key not in data:
            continue
        m, mu, sd = extract_rrmse_stats(data[key], field_i)
        if m is None:
            continue

        x = m.astype(float) + offsets[idx]

        if prefix == "RC":
            color, ls, lw, ms, alpha, zo = C_RC, LS_RC, LW_RC, MS_RC, A_RC, Z_RC
            label  = "RC"
            marker = M_RC
        elif prefix == "CI":
            color, ls, lw, ms, alpha, zo = C_CI[k], LS_CI, LW_CI, MS_CI, A_CI, Z_CI
            label  = f"CI (l={k})"
            marker = M_CI
        elif prefix == "FI":
            color, ls, lw, ms, alpha, zo = C_FI[k], LS_FI, LW_FI, MS_FI, A_FI, Z_FI
            label  = f"FI (l={k})"
            marker = M_FI
        else:  # FM
            color, ls, lw, ms, alpha, zo = C_FM[k], LS_FM, LW_FM, MS_FM, A_FM, Z_FM
            label  = f"FM (l={k})"
            marker = M_FM

        if show_errorbars:
            ax.errorbar(
                x, mu, yerr=sd,
                color=color, alpha=alpha, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms,
                elinewidth=EB_LW, capsize=EB_CAPSIZE, capthick=EB_CAPTHICK,
                zorder=zo, label=label,
            )
        else:
            ax.plot(
                x, mu,
                color=color, alpha=alpha, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, zorder=zo, label=label,
            )

    ylabel = "RRMSE" if show_ylabel else None
    _style_ax(ax, fs, ylabel=ylabel, xlim=M_XLIM, log_y=True,
              title=title or FIELD_NAMES_LATEX[field_i])
    ax.set_xticks(range(1, 9))


# ──────────────────────────────────────────────────────────────────────────────
# Legend — 4-block layout:  RC  |  CI  |  FI  |  FM
# ──────────────────────────────────────────────────────────────────────────────

def _add_grouped_legend(
    fig,
    fs:             dict,
    variant_filter: Optional[dict] = None,
    y_anchor:       float          = -0.10,
) -> None:
    """Place four separate Legend artists (RC | CI | FI | FM) below the figure."""
    def _handle(prefix: str, k: int) -> mlines.Line2D:
        if prefix == "RC":
            color, ls, lw, ms, marker, label = \
                C_RC, LS_RC, LW_RC, MS_RC, M_RC, "RC"
        elif prefix == "CI":
            color, ls, lw, ms, marker, label = \
                C_CI[k], LS_CI, LW_CI, MS_CI, M_CI, f"CI (l={k})"
        elif prefix == "FI":
            color, ls, lw, ms, marker, label = \
                C_FI[k], LS_FI, LW_FI, MS_FI, M_FI, f"FI (l={k})"
        else:  # FM
            color, ls, lw, ms, marker, label = \
                C_FM[k], LS_FM, LW_FM, MS_FM, M_FM, f"FM (l={k})"
        return mlines.Line2D([], [], color=color, marker=marker,
                             markersize=ms, linewidth=lw,
                             linestyle=ls, label=label)

    ci_v = _get_variants("CI", variant_filter)
    fi_v = _get_variants("FI", variant_filter)
    fm_v = _get_variants("FM", variant_filter)

    groups = [
        ([_handle("RC", -1)],
         ["RC"],
         1),
        ([_handle("CI", k) for k in ci_v],
         [f"CI (l={k})" for k in ci_v],
         min(2, max(1, len(ci_v)))),
        ([_handle("FI", k) for k in fi_v],
         [f"FI (l={k})" for k in fi_v],
         min(2, max(1, len(fi_v)))),
        ([_handle("FM", k) for k in fm_v],
         [f"FM (l={k})" for k in fm_v],
         min(2, max(1, len(fm_v)))),
    ]
    active = [(hdl, lbl, nc) for hdl, lbl, nc in groups if hdl]
    if not active:
        return

    x_positions = np.linspace(0.10, 0.90, len(active))
    common = dict(
        frameon=True, framealpha=0.95, edgecolor="#ccc",
        handlelength=2.0, handletextpad=0.4, borderpad=0.5, labelspacing=0.35,
        prop={"size": fs["legend"], "weight": "bold"},
        bbox_transform=fig.transFigure,
    )
    for (handles, labels, ncol), x_pos in zip(active, x_positions):
        leg = Legend(fig, handles, labels,
                     loc="lower center",
                     bbox_to_anchor=(x_pos, y_anchor),
                     ncol=ncol, columnspacing=0.8, **common)
        fig.legends.append(leg)


# ──────────────────────────────────────────────────────────────────────────────
# Figure builder
# ──────────────────────────────────────────────────────────────────────────────

def make_all_fields_figure_nozoom(
    agg_by_n:       dict,
    data:           dict,
    n_label:        str,
    fs:             dict,
    save:           bool            = True,
    variant_filter: Optional[dict] = None,
    outdir:         str             = ".",
) -> plt.Figure:
    """Poster figure: dominance + full-range RRMSE for all 3 CFD fields.

    Layout (1 row × 4 cols)::

        [Thm. 3 dominance | RRMSE τ₁₁ | RRMSE τ₂₂ | RRMSE k]

    Parameters
    ----------
    agg_by_n       : output of ``load_all_data`` (dominance metrics per N label)
    data           : output of ``load_all_data`` (RRMSE records)
    n_label        : the N label string (e.g. "N=50")
    fs             : font-size dict from ``_poster_fs()``
    save           : if True, save PDF / PNG / SVG to ``outdir``
    variant_filter : dict {prefix: [k, ...]} to restrict shown variants, or None
    outdir         : output directory
    """
    ncols = 1 + Q   # 4 columns: dominance + 3 fields
    fig   = plt.figure(figsize=(5 * ncols, 4.5))
    fig.patch.set_facecolor("white")
    gs    = GridSpec(1, ncols, figure=fig,
                     width_ratios=[1.05] + [1] * Q,
                     wspace=0.30)

    # ── Dominance panel ───────────────────────────────────────────────────────
    ax_dom = fig.add_subplot(gs[0, 0])
    plot_dominance(ax_dom, agg_by_n, fs, [n_label])

    # ── RRMSE panels — one per field ──────────────────────────────────────────
    for j in range(Q):
        ax = fig.add_subplot(gs[0, j + 1])
        plot_rrmse(
            ax, data, n_label, j, fs,
            show_ylabel=(j == 0),
            variant_filter=variant_filter,
            title=FIELD_NAMES_LATEX[j],
        )

    _add_grouped_legend(fig, fs, variant_filter, y_anchor=-0.12)
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])

    if save:
        os.makedirs(outdir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            fname = os.path.join(outdir, f"poster_A_cfd_all_fields_nozoom.{ext}")
            dpi   = 300 if ext == "pdf" else 200
            fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✓ {fname}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="CFD benchmark: Theorem 3 dominance + RRMSE figure")
    p.add_argument("--prefix", default="results_n=20_cfd",
                   help="Zarr file prefix (default: results_n=50_cfd)")
    p.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22,23,24,25,26,27,28], #51, 52, 53
                   help="Seed list (default: 51 52 53)")
    p.add_argument("--n_label", default="N=50",
                   help="Label for the training set size (default: N=50)")
    p.add_argument("--variant_filter", nargs="+", default=None,
                   metavar="PREFIX:k1,k2",
                   help="Variants to show per prefix, e.g. CI:0,2 FI:0,2 FM:0,2")
    p.add_argument("--outdir", default="results_cfd",
                   help="Output directory for figures (default: results_cfd)")
    p.add_argument("--no_show", action="store_true",
                   help="Do not call plt.show()")
    return p.parse_args()


def _parse_variant_filter(args_vf: Optional[List[str]]) -> Optional[dict]:
    """Parse CLI variant filter strings 'CI:0,2' into dict {'CI': [0, 2]}."""
    if args_vf is None:
        return None
    result = {}
    for item in args_vf:
        prefix, vals_str = item.split(":")
        result[prefix] = [int(v) for v in vals_str.split(",")]
    return result


if __name__ == "__main__":
    args = _parse_args()

    zarr_files = [os.path.join(_RESULTS_CFD, f"{args.prefix}_seed{s}.zarr")
                  for s in args.seeds]
    print(f"Zarr files: {zarr_files}")

    zarr_specs     = [(zarr_files, args.n_label)]
    agg_by_n, data, n_label = load_all_data(zarr_specs)

    fs             = _poster_fs(scale=1.0)
    variant_filter = _parse_variant_filter(args.variant_filter)

    print(f"\nBuilding figure for {n_label} …")
    fig = make_all_fields_figure_nozoom(
        agg_by_n, data, n_label, fs,
        save=True,
        variant_filter=variant_filter,
        outdir=args.outdir,
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
