"""
Proposal A (v5) — Poster / Paper figure generator

Layouts produced (one file per output field p, q, r, s):
  combined  poster_AA_field_{f}.{ext}       — dominance + full-range + zoom
  nozoom    poster_A_field_{f}_nozoom.{ext} — dominance + full-range only
  zoom_only paper_A_field_{f}_zoom.{ext}    — zoom panels only (paper use)
  decomp    paper_decomp_field_{f}.{ext}    — stacked bar: PCA recon vs PCA+GP

Features
--------
  - RC, CI, FI, FM all supported (FM in muted purple matching ms_analyse_compare_n.py)
  - Any number of n_train sizes: generalises columns automatically
  - variant_filter : dict mapping prefix to list of fixed_idx to display
      e.g. {"CI": [0, 2], "FI": [0, 2], "FM": [0, 2]}
      → only two variants per deductive method instead of four
  - Decomposition figure: for each chosen m, shows PCA reconstruction RRMSE
      (solid bar) + GP overhead (red-hatched extension). Best and worst
      variant per deductive method auto-detected or overrideable.
  - load_all_data() accepts a zarr_specs list for arbitrary n_train sets

Usage
-----
    python plot_proposal_A.py                       # default N=30, N=10
    # or edit zarr_specs / variant_filter in main() below
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.legend import Legend
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer

# Default results directory — relative to this script's location
_RESULTS_LV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'results', 'lotka_volterra')
from benchmark_pca_gp.postprocessing.ms_analyse_compare_n import (
    compute_pca_metrics,
    _collect_records,
    _pca_n_style,
)


# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────

# RC (row-wise / PCA-CGP) — bold red, always in front
C_RC  = "#D62728"
M_RC  = "o"
LS_RC = "-"
LW_RC = 2.6
MS_RC = 6
A_RC  = 1.00
Z_RC  = 10

# CI (col-wise indep.) — blue shades, l=0→3 lighten
C_CI  = {0: "#ff7f00", 1: "#ff9933", 2: "#ffb266", 3: "#ffcc99"} # {0: "#6baed6", 1: "#7cb7da", 2: "#8dc0df", 3: "#9ecae4"}
M_CI  = "s"
LS_CI = "--"
LW_CI = 1.3
MS_CI = 3
A_CI  = 0.82
Z_CI  = 6

# FI (field-wise indep.) — green shades, l=0→3 lighten
C_FI  = {0: "#417C03", 1: "#519204", 2: "#61A805", 3: "#71BE06"} # {0: "#74c476", 1: "#84ca85", 2: "#94d195", 3: "#a4d8a5"}
M_FI  = "^"
LS_FI = ":"
LW_FI = 1.3
MS_FI = 3
A_FI  = 0.82
Z_FI  = 5

# FM (field-wise MOGP-LCM) — muted purple shades (as in ms_analyse_compare_n)
C_FM  = {0: "#5e239d", 1: "#6f29b0", 2: "#802fbf", 3: "#9135ce"} # {0: "#9E9AC8", 1: "#ABA6D2", 2: "#B8B3DB", 3: "#C5C0E4"}
M_FM  = "D"
LS_FM = "-."
LW_FM = 1.3
MS_FM = 3
A_FM  = 0.82
Z_FM  = 4

# N-label palette — cycled for arbitrary number of n_train values
_N_PALETTE = [
    dict(color="#1f77b4", ls="-",         marker="o"),   # N=30 (default first)
    dict(color="#D2691E", ls="--",        marker="s"),   # N=10
    dict(color="#2ca02c", ls="-.",        marker="^"),   # N=15 (or any third)
    dict(color="#9467bd", ls=":",         marker="D"),
    dict(color="#8c564b", ls=(0, (5, 1)), marker="v"),
]

# Error-bar geometry
EB_LW       = 0.8
EB_CAPSIZE  = 3.0
EB_CAPTHICK = 0.8
JITTER      = 0.20

BG_COLOR     = "#f8fafc"
Q            = 4
OUTPUT_NAMES = ["p", "q", "r", "s"]
K_STARS      = [0, 1, 2, 3]
M_XLIM       = (0.5, 10.5)


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _n_styles(n_labels: List[str]) -> Dict[str, dict]:
    """Map each n_label to a colour / linestyle / marker dict."""
    return {lbl: _N_PALETTE[i % len(_N_PALETTE)]
            for i, lbl in enumerate(n_labels)}


def _get_variants(prefix: str, variant_filter: Optional[dict]) -> List[int]:
    """Return the fixed_idx list to show for a given prefix."""
    if prefix == "RC":
        return [-1]
    if variant_filter is None:
        return K_STARS
    return variant_filter.get(prefix, K_STARS)


def extract_rrmse_stats(
    records: List[dict], field_i: int
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


def _poster_fs(scale: float = 1.0) -> dict:
    return dict(title=10*scale, label=9*scale, tick=9*scale,
                legend=8*scale, annot=8*scale)


def _style_ax(ax, fs: dict, xlabel: str = r"Latent dimension $m$",
              ylabel: str = None, title: str = None,
              log_y: bool = False, xlim: tuple = M_XLIM) -> None:
    """Shared axis cosmetics."""
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
    for lbl in ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'):
        lbl.set_fontweight("bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(BG_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_data(
    zarr_specs: Optional[List[Tuple[List[str], str]]] = None,
) -> Tuple[dict, dict, List[str]]:
    """Load zarr stores and compute all metrics.

    Parameters
    ----------
    zarr_specs : list of (zarr_path_list, label).
        If None, defaults to the two-size setup N=30 / N=10.
        Example for three sizes::
            zarr_specs = [
                ([f"results_N=30_lv_seed{i}.zarr" for i in range(10)], "N=30"),
                ([f"results_N=15_lv_seed{i}.zarr" for i in range(10)], "N=15"),
                ([f"results_N=10_lv_seed{i}.zarr" for i in range(10)], "N=10"),
            ]

    Returns
    -------
    agg_by_n : dict[label → aggregated PCA metrics]
    data     : dict[(n_label, prefix, k) → records]
    n_labels : list of labels in the same order as zarr_specs
    """
    if zarr_specs is None:
        zarr_specs = [
            ([os.path.join(_RESULTS_LV, f"results_N_=30_lv_seed{i}.zarr") for i in range(1, 10)], "N=30"),
            ([os.path.join(_RESULTS_LV, f"results_N_=10_lv_seed{i}.zarr") for i in range(1, 10)], "N=10"),
        ]

    print("Loading zarr stores …")
    analyzers = [(MultiSeedAnalyzer(paths), label) for paths, label in zarr_specs]
    n_labels  = [label for _, label in zarr_specs]

    print("Computing PCA dominance metrics …")
    agg_by_n = compute_pca_metrics(analyzers, max_m=10)

    print("Collecting RRMSE records …")
    data: dict = {}
    for ana, n_label in analyzers:
        data[(n_label, "RC", -1)] = _collect_records(ana, n_label, ["RC"], [-1])
        for k in K_STARS:
            data[(n_label, "CI", k)] = _collect_records(ana, n_label, ["CI"], [k])
            data[(n_label, "FI", k)] = _collect_records(ana, n_label, ["FI"], [k])
            data[(n_label, "FM", k)] = _collect_records(ana, n_label, ["FM"], [k])

    return agg_by_n, data, n_labels


# ──────────────────────────────────────────────────────────────────────────────
# Panel: Dominance
# ──────────────────────────────────────────────────────────────────────────────

def plot_dominance(ax, agg_by_n: dict, fs: dict, n_labels: List[str]) -> None:
    """(E^col − E^row)/T, one curve per N label."""
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
    ax.legend(fontsize=fs["legend"], loc="lower left",
              framealpha=0.92, edgecolor="#ccc")


# ──────────────────────────────────────────────────────────────────────────────
# Panel: RRMSE
# ──────────────────────────────────────────────────────────────────────────────

def plot_rrmse(
    ax,
    data: dict,
    n_label: str,
    field_i: int,
    fs: dict,
    show_ylabel: bool = True,
    zoomed: bool = False,
    show_errorbars: bool = True,
    variant_filter: Optional[dict] = None,
    title: Optional[str] = None,
) -> None:
    """Draw RC + CI + FI + FM RRMSE curves on `ax`.

    Parameters
    ----------
    variant_filter : dict {prefix: [fixed_idx, ...]}, or None for all.
        Controls which variants of each deductive method are shown.
        e.g. {"CI": [0, 2], "FI": [0, 2], "FM": [0, 2]}
    title : optional panel title (overrides the default N-label title).
    """
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

        if zoomed:
            mask = (m >= 5) & (m <= 10)
            m, mu, sd = m[mask], mu[mask], sd[mask]
            if len(m) == 0:
                continue

        x = m.astype(float) + offsets[idx]

        if prefix == "RC":
            color, ls, lw, ms, alpha, zo = C_RC, LS_RC, LW_RC, MS_RC, A_RC, Z_RC
            label  = "Row-CMO"
            marker = M_RC
        elif prefix == "CI":
            color, ls, lw, ms, alpha, zo = C_CI[k], LS_CI, LW_CI, MS_CI, A_CI, Z_CI
            label  = f"Col-Indep (l={k})"
            marker = M_CI
        elif prefix == "FI":
            color, ls, lw, ms, alpha, zo = C_FI[k], LS_FI, LW_FI, MS_FI, A_FI, Z_FI
            label  = f"Fw-Indep (l={k})"
            marker = M_FI
        else:  # FM
            color, ls, lw, ms, alpha, zo = C_FM[k], LS_FM, LW_FM, MS_FM, A_FM, Z_FM
            label  = f"Fw-LCM (l={k})"
            marker = M_FM

        # Boost visibility in zoom view
        if zoomed and prefix in ["CI", "FI", "FM"]:
            lw *= 1.8
            ms *= 1.5

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
    if zoomed:
        _style_ax(ax, fs, ylabel=ylabel, xlim=(4.5, 10.5), log_y=True,
                  title=title)
        ax.set_xticks(range(5, 11))
    else:
        _style_ax(ax, fs, ylabel=ylabel, xlim=M_XLIM, log_y=True,
                  title=title or f"{n_label}")
        ax.set_xticks(range(1, 11))


# ──────────────────────────────────────────────────────────────────────────────
# Legend — 4-block layout:  RC  |  CI  |  FI  |  FM
# ──────────────────────────────────────────────────────────────────────────────

def _add_grouped_legend(
    fig,
    fs: dict,
    variant_filter: Optional[dict] = None,
    y_anchor: float = -0.10,
) -> None:
    """Place four separate Legend artists (RC | CI | FI | FM) below the figure.

    Empty groups (all variants filtered out) are silently skipped.
    X positions are distributed evenly across the active groups.
    """
    def _handle(prefix: str, k: int) -> mlines.Line2D:
        if prefix == "RC":
            color, ls, lw, ms, marker, label = \
                C_RC, LS_RC, LW_RC, MS_RC, M_RC, "Row-CMO"
        elif prefix == "CI":
            color, ls, lw, ms, marker, label = \
                C_CI[k], LS_CI, LW_CI, MS_CI, M_CI, f"Col-Indep (l={k})"
        elif prefix == "FI":
            color, ls, lw, ms, marker, label = \
                C_FI[k], LS_FI, LW_FI, MS_FI, M_FI, f"Fw-Indep (l={k})"
        else:  # FM
            color, ls, lw, ms, marker, label = \
                C_FM[k], LS_FM, LW_FM, MS_FM, M_FM, f"Fw-LCM (l={k})"
        return mlines.Line2D([], [], color=color, marker=marker,
                             markersize=ms, linewidth=lw,
                             linestyle=ls, label=label)

    ci_v = _get_variants("CI", variant_filter)
    fi_v = _get_variants("FI", variant_filter)
    fm_v = _get_variants("FM", variant_filter)

    groups = [
        ([_handle("RC", -1)],           ["Row-CMO"],                              1),
        ([_handle("CI", k) for k in ci_v], [f"Col-Indep (l={k})" for k in ci_v],
         min(2, max(1, len(ci_v)))),
        ([_handle("FI", k) for k in fi_v], [f"Fw-Indep (l={k})" for k in fi_v],
         min(2, max(1, len(fi_v)))),
        ([_handle("FM", k) for k in fm_v], [f"Fw-LCM (l={k})" for k in fm_v],
         min(2, max(1, len(fm_v)))),
    ]
    active = [(hdl, lbl, nc) for hdl, lbl, nc in groups if hdl]
    if not active:
        return

    if len(active) == 4:
        x_positions = [0.13, 0.31, 0.56, 0.81]
    elif len(active) == 3:
        x_positions = [0.32, 0.50, 0.68]
    elif len(active) == 2:
        x_positions = [0.40, 0.60]
    else:
        x_positions = [0.50]

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
# Figure builders
# ──────────────────────────────────────────────────────────────────────────────

def _add_zoom_connectors(ax_full, ax_zoom, x_lo: float = 4.5,
                          x_hi: float = 10.5) -> None:
    """Draw dashed verticals and dotted connectors between full and zoom axes."""
    for xv in [x_lo, x_hi]:
        ax_full.axvline(xv, color="black", linestyle="--",
                        linewidth=1.0, alpha=0.5, zorder=0)
    ax_zoom.axvline(x_hi, color="black", linestyle="--",
                    linewidth=1.0, alpha=0.5, zorder=0)
    for xv in [x_lo, x_hi]:
        con = ConnectionPatch(
            xyA=(xv, ax_full.get_ylim()[0]),
            xyB=(xv, ax_zoom.get_ylim()[1]),
            coordsA="data", coordsB="data",
            axesA=ax_full, axesB=ax_zoom,
            color="black", linestyle="dotted", alpha=0.3,
        )
        ax_full.add_artist(con)


def make_single_field_figure(
    agg_by_n: dict,
    data: dict,
    n_labels: List[str],
    field_i: int,
    fs: dict,
    save: bool = True,
    variant_filter: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Combined poster figure: dominance + full-range RRMSE + zoom RRMSE.

    Layout (K = number of n_train values)::

        Row 0 : [Dominance (rows 0–1)] | [RRMSE N=n1] … [RRMSE N=nK]
        Row 1 :        ↕ span          | [Zoom  N=n1] … [Zoom  N=nK]
    """
    K     = len(n_labels)
    ncols = 1 + K
    if figsize is None:
        figsize = (5 * ncols, 6)
    fig   = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    gs    = GridSpec(2, ncols, figure=fig,
                     width_ratios=[1.05] + [1] * K,
                     height_ratios=[1, 1],
                     hspace=0.35, wspace=0.45)

    ax_dom = fig.add_subplot(gs[:, 0])     # spans both rows
    plot_dominance(ax_dom, agg_by_n, fs, n_labels)

    axes_full, axes_zoom = [], []
    fname_field = OUTPUT_NAMES[field_i]
    for j, n_label in enumerate(n_labels):
        ax_f = fig.add_subplot(gs[0, j + 1])
        plot_rrmse(ax_f, data, n_label, field_i, fs,
                   show_ylabel=(j == 0), variant_filter=variant_filter,
                   title=f"Output: {fname_field}  —  {n_label}")
        axes_full.append(ax_f)

        ax_z = fig.add_subplot(gs[1, j + 1])
        plot_rrmse(ax_z, data, n_label, field_i, fs,
                   show_ylabel=(j == 0), zoomed=True, show_errorbars=False,
                   variant_filter=variant_filter)
        axes_zoom.append(ax_z)

    for ax_f, ax_z in zip(axes_full, axes_zoom):
        _add_zoom_connectors(ax_f, ax_z)

    _add_grouped_legend(fig, fs, variant_filter, y_anchor=-0.14)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    
    for ax_ in fig.axes:
        for lbl in ax_.get_xticklabels(which='both') + ax_.get_yticklabels(which='both'):
            lbl.set_fontweight("bold")
            lbl.set_fontsize(fs["tick"])

    if save:
        for ext in ["pdf", "png", "svg"]:
            fname = f"poster_AA_field_{fname_field}.{ext}"
            dpi   = 300 if ext == "pdf" else 200
            fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✓ {fname}")
    return fig


def make_single_field_figure_nozoom(
    agg_by_n: dict,
    data: dict,
    n_labels: List[str],
    field_i: int,
    fs: dict,
    save: bool = True,
    variant_filter: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    include_dominance: bool = True,
) -> plt.Figure:
    """Poster figure (no zoom row): dominance + full-range RRMSE columns."""
    K     = len(n_labels)
    original_ncols = 1 + K
    ncols = original_ncols if include_dominance else K
    if figsize is None:
        figsize = (5 * original_ncols, 4.5)
    fig   = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    
    if include_dominance:
        gs    = GridSpec(1, ncols, figure=fig,
                         width_ratios=[1.05] + [1] * K,
                         wspace=0.45)
        ax_dom = fig.add_subplot(gs[0, 0])
        plot_dominance(ax_dom, agg_by_n, fs, n_labels)
    else:
        gs    = GridSpec(1, ncols, figure=fig,
                         width_ratios=[1] * K,
                         wspace=0.45)

    fname_field = OUTPUT_NAMES[field_i]
    for j, n_label in enumerate(n_labels):
        col_idx = j + 1 if include_dominance else j
        ax = fig.add_subplot(gs[0, col_idx])
        plot_rrmse(ax, data, n_label, field_i, fs,
                   show_ylabel=(j == 0), variant_filter=variant_filter,
                   title=f"Output: {fname_field}  —  {n_label}")

    _add_grouped_legend(fig, fs, variant_filter, y_anchor=-0.20)
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])
    
    for ax_ in fig.axes:
        for lbl in ax_.get_xticklabels(which='both') + ax_.get_yticklabels(which='both'):
            lbl.set_fontweight("bold")
            lbl.set_fontsize(fs["tick"])

    if save:
        suffix = "nozoom" if include_dominance else "nozoom_rrmse_only"
        for ext in ["pdf", "png", "svg"]:
            fname = f"poster_A_field_{fname_field}_{suffix}.{ext}"
            dpi   = 300 if ext == "pdf" else 200
            fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✓ {fname}")
    return fig


def make_single_field_figure_zoom_only(
    agg_by_n: dict,
    data: dict,
    n_labels: List[str],
    field_i: int,
    fs: dict,
    save: bool = True,
    variant_filter: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Paper figure: zoomed RRMSE panels only (m=5..10), one per N label.

    No dominance panel — intended for direct inclusion in a paper.
    """
    K   = len(n_labels)
    if figsize is None:
        figsize = (5 * K, 4.5)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    gs  = GridSpec(1, K, figure=fig, wspace=0.40)

    fname_field = OUTPUT_NAMES[field_i]
    for j, n_label in enumerate(n_labels):
        ax = fig.add_subplot(gs[0, j])
        plot_rrmse(ax, data, n_label, field_i, fs,
                   show_ylabel=(j == 0), zoomed=True, show_errorbars=False,
                   variant_filter=variant_filter,
                   title=f"Output: {fname_field}  —  {n_label}")

    _add_grouped_legend(fig, fs, variant_filter, y_anchor=-0.20)
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])
    
    for ax_ in fig.axes:
        for lbl in ax_.get_xticklabels(which='both') + ax_.get_yticklabels(which='both'):
            lbl.set_fontweight("bold")
            lbl.set_fontsize(fs["tick"])

    if save:
        for ext in ["pdf", "png", "svg"]:
            fname = f"paper_A_field_{fname_field}_zoom.{ext}"
            dpi   = 300 if ext == "pdf" else 200
            fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✓ {fname}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PCA-reconstruction  vs  PCA+GP decomposition figure
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_M_LIST   = [2, 5, 8]    # default latent dimensions to display
_OVERHEAD_HATCH   = "//"         # hatch pattern for GP-overhead portion
_OVERHEAD_COLOR   = "#d73027"    # red for GP overhead
_OVERHEAD_ALPHA   = 0.30
_BAR_W            = 0.10         # bar width (in x-axis units)
_GAP_INNER        = 0.08         # gap between bars in the same group
_GAP_OUTER        = 0.30         # gap between method groups

# Default constraint vector for Lotka-Volterra (u=[1,1,1,1])
_LV_U = np.array([1.0, 1.0, 1.0, 1.0])

# ── Module-level caches (populated once, reused across all calls) ─────────────
# Avoids reloading ground truth and refitting PCA for each model/filter call.
_gt_cache:    dict = {}   # zarr_path → (fields_train_c, fields_test_c, means_train)
_recon_cache: dict = {}   # (zarr_path, pca_type, n_modes, fixed_idx) → (M,Q) or None


def _get_gt_cached(storage) -> tuple:
    """Load and cache ground truth for a zarr file (load once per path)."""
    path = storage.path
    if path not in _gt_cache:
        gt = storage.load_ground_truth()
        ft, fe = gt["fields_train"], gt["fields_test"]
        mt = [f.mean(axis=0) for f in ft]
        _gt_cache[path] = (
            [ft[i] - mt[i] for i in range(len(ft))],
            [fe[i] - mt[i] for i in range(len(fe))],
            mt,
        )
    return _gt_cache[path]


def _compute_recon_rrmse_from_zarr(storage, model_name: str,
                                   u: np.ndarray) -> Optional[np.ndarray]:
    """Recompute cumulative PCA reconstruction RRMSE from stored ground truth.

    Ground truth is loaded ONCE per zarr (cached in ``_gt_cache``).
    PCA is fitted ONCE per unique ``(path, pca_type, n_modes, fixed_idx)``
    combination (cached in ``_recon_cache``).

    Parameters
    ----------
    storage    : ZarrBenchmarkStorage (opened in read mode)
    model_name : name of the model inside the zarr
    u          : constraint vector (e.g. [1,1,1,1] for LotkaVolterra)

    Returns
    -------
    cum_rrmse_recon : (M, Q) numpy array, or None if pca_type is unknown.
    """
    from benchmark_pca_gp.reduction.rowwise   import RowwisePCA
    from benchmark_pca_gp.reduction.colwise   import ColwisePCA
    from benchmark_pca_gp.reduction.fieldwise import FieldwisePCA

    cfg       = dict(storage.store[f"models/{model_name}"].attrs)
    pca_type  = cfg.get("pca_type", "")
    n_modes   = int(cfg.get("n_modes", 10))
    fixed_raw = cfg.get("fixed_idx", -1)
    fixed_idx = None if (fixed_raw is None or fixed_raw == -1) else int(fixed_raw)

    cache_key = (storage.path, pca_type, n_modes, fixed_idx)
    if cache_key in _recon_cache:
        return _recon_cache[cache_key]

    # Load GT once per zarr (cached)
    fields_train_c, fields_test_c, means_train = _get_gt_cached(storage)

    if pca_type == "RowwisePCA":
        reducer = RowwisePCA(n_modes=n_modes)
    elif pca_type == "ColwisePCA":
        if fixed_idx is None:
            _recon_cache[cache_key] = None
            return None
        reducer = ColwisePCA(n_modes=n_modes, fixed_idx=fixed_idx)
    elif pca_type == "FieldwisePCA":
        reducer = FieldwisePCA(n_modes=n_modes)
    else:
        _recon_cache[cache_key] = None
        return None

    reducer.fit(fields_train_c)
    result = reducer.cumulative_reconstruction_error(
        fields_test_c, means_train, "test",
        fixed_idx=fixed_idx, u=u,
    )
    recon = result.get("cumulative_rrmse_test")   # (M, Q)
    _recon_cache[cache_key] = recon
    return recon


def _warm_up_single_zarr(zarr_path: str, u: np.ndarray) -> None:
    """Pre-populate GT and recon caches for all models in one zarr file.

    Safe to call concurrently for different paths (each path → independent
    ZarrBenchmarkStorage object; CPython dict writes are GIL-protected).
    """
    import os
    if not os.path.exists(zarr_path):
        return
    from benchmark_pca_gp.benchmark.storage import ZarrBenchmarkStorage
    try:
        storage = ZarrBenchmarkStorage(zarr_path, mode="r")
        _get_gt_cached(storage)                # cache GT
        for name in storage.list_models():
            res   = storage.load_model_result_light(name)
            inter = res.get("intermediate", {})
            if inter.get("cumulative_rrmse_recon_test") is not None:
                continue   # already stored, no need to compute
            try:
                _compute_recon_rrmse_from_zarr(storage, name, u)
            except Exception:
                pass
    except Exception:
        pass


def _warm_up_recon_parallel(paths: List[str], u: np.ndarray,
                             max_workers: int = 4) -> None:
    """Warm up GT/recon caches across zarr files using a thread pool.

    Uses threads (not processes) because:
    - Zarr I/O releases the GIL, so threads get real I/O parallelism.
    - Caches are module-level dicts shared between threads (no pickling).
    - For small N (LV), the PCA is trivial; I/O is the bottleneck.
    """
    new_paths = [p for p in paths if p not in _gt_cache]
    if not new_paths:
        return
    n = min(max_workers, len(new_paths))
    if n <= 1:
        for p in new_paths:
            _warm_up_single_zarr(p, u)
        return
    from concurrent.futures import ThreadPoolExecutor
    print(f"  Parallel PCA warm-up: {len(new_paths)} zarr files, {n} threads …",
          flush=True)
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda p: _warm_up_single_zarr(p, u), new_paths))


def _collect_records_with_recon_all(ana, n_label: str,
                                    u: np.ndarray) -> dict:
    """Single-pass over all seeds × models: returns dict (prefix, fixed_idx) → records.

    One call replaces the 13 separate filter calls that ``load_recon_data``
    used to issue.  Ground truth and PCA are loaded/fitted at most once per
    zarr file thanks to ``_gt_cache`` / ``_recon_cache``.
    """
    all_records: dict = {}
    for seed, single_ana in zip(ana._seeds, ana._analyzers):
        for name in single_ana.list_models():
            prefix    = name.split("_")[0]
            res       = single_ana.storage.load_model_result_light(name)
            cfg       = res.get("config", {})
            inter     = res.get("intermediate", {})
            fixed_idx = int(cfg.get("fixed_idx", -1))

            cum_pred  = inter.get("cumulative_rrmse_pred_test")
            if cum_pred is None:
                continue

            cum_recon = inter.get("cumulative_rrmse_recon_test")
            if cum_recon is None:
                try:
                    cum_recon = _compute_recon_rrmse_from_zarr(
                        single_ana.storage, name, u)
                except Exception:
                    cum_recon = None

            key = (prefix, fixed_idx)
            all_records.setdefault(key, []).append({
                "seed":            seed,
                "prefix":          prefix,
                "fixed_idx":       fixed_idx,
                "cum_rrmse":       np.array(cum_pred),
                "cum_rrmse_recon": np.array(cum_recon) if cum_recon is not None else None,
                "n_train_label":   n_label,
            })
    return all_records


def load_recon_data(
    zarr_specs:  Optional[List[Tuple[List[str], str]]] = None,
    u:           Optional[np.ndarray] = None,
    max_workers: int = 4,
) -> Tuple[dict, List[str]]:
    """Load records including PCA reconstruction RRMSE.

    If ``cumulative_rrmse_recon_test`` is absent from the zarr intermediate
    dict (old zarr files), it is recomputed automatically from the stored
    ground-truth fields by re-fitting the PCA.

    Performance
    -----------
    - Ground truth loaded ONCE per zarr file (cached).
    - PCA fitted ONCE per unique ``(pca_type, n_modes, fixed_idx)`` per zarr
      (cached).  With e.g. 3 n_modes values and 4 model types, this means
      ~21 PCA fits per zarr instead of ~48 × 13 = 624 in the naïve approach.
    - Warm-up runs in parallel across zarr files (``max_workers`` threads).

    Parameters
    ----------
    zarr_specs  : same format as load_all_data().  None = default N=30/N=10.
    u           : constraint vector.  Defaults to [1,1,1,1] (LotkaVolterra).
    max_workers : number of threads for parallel warm-up (set to 1 to disable).

    Returns
    -------
    data_recon : dict[(n_label, prefix, k) → records]
    n_labels   : list of N labels in input order.
    """
    if zarr_specs is None:
        zarr_specs = [
            ([os.path.join(_RESULTS_LV, f"results_N_=30_lv_seed{i}.zarr") for i in range(1, 10)], "N=30"),
            ([os.path.join(_RESULTS_LV, f"results_N_=10_lv_seed{i}.zarr") for i in range(1, 10)], "N=10"),
        ]
    if u is None:
        u = _LV_U

    # ── Parallel warm-up: populate GT + recon caches across all zarr files ──
    all_paths = [p for paths, _ in zarr_specs for p in paths]
    _warm_up_recon_parallel(all_paths, u, max_workers=max_workers)

    # ── Single-pass collection per analyzer ──────────────────────────────────
    analyzers = [(MultiSeedAnalyzer(paths), label) for paths, label in zarr_specs]
    n_labels  = [label for _, label in zarr_specs]
    data_recon: dict = {}

    for ana, n_label in analyzers:
        all_recs = _collect_records_with_recon_all(ana, n_label, u=u)
        data_recon[(n_label, "RC", -1)] = all_recs.get(("RC", -1), [])
        for k in K_STARS:
            for prefix in ["CI", "FI", "FM"]:
                data_recon[(n_label, prefix, k)] = all_recs.get((prefix, k), [])

    return data_recon, n_labels


def _bar_stats_at_m(
    records: List[dict], field_i: int, m_val: int
) -> Tuple[Optional[float], float, Optional[float]]:
    """Return (pred_mean, pred_std, recon_mean) at latent dimension m_val."""
    m_idx = m_val - 1   # 1-indexed → 0-indexed
    preds, recons = [], []
    for r in records:
        arr_p = r.get("cum_rrmse")
        if arr_p is not None:
            a = np.asarray(arr_p)
            if m_idx < a.shape[0]:
                preds.append(float(a[m_idx, field_i]))
        arr_r = r.get("cum_rrmse_recon")
        if arr_r is not None:
            a = np.asarray(arr_r)
            if m_idx < a.shape[0]:
                recons.append(float(a[m_idx, field_i]))
    pred_mean  = float(np.mean(preds))  if preds  else None
    pred_std   = float(np.std(preds))   if preds  else 0.0
    recon_mean = float(np.mean(recons)) if recons else None
    return pred_mean, pred_std, recon_mean


def _auto_best_worst(
    data_recon: dict, n_labels: List[str], field_i: int
) -> Dict[str, Tuple[int, int]]:
    """For each deductive prefix (CI/FI/FM) find (best_k, worst_k)
    by mean pred-RRMSE over all stored m values and all N labels."""
    result: dict = {}
    for prefix in ["CI", "FI", "FM"]:
        scores: dict = {}
        for k in K_STARS:
            vals = []
            for n_label in n_labels:
                key = (n_label, prefix, k)
                if key in data_recon:
                    _, mu, _ = extract_rrmse_stats(data_recon[key], field_i)
                    if mu is not None:
                        vals.append(float(mu.mean()))
            if vals:
                scores[k] = float(np.mean(vals))
        if not scores:
            continue
        best  = min(scores, key=scores.get)
        worst = max(scores, key=scores.get)
        result[prefix] = (best, worst)
    return result


def _bar_x_layout(bw: dict) -> List[Tuple[str, int, str, float, str]]:
    """Build bar layout: list of (prefix, k, tick_label, x_pos, group_label).

    Groups: RC alone, then CI/FI/FM each with (best, worst).
    Positions computed with _BAR_W, _GAP_INNER, _GAP_OUTER.
    """
    layout = []  # (prefix, k, tick_label, x_pos, group_label)
    cursor = 0.0

    def _add_group(prefix, ks, group_label):
        nonlocal cursor
        for j, k in enumerate(ks):
            if j > 0:
                cursor += _BAR_W + _GAP_INNER
            lbl = prefix if k == -1 else f"l={k}"
            layout.append((prefix, k, lbl, cursor, group_label))
        cursor += _BAR_W + _GAP_OUTER

    _add_group("RC", [-1], "Row-CMO")
    for prefix, group_label in [("CI", "Col-Indep"), ("FI", "Fw-Indep"), ("FM", "Fw-LCM")]:
        if prefix in bw:
            best, worst = bw[prefix]
            ks = [best] if best == worst else [best, worst]
            _add_group(prefix, ks, group_label)

    return layout


def _draw_bar(
    ax, x: float, pred_mean: float, pred_std: float,
    recon_mean: Optional[float], color: str,
    first_overhead: bool,
) -> None:
    """Draw one stacked bar: PCA recon (solid) + GP overhead (red hatch)."""
    recon_h  = max(recon_mean if recon_mean is not None else 0.0, 0.0)
    pred_h   = max(pred_mean, recon_h)
    overhead = pred_h - recon_h

    if recon_mean is not None:
        # Solid base: PCA reconstruction
        ax.bar(x, recon_h, width=_BAR_W, color=color, alpha=0.88,
               edgecolor="white", linewidth=0.2, zorder=3)
        # Hatched top: GP overhead
        if overhead > 1e-8 * pred_h:
            lbl = "GP overhead" if first_overhead else None
            ax.bar(x, overhead, bottom=recon_h, width=_BAR_W,
                   color=_OVERHEAD_COLOR, alpha=_OVERHEAD_ALPHA,
                   hatch=_OVERHEAD_HATCH, edgecolor=_OVERHEAD_COLOR,
                   linewidth=0.2, zorder=4, label=lbl)
    else:
        # Recon not available: draw total bar only, lighter
        ax.bar(x, pred_h, width=_BAR_W, color=color, alpha=0.6,
               edgecolor="white", linewidth=0.2, zorder=3)

    # Seed-std error bar at the top of the pred bar
    if pred_std > 0:
        ax.errorbar(x, pred_h, yerr=pred_std,
                    color="black", linewidth=0.9,
                    capsize=3.5, capthick=0.9, zorder=10)


def _add_group_brackets(ax, layout, fs: dict, y_frac: float = 1.04) -> None:
    """Draw horizontal bracket + group label above each method group."""
    groups: dict = {}
    for prefix, k, _, x, group_label in layout:
        groups.setdefault(group_label, []).append(x)

    y_top = ax.get_ylim()[1] * y_frac

    for g_label, xs in groups.items():
        x_lo, x_hi = min(xs) - _BAR_W * 0.4, max(xs) + _BAR_W * 0.4
        x_mid = (x_lo + x_hi) / 2
        # Bracket line
        ax.annotate("",
                    xy=(x_hi, y_top), xytext=(x_lo, y_top),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="dimgray", lw=1.0))
        # Label
        ax.text(x_mid, y_top * 1.01, g_label,
                ha="center", va="bottom",
                fontsize=fs["annot"], fontweight="bold", color="dimgray")


def _method_color(prefix: str, k: int) -> str:
    if prefix == "RC": return C_RC
    if prefix == "CI": return C_CI.get(k, C_CI[0])
    if prefix == "FI": return C_FI.get(k, C_FI[0])
    return C_FM.get(k, C_FM[0])


def make_pca_gp_decomposition_figure(
    data_recon:          dict,
    n_labels:            List[str],
    field_i:             int,
    fs:                  dict,
    m_list:              Optional[List[int]] = None,
    best_worst_override: Optional[dict]      = None,
    save:                bool                = True,
    figsize:             Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """PCA-reconstruction vs full (PCA + GP) RRMSE stacked bar figure.

    Layout: ``len(n_labels)`` rows × ``len(m_list)`` columns.

    Each panel (N, m):
      - X axis : method variants — RC | CI(best, worst) | FI(best, worst) | FM(best, worst)
      - Y axis : RRMSE (linear)
      - Bar anatomy:
          solid base (method colour)  = PCA reconstruction RRMSE
          red-hatched extension       = GP overhead (Final − PCA)
          black error bar at top      = ±1 std over seeds

    Parameters
    ----------
    m_list : latent dimensions to display, e.g. [2, 5, 8]. Default: [2, 5, 8].
    best_worst_override : dict {prefix: (best_k, worst_k)} to fix which variants
        are shown instead of auto-detecting from the data.
    """
    if m_list is None:
        m_list = _DEFAULT_M_LIST

    bw     = best_worst_override or _auto_best_worst(data_recon, n_labels, field_i)
    layout = _bar_x_layout(bw)

    tick_xs    = [entry[3] for entry in layout]
    tick_lbls  = [entry[2] for entry in layout]
    groups     = {entry[4] for entry in layout}
    all_x_span = (min(tick_xs) - _BAR_W, max(tick_xs) + _BAR_W)

    n_rows = len(n_labels)
    n_cols = len(m_list)
    if figsize is None:
        figsize = (max(4, 1.6 * len(layout)) * n_cols, 3.8 * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    fname_field = OUTPUT_NAMES[field_i]

    for row_i, n_label in enumerate(n_labels):
        for col_i, m_val in enumerate(m_list):
            ax = axes[row_i, col_i]
            ax.set_facecolor(BG_COLOR)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            first_overhead = True
            for prefix, k, tick_lbl, x_pos, group_label in layout:
                key = (n_label, prefix, k)
                recs = data_recon.get(key, [])
                pred_mean, pred_std, recon_mean = _bar_stats_at_m(recs, field_i, m_val)
                if pred_mean is None:
                    continue
                color = _method_color(prefix, k)
                _draw_bar(ax, x_pos, pred_mean, pred_std, recon_mean,
                          color, first_overhead)
                if first_overhead and recon_mean is not None:
                    first_overhead = False   # only label the legend once

            # Cosmetics
            ax.set_xticks(tick_xs)
            ax.set_xticklabels(tick_lbls, fontsize=fs["tick"] * 0.85,
                                fontweight="bold", rotation=0)
            ax.tick_params(labelsize=fs["tick"])
            for lbl in ax.get_yticklabels(which='both'):
                lbl.set_fontweight("bold")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.set_axisbelow(True)
            ax.set_xlim(all_x_span)

            if col_i == 0:
                ax.set_ylabel(
                    f"{n_label}\nRRMSE",
                    fontsize=fs["label"], fontweight="bold",
                )
            ax.set_title(
                f"$m = {m_val}$",
                fontsize=fs["title"], fontweight="bold", pad=6,
            )
            if row_i == n_rows - 1:
                ax.set_xlabel("Method / variant",
                              fontsize=fs["label"], fontweight="bold")

    # Group bracket annotations — leave headroom at top (rect y1=0.92)
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    for ax_row in axes:
        for ax in ax_row:
            _add_group_brackets(ax, layout, fs, y_frac=1.06)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Method colour patches
    legend_handles = []
    for prefix, group_label in [("RC","Row-CMO"), ("CI","Col-Indep"), ("FI","Fw-Indep"), ("FM","Fw-LCM")]:
        if group_label not in groups:
            continue
        if prefix == "RC":
            k = -1
        else:
            k = bw[prefix][0] if prefix in bw else 0  # best k for colour sample
        legend_handles.append(mpatches.Patch(
            color=_method_color(prefix, k), alpha=0.88, label=group_label))
    # PCA recon sentinel
    legend_handles.append(mpatches.Patch(
        facecolor="dimgray", alpha=0.55, label="PCA recon. (base)"))
    # GP overhead sentinel
    legend_handles.append(mpatches.Patch(
        facecolor=_OVERHEAD_COLOR, alpha=_OVERHEAD_ALPHA,
        hatch=_OVERHEAD_HATCH, label="GP overhead"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=True, framealpha=0.95, edgecolor="#ccc",
        prop={"size": fs["legend"], "weight": "bold"},
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(bottom=0.14)

    if save:
        for ext in ["pdf", "png", "svg"]:
            fname = f"paper_decomp_field_{fname_field}.{ext}"
            dpi   = 300 if ext == "pdf" else 200
            fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✓ {fname}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic table: PCA recon vs PCA+GP prediction
# ──────────────────────────────────────────────────────────────────────────────

def print_decomp_table(
    data_recon: dict,
    n_labels:   List[str],
    m_list:     Optional[List[int]] = None,
    fields:     Optional[List[int]] = None,
) -> None:
    """Print a formatted table of pred vs recon RRMSE for every method/variant.

    Helps diagnose:
    - Whether ``cumulative_rrmse_recon_test`` is stored in the zarr files
      (shows "N/A" in Recon column if absent).
    - Whether the GP overhead is truly negligible or just too small to render.

    Parameters
    ----------
    fields : list of field indices to print (default: all Q fields).
    """
    if m_list is None:
        m_list = _DEFAULT_M_LIST
    if fields is None:
        fields = list(range(Q))

    for field_i in fields:
        fname = OUTPUT_NAMES[field_i]
        for n_label in n_labels:

            # ── Header ────────────────────────────────────────────────────────
            title = f"  Field={fname}  |  {n_label}  "
            print(f"\n{'─' * 88}")
            print(f"{title:^88}")
            print(f"{'─' * 88}")

            # Column headers: one block per m value
            hdr_parts = [f"{'Method':>8}", f"{'l':>3}"]
            for m_val in m_list:
                hdr_parts.append(f"{'m='+str(m_val):>24}")
            print("  " + "  ".join(hdr_parts))

            sub_hdr_parts = [f"{'':>8}", f"{'':>3}"]
            for _ in m_list:
                sub_hdr_parts.append(
                    f"{'Pred':>7}  {'Recon':>7}  {'OH%':>6}")
            print("  " + "  ".join(sub_hdr_parts))
            print(f"  {'─' * 84}")

            # ── RC ────────────────────────────────────────────────────────────
            row_parts = [f"{'RC':>8}", f"{'-':>3}"]
            recon_missing = False
            for m_val in m_list:
                p, _, r = _bar_stats_at_m(
                    data_recon.get((n_label, "RC", -1), []), field_i, m_val)
                if p is None:
                    row_parts.append(f"{'no data':>24}")
                    continue
                if r is None:
                    recon_missing = True
                    row_parts.append(
                        f"{p:>7.4f}  {'N/A':>7}  {'N/A':>6}")
                else:
                    oh = p - r
                    oh_pct = oh / p * 100 if p > 1e-15 else 0.0
                    row_parts.append(
                        f"{p:>7.4f}  {r:>7.4f}  {oh_pct:>5.1f}%")
            print("  " + "  ".join(row_parts))
            if recon_missing:
                print("    ⚠  recon key absent for RC")

            # ── CI / FI / FM ──────────────────────────────────────────────────
            for prefix in ["CI", "FI", "FM"]:
                print(f"  {'─' * 40}")
                for k in K_STARS:
                    row_parts = [f"{prefix:>8}", f"{k:>3}"]
                    recon_missing = False
                    any_data = False
                    for m_val in m_list:
                        p, _, r = _bar_stats_at_m(
                            data_recon.get((n_label, prefix, k), []),
                            field_i, m_val)
                        if p is None:
                            row_parts.append(f"{'no data':>24}")
                            continue
                        any_data = True
                        if r is None:
                            recon_missing = True
                            row_parts.append(
                                f"{p:>7.4f}  {'N/A':>7}  {'N/A':>6}")
                        else:
                            oh = p - r
                            oh_pct = oh / p * 100 if p > 1e-15 else 0.0
                            flag = " ◀" if oh_pct > 20 else (
                                   " ▼" if oh_pct < 0 else "")
                            row_parts.append(
                                f"{p:>7.4f}  {r:>7.4f}  {oh_pct:>5.1f}%{flag}")
                    if any_data:
                        print("  " + "  ".join(row_parts))
                    if recon_missing:
                        print(f"    ⚠  recon key absent for {prefix} l={k}")

        print(f"\n{'═' * 88}")

    # ── Summary: recon availability ───────────────────────────────────────────
    print("\n── Recon data availability ──")
    for n_label in n_labels:
        for prefix in ["RC", "CI", "FI", "FM"]:
            ks = [-1] if prefix == "RC" else K_STARS
            for k in ks:
                recs = data_recon.get((n_label, prefix, k), [])
                n_total = len(recs)
                n_recon = sum(
                    1 for r in recs if r.get("cum_rrmse_recon") is not None)
                status = "✓ all" if n_recon == n_total and n_total > 0 \
                    else (f"✗ 0/{n_total}" if n_recon == 0
                          else f"⚠ {n_recon}/{n_total}")
                k_str = str(k) if k != -1 else " -"
                print(f"  {n_label:>5}  {prefix:>3}  l={k_str}  "
                      f"seeds={n_total:>3}  recon={status}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Data sources ─────────────────────────────────────────────────────────
    # Default: two sizes N=30 and N=10.
    # For three sizes, uncomment and edit the block below:
    #
    zarr_specs = [
        ([os.path.join(_RESULTS_LV, f"results_N_=30_lv_seed{i}.zarr") for i in range(1, 10)], "N=30"),
        ([os.path.join(_RESULTS_LV, f"results_N_=15_lv_seed{i}.zarr") for i in range(1, 10)], "N=15"),
        ([os.path.join(_RESULTS_LV, f"results_N_=10_lv_seed{i}.zarr") for i in range(1, 10)], "N=10"),
    ]
    # zarr_specs = None   # use built-in default (N=30, N=10)

    # ── Variant filter ────────────────────────────────────────────────────────
    # Show only two variants per deductive method (e.g. l=0 and l=2).
    # Set to None to display all four variants for each method.
    #
    # variant_filter = {"CI": [0, 2], "FI": [0, 2], "FM": [0, 2]}
    variant_filter = None

    # ── Paramètres d'apparence globaux ───────────────────────────────────────
    # Taille globale des figures (largeur, hauteur) en pouces
    GLOBAL_FIGSIZE = (18, 6)
    
    # Taille pour la figure de décomposition (si None, auto-calculée)
    DECOMP_FIGSIZE = (18, 6)

    # Échelle de la taille de police (plus grand = polices plus grosses)
    FONT_SCALE = 1.6

    # Force les labels d'axes (qui sont en texte mathématique pour les échelles log) à hériter du style normal (permettant le gras manuel)
    plt.rcParams["mathtext.default"] = "regular"
    # Utilise la police 'cm' (Computer Modern) pour les maths afin d'avoir le caractère \mathcal{E} en gras sans erreur
    plt.rcParams["mathtext.fontset"] = "cm"
    # ─────────────────────────────────────────────────────────────────────────

    # ── m values to show in the decomposition figure ─────────────────────────
    # e.g. [2, 5, 8] or [2, 4, 6, 8]
    decomp_m_list = [2, 5, 8]

    # ── Load from saved data ──────────────────────────────────────────────────
    import pickle
    import sys
    try:
        with open("plot_lv_rrmse_data.pkl", "rb") as f:
            saved_data = pickle.load(f)
        agg_by_n = saved_data["agg_by_n"]
        data = saved_data["data"]
        n_labels = saved_data["n_labels"]
        data_recon = saved_data["data_recon"]
        print("\nData loaded successfully from 'plot_lv_rrmse_data.pkl'.")
    except FileNotFoundError:
        print("Error: 'plot_lv_rrmse_data.pkl' not found.")
        print("Please run 'plot_lv_rrmse.py' first to generate the data.")
        sys.exit(1)

    fs = _poster_fs(scale=FONT_SCALE)
    u = _LV_U

    # ── Diagnostic table (always printed; helps verify recon data is present) ─
    print_decomp_table(data_recon, n_labels, m_list=decomp_m_list)

    # ── Generate figures ──────────────────────────────────────────────────────
    for i in range(Q):
        print(f"\n── Field {OUTPUT_NAMES[i]} ({i}) ──")

        # Combined: dominance + full-range + zoom
        fig = make_single_field_figure(
            agg_by_n, data, n_labels, i, fs,
            save=True, variant_filter=variant_filter, figsize=GLOBAL_FIGSIZE)
        plt.close(fig)

        # Full-range only (no zoom row)
        fig_nozoom = make_single_field_figure_nozoom(
            agg_by_n, data, n_labels, i, fs,
            save=True, variant_filter=variant_filter, figsize=GLOBAL_FIGSIZE,
            include_dominance=True)
        plt.close(fig_nozoom)

        # Full-range only (no zoom row, no dominance)
        fig_nozoom_rrmse = make_single_field_figure_nozoom(
            agg_by_n, data, n_labels, i, fs,
            save=True, variant_filter=variant_filter, figsize=GLOBAL_FIGSIZE,
            include_dominance=False)
        plt.close(fig_nozoom_rrmse)

        # Zoom only (paper version, no dominance panel)
        fig_zoom = make_single_field_figure_zoom_only(
            agg_by_n, data, n_labels, i, fs,
            save=True, variant_filter=variant_filter, figsize=GLOBAL_FIGSIZE)
        plt.close(fig_zoom)

        # PCA-recon vs PCA+GP decomposition (stacked bar)
        fig_decomp = make_pca_gp_decomposition_figure(
            data_recon, n_labels, i, fs,
            m_list=decomp_m_list,
            save=True, figsize=DECOMP_FIGSIZE)
        plt.close(fig_decomp)

    print(f"\n✓ All {Q} fields done — 4 figure variants per field.")


if __name__ == "__main__":
    main()
