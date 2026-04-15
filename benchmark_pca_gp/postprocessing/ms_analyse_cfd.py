"""
Post-processing analysis for the CFD diffuser benchmark.

Three output fields: τ₁₁ (tau11), τ₂₂ (tau22), k
Constraint : τ₁₁ + τ₂₂ − (4/3)k = 0

Figures produced
----------------
1. Cumulative Q²   vs modes retained (GP-prediction quality, k=1..M)
2. Cumulative RRMSE vs modes retained
3. Final Q² / RRMSE per field — grouped bar chart  (mean ± std over seeds)
4. Constraint error (max) per model
5. Latent GP Q² per mode     (heatmap-style grid or line plot)

Usage
-----
    python ms_analyse_cfd.py                        # seeds 1 & 2
    python ms_analyse_cfd.py --prefix results_cfd --seeds 1 2 3
    python ms_analyse_cfd.py --prefix results_cfd --seeds 1 2 --outdir figures_cfd/
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import List, Optional, Tuple, Dict

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# CFD-specific constants
# ─────────────────────────────────────────────────────────────────────────────

FIELD_NAMES = [r"$\tau_{11}$", r"$\tau_{22}$", r"$k$"]

# ─────────────────────────────────────────────────────────────────────────────
# Style tables  (same conventions as ms_analyse_compare_n.py)
# ─────────────────────────────────────────────────────────────────────────────

_PREFIX_STYLE = {
    "RC": ("#D62728", "-",  2.6, 1.00, "o", True),
    "CI": ("#6BAED6", "--", 1.3, 0.82, "s", False),
    "FI": ("#74C476", ":",  1.3, 0.82, "^", False),
    "FM": ("#9E9AC8", "-.", 1.3, 0.82, "D", False),
}


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def _lighten(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor)


def _build_scenario_styles(records: list) -> dict:
    by_prefix: dict = {}
    for r in records:
        by_prefix.setdefault(r["prefix"], set()).add(r["fixed_idx"])
    styles = {}
    for pfx, p_set in by_prefix.items():
        p_list = sorted(p_set)
        n = len(p_list)
        base_hex, ls, lw, alpha, marker, is_ours = _PREFIX_STYLE.get(
            pfx, ("#888888", "--", 1.3, 0.8, "o", False))
        for i, p in enumerate(p_list):
            shade = 0.35 * (i / max(n-1, 1))
            color = base_hex if i == 0 else _lighten(base_hex, shade)
            styles[(pfx, p)] = dict(
                color=color, ls=ls, lw=lw, alpha=alpha,
                marker=marker, zorder=6 if is_ours else 3,
            )
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Scale / font helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_scale(figsize: tuple, Q: int, n_rows: int) -> float:
    cell_w = figsize[0] / Q
    cell_h = figsize[1] / n_rows
    return max(0.45, min(cell_w / 5.0, cell_h / 4.0))


def _font_sizes(scale: float) -> dict:
    s = scale
    return {
        "tick":   max(7,  round(10 * s)),
        "label":  max(8,  round(11 * s)),
        "title":  max(9,  round(12 * s)),
        "annot":  max(8,  round(11 * s)),
        "legend": max(7,  round( 9 * s)),
    }


def _apply_bold_ticks(ax, fs: dict) -> None:
    ax.tick_params(axis="both", which="both", labelsize=fs["tick"])
    
    try:
        ax.figure.canvas.draw()
    except Exception:
        pass
        
    for lbl in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
        lbl.set_fontweight("bold")
        lbl.set_fontsize(fs["tick"])
        text = lbl.get_text()
        if text.startswith("$") and text.endswith("$") and "\\mathbf" not in text and "\\boldsymbol" not in text:
            inner = text[1:-1].replace("\\mathdefault", "")
            lbl.set_text(rf"$\mathbf{{{inner}}}$")


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_records(
    ana: MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
) -> List[dict]:
    """Collect per-model records with cumulative metrics (cum_q2, cum_rrmse)."""
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
            n_modes   = cfg.get("n_modes", np.nan)

            if model_types   is not None and prefix    not in model_types:
                continue
            if fixed_indices is not None and fixed_idx not in fixed_indices:
                continue

            records.append({
                "seed":       seed,
                "model_name": name,
                "prefix":     prefix,
                "n_modes":    n_modes,
                "fixed_idx":  fixed_idx,
                "cum_q2":     np.array(cum_q2),
                "cum_rrmse":  np.array(cum_rrmse),
            })
    return records


def _collect_final_records(
    ana: MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
) -> List[dict]:
    """Collect per-model records with final metrics (q2, rrmse, constraint)."""
    records = []
    for seed, single_ana in zip(ana._seeds, ana._analyzers):
        for name in single_ana.list_models():
            res = single_ana.storage.load_model_result_light(name)
            cfg = res.get("config", {})
            met = res.get("metrics", {})

            prefix    = name.split("_")[0]
            fixed_idx = int(cfg.get("fixed_idx", -1))

            if model_types   is not None and prefix    not in model_types:
                continue
            if fixed_indices is not None and fixed_idx not in fixed_indices:
                continue

            records.append({
                "seed":           seed,
                "model_name":     name,
                "prefix":         prefix,
                "fixed_idx":      fixed_idx,
                "n_modes":        cfg.get("n_modes", np.nan),
                "q2":             np.array(met.get("q2", [])),
                "rrmse":          np.array(met.get("rrmse", [])),
                "constraint_max": float(met.get("constraint_max", [np.nan])[0]),
                "constraint_mean":float(met.get("constraint_mean", [np.nan])[0]),
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 & 2 — Cumulative Q² / RRMSE vs modes
# ─────────────────────────────────────────────────────────────────────────────

_JITTER = 0.0 # 0.18


def _plot_cumulative_cell(ax, records, field_i, key, scenarios, styles,
                          scale: float = 1.0) -> None:
    n_scen  = len(scenarios)
    offsets = (np.arange(n_scen) - (n_scen-1)/2.0) * (_JITTER / max(n_scen-1, 1))

    def _priority(sc):
        return 0 if _PREFIX_STYLE.get(sc[0], ("",)*6)[5] else 1

    draw_order = sorted(range(n_scen), key=lambda i: _priority(scenarios[i]))
    M_global   = 1

    for ord_i in draw_order:
        prefix, p_idx = scenarios[ord_i]
        sub = [r for r in records
               if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
        if not sub:
            continue

        # Average over multi-models per seed (shouldn't happen for fixed config)
        by_seed: dict = {}
        for r in sub:
            by_seed.setdefault(r["seed"], []).append(r[key][:, field_i])

        seed_vals = np.array([np.mean(vs, axis=0) for vs in by_seed.values()])
        M_cur     = seed_vals.shape[1]
        M_global  = max(M_global, M_cur)
        x         = np.arange(1, M_cur+1, dtype=float) + offsets[ord_i]

        means = seed_vals.mean(axis=0)
        stds  = seed_vals.std(axis=0) if seed_vals.shape[0] > 1 else np.zeros(M_cur)

        st      = styles[(prefix, p_idx)]
        is_ours = _PREFIX_STYLE.get(prefix, ("",)*6)[5]
        lbl     = prefix if p_idx == -1 else f"{prefix} (p={p_idx})"

        ax.errorbar(
            x, means, yerr=stds,
            color=st["color"], alpha=st["alpha"],
            linestyle=st["ls"],
            linewidth=st["lw"] * scale,
            marker=st["marker"],
            markersize=(6 if is_ours else 3) * scale,
            elinewidth=0.8 * scale,
            capsize=3  * scale,
            capthick=0.8 * scale,
            zorder=st["zorder"],
            label=lbl,
        )

    ax.set_xticks(range(1, M_global+1))
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cumulative_metric(
    ana:           MultiSeedAnalyzer,
    metric:        str = "q2",
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    field_names:   Optional[List[str]] = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """Single-row figure: cumulative GP-prediction Q² or RRMSE vs modes retained.

    Parameters
    ----------
    ana          : MultiSeedAnalyzer
    metric       : "q2" or "rrmse"
    model_types  : list of prefixes to include, e.g. ["RC", "FI", "CI"]
    fixed_indices: list of fixed_idx values to include
    field_names  : axis titles per field; defaults to FIELD_NAMES
    """
    records = _collect_records(ana, model_types, fixed_indices)
    if not records:
        print(f"[plot_cumulative_metric] No records found.")
        return plt.figure()

    key    = "cum_q2"   if metric == "q2"   else "cum_rrmse"
    ylabel = "Q²"       if metric == "q2"   else "RRMSE"

    Q      = records[0][key].shape[1]
    names  = field_names or FIELD_NAMES[:Q]
    fsize  = figsize or (5 * Q, 6)
    scale  = _compute_scale(fsize, Q, 1)
    fs     = _font_sizes(scale)

    fig, axes = plt.subplots(1, Q, figsize=fsize, sharey=True, sharex=True,
                             squeeze=False)

    scenarios = sorted({(r["prefix"], r["fixed_idx"]) for r in records})
    styles    = _build_scenario_styles(records)

    for ci in range(Q):
        ax = axes[0, ci]
        _plot_cumulative_cell(ax, records, ci, key, scenarios, styles, scale)
        ax.set_title(names[ci], fontsize=fs["title"], fontweight="bold")
        ax.set_xlabel("Modes retained", fontsize=fs["label"], fontweight="bold")
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=fs["label"], fontweight="bold")
        ax.set_yscale("log")

    plt.tight_layout() 
    for ax in axes.flatten():
        _apply_bold_ticks(ax, fs)

    # ── Legend ────────────────────────────────────────────────────────────────
    def _leg_handle(pfx, p):
        st  = styles[(pfx, p)]
        lbl = pfx if p == -1 else f"{pfx} (p={p})"
        if pfx == "RC":
            lbl += "  ★"
        return mlines.Line2D(
            [], [],
            color=st["color"], linestyle=st["ls"],
            linewidth=st["lw"] * scale,
            marker=st["marker"],
            markersize=max(3, 5*scale) * (1.4 if pfx == "RC" else 1.0),
            alpha=st["alpha"],
            label=lbl,
        )

    handles = [_leg_handle(pfx, p) for pfx, p in sorted(scenarios)]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(handles), 5),
        frameon=True,
        edgecolor="black",
        fontsize=fs["legend"],
        prop={"size": fs["legend"], "weight": "bold"},
    )
    plt.subplots_adjust(bottom=max(0.18, 0.24 * scale))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Final metrics bar chart (Q² and RRMSE per field)
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_metrics(
    ana:           MultiSeedAnalyzer,
    metric:        str = "q2",
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    field_names:   Optional[List[str]] = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """Grouped bar chart: final Q² or RRMSE per model, one subplot per field.

    Models are grouped by scenario (prefix, fixed_idx).
    Bar height = mean over seeds; error bar = std over seeds.
    """
    records = _collect_final_records(ana, model_types, fixed_indices)
    if not records:
        print(f"[plot_final_metrics] No records found.")
        return plt.figure()

    key    = "q2"   if metric == "q2"   else "rrmse"
    ylabel = "Q²"   if metric == "q2"   else "RRMSE"
    Q      = len(records[0][key])
    names  = field_names or FIELD_NAMES[:Q]
    fsize  = figsize or (5 * Q, 4)
    scale  = _compute_scale(fsize, Q, 1)
    fs     = _font_sizes(scale)

    # Gather unique (prefix, fixed_idx) scenarios
    scenarios = sorted({(r["prefix"], r["fixed_idx"]) for r in records})
    styles    = _build_scenario_styles(records)
    n_scen    = len(scenarios)
    bar_w     = 0.9 / n_scen

    fig, axes = plt.subplots(1, Q, figsize=fsize, squeeze=False)

    for ci in range(Q):
        ax = axes[0, ci]
        for si, (prefix, p_idx) in enumerate(scenarios):
            sub = [r for r in records
                   if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
            vals = np.array([r[key][ci] for r in sub])
            mean = vals.mean()
            std  = vals.std() if len(vals) > 1 else 0.0
            x    = si + (si - (n_scen-1)/2.0) * bar_w * 0.0   # single position
            x    = si

            st  = styles[(prefix, p_idx)]
            lbl = prefix if p_idx == -1 else f"{prefix}\np={p_idx}"
            ax.bar(si, mean, bar_w * 0.85,
                   color=st["color"], alpha=st["alpha"],
                   linewidth=0.8*scale, edgecolor="white",
                   zorder=3)
            ax.errorbar(si, mean, yerr=std,
                        fmt="none", color="black",
                        elinewidth=1.0*scale, capsize=4*scale,
                        capthick=0.8*scale, zorder=5)

        ax.set_xticks(range(n_scen))
        ax.set_xticklabels(
            [pfx if p == -1 else f"{pfx}\np={p}" for pfx, p in scenarios],
            fontsize=fs["tick"], fontweight="bold", rotation=45, ha="right",
        )
        ax.set_title(names[ci], fontsize=fs["title"], fontweight="bold")
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=fs["label"], fontweight="bold")
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ax in axes.flatten():
        _apply_bold_ticks(ax, fs)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Constraint error bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_constraint_error(
    ana:           MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """Bar chart of max constraint error per model (mean ± std over seeds)."""
    records = _collect_final_records(ana, model_types, fixed_indices)
    if not records:
        return plt.figure()

    scenarios = sorted({(r["prefix"], r["fixed_idx"]) for r in records})
    styles    = _build_scenario_styles(records)
    n_scen    = len(scenarios)

    fsize  = figsize or (max(6, n_scen * 0.9), 4)
    scale  = _compute_scale(fsize, 1, 1)
    fs     = _font_sizes(scale)

    fig, ax = plt.subplots(figsize=fsize)

    for si, (prefix, p_idx) in enumerate(scenarios):
        sub  = [r for r in records
                if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
        vals = np.array([r["constraint_max"] for r in sub])
        mean = vals.mean()
        std  = vals.std() if len(vals) > 1 else 0.0
        st   = styles[(prefix, p_idx)]

        ax.bar(si, mean, 0.65,
               color=st["color"], alpha=st["alpha"],
               edgecolor="white", linewidth=0.8*scale, zorder=3)
        ax.errorbar(si, mean, yerr=std,
                    fmt="none", color="black",
                    elinewidth=1.0*scale, capsize=4*scale,
                    capthick=0.8*scale, zorder=5)

    ax.set_xticks(range(n_scen))
    ax.set_xticklabels(
        [pfx if p == -1 else f"{pfx}\np={p}" for pfx, p in scenarios],
        fontsize=fs["tick"], fontweight="bold", rotation=45, ha="right",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Max constraint error  |u·f|", fontsize=fs["label"], fontweight="bold")
    ax.set_title("Constraint satisfaction", fontsize=fs["title"], fontweight="bold")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _apply_bold_ticks(ax, fs)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Latent GP Q² per mode
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_q2(
    ana:           MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    field_names:   Optional[List[str]] = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """Line plot: latent GP Q² per mode m, aggregated over seeds.

    One subplot per field (or per output component for RC).
    """
    # Collect latent_q2 from intermediate
    scene_data: Dict[tuple, List[np.ndarray]] = {}   # (prefix, p_idx) → list of (M, q) arrays

    for seed_i, (seed, single_ana) in enumerate(zip(ana._seeds, ana._analyzers)):
        for name in single_ana.list_models():
            res   = single_ana.storage.load_model_result_light(name)
            cfg   = res.get("config", {})
            inter = res.get("intermediate", {})
            lq2   = inter.get("latent_q2")
            if lq2 is None:
                continue

            prefix    = name.split("_")[0]
            fixed_idx = int(cfg.get("fixed_idx", -1))

            if model_types   is not None and prefix    not in model_types:
                continue
            if fixed_indices is not None and fixed_idx not in fixed_indices:
                continue

            key = (prefix, fixed_idx)
            scene_data.setdefault(key, []).append(np.array(lq2))  # (M, q)

    if not scene_data:
        print("[plot_latent_q2] No latent_q2 data found.")
        return plt.figure()

    # Q = number of fields shown = columns
    # Use field_names length to set Q; fall back to 3
    Q     = len(field_names or FIELD_NAMES)
    names = field_names or FIELD_NAMES[:Q]

    scenarios = sorted(scene_data.keys())
    styles    = _build_scenario_styles(
        [{"prefix": p, "fixed_idx": i} for p, i in scenarios])
    n_scen    = len(scenarios)

    fsize  = figsize or (5 * Q, 4)
    scale  = _compute_scale(fsize, Q, 1)
    fs     = _font_sizes(scale)

    fig, axes = plt.subplots(1, Q, figsize=fsize, sharey=True, squeeze=False)

    for (prefix, p_idx), arrays in scene_data.items():
        # arrays: list over seeds of (M, q) arrays
        # For RC: q=Q (one per field); for CI: q=1 (scalar per mode)
        # For FI/FM: q depends on scenario
        st      = styles[(prefix, p_idx)]
        is_ours = _PREFIX_STYLE.get(prefix, ("",)*6)[5]

        # Stack over seeds: (n_seeds, M, q)
        arr3 = np.array(arrays)   # (S, M, q)
        M    = arr3.shape[1]
        q    = arr3.shape[2]
        x    = np.arange(1, M+1)

        for ci in range(Q):
            if ci >= q and prefix != "RC":
                continue   # this component not available for this model
            ax    = axes[0, ci]
            qi    = ci if q > 1 else 0   # RC: q=Q; CI: q=1; FI: q varies

            # Mean & std over seeds for this field component
            col_vals = arr3[:, :, min(qi, q-1)]   # (n_seeds, M)
            means    = col_vals.mean(axis=0)
            stds     = col_vals.std(axis=0) if col_vals.shape[0] > 1 else np.zeros(M)

            lbl = prefix if p_idx == -1 else f"{prefix} (p={p_idx})"
            ax.errorbar(
                x, means, yerr=stds,
                color=st["color"], alpha=st["alpha"],
                linestyle=st["ls"],
                linewidth=st["lw"] * scale,
                marker=st["marker"],
                markersize=(5 if is_ours else 3) * scale,
                elinewidth=0.7*scale, capsize=2.5*scale,
                zorder=st["zorder"],
                label=lbl,
            )

    for ci in range(Q):
        ax = axes[0, ci]
        ax.set_title(names[ci], fontsize=fs["title"], fontweight="bold")
        ax.set_xlabel("Mode  m", fontsize=fs["label"], fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Latent Q²", fontsize=fs["label"], fontweight="bold")
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(range(1, max(1, M)+1) if 'M' in dir() else [1])

    plt.tight_layout()
    for ax in axes.flatten():
        _apply_bold_ticks(ax, fs)
        handles, labels_ = ax.get_legend_handles_labels()
        by_lbl = dict(zip(labels_, handles))
        if by_lbl:
            ax.legend(by_lbl.values(), by_lbl.keys(),
                      fontsize=fs["legend"],
                      prop={"size": fs["legend"], "weight": "bold"})

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all figures at once
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    ana:           MultiSeedAnalyzer,
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    field_names:   Optional[List[str]] = None,
    outdir:        Optional[str]       = "results_cfd",
    show:          bool                = False,
) -> dict:
    """Run all analysis figures and optionally save them.

    Parameters
    ----------
    ana          : MultiSeedAnalyzer built from CFD zarr files
    model_types  : filter prefixes, e.g. ["RC", "FI", "CI"]
    fixed_indices: filter fixed_idx values
    field_names  : overrides FIELD_NAMES
    outdir       : directory to save PNG figures; None = don't save
    show         : call plt.show() at the end

    Returns
    -------
    dict of figure objects keyed by name
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    def _path(name):
        return os.path.join(outdir, name) if outdir else None

    fnames = field_names or FIELD_NAMES

    # Summary
    print("=" * 70)
    print("SUMMARY — CFD benchmark")
    print("=" * 70)
    ana.print_summary()
    print()

    figs = {}

    figs["cum_q2"] = plot_cumulative_metric(
        ana, metric="q2",
        model_types=model_types, fixed_indices=fixed_indices,
        field_names=fnames,
        figsize=(5 * len(fnames), 4),
        output_path=_path("cfd_cumulative_q2.pdf"),
    )

    figs["cum_rrmse"] = plot_cumulative_metric(
        ana, metric="rrmse",
        model_types=model_types, fixed_indices=fixed_indices,
        field_names=fnames,
        figsize=(5 * len(fnames), 5),
        output_path=_path("cfd_cumulative_rrmse.pdf"),
    )

    figs["final_q2"] = plot_final_metrics(
        ana, metric="q2",
        model_types=model_types, fixed_indices=fixed_indices,
        field_names=fnames,
        figsize=(5 * len(fnames), 4),
        output_path=_path("cfd_final_q2.pdf"),
    )

    figs["final_rrmse"] = plot_final_metrics(
        ana, metric="rrmse",
        model_types=model_types, fixed_indices=fixed_indices,
        field_names=fnames,
        figsize=(5 * len(fnames), 5),
        output_path=_path("cfd_final_rrmse.pdf"),
    )

    figs["constraint"] = plot_constraint_error(
        ana,
        model_types=model_types, fixed_indices=fixed_indices,
        figsize=(8, 4),
        output_path=_path("cfd_constraint_error.pdf"),
    )

    figs["latent_q2"] = plot_latent_q2(
        ana,
        model_types=model_types, fixed_indices=fixed_indices,
        field_names=fnames,
        figsize=(5 * len(fnames), 4),
        output_path=_path("cfd_latent_q2.pdf"),
    )

    if show:
        plt.show()

    return figs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="CFD benchmark post-processing")
    p.add_argument("--prefix", default="results_n=50_cfd",
                   help="Zarr file prefix (default: results_cfd)")
    p.add_argument("--seeds", type=int, nargs="+", default=[51,52,53],
                   help="Seed list (default: 1 2)")
    p.add_argument("--model_types", nargs="+", default=None,
                   help="Prefixes to include e.g. RC FI CI FM")
    p.add_argument("--fixed_indices", type=int, nargs="+", default=None,
                   help="Fixed output indices to include")
    p.add_argument("--outdir", default="results_cfd",
                   help="Directory to save figures")
    p.add_argument("--no_show", action="store_true",
                   help="Do not call plt.show()")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    zarr_files = [f"{args.prefix}_seed{s}.zarr" for s in args.seeds]
    print(f"Loading: {zarr_files}")
    ana = MultiSeedAnalyzer(zarr_files)

    run_all(
        ana,
        model_types=args.model_types,
        fixed_indices=args.fixed_indices,
        outdir=args.outdir,
        show=not args.no_show,
    )
