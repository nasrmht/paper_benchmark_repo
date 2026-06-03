"""
Comparative analysis across multiple training-set sizes (N=30 / N=15 / N=10).

Results are loaded from .pkl files produced by run_lotka_volterra.py:

  results/lotka_volterra/results_N_=30_lv_seed{1..10}.pkl  →  n_train = 30
  results/lotka_volterra/results_N_=15_lv_seed{1..10}.pkl  →  n_train = 15
  results/lotka_volterra/results_N_=10_lv_seed{1..10}.pkl  →  n_train = 10

Layout of the combined figures (one per metric: Q² and RRMSE):

    row 0 (top)    → first entry in `analyzers`  (pass N=30 first)
    row 1 (middle) → second entry                (e.g. N=15)
    row 2 (bottom) → third entry                 (e.g. N=10)
    columns        → one per output field (p, q, r, s)

The y-axis is shared within each column (`sharey='col'`) so that the
difference between training sizes is shown on an identical scale.

Visual encoding:
  - Our model (RC)  : bold solid red line, drawn on top.
  - Baselines (CI/FI/FM) : thin non-solid lines, muted colours.
  - Small x-jitter per scenario so error bars don't overlap.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import colorsys
import numpy as np

# Default results directory (relative to this script)
_RESULTS_LV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'results', 'lotka_volterra')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["mathtext.default"] = "bf"

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Style tables
# ─────────────────────────────────────────────────────────────────────────────

# RC (row-wise / PCA-CGP) — bold red, always in front
C_RC  = "#D62728"
M_RC  = "o"
LS_RC = "-"
LW_RC = 2.6
MS_RC = 6
A_RC  = 1.00
Z_RC  = 10

# CI (col-wise indep.) — blue shades, l=0→3 lighten
C_CI  = {0: "#ff7f00", 1: "#ff9933", 2: "#ffb266", 3: "#ffcc99"} 
M_CI  = "s"
LS_CI = "--"
LW_CI = 1.3
MS_CI = 3
A_CI  = 0.82
Z_CI  = 6

# FI (field-wise indep.) — green shades, l=0→3 lighten
C_FI  = {0: "#417C03", 1: "#519204", 2: "#61A805", 3: "#71BE06"}
M_FI  = "^"
LS_FI = ":"
LW_FI = 1.3
MS_FI = 3
A_FI  = 0.82
Z_FI  = 5

# FM (field-wise MOGP-LCM) — muted purple shades
C_FM  = {0: "#5e239d", 1: "#6f29b0", 2: "#802fbf", 3: "#9135ce"} 
M_FM  = "D"
LS_FM = "-."
LW_FM = 1.3
MS_FM = 3
A_FM  = 0.82
Z_FM  = 4

# jitter window shared across all scenarios in a subplot
_JITTER = 0.20    # total spread in x units across all scenarios at one k-value


def _build_scenario_styles(records: list) -> dict:
    """
    Returns styles[(prefix, fixed_idx)] = dict(color, ls, lw, alpha, marker, zorder).
    """
    by_prefix: dict = {}
    for r in records:
        by_prefix.setdefault(r["prefix"], set()).add(r["fixed_idx"])

    styles = {}
    for pfx, p_set in by_prefix.items():
        for p in p_set:
            if pfx == "RC":
                st = dict(color=C_RC, ls=LS_RC, lw=LW_RC, alpha=A_RC, marker=M_RC, zorder=Z_RC)
            elif pfx == "CI":
                color = C_CI.get(p, C_CI[0])
                st = dict(color=color, ls=LS_CI, lw=LW_CI, alpha=A_CI, marker=M_CI, zorder=Z_CI)
            elif pfx == "FI":
                color = C_FI.get(p, C_FI[0])
                st = dict(color=color, ls=LS_FI, lw=LW_FI, alpha=A_FI, marker=M_FI, zorder=Z_FI)
            elif pfx == "FM":
                color = C_FM.get(p, C_FM[0])
                st = dict(color=color, ls=LS_FM, lw=LW_FM, alpha=A_FM, marker=M_FM, zorder=Z_FM)
            else:
                st = dict(color="#888888", ls="--", lw=1.3, alpha=0.8, marker="o", zorder=3)
            styles[(pfx, p)] = st
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_records(
    ana: MultiSeedAnalyzer,
    n_train_label: str,
    model_types: Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    n_modes_filter: Optional[int] = None,
) -> List[dict]:
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
            if n_modes_filter is not None and n_modes != n_modes_filter:
                continue

            records.append({
                "seed":          seed,
                "model_name":    name,
                "prefix":        prefix,
                "n_modes":       n_modes,
                "fixed_idx":     fixed_idx,
                "cum_q2":        np.array(cum_q2),
                "cum_rrmse":     np.array(cum_rrmse),
                "n_train_label": n_train_label,
            })
    return records


def _pick_n_modes(records: List[dict], n_modes_filter: Optional[int]):
    if not records:
        return records, None
    all_n = sorted({r["n_modes"] for r in records if not np.isnan(r["n_modes"])})
    if n_modes_filter is not None:
        return [r for r in records if r["n_modes"] == n_modes_filter], n_modes_filter
    if len(all_n) > 1:
        chosen = all_n[-1]
        print(f"  Multiple n_modes {all_n} — using n_modes={chosen}. "
              "Pass n_modes= to override.")
        return [r for r in records if r["n_modes"] == chosen], chosen
    return records, (all_n[0] if all_n else None)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_scale(figsize: tuple, Q: int, n_rows: int) -> float:
    """
    Scale factor relative to a reference cell of 5 × 4 inches.
    Drives fonts, line widths and marker sizes so they remain readable
    regardless of the requested figsize.
    """
    cell_w = figsize[0] / Q
    cell_h = figsize[1] / n_rows
    return max(0.45, min(cell_w / 5.0, cell_h / 4.0))


def _font_sizes(scale: float) -> dict:
    """Return a dict of absolute font sizes derived from `scale`."""
    s = scale
    return {
        "tick":   max(11,  round(10 * s)),
        "label":  max(11,  round(11 * s)),
        "title":  max(11,  round(12 * s)),
        "annot":  max(11,  round(11 * s)),
        "legend": max(11,  round( 9 * s)),
    }


def _apply_bold_ticks(ax, fs: dict) -> None:
    """Make all tick labels bold and correctly sized on `ax`."""
    ax.tick_params(axis="both", labelsize=fs["tick"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(fs["tick"])


# ─────────────────────────────────────────────────────────────────────────────
# Single-cell renderer
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_cell(ax, records, field_i, key, scenarios, styles, scale: float = 1.0):
    """
    Draw all curves for one (row, col) subplot.

    Scenarios are drawn from back (baseline) to front (RC) via zorder.
    A small x-jitter separates error bars so they don't overlap.
    All line widths, marker sizes and cap sizes scale with `scale`.
    """
    n_scen  = len(scenarios)
    offsets = (np.arange(n_scen) - (n_scen - 1) / 2.0) * (_JITTER / max(n_scen - 1, 1))

    def _sort_key(sc):
        return 0 if sc[0] == "RC" else 1

    draw_order = sorted(range(n_scen), key=lambda i: _sort_key(scenarios[i]))

    M_global = 1
    for ord_i in draw_order:
        prefix, p_idx = scenarios[ord_i]
        sub = [r for r in records
               if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
        if not sub:
            continue

        by_seed: dict = {}
        for r in sub:
            by_seed.setdefault(r["seed"], []).append(r[key][:, field_i])

        seed_vals = np.array(
            [np.mean(vs, axis=0) for vs in by_seed.values()])   # (S, M)
        M_cur    = seed_vals.shape[1]
        M_global = max(M_global, M_cur)
        x        = np.arange(1, M_cur + 1, dtype=float) + offsets[ord_i]

        means = seed_vals.mean(axis=0)
        stds  = seed_vals.std(axis=0) if seed_vals.shape[0] > 1 \
                else np.zeros(M_cur)

        st       = styles[(prefix, p_idx)]
        is_ours  = (prefix == "RC")
        if prefix == "RC":
            lbl = "Row-CMO"
        elif prefix == "CI":
            lbl = f"Col-Indep (l={p_idx})"
        elif prefix == "FI":
            lbl = f"Fw-Indep (l={p_idx})"
        elif prefix == "FM":
            lbl = f"Fw-LCM (l={p_idx})"
        else:
            lbl = f"{prefix} (l={p_idx})"

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

    ax.set_xticks(range(1, M_global + 1))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Core 2-row figure builder
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metric_rows(
    all_records:    List[dict],
    n_train_labels: List[str],
    metric:         str,
    n_modes:        Optional[int] = None,
    figsize:        Optional[tuple] = None,
    output_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Grid: n_rows × Q subplots.
      - Each row    = one n_train class (top = n_train_labels[0]).
      - Each column = one output field.
      - sharey='col': same y-scale for the two rows of a column.
    """
    if not all_records:
        print(f"[_plot_metric_rows] No records for metric={metric}.")
        return plt.figure()

    key    = "cum_q2"  if metric == "q2"   else "cum_rrmse"
    ylabel = "Q²"      if metric == "q2"   else "RRMSE"

    Q      = all_records[0][key].shape[1]
    n_rows = len(n_train_labels)
    fsize  = figsize or (5 * Q, 4 * n_rows)

    # ── Scale factor: drives all sizes consistently with figsize ──────────────
    scale = _compute_scale(fsize, Q, n_rows)
    fs    = _font_sizes(scale)

    fig, axes = plt.subplots(
        n_rows, Q,
        figsize=fsize,
        sharey="col",
        sharex=True,
        squeeze=False,
    )

    scenarios = sorted({(r["prefix"], r["fixed_idx"]) for r in all_records})
    styles    = _build_scenario_styles(all_records)
    liste_output = ['p','q', 'r', 's']
    for row_i, n_label in enumerate(n_train_labels):
        row_recs = [r for r in all_records if r["n_train_label"] == n_label]

        for col_i in range(Q):
            ax = axes[row_i, col_i]
            _plot_one_cell(ax, row_recs, col_i, key, scenarios, styles, scale)

            if row_i == 0:
                ax.set_title(f"{liste_output[col_i]}", fontsize=fs["title"], fontweight="bold")

            if col_i == 0:
                ax.set_ylabel(
                    f"{n_label}\n{ylabel}",
                    fontsize=fs["label"], fontweight="bold", labelpad=8,
                )

            if row_i == n_rows - 1:
                ax.set_xlabel("Latent dimension  ", fontsize=fs["label"], fontweight="bold")

            ax.set_facecolor("#F8FAFC" if row_i % 2 == 0 else "#FFFEF8")

    ax.set_yscale("log")
    # ── Apply bold ticks after content is drawn ───────────────────────────────
    # (tight_layout is called first so tick labels are instantiated)
    plt.tight_layout()
    for ax in axes.flatten():
        _apply_bold_ticks(ax, fs)

    # Right-margin row annotation
    for row_i, n_label in enumerate(n_train_labels):
        axes[row_i, -1].annotate(
            n_label,
            xy=(1.04, 0.5), xycoords="axes fraction",
            fontsize=fs["annot"], fontweight="bold", color="dimgray",
            va="center", ha="left", rotation=270,
        )

    # Dashed separator between rows
    if n_rows > 1:
        sep_y = axes[1, 0].get_position().y1 + 0.006
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [sep_y, sep_y],
            transform=fig.transFigure,
            color="silver", linewidth=max(0.5, 0.8 * scale), linestyle="--",
        ))

    # ── Legend — 4 colonnes : RC | CI | FI | FM ──

    def _leg_handle(pfx, p, boost=False):
        st  = styles[(pfx, p)]
        if pfx == "RC":
            lbl = "Row-CMO"
        elif pfx == "CI":
            lbl = f"Col-Indep (l={p})"
        elif pfx == "FI":
            lbl = f"Fw-Indep (l={p})"
        elif pfx == "FM":
            lbl = f"Fw-LCM (l={p})"
        else:
            lbl = f"{pfx} (l={p})"
            
        return mlines.Line2D([], [],
                             color=st["color"], linestyle=st["ls"],
                             linewidth=st["lw"] * scale * (1.4 if boost else 1.0),
                             marker=st["marker"],
                             markersize=max(3, 5 * scale) * (1.5 if boost else 1.0),
                             alpha=st["alpha"],
                             label=lbl)

    def _colmajor(items, ncol):
        """Permute items so that the row-major display of matplotlib
        visually gives ncol columns filled from top to bottom."""
        if not items:
            return []
        nrow = -(-len(items) // ncol)     # ceil division
        return [items[j * nrow + i]
                for i in range(nrow) for j in range(ncol)
                if j * nrow + i < len(items)]

    _leg_kw = dict(frameon=True, edgecolor="black",
                   bbox_transform=fig.transFigure,
                   fontsize=fs["legend"],
                   prop={"size": fs["legend"], "weight": "bold"})

    rc_pairs = sorted((pfx, p) for pfx, p in scenarios if pfx == "RC")
    ci_pairs = _colmajor(sorted([(pfx, p) for pfx, p in scenarios if pfx == "CI"], key=lambda x: x[1]), ncol=2)
    fi_pairs = _colmajor(sorted([(pfx, p) for pfx, p in scenarios if pfx == "FI"], key=lambda x: x[1]), ncol=2)
    fm_pairs = _colmajor(sorted([(pfx, p) for pfx, p in scenarios if pfx == "FM"], key=lambda x: x[1]), ncol=2)

    for pairs, x_pos, boost, ncol in [
        (rc_pairs, 0.12, False, 1),
        (ci_pairs, 0.32, False, min(2, max(1, len(ci_pairs)))),
        (fi_pairs, 0.6, False, min(2, max(1, len(fi_pairs)))),
        (fm_pairs, 0.87, False, min(2, max(1, len(fm_pairs)))),
    ]:
        if not pairs:
            continue
        leg = fig.legend(
            handles=[_leg_handle(pfx, p, boost) for pfx, p in pairs],
            loc="lower center", bbox_to_anchor=(x_pos, -0.02),
            ncol=ncol, columnspacing=0.8, **_leg_kw)

    plt.subplots_adjust(bottom=max(0.14, 0.20 * scale))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_q2(
    analyzers:     List[Tuple[MultiSeedAnalyzer, str]],
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    n_modes:       Optional[int]       = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """
    Q² figure: n_rows × Q subplots, sharey per column.

    Parameters
    ----------
    analyzers : ordered list of (MultiSeedAnalyzer, label).
                Row order = list order. Pass N=30 first for top row.
    model_types   : prefixes to include e.g. ["RC", "FI", "CI"]. None = all.
    fixed_indices : fixed_idx values to include. None = all.
    n_modes       : filter on training n_modes. None = use largest.
    """
    all_records = []
    for ana, label in analyzers:
        all_records.extend(
            _collect_records(ana, label, model_types, fixed_indices, n_modes))

    all_records, chosen_n = _pick_n_modes(all_records, n_modes)
    return _plot_metric_rows(
        all_records, [lbl for _, lbl in analyzers],
        metric="q2", n_modes=chosen_n,
        figsize=figsize, output_path=output_path,
    )


def plot_combined_rmse(
    analyzers:     List[Tuple[MultiSeedAnalyzer, str]],
    model_types:   Optional[List[str]] = None,
    fixed_indices: Optional[List[int]] = None,
    n_modes:       Optional[int]       = None,
    figsize:       Optional[tuple]     = None,
    output_path:   Optional[str]       = None,
) -> plt.Figure:
    """RRMSE figure: n_rows × Q subplots. Same signature as plot_combined_q2."""
    all_records = []
    for ana, label in analyzers:
        all_records.extend(
            _collect_records(ana, label, model_types, fixed_indices, n_modes))

    all_records, chosen_n = _pick_n_modes(all_records, n_modes)
    return _plot_metric_rows(
        all_records, [lbl for _, lbl in analyzers],
        metric="rmse", n_modes=chosen_n,
        figsize=figsize, output_path=output_path,
    )


# =============================================================================
# PCA COMPARISON ANALYSIS  ( Theorem 3)
#
# Reproduces the analysis from numerical_application_pca.py but:
#   - data comes from zarr benchmark files (ground-truth training fields)
#   - metrics are averaged over seeds (mean ± std shown as shaded bands)
#   - two n_train classes are shown side by side on the same subplots
# =============================================================================

# ── Low-level math helpers ────────────────────────────────────────────────────

def _load_train_fields(single_ana, dt_override: float = None) -> dict:
    """Load raw training fields from one ResultsAnalyzer.

    Returns {field_0: ndarray(n_train, S), field_1: …, …}

    For zarr storage: fields are read directly from the archive.
    For pkl storage (fields not saved): regenerated from the dataset using the
    stored config (seed, n_train, n_total, t_end, dt).  Requires t_end/dt to
    be present in the config (runner stores them since the pkl migration).
    Fields are converted to float64 to avoid float32 precision artefacts
    in SVD / eigendecomposition (especially when S is large).

    Parameters
    ----------
    dt_override : when set, always regenerate trajectories using this dt
        instead of the config value.  Pass 0.05 for fast approximate
        loading (S=400 steps) when exact reconstruction is not required.
    """
    gt = single_ana.storage.load_ground_truth()
    if gt["fields_train"] is not None and dt_override is None:
        return {f"field_{i}": np.array(arr, dtype=np.float64)
                for i, arr in enumerate(gt["fields_train"])}

    # ── PKL backend or dt_override: regenerate from dataset config ───────────
    if hasattr(single_ana.storage, "_data"):
        cfg = single_ana.storage._data.get("config", {})
    else:
        cfg = dict(single_ana.storage.store["config"].attrs)

    dataset_name = cfg.get("dataset", "")
    seed    = int(cfg.get("seed",    0))
    n_train = int(cfg.get("n_train", 30))
    n_total = int(cfg.get("n_total", 100))
    t_end   = float(cfg["t_end"]) if cfg.get("t_end") is not None else 20.0
    # dt_override takes priority; fall back to config dt, then to 0.05.
    # Config dt=0.001 (original benchmark) takes >60s/seed — use dt_override=0.05
    # when exact fields are not required (e.g. Theorem 3 dominance plot).
    dt = (dt_override if dt_override is not None
          else (float(cfg["dt"]) if cfg.get("dt") is not None else 0.05))

    if "LotkaVolterra" in dataset_name:
        from benchmark_pca_gp.data.lotka_volterra import LotkaVolterraDataset
        dataset = LotkaVolterraDataset(t_end=t_end, dt=dt, n_train=n_train)
        X, fields = dataset.generate(n_total, seed)
        X_tr, X_te, f_tr, f_te = dataset.split_train_test(X, fields, n_train, seed)
        # Reproduce the z-space NC transform used by the runner before storage
        f_tr_z, _, means_z = dataset.center(f_tr, f_te, X_train=X_tr, X_test=X_te)
        Q = dataset.n_outputs
        f_tr_model = [f_tr_z[k] + means_z[k] for k in range(Q)]
        return {f"field_{i}": np.array(f, dtype=np.float64)
                for i, f in enumerate(f_tr_model)}

    raise RuntimeError(
        f"Cannot load training fields: pkl storage does not save fields_train "
        f"and dataset '{dataset_name}' has no regeneration support."
    )


def _center(Y_dict: dict) -> dict:
    return {k: Y - Y.mean(axis=0) for k, Y in Y_dict.items()}


def _svd(Y: np.ndarray):
    """SVD — returns (U, singular_values, V) where V has shape (S, r)."""
    U, D, Vt = np.linalg.svd(Y, full_matrices=False)
    return U, D, Vt.T


def _gram(Y: np.ndarray) -> np.ndarray:
    n, S = Y.shape
    return (1.0 / S) * (Y @ Y.T)    # (n, n)


def _eigengaps(M: np.ndarray):
    """Sorted eigenvalues (descending) and their successive gaps."""
    vals = np.maximum(np.linalg.eigvalsh(M)[::-1], 0.0)
    gaps = np.zeros_like(vals)
    if len(vals) > 1:
        gaps[:-1] = vals[:-1] - vals[1:]
    gaps[-1] = vals[-1]
    return vals, gaps


def _err_right(Y, V_m):
    """||Y − Y Vm Vm^T||_F²  (row-wise / field-wise projection)."""
    return float(np.sum((Y - (Y @ V_m) @ V_m.T) ** 2))


def _err_left(Y, U_m):
    """||Y − Um Um^T Y||_F²  (col-wise projection)."""
    return float(np.sum((Y - U_m @ (U_m.T @ Y)) ** 2))




# ── Theorem-3 ──────────────────────────────────────────────────

def _pca_metrics_one_seed_thm3(Y_raw: dict, max_m: int) -> dict:
    """
    Compute the quantities required for Theorem 3.
    """
    fields = sorted(Y_raw.keys())
    Q = len(fields)
    Y = _center(Y_raw)
    n, S = Y[fields[0]].shape

    T = sum(float(np.sum(Y[f] ** 2)) for f in fields)

    K_avg = sum(_gram(Y[f]) for f in fields) / Q

    Y_row = np.vstack([Y[f] for f in fields])   # (n*Q, S)
    V_row = _svd(Y_row)[2]
    U_col = _svd(np.hstack([Y[f] for f in fields]))[0]

    M_row    = (1.0 / (n * Q)) * (Y_row @ Y_row.T)
    vals_row = _eigengaps(M_row)[0]
    vals_col = _eigengaps(K_avg)[0]

    m_range = np.arange(1, max_m + 1)
    err_row = {f: np.zeros(max_m) for f in fields}
    err_col = {f: np.zeros(max_m) for f in fields}

    for i, m in enumerate(m_range):
        V_row_m = V_row[:, :m]
        U_col_m = U_col[:, :m]
        for f in fields:
            Yf = Y[f]
            err_row[f][i] = _err_right(Yf, V_row_m)
            err_col[f][i] = _err_left(Yf, U_col_m)

    total_row  = np.array([sum(err_row[f][i] for f in fields) for i in range(max_m)])
    total_col  = np.array([sum(err_col[f][i] for f in fields) for i in range(max_m)])
    diff_exact = total_col - total_row
    lambda_row = np.array([n * Q * np.sum(vals_row[:m]) for m in m_range])
    lambda_col = np.array([Q * S * np.sum(vals_col[:m]) for m in m_range])
    diff_tr    = lambda_row - lambda_col

    return dict(fields=fields, m_range=m_range, T=T,
                total_row=total_row, total_col=total_col,
                diff_exact=diff_exact,
                lambda_row=lambda_row, lambda_col=lambda_col,
                diff_tr=diff_tr)


def _agg_thm3(metrics_list: list) -> dict:
    """Mean ± std over seeds — Theorem 3 quantities only."""
    if not metrics_list:
        return {}
    ref    = metrics_list[0]
    result = {"fields": ref["fields"], "m_range": ref["m_range"]}

    T_arr = np.array([m["T"] for m in metrics_list])
    result["T_mean"] = float(T_arr.mean())
    result["T_std"]  = float(T_arr.std())

    for key in ("total_row", "total_col", "diff_exact", "diff_tr",
                "lambda_row", "lambda_col"):
        arr = np.array([m[key] for m in metrics_list])
        result[f"{key}_mean"] = arr.mean(axis=0)
        result[f"{key}_std"]  = arr.std(axis=0)
    return result


# ── Public data-collection entry point ───────────────────────────────────────

def compute_pca_metrics(
    analyzers:  List[Tuple[MultiSeedAnalyzer, str]],
    max_m:      int   = 10,
    thm3_only:  bool  = False,
    dt_fast:    float = None,
) -> dict:
    """Compute PCA comparison metrics for every n_train class.

    Parameters
    ----------
    analyzers  : list of (MultiSeedAnalyzer, label) in the same order used
                 for the combined GP-performance plots.
    max_m      : maximum number of PCA modes to evaluate.
    thm3_only  : skip Theorem 1 & 2 computations (much faster).
                 Returns only the fields needed for plot_dominance.
    dt_fast    : when set, override config dt for field regeneration.
                 Combine with thm3_only=True and dt_fast=0.05 for fast
                 approximate dominance curves (S=400 vs S=20000).

    Returns
    -------
    dict[label] → aggregated metrics dict  (mean ± std over seeds)
    """
    results = {}
    for ana, label in analyzers:
        per_seed = []
        for single_ana in ana._analyzers:
            try:
                Y_raw = _load_train_fields(single_ana, dt_override=dt_fast)
                metric_fn = _pca_metrics_one_seed_thm3 
                per_seed.append(metric_fn(Y_raw, max_m))
            except Exception as e:
                print(f"  Warning (seed skipped for {label}): {e}")
        agg_fn = _agg_thm3 
        results[label] = agg_fn(per_seed)
        print(f"  PCA metrics — {label}: {len(per_seed)} seeds aggregated.")
    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────

# Line-style encoding for n_train classes (reuses the same convention as the
# GP-performance plots so the figures look consistent).
_PCA_N_STYLE = {
    "N=30": dict(lw=2.2, ls="-",  alpha=1.00, marker="o"),
    "N=15": dict(lw=1.8, ls="-.",  alpha=1.00, marker="^"),
    "N=10": dict(lw=1.4, ls="--", alpha=0.80, marker="s"),
}

def _pca_n_style(label: str) -> dict:
    return _PCA_N_STYLE.get(label, dict(lw=1.6, ls="-.", alpha=0.85, marker="^"))


def _band(ax, x, mean, std, color, label, **kw):
    """Plot mean line + ±std shaded band."""
    ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def _finish_ax(ax, fs, xlabel="modes  m", yscale="log"):
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=fs["label"], fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _apply_bold_ticks(ax, fs)
    handles, labels_ = ax.get_legend_handles_labels()
    by_lbl = dict(zip(labels_, handles))
    if by_lbl:
        ax.legend(by_lbl.values(), by_lbl.keys(),
                  fontsize=fs["legend"], loc="upper right")




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # headless rendering

    # ── Data sources ─────────────────────────────────────────────────────────
    # Seeds 1..10 for each N.  Filter out missing files gracefully.
    def _pkl_paths(n: int):
        return [p for i in range(1, 11)
                if os.path.exists(
                    p := os.path.join(_RESULTS_LV,
                                      f"results_N_={n}_lv_seed{i}.pkl"))]

    list_n30 = _pkl_paths(30)
    list_n15 = _pkl_paths(15)
    list_n10 = _pkl_paths(10)

    for label, paths in [("N=30", list_n30), ("N=15", list_n15), ("N=10", list_n10)]:
        print(f"  {label}: {len(paths)} seeds found")

    ana_n30 = MultiSeedAnalyzer(list_n30)
    ana_n15 = MultiSeedAnalyzer(list_n15)
    ana_n10 = MultiSeedAnalyzer(list_n10)

    analyzers = [(ana_n30, "N=30"), (ana_n15, "N=15"), (ana_n10, "N=10")]

    for ana, label in analyzers:
        print("=" * 60)
        print(f"Summary  {label}")
        print("=" * 60)
        ana.print_summary()
        print()

    model_types = ["RC", "FI", "CI", "FM"]

    # ── RRMSE / Q² combined figures ───────────────────────────────────────────
    fig_q2 = plot_combined_q2(
        analyzers,
        model_types=model_types,
        output_path="combined_q2_vs_modes.pdf",
        figsize=(14, 9)
    )
    plt.close(fig_q2)

    fig_rmse = plot_combined_rmse(
        analyzers,
        model_types=model_types,
        output_path="combined_rmse_vs_modes.pdf",
        figsize=(14, 9)
    )
    plt.close(fig_rmse)

    print("\nDone — figures saved in the current directory.")
