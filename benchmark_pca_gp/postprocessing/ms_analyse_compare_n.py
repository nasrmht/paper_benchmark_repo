"""
Comparative analysis across two training-set sizes.

  results_lv11_seed*.zarr  →  n_train = 10
  results_lv10_seed*.zarr  →  n_train = 30

Layout of the combined figures (one per metric: Q² and RMSE):

    row 0 (top)    → first entry in `analyzers`  (pass N=30 first)
    row 1 (bottom) → second entry in `analyzers` (pass N=10 second)
    columns        → one per output field (f0, f1, …)

The y-axis is shared within each column (`sharey='col'`) so that the
difference between N=30 and N=10 is shown on an identical scale.

Visual encoding:
  - Our model (RC)  : bold solid line, thicker, vivid colour, drawn on top.
  - Baselines (CI/FI/FM) : thin non-solid lines, muted colours.
  - Small x-jitter per scenario so error bars don't overlap.
"""

import sys
sys.path.insert(0, '..')

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple

from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Style tables
# ─────────────────────────────────────────────────────────────────────────────

# Our model gets a vivid warm colour; baselines get cool/muted families.
_PREFIX_STYLE = {
    # prefix : (base_hex,   linestyle,  lw,   alpha,  marker, is_ours)
    "RC":     ("#D62728",   "-",        2.6,  1.00,   "o",    True),   # bold red
    "CI":     ("#6BAED6",   "--",       1.3,  0.82,   "s",    False),  # muted blue
    "FI":     ("#74C476",   ":",        1.3,  0.82,   "^",    False),  # muted green
    "FM":     ("#9E9AC8",   "-.",       1.3,  0.82,   "D",    False),  # muted purple
}

# jitter window shared across all scenarios in a subplot
_JITTER = 0.20    # total spread in x units across all scenarios at one k-value


# ─────────────────────────────────────────────────────────────────────────────
# Colour utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255))


def _lighten(hex_color: str, factor: float) -> str:
    """Mix hex_color with white by `factor` (0 = original, 1 = white)."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r + (1 - r) * factor,
                       g + (1 - g) * factor,
                       b + (1 - b) * factor)


def _build_scenario_styles(records: list) -> dict:
    """
    Returns styles[(prefix, fixed_idx)] = dict(color, ls, lw, alpha, marker, zorder).

    Within each prefix, scenarios with different fixed_idx are distinguished by
    progressively lighter shades of the base colour.
    """
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
            # Lighten progressively for i > 0 (deeper = lower index)
            shade = 0.0 + 0.35 * (i / max(n - 1, 1))   # 0 → 0.35 lightening
            color = base_hex if i == 0 else _lighten(base_hex, shade)
            styles[(pfx, p)] = dict(
                color=color,
                ls=ls,
                lw=lw,
                alpha=alpha,
                marker=marker,
                zorder=6 if is_ours else 3,
            )
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
        "tick":   max(7,  round(10 * s)),
        "label":  max(8,  round(11 * s)),
        "title":  max(9,  round(12 * s)),
        "annot":  max(8,  round(11 * s)),
        "legend": max(7,  round( 9 * s)),
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
        return 0 if _PREFIX_STYLE.get(sc[0], ("", "", 0, 0, "", False))[5] else 1

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
        is_ours  = _PREFIX_STYLE.get(prefix, ("",) * 6)[5]
        lbl      = prefix if p_idx == -1 else f"{prefix} (p={p_idx})"

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

    # ── Legend — 3 colonnes : RC seul (légèrement agrandi) | CI 2×2 | FI 2×2 ──

    def _leg_handle(pfx, p, boost=False):
        st  = styles[(pfx, p)]
        lbl = pfx if p == -1 else f"{pfx} (k_*={p})"
        if pfx == "RC":
            lbl += "  ★"
        return mlines.Line2D([], [],
                             color=st["color"], linestyle=st["ls"],
                             linewidth=st["lw"] * scale * (1.4 if boost else 1.0),
                             marker=st["marker"],
                             markersize=max(3, 5 * scale) * (1.5 if boost else 1.0),
                             alpha=st["alpha"],
                             label=lbl)

    def _colmajor(items, ncol):
        """Permute items so que l'affichage row-major de matplotlib
        donne visuellement ncol colonnes remplies de haut en bas."""
        nrow = -(-len(items) // ncol)     # ceil division
        return [items[j * nrow + i]
                for i in range(nrow) for j in range(ncol)
                if j * nrow + i < len(items)]

    _leg_kw = dict(frameon=True, edgecolor="black",
                   bbox_transform=fig.transFigure,
                   fontsize=fs["legend"],
                   prop={"size": fs["legend"], "weight": "bold"})

    rc_pairs = sorted((pfx, p) for pfx, p in scenarios if pfx == "RC")
    ci_pairs = _colmajor(sorted([(pfx, p) for pfx, p in scenarios if pfx == "CI"],
                                key=lambda x: x[1]), ncol=2)
    fi_pairs = _colmajor(sorted([(pfx, p) for pfx, p in scenarios if pfx == "FI"],
                                key=lambda x: x[1]), ncol=2)

    fig.suptitle(...)   # uncomment if a super-title is desired
    for pairs, x_pos, title, boost, ncol in [
        (rc_pairs, 0.12, "RC", True,  1),
        (ci_pairs, 0.45, "CI", False, 2),
        (fi_pairs, 0.78, "FI", False, 2),
    ]:
        if not pairs:
            continue
        leg = fig.legend(
            handles=[_leg_handle(pfx, p, boost) for pfx, p in pairs],
            loc="lower center", bbox_to_anchor=(x_pos, -0.02),
            ncol=ncol, **_leg_kw)
        #leg.set_title(title, prop={"size": fs["legend"] + 1, "weight": "bold"})

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
# PCA COMPARISON ANALYSIS  (Theorem 2 / Theorem 3 / Lemma 5)
#
# Reproduces the analysis from numerical_application_pca.py but:
#   - data comes from zarr benchmark files (ground-truth training fields)
#   - metrics are averaged over seeds (mean ± std shown as shaded bands)
#   - two n_train classes are shown side by side on the same subplots
# =============================================================================

# ── Low-level math helpers ────────────────────────────────────────────────────

def _load_train_fields(single_ana) -> dict:
    """Load raw training fields from one ResultsAnalyzer.

    Returns {field_0: ndarray(n_train, S), field_1: …, …}

    Fields are converted to float64 to avoid float32 precision artefacts
    in SVD / eigendecomposition (especially when S is large).
    """
    gt = single_ana.storage.load_ground_truth()
    return {f"field_{i}": np.array(arr, dtype=np.float64)
            for i, arr in enumerate(gt["fields_train"])}


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


def _sigma_op_diff(Vi: np.ndarray, di: np.ndarray,
                   Vj: np.ndarray, dj: np.ndarray, n: int) -> float:
    """||Σ_i − Σ_j||_op  where  Σ_k = (1/n) Y_k^T Y_k  (S×S).

    Uses rank-2r reduction:
      Y_i^T Y_i − Y_j^T Y_j = [Vi, Vj] · diag(di², −dj²) · [Vi, Vj]^T
    QR-decompose W = [Vi, Vj] (S × 2r) → W = QR, then
      ||op|| = max|eig(R · Ω · R^T)| / n ,  Ω = diag(di², −dj²).
    """
    W = np.hstack([Vi, Vj])                          # (S, 2r)
    _, R = np.linalg.qr(W, mode="reduced")           # R: (2r, 2r)
    Omega = np.diag(np.concatenate([di ** 2, -dj ** 2]))
    M = R @ Omega @ R.T
    return float(np.max(np.abs(np.linalg.eigvalsh(M)))) / n


def _pca_metrics_one_seed(Y_raw: dict, max_m: int) -> dict:
    """All PCA comparison metrics for one seed.

    Implements the three theorems from the theoretical analysis:

    Theorem 1 (row-wise excess error bound):
        ΔE^row_{k,m} ≤ (2n√2 ‖Σ_k‖_F / δ_Σ_k) · min(√m/Q Σ_j ‖Σ_j−Σ_k‖_op,
                                                         1/Q  Σ_j ‖Σ_j−Σ_k‖_F)
        where Σ_k = (1/n) Y_k^T Y_k  (S×S spatial covariance)
        and δ_Σ_k = m-th spectral gap of Σ_k.

        Note: ||Σ_k||_F = (S/n)·||K_k||_F  (same numeric value via trace cycling),
        but pairwise differences ||Σ_j−Σ_k||_F  ≠  (S/n)·||K_j−K_k||_F  in general
        (different eigenvectors). Computed correctly via n×n cross-products.

    Theorem 2 (col-wise excess error bound):
        ΔE^col_{k,m} ≤ (2n√2 ‖K_k‖_F / δ_k) · min(√m/Q Σ_j ‖K_j−K_k‖_op,
                                                       1/Q  Σ_j ‖K_j−K_k‖_F)
        where K_k = (1/S) Y_k Y_k^T  (n×n Gram matrix) and δ_k its m-th spectral gap.

    Theorem 3 (col-wise vs row-wise dominance):
        E^row_m < E^col_m  ⟺  Λ_{row,m} > Λ_{col,m}
        where Λ_{row,m} = nQ Σ_{i≤m} λ_i^{row},  Λ_{col,m} = QS Σ_{i≤m} λ_i^{col}.
    """
    fields = sorted(Y_raw.keys())
    Q = len(fields)
    Y = _center(Y_raw)
    n, S = Y[fields[0]].shape

    # K_k = (1/S) Y_k Y_k^T  (n×n Gram matrix, used in Theorems 1 & 2)
    K     = {f: _gram(Y[f]) for f in fields}
    K_avg = sum(K[f] for f in fields) / Q

    # T = total squared norm (same whether computed from Y_all or Y_all_h)
    T = sum(float(np.sum(Y[f] ** 2)) for f in fields)

    # Σ_k = (1/n) Y_k^T Y_k  is an S×S spatial covariance matrix.
    # K_k = (1/S) Y_k Y_k^T  is an n×n Gram matrix.
    # These are DIFFERENT matrices (different sizes, different eigenvectors),
    # so pairwise differences are NOT simply related by (S/n).
    #
    # Key identities (via trace cycling):
    #   ||Σ_k||_F = (1/n)||Y_k Y_k^T||_F  =  (S/n)·||K_k||_F  ← same numeric value
    #   ||Σ_j−Σ_k||_F² = (1/n²)(||Y_j Y_j^T||_F² − 2||Y_j Y_k^T||_F² + ||Y_k Y_k^T||_F²)
    #   ||Σ_j−Σ_k||_op via QR trick on [Vj, Vk]  (see _sigma_op_diff)
    scale_Sn = S / n   # kept only for gaps_Sigma (eigenvalue scaling)

    # Precompute n×n Gram products for Σ Frobenius distances
    self_fro2  = {f: float(np.sum((Y[f] @ Y[f].T) ** 2)) for f in fields}
    cross_fro2 = {(j, k): float(np.sum((Y[j] @ Y[k].T) ** 2))
                  for j in fields for k in fields}

    # Field-wise PCA — store singular values too (needed for Σ op-norm differences)
    V_fw, D_fw = {}, {}
    for f in fields:
        _, D_fw[f], V_fw[f] = _svd(Y[f])   # D_fw[f]: (r,), V_fw[f]: (S, r)

    # Row-wise PCA  — stack vertically: (n*Q, S)
    V_row = _svd(np.vstack([Y[f] for f in fields]))[2]

    # Col-wise PCA  — stack horizontally: (n, S*Q)
    U_col = _svd(np.hstack([Y[f] for f in fields]))[0]

    # Eigenvalues for Theorem 3 trace formula
    M_row    = (1.0 / (n * Q)) * (np.vstack([Y[f] for f in fields]) @
                                   np.vstack([Y[f] for f in fields]).T)
    vals_row = _eigengaps(M_row)[0]
    vals_col = _eigengaps(K_avg)[0]

    # Spectral gaps of K_k (Theorem 2) and Σ_k (Theorem 1)
    vals_K, gaps_K = {}, {}
    for f in fields:
        vK, gK    = _eigengaps(K[f])
        vals_K[f] = vK
        gaps_K[f] = gK
    # gaps_Sigma[f] = (S/n) * gaps_K[f]  (eigenvalues of Sigma scale by S/n)
    gaps_Sigma = {f: scale_Sn * gaps_K[f] for f in fields}

    # ── Pairwise K distances (Theorem 2) ─────────────────────────────────────
    k_norm_f  = {f: np.linalg.norm(K[f], "fro") for f in fields}
    sum_fro_K = {f: sum(np.linalg.norm(K[j]-K[f],"fro") for j in fields)/Q
                 for f in fields}
    sum_op_K  = {f: sum(np.linalg.norm(K[j]-K[f], ord=2) for j in fields)/Q
                 for f in fields}

    # ── Pairwise Σ distances (Theorem 1) — correct formulas ─────────────────
    sig_norm_f  = {f: np.sqrt(self_fro2[f]) / n for f in fields}  # ||Σ_k||_F
    sum_fro_Sig = {
        f: sum(
            np.sqrt(max(
                (self_fro2[j] - 2 * cross_fro2[(j, f)] + self_fro2[f]) / n ** 2,
                0.0))
            for j in fields) / Q
        for f in fields}
    sum_op_Sig  = {
        f: sum(
            _sigma_op_diff(V_fw[j], D_fw[j], V_fw[f], D_fw[f], n)
            for j in fields) / Q
        for f in fields}

    m_range = np.arange(1, max_m + 1)
    err_fw  = {f: np.zeros(max_m) for f in fields}
    err_row = {f: np.zeros(max_m) for f in fields}
    err_col = {f: np.zeros(max_m) for f in fields}
    bnd_row = {f: np.zeros(max_m) for f in fields}
    bnd_col = {f: np.zeros(max_m) for f in fields}

    for i, m in enumerate(m_range):
        V_row_m = V_row[:, :m]
        U_col_m = U_col[:, :m]

        for f in fields:
            Yf = Y[f]
            err_fw[f][i]  = _err_right(Yf, V_fw[f][:, :m])
            err_row[f][i] = _err_right(Yf, V_row_m)
            err_col[f][i] = _err_left(Yf, U_col_m)

            # ── Theorem 1: row-wise bound using Σ_k = (S/n)·K_k ──────────
            # 2n√2 · ‖Σ_k‖_F / δ_Σ_k · min(√m/Q·Σ‖Σ_j−Σ_k‖_op, 1/Q·Σ‖Σ_j−Σ_k‖_F)
            # The (S/n) factor cancels in ‖Σ_k‖_F/δ_Σ_k and appears once in min_term,
            # making bnd_row numerically equal to (2S√2·‖K_k‖_F/δ_k)·min_term_K.
            sig_norm   = sig_norm_f[f]
            delta_sig  = max(gaps_Sigma[f][m - 1], 1e-12)
            min_term_sig = min(np.sqrt(m) * sum_op_Sig[f], sum_fro_Sig[f])
            bnd_row[f][i] = (2 * n * np.sqrt(2) * sig_norm / delta_sig) * min_term_sig

            # ── Theorem 2: col-wise bound using K_k ───────────────────────
            # 2n√2 · ‖K_k‖_F / δ_k · min(√m/Q·Σ‖K_j−K_k‖_op, 1/Q·Σ‖K_j−K_k‖_F)
            k_norm   = k_norm_f[f]
            delta_k  = max(gaps_K[f][m - 1], 1e-12)
            min_term_K = min(np.sqrt(m) * sum_op_K[f], sum_fro_K[f])
            bnd_col[f][i] = (2 * n * np.sqrt(2) * k_norm / delta_k) * min_term_K

    # ── Theorem 3 (col-wise vs row-wise dominance — trace formula) ───────────
    # Exact equality holds:  E^col_m − E^row_m  =  Λ_{row,m} − Λ_{col,m}
    # because E^row_m = T − Λ_{row,m}  and  E^col_m = T − Λ_{col,m}  (Eckart-Young).
    # Any numerical discrepancy is a floating-point artefact (avoided via float64).
    total_row  = np.array([sum(err_row[f][i] for f in fields) for i in range(max_m)])
    total_col  = np.array([sum(err_col[f][i] for f in fields) for i in range(max_m)])
    diff_exact = total_col - total_row                                         # E^col − E^row
    lambda_row = np.array([n * Q * np.sum(vals_row[:m]) for m in m_range])    # Λ_{row,m}
    lambda_col = np.array([Q * S * np.sum(vals_col[:m]) for m in m_range])    # Λ_{col,m}
    diff_tr    = lambda_row - lambda_col                                        # Λ_{row} − Λ_{col}

    return dict(fields=fields, m_range=m_range, T=T,
                err_fw=err_fw, err_row=err_row, err_col=err_col,
                bnd_row=bnd_row, bnd_col=bnd_col,
                gaps_K={f: gaps_K[f][:max_m]     for f in fields},
                gaps_Sigma={f: gaps_Sigma[f][:max_m] for f in fields},
                total_row=total_row, total_col=total_col,
                diff_exact=diff_exact,
                lambda_row=lambda_row, lambda_col=lambda_col,
                diff_tr=diff_tr)


def _agg(metrics_list: list) -> dict:
    """Mean ± std over seeds."""
    if not metrics_list:
        return {}
    ref    = metrics_list[0]
    result = {"fields": ref["fields"], "m_range": ref["m_range"]}

    # T is a scalar per seed: store mean and std
    T_arr = np.array([m["T"] for m in metrics_list])
    result["T_mean"] = float(T_arr.mean())
    result["T_std"]  = float(T_arr.std())

    for key in ("total_row", "total_col", "diff_exact", "diff_tr",
                "lambda_row", "lambda_col"):
        arr = np.array([m[key] for m in metrics_list])
        result[f"{key}_mean"] = arr.mean(axis=0)
        result[f"{key}_std"]  = arr.std(axis=0)

    for key in ("err_fw", "err_row", "err_col", "bnd_row", "bnd_col",
                "gaps_K", "gaps_Sigma"):
        result[f"{key}_mean"] = {}
        result[f"{key}_std"]  = {}
        for f in ref["fields"]:
            arr = np.array([m[key][f] for m in metrics_list])
            result[f"{key}_mean"][f] = arr.mean(axis=0)
            result[f"{key}_std"][f]  = arr.std(axis=0)
    return result


# ── Public data-collection entry point ───────────────────────────────────────

def compute_pca_metrics(
    analyzers: List[Tuple[MultiSeedAnalyzer, str]],
    max_m:     int = 10,
) -> dict:
    """Compute PCA comparison metrics for every n_train class.

    Parameters
    ----------
    analyzers : list of (MultiSeedAnalyzer, label) in the same order used
                for the combined GP-performance plots.
    max_m     : maximum number of PCA modes to evaluate.

    Returns
    -------
    dict[label] → aggregated metrics dict  (mean ± std over seeds)
    """
    results = {}
    for ana, label in analyzers:
        per_seed = []
        for single_ana in ana._analyzers:
            try:
                Y_raw = _load_train_fields(single_ana)
                per_seed.append(_pca_metrics_one_seed(Y_raw, max_m))
            except Exception as e:
                print(f"  Warning (seed skipped for {label}): {e}")
        results[label] = _agg(per_seed)
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


# ── Figure: Theorem 1  (row-wise excess error bound) ─────────────────────────

def plot_pca_theorem1(
    agg_by_n:    dict,
    figsize:     Optional[tuple] = None,
    output_path: Optional[str]   = None,
) -> plt.Figure:
    """2 rows × Q cols.
    Row 0 — empirical ΔE^row_{k,m} = E^field − E^row  vs Theorem-1 bound.
    Row 1 — spectral gap δ_Σ_k(m) of Σ_k = (1/n) Y_k Y_k^T.

    Theorem 1 bound (as stated in the theorem):
        (2n√2 ‖Σ_k‖_F / δ_Σ_k) · min(√m/Q · Σ_j ‖Σ_j−Σ_k‖_op,
                                         1/Q  · Σ_j ‖Σ_j−Σ_k‖_F)
        with Σ_k = (1/n) Y_k Y_k^T  (= (S/n)·K_k).
    """
    ref     = next(iter(agg_by_n.values()))
    fields  = ref["fields"];  m_range = ref["m_range"];  Q = len(fields)
    fsize   = figsize or (5 * Q, 8)
    scale   = _compute_scale(fsize, Q, 2);  fs = _font_sizes(scale)

    fig, axes = plt.subplots(2, Q, figsize=fsize, sharex=True, squeeze=False)

    C_emp, C_bnd, C_gap = "#2166ac", "#d73027", "#7b2d8b"

    for ci, f in enumerate(fields):
        ax_e, ax_g = axes[0, ci], axes[1, ci]
        for n_label, agg in agg_by_n.items():
            st  = _pca_n_style(n_label)
            kw  = dict(linewidth=st["lw"], linestyle=st["ls"],
                       alpha=st["alpha"], marker=st["marker"], markersize=3*scale)

            # Empirical excess error (clipped to avoid log(0))
            delta = np.maximum(agg["err_row_mean"][f] - agg["err_fw_mean"][f], 1e-10)
            _band(ax_e, m_range, delta, agg["err_row_std"][f],
                  C_emp, fr"$\Delta E^{{row}}$  [{n_label}]", **kw)

            # Theorem 1 bound
            kw_bnd = dict(kw, linestyle=(":" if st["ls"] == "-" else "-."), marker="")
            _band(ax_e, m_range, agg["bnd_row_mean"][f], agg["bnd_row_std"][f],
                  C_bnd, f"Bound Thm 1  [{n_label}]", **kw_bnd)

            # Spectral gap δ_Σ_k of Σ_k = (1/n) Y_k Y_k^T  (= (S/n) × gap of K_k)
            _band(ax_g, m_range, agg["gaps_Sigma_mean"][f], agg["gaps_Sigma_std"][f],
                  C_gap, fr"$\delta_{{\Sigma_k}}(m)$  [{n_label}]", **kw)

        if ci == 0:
            ax_e.set_ylabel(r"$\Delta E^{row}_{k,m}$",
                            fontsize=fs["label"], fontweight="bold")
            ax_g.set_ylabel(r"Spectral gap $\delta_{\Sigma_k}(m)$",
                            fontsize=fs["label"], fontweight="bold")

        ax_e.set_title(f.replace("field_", "f"),
                       fontsize=fs["title"], fontweight="bold")
        _finish_ax(ax_e, fs)
        _finish_ax(ax_g, fs)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# Backward-compatible alias
plot_pca_theorem2 = plot_pca_theorem1


# ── Figure: Theorem 2  (col-wise excess error bound) ─────────────────────────

def plot_pca_theorem2_col(
    agg_by_n:    dict,
    figsize:     Optional[tuple] = None,
    output_path: Optional[str]   = None,
) -> plt.Figure:
    """2 rows × Q cols.
    Row 0 — empirical ΔE^col_{k,m} = E^field − E^col  vs Theorem-2 bound.
    Row 1 — spectral gap δ_k(m) of K_k.

    Theorem 2 bound:
        (2n√2 ‖K_k‖_F / δ_k) · min(√m · Σ_j ‖K_j−K_k‖_op / Q,
                                      Σ_j ‖K_j−K_k‖_F / Q)
    Same structure as Theorem 1, only prefactor n instead of S.
    """
    ref     = next(iter(agg_by_n.values()))
    fields  = ref["fields"];  m_range = ref["m_range"];  Q = len(fields)
    fsize   = figsize or (5 * Q, 8)
    scale   = _compute_scale(fsize, Q, 2);  fs = _font_sizes(scale)

    fig, axes = plt.subplots(2, Q, figsize=fsize, sharex=True, squeeze=False)

    C_emp, C_bnd, C_gap = "#1a9641", "#d73027", "#e07b00"

    for ci, f in enumerate(fields):
        ax_e, ax_g = axes[0, ci], axes[1, ci]
        for n_label, agg in agg_by_n.items():
            st  = _pca_n_style(n_label)
            kw  = dict(linewidth=st["lw"], linestyle=st["ls"],
                       alpha=st["alpha"], marker=st["marker"], markersize=3*scale)

            delta = np.maximum(agg["err_col_mean"][f] - agg["err_fw_mean"][f], 1e-10)
            _band(ax_e, m_range, delta, agg["err_col_std"][f],
                  C_emp, fr"$\Delta E^{{col}}$  [{n_label}]", **kw)

            kw_bnd = dict(kw, linestyle=(":" if st["ls"] == "-" else "-."), marker="")
            _band(ax_e, m_range, agg["bnd_col_mean"][f], agg["bnd_col_std"][f],
                  C_bnd, f"Bound Thm 2  [{n_label}]", **kw_bnd)

            _band(ax_g, m_range, agg["gaps_K_mean"][f], agg["gaps_K_std"][f],
                  C_gap, fr"$\delta_k(m)$  [{n_label}]", **kw)

        if ci == 0:
            ax_e.set_ylabel(r"$\Delta E^{col}_{k,m}$",
                            fontsize=fs["label"], fontweight="bold")
            ax_g.set_ylabel(r"Spectral gap $\delta_k(m)$",
                            fontsize=fs["label"], fontweight="bold")

        ax_e.set_title(f.replace("field_", "f"),
                       fontsize=fs["title"], fontweight="bold")
        _finish_ax(ax_e, fs)
        _finish_ax(ax_g, fs)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# Backward-compatible alias
plot_pca_theorem3 = plot_pca_theorem2_col


# ── Figure: Theorem 3  (col-wise vs row-wise dominance — trace formula) ──────

def plot_pca_theorem3_dominance(
    agg_by_n:    dict,
    figsize:     Optional[tuple] = None,
    output_path: Optional[str]   = None,
) -> plt.Figure:
    """1 row × n_classes cols — normalised by total variance T.

    Theorem 3: E^row_m < E^col_m  ⟺  Λ_{row,m} > Λ_{col,m}

    Mathematically the two quantities are exactly equal:
        E^col_m − E^row_m  =  Λ_{row,m} − Λ_{col,m}
    (both equal T − Λ_{col} − (T − Λ_{row}) by Eckart-Young).

    Both are normalised by T (total squared norm) so the y-axis reads as
    "fraction of total variance" in (−1, 1).  If the two curves overlap
    perfectly the equality is confirmed numerically; any residual gap is
    a floating-point artefact.

    Solid  (dark)  : (E^col_m − E^row_m) / T   (direct projection error)
    Dashed (red)   : (Λ_{row,m} − Λ_{col,m}) / T  (trace formula, Thm 3)
    """
    n_cols = len(agg_by_n)
    fsize  = figsize or (6 * n_cols, 5)
    scale  = _compute_scale(fsize, n_cols, 1);  fs = _font_sizes(scale)

    fig, axes = plt.subplots(1, n_cols, figsize=fsize, sharey=True, squeeze=False)

    C_exact, C_lambda = "#2c2c2c", "#d73027"

    global_y_abs = 1e-12
    for agg in agg_by_n.values():
        T = agg["T_mean"]
        diff_exact_n = agg["diff_exact_mean"] / T
        diff_tr_n    = agg["diff_tr_mean"] / T
        global_y_abs = max(global_y_abs, np.abs(np.concatenate([diff_exact_n, diff_tr_n])).max() * 1.3)

    for ci, (n_label, agg) in enumerate(agg_by_n.items()):
        ax      = axes[0, ci]
        m_range = agg["m_range"]
        st      = _pca_n_style(n_label)
        lw      = st["lw"] * scale
        T       = agg["T_mean"]          # mean total variance over seeds

        # ── Normalise by T ────────────────────────────────────────────────────
        diff_exact_n = agg["diff_exact_mean"] / T
        std_exact_n  = agg["diff_exact_std"]  / T
        diff_tr_n    = agg["diff_tr_mean"]    / T
        std_tr_n     = agg["diff_tr_std"]     / T

        # ── Exact error difference  (E^col − E^row) / T ──────────────────────
        _band(ax, m_range, diff_exact_n, std_exact_n,
              C_exact,
              r"$(E^{col}_m - E^{row}_m)\,/\,T$  (direct)",
              linewidth=lw, linestyle="-",
              marker="o", markersize=4*scale, alpha=st["alpha"])

        # ── Trace formula  (Λ_{row} − Λ_{col}) / T ───────────────────────────
        _band(ax, m_range, diff_tr_n, std_tr_n,
              C_lambda,
              r"$(\Lambda_{row,m} - \Lambda_{col,m})\,/\,T$  (Thm 3)",
              linewidth=lw, linestyle="--",
              marker="", markersize=0, alpha=st["alpha"])

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8*scale)

        # ── Shaded dominance regions ──────────────────────────────────────────
        y_abs = global_y_abs
        ax.set_ylim(-y_abs, y_abs)
        ax.fill_between(m_range, 0,  y_abs, color="#2166ac", alpha=0.04,
                        label="col dominates (E^col < E^row)")
        ax.fill_between(m_range, -y_abs, 0, color="#d73027", alpha=0.04,
                        label="row dominates (E^row < E^col)")

        # ── Labels ────────────────────────────────────────────────────────────
        ax.set_title(n_label, fontsize=fs["title"], fontweight="bold")
        ax.set_xlabel("modes  m", fontsize=fs["label"], fontweight="bold")
        if ci == 0:
            ax.set_ylabel(r"$(E^{col}_m - E^{row}_m)\,/\,T$",
                          fontsize=fs["label"], fontweight="bold")
        _finish_ax(ax, fs, yscale="linear")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# Backward-compatible alias
plot_pca_lemma5 = plot_pca_theorem3_dominance


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    list_n30 = [f"results_lv_n=10_30/results_N=30_lv_seed{i}.zarr" for i in range(10)]
    list_n15 = [f"results_lv_n=10_30/results_N=15_lv_seed{i}.zarr" for i in range(10)]
    list_n10 = [f"results_lv_n=10_30/results_N=10_lv_seed{i}.zarr" for i in range(10)]

    ana_n30 = MultiSeedAnalyzer(list_n30)
    ana_n15 = MultiSeedAnalyzer(list_n15)
    ana_n10 = MultiSeedAnalyzer(list_n10)

    # N=30 → top row;   N=10 → bottom row
    analyzers = [(ana_n30, "N=30"), (ana_n15, "N=15"), (ana_n10, "N=10")]

    for ana, label in analyzers:
        print("=" * 60)
        print(f"Summary  {label}")
        print("=" * 60)
        ana.print_summary()
        print()

    model_types = ["RC", "FI", "CI", "FM"]

    fig_q2 = plot_combined_q2(
        analyzers,
        model_types=model_types,
        output_path="combined_q2_vs_modes.png",
        figsize=(11,4)
    )

    fig_rmse = plot_combined_rmse(
        analyzers,
        model_types=model_types,
        output_path="combined_rmse_vs_modes.pdf",
        figsize=(14,8)
    )

    # # ── PCA comparison analysis ───────────────────────────────────────────────
    # print("\n" + "=" * 60)
    # print("PCA COMPARISON  (Theorem 1 / Theorem 2 / Theorem 3)")
    # print("=" * 60)

    # agg_by_n = compute_pca_metrics(analyzers, max_m=10)

    # fig_thm1 = plot_pca_theorem1(
    #     agg_by_n,
    #     figsize=(11, 8),
    #     output_path="pca_theorem1_row_bound.svg",
    # )

    # fig_thm2 = plot_pca_theorem2_col(
    #     agg_by_n,
    #     figsize=(11, 8),
    #     output_path="pca_theorem2_col_bound.svg",
    # )

    # fig_thm3 = plot_pca_theorem3_dominance(
    #     agg_by_n,
    #     figsize=(10, 5),
    #     output_path="pca_theorem3_dominance.svg",
    # )

    # plt.show()
