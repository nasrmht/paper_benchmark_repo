#!/usr/bin/env python3
"""
Ranking-based comparison — % of being 1st, 2nd, or 3rd.

No statistical test. For each (N, output, scenario), the 3 models are ranked
by RRMSE on each run, and the fraction of runs where each model ranks 1st/2nd/3rd
is computed.

Structure of the comparison:
  - mogp_constrained : unique (does not depend on the deduced output)
  - indep / lcm      : 3 variants each (indep_deduced_f1/f2/f3, lcm_deduced_f1/f2/f3)

Full table (per N):
  - Rows    : deduction scenarios (f1*, f2*, f3*)
  - Col blocks : one per output (f1, f2, f3)
  - Within each block : one sub-column per model
  - Cell    : (1st %, 2nd %, 3rd %)

Condensed outputs (per N + across all N):
  A. Top-1 heatmap         — seaborn/imshow heatmap, % rank 1 colour-coded
  B. Stacked bar chart     — full rank distribution (1st+2nd+3rd stacked)
  C. Clean top-1 table     — wide table, one % per cell, RdYlGn colours
  D. Top-1 line vs N       — evolution of % rank 1 across training sizes

CSV exports:
  rank_percentages_N{n}.csv  — tidy CSV per N
  rank_percentages_all.csv   — all N concatenated
"""

import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd

try:
    from scipy.stats import rankdata as _scipy_rankdata
    def _rankdata(arr):
        return _scipy_rankdata(arr)
except ImportError:
    def _rankdata(arr):
        arr = np.asarray(arr, dtype=float)
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False
    print("seaborn not found — heatmap uses matplotlib imshow fallback.")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
METHOD_LABELS  = ['CMOGP', 'Indep. GP', 'LCM']
COLORS         = {'mogp': '#2E86AB', 'indep': '#F18F01', 'lcm': '#A23B72'}
METHOD_COLORS  = [COLORS['mogp'], COLORS['indep'], COLORS['lcm']]

OUTPUT_NAMES   = [r'RMSE $(\hat{f}_1)$', r'RMSE $(\hat{f}_2)$', r'RMSE $(\hat{f}_3)$']
OUTPUT_SHORT   = ['f1', 'f2', 'f3']
OUTPUT_COLORS  = ['#4E79A7', '#E15759', '#59A14F']   # blue / red / green

SCENARIOS      = ['deduced_f1', 'deduced_f2', 'deduced_f3']
SCENARIO_LABELS = ['l=1', 'l=2', 'l=3']

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16,
})

# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (same logic as visualisation_poster_cd.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_results(results_dir: str) -> dict:
    agg = os.path.join(results_dir, 'benchmark_complet_aggregated.pkl')
    if os.path.exists(agg):
        print(f"Loading aggregated file: {agg}")
        with open(agg, 'rb') as f:
            return pickle.load(f)

    files = sorted(glob.glob(
        os.path.join(results_dir, "benchmark_complet_seed=*.pkl")))
    if not files:
        print(f"No pickle files found in {results_dir}")
        return {}

    print(f"Loading {len(files)} seed files…")
    with open(files[0], 'rb') as f:
        first = pickle.load(f)
    n_list = sorted(first.keys())

    all_scen = (['mogp_constrained']
                + [f'lcm_{s}'   for s in SCENARIOS]
                + [f'indep_{s}' for s in SCENARIOS])
    data = {n: {s: {o: {'rrmse': []} for o in range(3)}
                for s in all_scen}
            for n in n_list}

    for fp in files:
        try:
            with open(fp, 'rb') as f:
                res = pickle.load(f)
            for n in n_list:
                if n not in res:
                    continue
                for s in all_scen:
                    if s not in res[n]:
                        continue
                    for o in range(3):
                        if o in res[n][s] and 'rrmse' in res[n][s][o]:
                            data[n][s][o]['rrmse'].extend(res[n][s][o]['rrmse'])
        except Exception as e:
            print(f"  Warning – {fp}: {e}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Ranking computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_rank_percentages(results: dict, n: int) -> dict:
    """
    Returns rank_data[out_idx][scenario] =
        {'CMoGP': [pct_1st, pct_2nd, pct_3rd],
         'Indep. GP': [...], 'LCM': [...]}
    or None if data is missing.
    """
    rank_data = {o: {} for o in range(3)}

    for out_idx in range(3):
        for scen in SCENARIOS:
            try:
                d_mogp  = np.array(results[n]['mogp_constrained'][out_idx]['rrmse'])
                d_indep = np.array(results[n][f'indep_{scen}'][out_idx]['rrmse'])
                d_lcm   = np.array(results[n][f'lcm_{scen}'][out_idx]['rrmse'])
            except KeyError:
                rank_data[out_idx][scen] = None
                continue

            min_len = min(len(d_mogp), len(d_indep), len(d_lcm))
            if min_len == 0:
                rank_data[out_idx][scen] = None
                continue

            # (N_runs, 3) – col order matches METHOD_LABELS: CMoGP, Indep, LCM
            mat   = np.stack([d_mogp[:min_len], d_indep[:min_len],
                               d_lcm[:min_len]], axis=1)
            ranks = np.apply_along_axis(_rankdata, 1, mat)  # lower RRMSE → rank 1

            rank_data[out_idx][scen] = {
                method: [round(100 * np.mean(ranks[:, i] == r), 1)
                         for r in [1, 2, 3]]
                for i, method in enumerate(METHOD_LABELS)
            }
    return rank_data


def _get(rank_data, out_idx, scen, method, rank_pos):
    """Safe accessor — returns 0.0 if data is missing."""
    d = rank_data[out_idx].get(scen)
    return d[method][rank_pos] if d is not None else 0.0


def build_tidy_df(rank_data: dict, n: int) -> pd.DataFrame:
    rows = []
    for out_idx, out_short in enumerate(OUTPUT_SHORT):
        for s_idx, scen in enumerate(SCENARIOS):
            d = rank_data[out_idx].get(scen)
            if d is None:
                continue
            for method, pcts in d.items():
                rows.append({
                    'N': n,
                    'Output': out_short,
                    'Scenario': SCENARIO_LABELS[s_idx],
                    'Method': method,
                    'pct_1st': pcts[0],
                    'pct_2nd': pcts[1],
                    'pct_3rd': pcts[2],
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helper
# ─────────────────────────────────────────────────────────────────────────────
def _pct_to_hex(pct: float, cmap='RdYlGn') -> str:
    return mcolors.to_hex(plt.get_cmap(cmap)(pct / 100.0))


def _style_header_cells(tbl, n_data_cols, method_colors, output_colors):
    """Apply colour to the header row of an ax.table."""
    for j in range(n_data_cols):
        cell = tbl[0, j]
        out_idx = j // 3
        cell.set_facecolor(output_colors[out_idx])
        cell.set_text_props(color='white', fontweight='bold')


def _style_row_label_cells(tbl, n_rows):
    for i in range(1, n_rows + 1):
        cell = tbl[i, -1]
        cell.set_facecolor('#DDEEFF')
        cell.set_text_props(fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Full table  (3 sub-tables, one per output)
# ─────────────────────────────────────────────────────────────────────────────
def plot_full_table(rank_data: dict, n: int, out_dir: str):
    """
    3 sub-tables side by side (one per output).
    Rows = scenarios.  Cols = methods.
    Cell = "1st: X%  2nd: Y%  3rd: Z%", background = RdYlGn(pct_1st).
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle(f'Ranking Percentages — Full Table  (N = {n})',
                 fontsize=14, fontweight='bold', y=1.03)

    for out_idx, (ax, out_name, out_col) in enumerate(
            zip(axes, OUTPUT_NAMES, OUTPUT_COLORS)):
        ax.axis('off')
        ax.set_title(out_name, fontsize=13, fontweight='bold',
                     color=out_col, pad=8)

        cell_text, cell_colors = [], []
        for scen in SCENARIOS:
            row_t, row_c = [], []
            d = rank_data[out_idx].get(scen)
            for method in METHOD_LABELS:
                if d is None:
                    row_t.append('N/A')
                    row_c.append('#DDDDDD')
                else:
                    p1, p2, p3 = d[method]
                    row_t.append(f'1st: {p1:.0f}%\n2nd: {p2:.0f}%\n3rd: {p3:.0f}%')
                    row_c.append(_pct_to_hex(p1))
            cell_text.append(row_t)
            cell_colors.append(row_c)

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=SCENARIO_LABELS,
            colLabels=METHOD_LABELS,
            cellColours=cell_colors,
            cellLoc='center',
            loc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 3.0)

        # Style column headers
        for j, color in enumerate(METHOD_COLORS):
            cell = tbl[0, j]
            cell.set_facecolor(color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)

        # Style row labels
        for i in range(1, len(SCENARIO_LABELS) + 1):
            cell = tbl[i, -1]
            cell.set_facecolor('#DDEEFF')
            cell.set_text_props(fontweight='bold', fontsize=11)

    plt.tight_layout()
    for ext in ['svg', 'pdf']:
        p = os.path.join(out_dir, f'full_table_N{n}.{ext}')
        plt.savefig(p, format=ext, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Condensed A : Top-1 heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_top1_heatmap(rank_data: dict, n: int, out_dir: str):
    """
    Heatmap — % Rank 1st.
    Rows : 9 = scenario (3) × method (3), grouped by scenario.
    Cols : 3 outputs.
    """
    row_labels = []
    matrix = np.full((9, 3), np.nan)

    for s_idx, (scen, scen_label) in enumerate(zip(SCENARIOS, SCENARIO_LABELS)):
        for m_idx, method in enumerate(METHOD_LABELS):
            r = s_idx * 3 + m_idx
            row_labels.append(f'{scen_label} · {method}')
            for out_idx in range(3):
                d = rank_data[out_idx].get(scen)
                if d is not None:
                    matrix[r, out_idx] = d[method][0]

    df_heat = pd.DataFrame(matrix, index=row_labels, columns=OUTPUT_SHORT)

    fig, ax = plt.subplots(figsize=(7, 8))

    if _SEABORN:
        sns.heatmap(
            df_heat, ax=ax, cmap='RdYlGn', vmin=0, vmax=100,
            annot=True, fmt='.0f', linewidths=0.6, linecolor='white',
            cbar_kws={'label': '% Ranked 1st', 'shrink': 0.8},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
        )
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        plt.colorbar(im, ax=ax, label='% Ranked 1st', shrink=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(OUTPUT_SHORT)
        ax.set_yticks(range(9))
        ax.set_yticklabels(row_labels, fontsize=10)
        for i in range(9):
            for j in range(3):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if v < 25 or v > 75 else 'black')

    # Horizontal separators between scenario groups
    for sep in [3, 6]:
        ax.axhline(sep, color='black', linewidth=2.5)

    # Right-side annotation: scenario labels centred on each group
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([1.5, 4.5, 7.5])
    ax2.set_yticklabels(SCENARIO_LABELS, fontsize=13, fontweight='bold')
    ax2.tick_params(length=0)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax.set_title(f'% Ranked 1st — N = {n}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Output', fontsize=12)
    ax.set_ylabel('')

    plt.tight_layout()
    p = os.path.join(out_dir, f'heatmap_top1_N{n}.svg')
    plt.savefig(p, format='svg', bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Condensed B : Stacked bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_stacked_bars(rank_data: dict, n: int, out_dir: str):
    """
    3 subplots (one per output).
    x = deduction scenarios, grouped bars per method, stacked by rank.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
   # fig.suptitle(f'1st Rank Distribution', fontsize=14, fontweight='bold')

    bw      = 0.22
    x       = np.arange(3)
    alphas  = [0.95, 0.55, 0.25]   # shade for 1st / 2nd / 3rd

    for out_idx, (ax, out_name) in enumerate(zip(axes, OUTPUT_NAMES)):
        for m_idx, (method, color) in enumerate(zip(METHOD_LABELS, METHOD_COLORS)):
            offset = (m_idx - 1) * (bw + 0.04)
            bx = x + offset

            p1s = np.array([_get(rank_data, out_idx, s, method, 0) for s in SCENARIOS])
            p2s = np.array([_get(rank_data, out_idx, s, method, 1) for s in SCENARIOS])
            p3s = np.array([_get(rank_data, out_idx, s, method, 2) for s in SCENARIOS])

            ax.bar(bx, p1s, bw, color=color, alpha=alphas[0],
                   edgecolor='white', linewidth=0.5)
            # ax.bar(bx, p2s, bw, bottom=p1s, color=color, alpha=alphas[1],
            #        edgecolor='white', linewidth=0.5)
            # ax.bar(bx, p3s, bw, bottom=p1s + p2s, color=color, alpha=alphas[2],
            #        edgecolor='white', linewidth=0.5)

            # # Annotate % rank-1 inside the bottom bar
            # for xi, v in zip(bx, p1s):
            #     if v >= 8:
            #         ax.text(xi, v / 2, f'{v:.0f}', ha='center', va='center',
            #                 fontsize=8.5, fontweight='bold', color='white')

        ax.set_title(out_name, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_LABELS, fontsize=16, fontweight='bold')
        ax.set_xlabel('Deduced output', fontsize=16, fontweight='bold')
        if out_idx == 0:
            ax.set_ylabel('Win rate (%)', fontsize=16, fontweight='bold')
        
        ax.set_yticks([0, 25, 50, 75, 100])
        #ax.tick_params(axis='y', labelsize=12)
        ax.set_yticklabels([0, 25, 50, 75, 100], fontsize=16, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    method_hdl = [mpatches.Patch(color=c, alpha=0.95, label=l)
                  for c, l in zip(METHOD_COLORS, METHOD_LABELS)]
   # shade_hdl  = [mpatches.Patch(facecolor='dimgray', alpha=a, label=lbl)
   #               for a, lbl in zip(alphas, ['1st', '2nd', '3rd'])]
    fig.legend(handles=method_hdl, #+ shade_hdl,
               loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.06),
               frameon=True, edgecolor='black', 
               prop={'size': 16, 'weight': 'bold'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    p = os.path.join(out_dir, f'stacked_bars_N{n}.pdf')
    plt.savefig(p, format='pdf', bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Condensed C : Clean Top-1 wide table
# ─────────────────────────────────────────────────────────────────────────────
def plot_top1_table(rank_data: dict, n: int, out_dir: str):
    """
    Wide table: rows = scenarios, cols = output × method (9 cols).
    Cell = '% rank 1', background colour = RdYlGn(pct_1st).
    Column headers coloured by output group.
    """
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis('off')
    ax.set_title(f'% Ranked 1st — Condensed Table  (N = {n})',
                 fontsize=14, fontweight='bold', pad=15)

    cell_text, cell_colors = [], []
    for scen in SCENARIOS:
        row_t, row_c = [], []
        for out_idx in range(3):
            d = rank_data[out_idx].get(scen)
            for method in METHOD_LABELS:
                if d is None:
                    row_t.append('—')
                    row_c.append('#EEEEEE')
                else:
                    p1 = d[method][0]
                    row_t.append(f'{p1:.0f}%')
                    row_c.append(_pct_to_hex(p1))
        cell_text.append(row_t)
        cell_colors.append(row_c)

    col_labels = [f'{m}\n({o})'
                  for o in OUTPUT_SHORT
                  for m in ['CMoGP', 'Ind.', 'LCM']]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=SCENARIO_LABELS,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.2)

    # Column headers: coloured by output group
    for j in range(9):
        out_idx = j // 3
        cell = tbl[0, j]
        cell.set_facecolor(OUTPUT_COLORS[out_idx])
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)

    # Row labels
    for i in range(1, len(SCENARIO_LABELS) + 1):
        cell = tbl[i, -1]
        cell.set_facecolor('#DDEEFF')
        cell.set_text_props(fontweight='bold', fontsize=12)

    # Colourbar legend
    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.015, pad=0.04)
    cbar.set_label('% Ranked 1st', fontsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    p = os.path.join(out_dir, f'top1_table_N{n}.svg')
    plt.savefig(p, format='svg', bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Condensed D : % Rank-1 vs N  (aggregated over scenarios)
# ─────────────────────────────────────────────────────────────────────────────
def plot_top1_vs_n(df_all: pd.DataFrame, out_dir: str):
    """
    Line plot: % rank 1 vs training size N for each method.
    One subplot per output, averaged over deduction scenarios.
    """
    if df_all.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('% Ranked 1st vs Training Size N', fontsize=14,
                 fontweight='bold')

    for out_idx, (ax, out_name) in enumerate(zip(axes, OUTPUT_NAMES)):
        sub = df_all[df_all['Output'] == OUTPUT_SHORT[out_idx]]
        agg = (sub.groupby(['N', 'Method'])['pct_1st']
               .mean().reset_index())

        for method, color in zip(METHOD_LABELS, METHOD_COLORS):
            m = agg[agg['Method'] == method].sort_values('N')
            ax.plot(m['N'], m['pct_1st'], 'o-',
                    color=color, linewidth=2.5, markersize=7, label=method)

        ax.axhline(33.3, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Random (33%)' if out_idx == 0 else '')
        ax.set_title(out_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training size N', fontsize=11)
        if out_idx == 0:
            ax.set_ylabel('% Ranked 1st (avg over scenarios)', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.06),
               frameon=True, edgecolor='black', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    p = os.path.join(out_dir, 'top1_vs_N.svg')
    plt.savefig(p, format='svg', bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Wide pivot table (paper-ready): best method highlighted per cell
# ─────────────────────────────────────────────────────────────────────────────
def plot_top1_pivot_highlighted(df_all: pd.DataFrame, out_dir: str,
                                n_values: list = None):
    """
    For each N (or averaged across N if n_values=None), plot a heatmap where:
      - rows = scenario × output  (9 rows)
      - cols = method             (3 cols)
      - value = % rank 1
      - best method per row is highlighted with a gold star annotation
    Clean, paper-friendly.
    """
    if df_all.empty:
        return

    ns = n_values if n_values is not None else sorted(df_all['N'].unique())

    for n in ns:
        sub = df_all[df_all['N'] == n]
        pivot = sub.pivot_table(
            index=['Output', 'Scenario'],
            columns='Method',
            values='pct_1st',
        )[METHOD_LABELS]   # ensure column order

        fig, ax = plt.subplots(figsize=(6, 7))

        if _SEABORN:
            sns.heatmap(
                pivot, ax=ax, cmap='Blues', vmin=0, vmax=100,
                annot=True, fmt='.0f', linewidths=0.5, linecolor='white',
                cbar_kws={'label': '% Ranked 1st', 'shrink': 0.8},
                annot_kws={'fontsize': 11},
            )
        else:
            mat = pivot.values
            im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=100, aspect='auto')
            plt.colorbar(im, ax=ax, label='% Ranked 1st', shrink=0.8)
            ax.set_xticks(range(3))
            ax.set_xticklabels(METHOD_LABELS)
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(
                [f'{o} · {s}' for o, s in pivot.index], fontsize=9)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center',
                            fontsize=10)

        # Highlight the best method in each row with a gold border
        for row_i, (_, row) in enumerate(pivot.iterrows()):
            best_col = row.values.argmax()
            ax.add_patch(plt.Rectangle(
                (best_col - 0.5, row_i - 0.5), 1, 1,
                fill=False, edgecolor='gold', linewidth=3, zorder=5))

        # Horizontal separators between output groups (every 3 rows)
        for sep in [3, 6]:
            ax.axhline(sep, color='black', linewidth=2)

        ax.set_title(f'% Ranked 1st — N = {n}  (★ = best per row)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel('')

        plt.tight_layout()
        p = os.path.join(out_dir, f'pivot_highlighted_N{n}.svg')
        plt.savefig(p, format='svg', bbox_inches='tight')
        print(f'  Saved: {p}')
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, '..', 'results', 'constraint_benchmark')
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    if not results:
        return

    out_dir = os.path.join(base_dir, 'figures', 'ranking_tables')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n{'='*60}")

    n_values = sorted(results.keys())
    all_dfs  = []

    for n in n_values:
        print(f"\n──── N = {n} ────")
        rank_data = compute_rank_percentages(results, n)

        # Tidy CSV per N
        df = build_tidy_df(rank_data, n)
        all_dfs.append(df)
        p = os.path.join(out_dir, f'rank_percentages_N{n}.csv')
        df.to_csv(p, index=False, float_format='%.1f')
        print(f'  Saved: {p}')

        # Fig 1 — full table (3 sub-tables)
        plot_full_table(rank_data, n, out_dir)

        # Fig 2 — heatmap top-1
        plot_top1_heatmap(rank_data, n, out_dir)

        # Fig 3 — stacked bars
        plot_stacked_bars(rank_data, n, out_dir)

        # Fig 4 — clean top-1 wide table
        plot_top1_table(rank_data, n, out_dir)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)

        # Global CSV
        p = os.path.join(out_dir, 'rank_percentages_all.csv')
        df_all.to_csv(p, index=False, float_format='%.1f')
        print(f'\n  Saved: {p}')

        # Fig 5 — % rank-1 vs N
        plot_top1_vs_n(df_all, out_dir)

        # Fig 6 — pivot table highlighted
        plot_top1_pivot_highlighted(df_all, out_dir, n_values=n_values)

    print(f"\n{'='*60}")
    print(f"Done. All files saved to: {out_dir}")
    print("="*60)
    print("""
Summary of outputs
──────────────────
Per N:
  full_table_N{n}.svg/.pdf     Full table: 3 sub-tables (1 per output), all 3 %
  heatmap_top1_N{n}.svg        Heatmap: % rank-1, rows=scenario×method, cols=output
  stacked_bars_N{n}.svg        Stacked bars: rank distribution per scenario/method
  top1_table_N{n}.svg          Wide condensed table: % rank-1 only, colour-coded
  rank_percentages_N{n}.csv    Tidy CSV

Global (across all N):
  rank_percentages_all.csv     Tidy CSV, all N
  top1_vs_N.svg                Line plot: % rank-1 vs N (avg over scenarios)
  pivot_highlighted_N{n}.svg   Pivot heatmap with gold border on best method
""")


if __name__ == '__main__':
    main()
