"""Post-processing and visualisation tools for benchmark results."""
import glob as _glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional, Dict, Any, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from ..benchmark.storage import open_storage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _short_label(model_name: str) -> str:
    """Short label for plot axes.

    Examples: RC_ConstMOGP_M5 → 'RC M5'
              CI_SOGP_M5_p2   → 'CI p=2'
    """
    parts = model_name.split("_")
    prefix = parts[0]
    m_part = next((p for p in parts if p.startswith("M") and p[1:].isdigit()), "")
    p_part = next((p for p in parts if p.startswith("p") and p[1:].isdigit()), "")
    if p_part:
        return f"{prefix} p={p_part[1:]}"
    return f"{prefix} {m_part}"


def _per_mode_aggregate(
    lat_rmse: np.ndarray,
    lat_q2: np.ndarray,
    pca_type: str,
    gp_type: str,
) -> tuple:
    """Converts latent metrics (variable shape) to vectors (M,).

    - FI (FieldwisePCA + IndepSOGP): shape (Q, M) → average over Q fields
    - Others (RC / CI / FM): shape (M, q) → average over q dimensions
    """
    if pca_type == "FieldwisePCA" and gp_type == "IndepSOGP":
        return lat_rmse.mean(axis=0), lat_q2.mean(axis=0)   # (M,)
    else:
        return lat_rmse.mean(axis=1), lat_q2.mean(axis=1)   # (M,)


# ---------------------------------------------------------------------------
# ResultsAnalyzer — single-seed (unchanged for compatibility)
# ---------------------------------------------------------------------------

class ResultsAnalyzer:
    """Loads, tabulates, and visualizes the results of a single zarr file.

    Parameters
    ----------
    storage_path : path to the zarr file written by BenchmarkRunner
    """

    def __init__(self, storage_path: str):
        self.storage = open_storage(storage_path, mode="r")
        self._df: Any = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        return self.storage.list_models()

    def load_model(self, model_name: str) -> Dict[str, Any]:
        return self.storage.load_model_result(model_name)

    def to_dataframe(self, reload: bool = False):
        """DataFrame with one row per model.

        Columns: model_name, pca_type, gp_type, n_modes, fixed_idx,
                 q2_f0...q2_fQ-1, rrmse_f0...rrmse_fQ-1,
                 iw_f0...iw_fQ-1, constraint_mean, constraint_max, fit_time_s.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        if self._df is not None and not reload:
            return self._df

        Q = self.storage.get_n_outputs()

        rows = []
        for name in self.list_models():
            res = self.storage.load_model_result_light(name)
            cfg = res.get("config", {})
            met = res.get("metrics", {})
            row = {
                "model_name": name,
                "pca_type":   cfg.get("pca_type", ""),
                "gp_type":    cfg.get("gp_type",  ""),
                "n_modes":    cfg.get("n_modes",   np.nan),
                "fixed_idx":  cfg.get("fixed_idx", -1),
                "fit_time_s": cfg.get("fit_time_s", np.nan),
            }
            q2    = met.get("q2",   np.full(Q, np.nan))
            rrmse = met.get("rrmse", np.full(Q, np.nan))
            iw    = met.get("interval_width", np.full(Q, np.nan))
            for i in range(Q):
                row[f"q2_f{i}"]    = float(q2[i])    if i < len(q2)    else np.nan
                row[f"rrmse_f{i}"] = float(rrmse[i]) if i < len(rrmse) else np.nan
                row[f"iw_f{i}"]    = float(iw[i])    if i < len(iw)    else np.nan
            row["constraint_mean"] = float(np.atleast_1d(
                met.get("constraint_mean", [np.nan]))[0])
            row["constraint_max"]  = float(np.atleast_1d(
                met.get("constraint_max",  [np.nan]))[0])
            rows.append(row)

        self._df = pd.DataFrame(rows)
        return self._df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Compact table: models × main metrics."""
        models = self.list_models()
        if not models:
            print("No model found.")
            return

        Q = self.storage.get_n_outputs()

        header = f"{'Model':<45} {'n_modes':>7} {'p':>3}  "
        header += "  ".join(f"Q²_f{i}" for i in range(Q))
        header += "   RRMSE_mean  Constr_max"
        print(header)
        print("-" * len(header))

        for name in sorted(models):
            res   = self.storage.load_model_result_light(name)
            cfg   = res.get("config", {})
            met   = res.get("metrics", {})
            q2    = met.get("q2",    np.full(Q, np.nan))
            rrmse = met.get("rrmse", np.full(Q, np.nan))
            cmax  = float(np.atleast_1d(met.get("constraint_max", [np.nan]))[0])
            row   = (f"{name:<45} {int(cfg.get('n_modes', 0)):>7} "
                     f"{int(cfg.get('fixed_idx', -1)):>3}  ")
            row  += "  ".join(f"{float(q2[i]):6.3f}" for i in range(len(q2)))
            row  += f"   {float(rrmse.mean()):9.4f}  {cmax:.2e}"
            print(row)

    # ------------------------------------------------------------------
    # Plots (mono-seed)
    # ------------------------------------------------------------------

    def plot_rrmse_vs_modes(
        self,
        field_idx: int = 0,
        model_types: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """RRMSE vs number of PCA modes for a given output field."""
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        df  = self.to_dataframe()
        fig, ax = (plt.subplots() if ax is None else (ax.get_figure(), ax))
        col = f"rrmse_f{field_idx}"
        for label, group in df.groupby("model_name"):
            if model_types and not any(t in label for t in model_types):
                continue
            sub = group.sort_values("n_modes")
            ax.plot(sub["n_modes"], sub[col], marker="o", label=label)
        ax.set_xlabel("Number of PCA modes")
        ax.set_ylabel(f"RRMSE (field {field_idx})")
        ax.set_title(f"RRMSE vs modes — field {field_idx}")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

    def plot_q2_bar(
        self,
        n_modes: Optional[int] = None,
        fixed_idx: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Bar chart: Q² per output field and per model."""
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        df = self.to_dataframe()
        if n_modes is not None:
            df = df[df["n_modes"] == n_modes]
        if fixed_idx is not None:
            df = df[(df["fixed_idx"] == fixed_idx) | (df["fixed_idx"] == -1)]

        Q = self.storage.get_n_outputs()
        q2_cols = [f"q2_f{i}" for i in range(Q)]

        fig, ax = (plt.subplots(figsize=(max(8, len(df) * 1.5), 4))
                   if ax is None else (ax.get_figure(), ax))
        x = np.arange(Q)
        width = 0.8 / max(len(df), 1)
        for j, (_, row) in enumerate(df.iterrows()):
            vals = [row[c] for c in q2_cols]
            ax.bar(x + j * width, vals, width, label=row["model_name"])
        ax.set_xticks(x + width * len(df) / 2)
        ax.set_xticklabels([f"f{i}" for i in range(Q)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Q²")
        ax.set_title("Q² per output field")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

    def plot_constraint_satisfaction(
        self, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Bars: max constraint violation per model."""
        models = self.list_models()
        names, vals = [], []
        for name in sorted(models):
            res  = self.storage.load_model_result_light(name)
            cmax = float(np.atleast_1d(
                res["metrics"].get("constraint_max", [np.nan]))[0])
            names.append(name.replace("_M", "\nM").replace("_p", "\np"))
            vals.append(cmax)

        fig, ax = (plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
                   if ax is None else (ax.get_figure(), ax))
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Max |u.T @ f|")
        ax.set_title("Constraint satisfaction")
        ax.set_yscale("log")
        fig.tight_layout()
        return fig

    def compare_models(
        self,
        metric: str = "rrmse",
        field_idx: int = 0,
        n_modes: Optional[int] = None,
    ):
        """Sorted DataFrame for a given metric and field."""
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        df  = self.to_dataframe()
        col = f"{metric}_f{field_idx}"
        if n_modes is not None:
            df = df[df["n_modes"] == n_modes]
        return df[["model_name", "pca_type", "gp_type", "n_modes",
                   "fixed_idx", col]].sort_values(col)


# ---------------------------------------------------------------------------
# MultiSeedAnalyzer — multi-seeds aggregation
# ---------------------------------------------------------------------------

class MultiSeedAnalyzer:
    """Aggregates the results of multiple zarr files (one per seed).

    Parameters
    ----------
    paths : list of zarr paths, glob pattern (e.g. "results_lv_seed*.zarr"),
            or a single path.

    Examples
    --------
    >>> ana = MultiSeedAnalyzer("results_lv_seed*.zarr")
    >>> ana = MultiSeedAnalyzer(["results_lv_seed0.zarr", "results_lv_seed1.zarr"])
    """

    def __init__(self, paths: Union[str, List[str]]):
        if isinstance(paths, str):
            if "*" in paths or "?" in paths:
                paths = sorted(_glob.glob(paths))
            else:
                paths = [paths]
        if not paths:
            raise ValueError("No results file found.")

        self.paths = list(paths)
        self._analyzers = [ResultsAnalyzer(p) for p in self.paths]

        # Extract the seed from the stored config or from the filename
        self._seeds: List[int] = []
        for p, a in zip(self.paths, self._analyzers):
            try:
                cfg = a.storage._data["config"] if hasattr(a.storage, "_data") \
                    else dict(a.storage.store["config"].attrs)
                seed = int(cfg["seed"])
            except Exception:
                m = re.search(r"seed(\d+)", os.path.basename(p))
                seed = int(m.group(1)) if m else -1
            self._seeds.append(seed)

        self._df: Any = None
        self._df_per_mode: Any = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """List of model names (from the first zarr)."""
        return self._analyzers[0].list_models()

    def to_dataframe(self, reload: bool = False):
        """DataFrame with one row per (seed, model).

        Columns identical to ResultsAnalyzer.to_dataframe() + 'seed' column.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        if self._df is not None and not reload:
            return self._df

        dfs = []
        for seed, ana in zip(self._seeds, self._analyzers):
            df = ana.to_dataframe().copy()
            df.insert(0, "seed", seed)
            dfs.append(df)
        self._df = pd.concat(dfs, ignore_index=True)
        return self._df

    def to_per_mode_dataframe(self, reload: bool = False):
        """DataFrame with one row per (seed, model, latent mode).

        Columns: seed, model_name, pca_type, gp_type, n_modes, fixed_idx,
                 mode_idx, mean_latent_q2, mean_latent_rmse.

        Only models with intermediate latent metrics are included.
        The average is taken over the latent output dimensions (q).
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        if self._df_per_mode is not None and not reload:
            return self._df_per_mode

        rows = []
        for seed, ana in zip(self._seeds, self._analyzers):
            for name in ana.list_models():
                res   = ana.storage.load_model_result_light(name)
                cfg   = res.get("config", {})
                inter = res.get("intermediate", {})
                if "latent_rmse" not in inter or "latent_q2" not in inter:
                    continue
                lat_rmse = inter["latent_rmse"]
                lat_q2   = inter["latent_q2"]
                pca_type = cfg.get("pca_type", "")
                gp_type  = cfg.get("gp_type",  "")
                rrmse_pm, q2_pm = _per_mode_aggregate(
                    lat_rmse, lat_q2, pca_type, gp_type
                )
                for m, (r, q) in enumerate(zip(rrmse_pm, q2_pm)):
                    rows.append({
                        "seed":             seed,
                        "model_name":       name,
                        "pca_type":         pca_type,
                        "gp_type":          gp_type,
                        "n_modes":          cfg.get("n_modes", np.nan),
                        "fixed_idx":        cfg.get("fixed_idx", -1),
                        "mode_idx":         m,
                        "mean_latent_q2":   float(q),
                        "mean_latent_rmse": float(r),
                    })

        self._df_per_mode = pd.DataFrame(rows)
        return self._df_per_mode

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Displays mean ± std of Q² and RRMSE per model, over all seeds."""
        if not _HAS_PANDAS:
            # fallback without pandas
            for seed, ana in zip(self._seeds, self._analyzers):
                print(f"--- seed={seed} ---")
                ana.print_summary()
            return

        df = self.to_dataframe()
        q2_cols    = sorted(c for c in df.columns if c.startswith("q2_f"))
        rrmse_cols = sorted(c for c in df.columns if c.startswith("rrmse_f"))
        models = sorted(df["model_name"].unique())

        hdr = (f"{'Model':<45} {'n_modes':>7} {'p':>3}  "
               f"{'mean Q²':>10}  {'std Q²':>8}  {'mean RRMSE':>10}  {'std RRMSE':>9}")
        print(hdr)
        print("-" * len(hdr))

        for name in models:
            sub  = df[df["model_name"] == name]
            row0 = sub.iloc[0]
            q2_vals    = sub[q2_cols].values.flatten()
            rrmse_vals = sub[rrmse_cols].values.flatten()
            print(
                f"{name:<45} {int(row0.get('n_modes', 0)):>7} "
                f"{int(row0.get('fixed_idx', -1)):>3}  "
                f"{q2_vals.mean():>10.4f}  {q2_vals.std():>8.4f}  "
                f"{rrmse_vals.mean():>10.4f}  {rrmse_vals.std():>9.4f}"
            )

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    # Palettes
    _COLOR_RC   = "#2166ac"   # blue
    _COLOR_CI   = "#d6604d"   # red/orange
    _COLOR_FI   = "#1a9641"   # green
    _COLOR_FM   = "#7b2d8b"   # purple
    _COLORS_BY_PREFIX = {"RC": _COLOR_RC, "CI": _COLOR_CI,
                          "FI": _COLOR_FI, "FM": _COLOR_FM}

    def _get_q2_cols(self) -> List[str]:
        df = self.to_dataframe()
        return sorted(c for c in df.columns if c.startswith("q2_f"))

    def _palette(self, names: List[str], base_color: str) -> List[str]:
        """Gradient of tints around base_color to distinguish scenarios."""
        import colorsys
        r, g, b = tuple(int(base_color.lstrip("#")[i:i+2], 16) / 255
                        for i in (0, 2, 4))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        n = max(len(names), 1)
        return [
            "#{:02x}{:02x}{:02x}".format(
                *[int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v * (0.5 + 0.5 * k / n))]
            )
            for k in range(n)
        ]

    def plot_comparison_violin_q2(
        self,
        n_modes: Optional[int] = None,
        figsize: Optional[tuple] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Comparative violin plot of Q² distribution over seeds.

        Figure 1x3: [RC vs CI | RC vs FI | RC vs FM].
        Each violin = distribution of Q² over (seeds x output fields).

        Parameters
        ----------
        n_modes     : filter by number of modes (None = all)
        figsize     : figure size (default: (15, 5))
        output_path : path to save the figure (None = do not save)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        df = self.to_dataframe()
        if n_modes is not None:
            df = df[df["n_modes"] == n_modes]
        q2_cols = self._get_q2_cols()

        comparisons = [
            ("CI", "RC vs CI (col-wise)"),
            ("FI", "RC vs FI (field-wise indep)"),
            ("FM", "RC vs FM (field-wise MOGP)"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 5), sharey=True)
        fig.suptitle(
            "Q² distribution over seeds"
            + (f"  [M={n_modes}]" if n_modes else ""),
            fontsize=13
        )

        for ax, (comp_prefix, title) in zip(axes, comparisons):
            rc_names   = sorted(df[df["model_name"].str.startswith("RC_")]
                                ["model_name"].unique())
            comp_names = sorted(df[df["model_name"].str.startswith(comp_prefix + "_")]
                                ["model_name"].unique())

            all_names = rc_names + comp_names
            if not all_names:
                ax.set_title(title + "\n(no data)")
                continue

            data_list, labels, colors = [], [], []
            base_comp_color = self._COLORS_BY_PREFIX.get(comp_prefix, "gray")
            comp_palette    = self._palette(comp_names, base_comp_color)

            for name in rc_names:
                sub = df[df["model_name"] == name]
                # Distribution over seeds x fields
                vals = sub[q2_cols].values.flatten()
                data_list.append(vals)
                labels.append(_short_label(name))
                colors.append(self._COLOR_RC)

            for i, name in enumerate(comp_names):
                sub = df[df["model_name"] == name]
                vals = sub[q2_cols].values.flatten()
                data_list.append(vals)
                labels.append(_short_label(name))
                colors.append(comp_palette[i])

            positions = list(range(len(all_names)))
            parts = ax.violinplot(
                data_list, positions=positions,
                showmedians=True, showextrema=True
            )

            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
            for key in ("cmedians", "cbars", "cmaxes", "cmins"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(1.2)

            # Separation line RC / comp
            if rc_names and comp_names:
                ax.axvline(len(rc_names) - 0.5, color="gray",
                           linestyle="--", linewidth=0.8, alpha=0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Q²" if ax == axes[0] else "")
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_comparison_rrmse_per_mode(
        self,
        n_modes: Optional[int] = None,
        figsize: Optional[tuple] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Latent GP RRMSE per mode, comparison RC vs CI / FI / FM.

        Figure 1x3: each subplot shows, for each latent mode m,
        the average RRMSE of the GP on the latent weights (mean ± std over seeds).

        - RC: one curve (no fixed output)
        - CI / FI / FM: Q curves, one per scenario (fixed_idx)

        Parameters
        ----------
        n_modes     : filter by number of modes (None = all)
        figsize     : figure size (default: (15, 5))
        output_path : path to save (None = do not save)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        df_pm = self.to_per_mode_dataframe()
        if df_pm.empty:
            print("No latent metric per mode found. "
                  "Verify that intermediate_metrics is correctly computed.")
            return plt.figure()

        if n_modes is not None:
            df_pm = df_pm[df_pm["n_modes"] == n_modes]

        comparisons = [
            ("CI", "RC vs CI"),
            ("FI", "RC vs FI"),
            ("FM", "RC vs FM"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 5))
        fig.suptitle(
            "Latent GP RRMSE per mode"
            + (f"  [M={n_modes}]" if n_modes else ""),
            fontsize=13
        )

        for ax, (comp_prefix, title) in zip(axes, comparisons):
            rc_mask   = df_pm["model_name"].str.startswith("RC_")
            comp_mask = df_pm["model_name"].str.startswith(comp_prefix + "_")
            sub_all   = df_pm[rc_mask | comp_mask]

            if sub_all.empty:
                ax.set_title(title + "\n(no data)")
                continue

            all_names = sorted(sub_all["model_name"].unique())
            mode_indices = sorted(sub_all["mode_idx"].unique())
            x = np.array(mode_indices) + 1   # 1-indexed

            base_comp_color = self._COLORS_BY_PREFIX.get(comp_prefix, "gray")
            rc_names   = [n for n in all_names if n.startswith("RC_")]
            comp_names = [n for n in all_names if n.startswith(comp_prefix + "_")]
            comp_palette = self._palette(comp_names, base_comp_color)

            def _plot_model(name, color, linestyle):
                sub = sub_all[sub_all["model_name"] == name]
                grp = sub.groupby("mode_idx")["mean_latent_rmse"]
                means = np.array([grp.get_group(m).mean() for m in mode_indices])
                stds  = np.array([grp.get_group(m).std()
                                  if len(grp.get_group(m)) > 1 else 0.0
                                  for m in mode_indices])
                lbl = _short_label(name)
                ax.plot(x, means, marker="o", linestyle=linestyle,
                        color=color, label=lbl, linewidth=1.8)
                ax.fill_between(x, means - stds, means + stds,
                                alpha=0.18, color=color)

            for name in rc_names:
                _plot_model(name, self._COLOR_RC, "-")

            # FI: all scenarios p share the same latent metrics
            # (all Q fields are trained) → we only display a single line
            if comp_prefix == "FI":
                # Choose the first scenario as representative
                rep = comp_names[0] if comp_names else None
                if rep:
                    _plot_model(rep, base_comp_color, "--")
                    ax.text(0.97, 0.97, "FI: identical metrics\nfor all p",
                            transform=ax.transAxes, fontsize=7,
                            ha="right", va="top", color="gray")
            else:
                for i, name in enumerate(comp_names):
                    _plot_model(name, comp_palette[i], "--")

            ax.set_xlabel("Latent mode m")
            ax.set_ylabel("Mean latent RRMSE" if ax == axes[0] else "")
            ax.set_title(title, fontsize=10)
            ax.set_xticks(x)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(linestyle=":", alpha=0.5)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_cumulative_pca_quality(
        self,
        n_modes: Optional[int] = None,
        figsize: Optional[tuple] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """PCA reconstruction Q² and RRMSE as a function of the number of retained modes.

        For each M-mode model, shows how the field reconstruction quality
        evolves when reconstructing with k = 1, 2, ..., M modes.
        Metrics are averaged over seeds; error bar = ± std over seeds.
        All scenarios of the same model (same prefix) share the same line style,
        while the background and edge colors are defined by the `fixed_idx` (p) index.

        Figure (Q rows x 2 columns):
          - left column  : PCA reconstruction Q²
          - right column : PCA reconstruction RRMSE
          - stacked/shifted bar chart per scenario.

        Parameters
        ----------
        n_modes     : filter by number of modes (None = all; if multiple
                      values exist, only the first is used)
        figsize     : figure size (default: (14, 4*Q))
        output_path : save path (None = do not save)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        # ── Data Collection ─────────────────────────────────────────────
        # For each (seed, model), we read cumulative_q2_test (M, Q)
        # and cumulative_rrmse_test (M, Q) from intermediate.
        records = []
        for seed, ana in zip(self._seeds, self._analyzers):
            for name in ana.list_models():
                res   = ana.storage.load_model_result_light(name)
                cfg   = res.get("config", {})
                inter = res.get("intermediate", {})
                cum_q2   = inter.get("cumulative_q2_test")
                cum_rrmse = inter.get("cumulative_rrmse_test")
                if cum_q2 is None or cum_rrmse is None:
                    continue
                records.append({
                    "seed":      seed,
                    "model_name": name,
                    "prefix":    name.split("_")[0],
                    "n_modes":   cfg.get("n_modes", np.nan),
                    "fixed_idx": int(cfg.get("fixed_idx", -1)),
                    "cum_q2":    np.array(cum_q2),    # (M, Q)
                    "cum_rrmse": np.array(cum_rrmse),  # (M, Q)
                })

        if not records:
            print(
                "[MultiSeedAnalyzer] No cumulative PCA data found. "
                "Rerun the benchmark to populate 'cumulative_q2_test'."
            )
            return plt.figure()

        # ── Filtering n_modes ─────────────────────────────────────────────────
        all_n_modes = sorted({r["n_modes"] for r in records if not np.isnan(r["n_modes"])})
        if n_modes is not None:
            records = [r for r in records if r["n_modes"] == n_modes]
        elif len(all_n_modes) > 1:
            n_modes = all_n_modes[0]
            records = [r for r in records if r["n_modes"] == n_modes]
            print(f"[MultiSeedAnalyzer] Multiple n_modes found {all_n_modes}. "
                  f"Displaying n_modes={n_modes}. "
                  "Pass n_modes= to choose another one.")

        if not records:
            print(f"[MultiSeedAnalyzer] No model with n_modes={n_modes}.")
            return plt.figure()

        M_max = max(r["cum_q2"].shape[0] for r in records)
        Q     = records[0]["cum_q2"].shape[1]
        x     = np.arange(1, M_max + 1)

        fig, axes = plt.subplots(
            Q, 2,
            figsize=figsize or (14, 4 * Q),
            sharex=True,
        )
        axes = np.array(axes).reshape(Q, 2)

        n_modes_label = f"  [M={n_modes}]" if n_modes else ""
        fig.suptitle(
            f"PCA reconstruction quality as a function of the number of retained modes{n_modes_label}",
            fontsize=13,
        )

        # Unique scenarios = (prefix, fixed_idx)
        scenarios = sorted(list(set((r["prefix"], r["fixed_idx"]) for r in records)))
        n_scenarios = len(scenarios)

        # Color <-> fixed_idx Association
        unique_p = sorted(list(set(r["fixed_idx"] for r in records)))
        color_map = {}
        cmap = plt.get_cmap("tab10")
        color_idx = 0
        from matplotlib.colors import to_rgba
        for p in unique_p:
            if p == -1:
                color_map[p] = self._COLOR_RC
            else:
                color_map[p] = cmap(color_idx % 10)
                color_idx += 1

        # Linestyle <-> prefix Association
        linestyle_map = {
            "RC": "-",
            "CI": "--",
            "FI": ":",
            "FM": "-."
        }

        # Total base bar width and per scenario
        total_width = 0.8
        w = total_width / max(1, n_scenarios)

        for i in range(Q):
            ax_q2 = axes[i, 0]
            ax_rr = axes[i, 1]

            for s_idx, (prefix, p_idx) in enumerate(scenarios):
                sub = [r for r in records if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
                if not sub:
                    continue

                by_seed: Dict[int, Dict] = {}
                for r in sub:
                    s = r["seed"]
                    M_r = r["cum_q2"].shape[0]
                    if s not in by_seed:
                        by_seed[s] = {"q2": [], "rr": [], "M": M_r}
                    by_seed[s]["q2"].append(r["cum_q2"][:, i])    # (M,)
                    by_seed[s]["rr"].append(r["cum_rrmse"][:, i]) # (M,)

                seed_q2 = np.array([np.mean(by_seed[s]["q2"], axis=0) for s in sorted(by_seed)])
                seed_rr = np.array([np.mean(by_seed[s]["rr"], axis=0) for s in sorted(by_seed)])

                M_cur = seed_q2.shape[1]
                x_cur = np.arange(1, M_cur + 1)
                
                # Offset for the barplot
                offset = (s_idx - n_scenarios / 2.0 + 0.5) * w
                x_pos = x_cur + offset

                c = color_map[p_idx]
                fc = to_rgba(c, 0.4)
                ls = linestyle_map.get(prefix, "-")
                lbl = f"{prefix} (p={p_idx})" if p_idx != -1 else prefix

                for ax, vals in [(ax_q2, seed_q2), (ax_rr, seed_rr)]:
                    means = vals.mean(axis=0)
                    stds  = vals.std(axis=0) if len(vals) > 1 else np.zeros_like(means)
                    ax.bar(
                        x_pos, means, width=w, yerr=stds, align='center',
                        facecolor=fc, edgecolor=c, linewidth=1.5, linestyle=ls,
                        error_kw={'ecolor': 'black', 'elinewidth': 1, 'capsize': 2, 'capthick': 1},
                        label=lbl
                    )

            ax_q2.set_ylabel(f"Q²  —  f{i}", fontsize=9)
            ax_rr.set_ylabel(f"RRMSE  —  f{i}", fontsize=9)

            if i == 0:
                ax_q2.set_title("Q² PCA  (mean ± std over seeds)", fontsize=10)
                ax_rr.set_title("RRMSE PCA  (mean ± std over seeds)", fontsize=10)

            for ax in (ax_q2, ax_rr):
                ax.grid(axis="y", linestyle=":", alpha=0.5)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_xticks(x[:M_max])

        for ax in axes[-1, :]:
            ax.set_xlabel("Number of retained modes k")

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_final_metrics_vs_modes(
        self,
        n_modes: Optional[int] = None,
        model_types: Optional[List[str]] = None,
        fixed_indices: Optional[List[int]] = None,
        figsize: Optional[tuple] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Q² and RRMSE of final predictions as a function of the number of latent modes k.

        For each M-mode model, this function shows the cumulative prediction quality:
        we reconstruct the field using the first k predicted GP projections
        and compare that to the original test fields. The `x` axis
        represents the number of projections `k` (1 to M).

        Figure (Q rows x 2 columns):
          - left column  : Q² per output field
          - right column : RRMSE per output field
          - stacked/shifted bar chart per scenario.

        All scenarios of the same model (same prefix) share the same line style,
        while the background and edge colors are defined by the `fixed_idx` (p) index.

        Parameters
        ----------
        n_modes       : filter by a number of training modes.
        model_types   : list of prefixes to keep (e.g. ["RC", "FI"]). None = all.
        fixed_indices : list of fixed_idx to keep (e.g. [-1, 0, 1]). None = all.
        figsize       : figure size (default: (14, 4*Q))
        output_path   : save path (None = do not save)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        records = []
        for seed, ana in zip(self._seeds, self._analyzers):
            for name in ana.list_models():
                res   = ana.storage.load_model_result_light(name)
                cfg   = res.get("config", {})
                inter = res.get("intermediate", {})
                
                # We read the new cumulative *prediction* errors
                cum_q2 = inter.get("cumulative_q2_pred_test")
                cum_rrmse = inter.get("cumulative_rrmse_pred_test")
                if cum_q2 is None or cum_rrmse is None:
                    continue
                
                prefix = name.split("_")[0]
                fixed_idx = int(cfg.get("fixed_idx", -1))
                
                if model_types is not None and prefix not in model_types:
                    continue
                if fixed_indices is not None and fixed_idx not in fixed_indices:
                    continue
                    
                records.append({
                    "seed":      seed,
                    "model_name": name,
                    "prefix":    prefix,
                    "n_modes":   cfg.get("n_modes", np.nan),
                    "fixed_idx": fixed_idx,
                    "cum_q2":    np.array(cum_q2),      # (M, Q)
                    "cum_rrmse": np.array(cum_rrmse),   # (M, Q)
                })

        if not records:
            print(
                "[MultiSeedAnalyzer] No cumulative *prediction* data found. "
                "Ensure that the model returns 'predicted_weights' in 'predictions' "
                "and that the benchmark has been rerun."
            )
            return plt.figure()

        all_n_modes = sorted({r["n_modes"] for r in records if not np.isnan(r["n_modes"])})
        if n_modes is not None:
            records = [r for r in records if r["n_modes"] == n_modes]
        elif len(all_n_modes) > 1:
            n_modes = all_n_modes[0]
            records = [r for r in records if r["n_modes"] == n_modes]
            print(f"[MultiSeedAnalyzer] Multiple trained n_modes found {all_n_modes}. "
                  f"Displaying cumulative prediction for the model with n_modes={n_modes}. "
                  "Pass n_modes= to choose another one.")
        else:
            n_modes = all_n_modes[0]

        if not records:
            print(f"[MultiSeedAnalyzer] No model with n_modes={n_modes}.")
            return plt.figure()
            
        M_max = max(r["cum_q2"].shape[0] for r in records)
        Q     = records[0]["cum_q2"].shape[1]
        x_positions = np.arange(1, M_max + 1)

        fig, axes = plt.subplots(
            Q, 2,
            figsize=figsize or (14, 4 * Q),
            sharex=True,
        )
        axes = np.array(axes).reshape(Q, 2)

        n_modes_label = f"  [Models trained with M={n_modes}]" if n_modes else ""

        scenarios = sorted(list(set((r["prefix"], r["fixed_idx"]) for r in records)))
        n_scenarios = len(scenarios)

        unique_p = sorted(list(set(r["fixed_idx"] for r in records)))
        color_map = {}
        cmap = plt.get_cmap("tab10")
        color_idx = 0
        from matplotlib.colors import to_rgba
        for p in unique_p:
            if p == -1:
                color_map[p] = self._COLOR_RC
            else:
                color_map[p] = cmap(color_idx % 10)
                color_idx += 1

        linestyle_map = {
            "RC": "-",
            "CI": "--",
            "FI": ":",
            "FM": "-."
        }

        total_width = 0.3
        w = total_width / max(1, n_scenarios)

        for i in range(Q):
            ax_q2 = axes[i, 0]
            ax_rr = axes[i, 1]

            for s_idx, (prefix, p_idx) in enumerate(scenarios):
                sub = [r for r in records if r["prefix"] == prefix and r["fixed_idx"] == p_idx]
                if not sub:
                    continue

                by_seed: Dict[int, Dict] = {}
                for r in sub:
                    s = r["seed"]
                    M_r = r["cum_q2"].shape[0]
                    if s not in by_seed:
                        by_seed[s] = {"q2": [], "rr": [], "M": M_r}
                    by_seed[s]["q2"].append(r["cum_q2"][:, i])    # (M,)
                    by_seed[s]["rr"].append(r["cum_rrmse"][:, i]) # (M,)

                seed_q2 = np.array([np.mean(by_seed[s]["q2"], axis=0) for s in sorted(by_seed)])
                seed_rr = np.array([np.mean(by_seed[s]["rr"], axis=0) for s in sorted(by_seed)])

                M_cur = seed_q2.shape[1]
                x_cur = np.arange(1, M_cur + 1)
                
                offset = (s_idx - n_scenarios / 2.0 + 0.5) * w
                x_pos_bars = x_cur + offset

                c = color_map[p_idx]
                ls = linestyle_map.get(prefix, "-")
                lbl = f"{prefix} (p={p_idx})" if p_idx != -1 else prefix

                for ax, vals in [(ax_q2, seed_q2), (ax_rr, seed_rr)]:
                    means = vals.mean(axis=0)
                    stds  = vals.std(axis=0) if len(vals) > 1 else np.zeros_like(means)
                    ax.errorbar(
                        x_pos_bars, means, yerr=stds,
                        color=c, linewidth=1.5, linestyle=ls,
                        marker='o', markersize=4,
                        elinewidth=1, capsize=3, capthick=1,
                        label=lbl
                    )

            ax_q2.set_ylabel(f"Q²  —  f{i}", fontsize=9)
            ax_rr.set_ylabel(f"RRMSE  —  f{i}", fontsize=9)

            if i == 0:
                ax_q2.set_title("Q² Prediction  (mean ± std over seeds)", fontsize=10)
                ax_rr.set_title("RRMSE Prediction  (mean ± std over seeds)", fontsize=10)

            for ax in (ax_q2, ax_rr):
                ax.grid(axis="y", linestyle=":", alpha=0.5)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_xticks(x_positions[:M_max])
                ax.set_xticklabels([f"{nm:g}" for nm in x_positions[:M_max]])

        for ax in axes[-1, :]:
            ax.set_xlabel("Number of retained modes k to reconstruct final predictions")

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig
