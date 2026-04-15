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
# Helpers internes
# ---------------------------------------------------------------------------

def _short_label(model_name: str) -> str:
    """Étiquette courte pour les axes de graphiques.

    Exemples : RC_ConstMOGP_M5 → 'RC M5'
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
    """Convertit les métriques latentes (forme variable) en vecteurs (M,).

    - FI (FieldwisePCA + IndepSOGP) : forme (Q, M) → moyenne sur les Q champs
    - Autres (RC / CI / FM)          : forme (M, q) → moyenne sur les q dims
    """
    if pca_type == "FieldwisePCA" and gp_type == "IndepSOGP":
        return lat_rmse.mean(axis=0), lat_q2.mean(axis=0)   # (M,)
    else:
        return lat_rmse.mean(axis=1), lat_q2.mean(axis=1)   # (M,)


# ---------------------------------------------------------------------------
# ResultsAnalyzer — single-seed (inchangé pour compatibilité)
# ---------------------------------------------------------------------------

class ResultsAnalyzer:
    """Charge, tabule et visualise les résultats d'un seul fichier zarr.

    Parameters
    ----------
    storage_path : chemin vers le zarr écrit par BenchmarkRunner
    """

    def __init__(self, storage_path: str):
        self.storage = open_storage(storage_path, mode="r")
        self._df: Any = None

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        return self.storage.list_models()

    def load_model(self, model_name: str) -> Dict[str, Any]:
        return self.storage.load_model_result(model_name)

    def to_dataframe(self, reload: bool = False):
        """DataFrame avec une ligne par modèle.

        Colonnes : model_name, pca_type, gp_type, n_modes, fixed_idx,
                   q2_f0…q2_fQ-1, rrmse_f0…rrmse_fQ-1,
                   iw_f0…iw_fQ-1, constraint_mean, constraint_max, fit_time_s.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas est requis pour to_dataframe()")
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
    # Résumé
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Tableau compact : modèles × métriques principales."""
        models = self.list_models()
        if not models:
            print("Aucun modèle trouvé.")
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
    # Graphiques (mono-seed)
    # ------------------------------------------------------------------

    def plot_rrmse_vs_modes(
        self,
        field_idx: int = 0,
        model_types: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """RRMSE vs nombre de modes PCA pour un champ de sortie donné."""
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")
        df  = self.to_dataframe()
        fig, ax = (plt.subplots() if ax is None else (ax.get_figure(), ax))
        col = f"rrmse_f{field_idx}"
        for label, group in df.groupby("model_name"):
            if model_types and not any(t in label for t in model_types):
                continue
            sub = group.sort_values("n_modes")
            ax.plot(sub["n_modes"], sub[col], marker="o", label=label)
        ax.set_xlabel("Nombre de modes PCA")
        ax.set_ylabel(f"RRMSE (champ {field_idx})")
        ax.set_title(f"RRMSE vs modes — champ {field_idx}")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

    def plot_q2_bar(
        self,
        n_modes: Optional[int] = None,
        fixed_idx: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Diagramme en barres : Q² par champ de sortie et par modèle."""
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")
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
        ax.set_title("Q² par champ de sortie")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

    def plot_constraint_satisfaction(
        self, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Barres : violation max de la contrainte par modèle."""
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
        ax.set_title("Satisfaction de la contrainte")
        ax.set_yscale("log")
        fig.tight_layout()
        return fig

    def compare_models(
        self,
        metric: str = "rrmse",
        field_idx: int = 0,
        n_modes: Optional[int] = None,
    ):
        """DataFrame trié pour une métrique et un champ donné."""
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")
        df  = self.to_dataframe()
        col = f"{metric}_f{field_idx}"
        if n_modes is not None:
            df = df[df["n_modes"] == n_modes]
        return df[["model_name", "pca_type", "gp_type", "n_modes",
                   "fixed_idx", col]].sort_values(col)


# ---------------------------------------------------------------------------
# MultiSeedAnalyzer — agrégation multi-seeds
# ---------------------------------------------------------------------------

class MultiSeedAnalyzer:
    """Agrège les résultats de plusieurs fichiers zarr (un par seed).

    Parameters
    ----------
    paths : liste de chemins zarr, glob pattern (ex: "results_lv_seed*.zarr"),
            ou chemin unique.

    Exemples
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
            raise ValueError("Aucun fichier de résultats trouvé.")

        self.paths = list(paths)
        self._analyzers = [ResultsAnalyzer(p) for p in self.paths]

        # Extraire le seed depuis le config stocké ou depuis le nom de fichier
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
    # Données
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Liste des noms de modèles (depuis le premier zarr)."""
        return self._analyzers[0].list_models()

    def to_dataframe(self, reload: bool = False):
        """DataFrame avec une ligne par (seed, modèle).

        Colonnes identiques à ResultsAnalyzer.to_dataframe() + colonne 'seed'.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")
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
        """DataFrame avec une ligne par (seed, modèle, mode latent).

        Colonnes : seed, model_name, pca_type, gp_type, n_modes, fixed_idx,
                   mode_idx, mean_latent_q2, mean_latent_rmse.

        Seuls les modèles ayant des métriques intermédiaires latentes sont inclus.
        La moyenne est prise sur les dimensions de sortie latentes (q).
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")
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
    # Résumé
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Affiche mean ± std de Q² et RRMSE par modèle, sur tous les seeds."""
        if not _HAS_PANDAS:
            # fallback sans pandas
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
    # Graphiques de comparaison
    # ------------------------------------------------------------------

    # Palettes
    _COLOR_RC   = "#2166ac"   # bleu
    _COLOR_CI   = "#d6604d"   # rouge/orange
    _COLOR_FI   = "#1a9641"   # vert
    _COLOR_FM   = "#7b2d8b"   # violet
    _COLORS_BY_PREFIX = {"RC": _COLOR_RC, "CI": _COLOR_CI,
                          "FI": _COLOR_FI, "FM": _COLOR_FM}

    def _get_q2_cols(self) -> List[str]:
        df = self.to_dataframe()
        return sorted(c for c in df.columns if c.startswith("q2_f"))

    def _palette(self, names: List[str], base_color: str) -> List[str]:
        """Dégradé de teintes autour de base_color pour distinguer les scénarios."""
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
        """Violin plot comparatif de la distribution de Q² sur les seeds.

        Figure 1×3 : [RC vs CI | RC vs FI | RC vs FM].
        Chaque violin = distribution de Q² sur (seeds × champs de sortie).

        Parameters
        ----------
        n_modes     : filtrer par nombre de modes (None = tous)
        figsize     : taille de figure (défaut: (15, 5))
        output_path : chemin pour sauvegarder la figure (None = ne pas sauver)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")

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
            "Distribution de Q² sur les seeds"
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
                ax.set_title(title + "\n(aucune donnée)")
                continue

            data_list, labels, colors = [], [], []
            base_comp_color = self._COLORS_BY_PREFIX.get(comp_prefix, "gray")
            comp_palette    = self._palette(comp_names, base_comp_color)

            for name in rc_names:
                sub = df[df["model_name"] == name]
                # Distribution sur seeds × champs
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

            # Ligne de séparation RC / comp
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
        """RRMSE GP latent par mode, comparaison RC vs CI / FI / FM.

        Figure 1×3 : chaque sous-graphe montre, pour chaque mode latent m,
        la RRMSE moyenne du GP sur les poids latents (mean ± std sur les seeds).

        - RC : une courbe (pas de sortie fixée)
        - CI / FI / FM : Q courbes, une par scénario (fixed_idx)

        Parameters
        ----------
        n_modes     : filtrer par nombre de modes (None = tous)
        figsize     : taille de figure (défaut: (15, 5))
        output_path : chemin pour sauvegarder (None = ne pas sauver)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")

        df_pm = self.to_per_mode_dataframe()
        if df_pm.empty:
            print("Aucune métrique latente par mode trouvée. "
                  "Vérifiez que intermediate_metrics est bien calculé.")
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
            "RRMSE GP latent par mode"
            + (f"  [M={n_modes}]" if n_modes else ""),
            fontsize=13
        )

        for ax, (comp_prefix, title) in zip(axes, comparisons):
            rc_mask   = df_pm["model_name"].str.startswith("RC_")
            comp_mask = df_pm["model_name"].str.startswith(comp_prefix + "_")
            sub_all   = df_pm[rc_mask | comp_mask]

            if sub_all.empty:
                ax.set_title(title + "\n(aucune donnée)")
                continue

            all_names = sorted(sub_all["model_name"].unique())
            mode_indices = sorted(sub_all["mode_idx"].unique())
            x = np.array(mode_indices) + 1   # 1-indexé

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

            # FI : tous les scénarios p partagent les mêmes métriques latentes
            # (tous les Q champs sont entraînés) → on n'affiche qu'une seule ligne
            if comp_prefix == "FI":
                # Choisir le premier scénario comme représentant
                rep = comp_names[0] if comp_names else None
                if rep:
                    _plot_model(rep, base_comp_color, "--")
                    ax.text(0.97, 0.97, "FI : métriques identiques\npour tous p",
                            transform=ax.transAxes, fontsize=7,
                            ha="right", va="top", color="gray")
            else:
                for i, name in enumerate(comp_names):
                    _plot_model(name, comp_palette[i], "--")

            ax.set_xlabel("Mode latent m")
            ax.set_ylabel("RRMSE latent moyen" if ax == axes[0] else "")
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
        """Q² et RRMSE de reconstruction PCA en fonction du nombre de modes retenus.

        Pour chaque modèle à M modes, montre comment la qualité de reconstruction
        des champs évolue quand on reconstruit avec k = 1, 2, …, M modes.
        Les métriques sont moyennées sur les seeds ; la barre d'erreur = ± std sur les seeds.
        Tous les scénarios d'un même modèle (même préfixe) partagent le même style de trait,
        tandis que la couleur de fond et des contours est définie par l'indice `fixed_idx` (p).

        Figure (Q lignes × 2 colonnes) :
          - colonne gauche  : Q² de reconstruction PCA
          - colonne droite  : RRMSE de reconstruction PCA
          - diagramme en barres empilées/décalées par scénario.

        Parameters
        ----------
        n_modes     : filtrer par nombre de modes (None = tous ; si plusieurs
                      valeurs existent, seule la première est utilisée)
        figsize     : taille de la figure (défaut : (14, 4*Q))
        output_path : chemin de sauvegarde (None = ne pas sauver)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")

        # ── Collecte des données ─────────────────────────────────────────────
        # Pour chaque (seed, modèle), on lit cumulative_q2_test (M, Q)
        # et cumulative_rrmse_test (M, Q) depuis l'intermediate.
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
                "[MultiSeedAnalyzer] Aucune donnée cumulative PCA trouvée. "
                "Relancez le benchmark pour peupler 'cumulative_q2_test'."
            )
            return plt.figure()

        # ── Filtrage n_modes ─────────────────────────────────────────────────
        all_n_modes = sorted({r["n_modes"] for r in records if not np.isnan(r["n_modes"])})
        if n_modes is not None:
            records = [r for r in records if r["n_modes"] == n_modes]
        elif len(all_n_modes) > 1:
            n_modes = all_n_modes[0]
            records = [r for r in records if r["n_modes"] == n_modes]
            print(f"[MultiSeedAnalyzer] Plusieurs n_modes trouvés {all_n_modes}. "
                  f"Affichage de n_modes={n_modes}. "
                  "Passez n_modes= pour en choisir un autre.")

        if not records:
            print(f"[MultiSeedAnalyzer] Aucun modèle avec n_modes={n_modes}.")
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
            f"Qualité de reconstruction PCA en fonction des modes retenus{n_modes_label}",
            fontsize=13,
        )

        # Scénarios uniques = (prefix, fixed_idx)
        scenarios = sorted(list(set((r["prefix"], r["fixed_idx"]) for r in records)))
        n_scenarios = len(scenarios)

        # Association Couleur <-> fixed_idx
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

        # Association Linestyle <-> prefix
        linestyle_map = {
            "RC": "-",
            "CI": "--",
            "FI": ":",
            "FM": "-."
        }

        # Largeur de barre de base totale et par scénario
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
                
                # Décalage pour le barplot
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
                ax_q2.set_title("Q² PCA  (mean ± std sur seeds)", fontsize=10)
                ax_rr.set_title("RRMSE PCA  (mean ± std sur seeds)", fontsize=10)

            for ax in (ax_q2, ax_rr):
                ax.grid(axis="y", linestyle=":", alpha=0.5)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_xticks(x[:M_max])

        for ax in axes[-1, :]:
            ax.set_xlabel("Nombre de modes retenus k")

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
        """Q² et RRMSE des prédictions finales en fonction du nombre de modes latent k.

        Pour chaque modèle à M modes, cette fonction montre la qualité de prédiction 
        cumulative : on reconstruit le champ en utilisant les k premières projections gp
        (prédites) et on compare ça aux champs de tests originaux. L'axe des `x`
        représente le nombre de projections `k` (1 à M).

        Figure (Q lignes × 2 colonnes) :
          - colonne gauche  : Q² par champ de sortie
          - colonne droite  : RRMSE par champ de sortie
          - diagramme en barres empilées/décalées par scénario.

        Tous les scénarios d'un même modèle (même préfixe) partagent le même style de trait,
        tandis que la couleur de fond et des contours est définie par l'indice `fixed_idx` (p).

        Parameters
        ----------
        n_modes       : filtrer par un nombre de modes d'entraînement.
        model_types   : liste de préfixes à conserver (ex: ["RC", "FI"]). None = tous.
        fixed_indices : liste de fixed_idx à conserver (ex: [-1, 0, 1]). None = tous.
        figsize       : taille de la figure (défaut : (14, 4*Q))
        output_path   : chemin de sauvegarde (None = ne pas sauver)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas requis")

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
                "[MultiSeedAnalyzer] Aucune donnée cumulative *de prédiction* trouvée. "
                "Assurez-vous que le modèle retourne 'predicted_weights' dans 'predictions' "
                "et que le benchmark a été relancé."
            )
            return plt.figure()

        all_n_modes = sorted({r["n_modes"] for r in records if not np.isnan(r["n_modes"])})
        if n_modes is not None:
            records = [r for r in records if r["n_modes"] == n_modes]
        elif len(all_n_modes) > 1:
            n_modes = all_n_modes[0]
            records = [r for r in records if r["n_modes"] == n_modes]
            print(f"[MultiSeedAnalyzer] Plusieurs n_modes entraînés trouvés {all_n_modes}. "
                  f"Affichage de la prédiction cumulative pour le modèle avec n_modes={n_modes}. "
                  "Passez n_modes= pour en choisir un autre.")
        else:
            n_modes = all_n_modes[0]

        if not records:
            print(f"[MultiSeedAnalyzer] Aucun modèle avec n_modes={n_modes}.")
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

        n_modes_label = f"  [Modèles entraînés avec M={n_modes}]" if n_modes else ""
        # fig.suptitle(
        #     f"Qualité cumulative des PRÉDICTIONS GP en fonction du nombre de modes retenus k{n_modes_label}",
        #     fontsize=13,
        # )

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
                ax_q2.set_title("Q² Prédiction  (mean ± std sur seeds)", fontsize=10)
                ax_rr.set_title("RRMSE Prédiction  (mean ± std sur seeds)", fontsize=10)

            for ax in (ax_q2, ax_rr):
                ax.grid(axis="y", linestyle=":", alpha=0.5)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_xticks(x_positions[:M_max])
                ax.set_xticklabels([f"{nm:g}" for nm in x_positions[:M_max]])

        for ax in axes[-1, :]:
            ax.set_xlabel("Nombre de modes retenus k pour reconstruire les prédictions finales")

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig
