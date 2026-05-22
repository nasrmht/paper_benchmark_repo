"""Benchmark Lotka-Volterra — script mono-seed (compatible cluster).

Usage
-----
    # Standard (storage auto-nommé d'après le seed)
    python run_lotka_volterra.py --seed 42

    # Storage explicite
    python run_lotka_volterra.py --seed 7 --storage my_run_seed7.pkl

    # Mode rapide pour tester
    python run_lotka_volterra.py --quick --seed 0

    # Un seul n_modes (pour benchmarks multi-seeds comparatifs)
    python run_lotka_volterra.py --seed 3 --n_modes 5

Résultats stockés dans  {storage_prefix}_seed{seed}.pkl  par défaut.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from benchmark_pca_gp.data.lotka_volterra import LotkaVolterraDataset
from benchmark_pca_gp.models.registry import ModelRegistry
from benchmark_pca_gp.benchmark.runner import BenchmarkRunner
from benchmark_pca_gp.postprocessing.analysis import ResultsAnalyzer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark LV — single seed")
    p.add_argument("--quick", action="store_true",
                   help="Mode rapide : peu d'échantillons et peu de modes")
    p.add_argument("--storage", default=None,
                   help="Chemin pkl explicite (écrase --storage_prefix)")
    p.add_argument("--storage_prefix", default="results_N_b_=10_lv",
                   help="Préfixe du nommage auto '{prefix}_seed{N}.zarr'")
    p.add_argument("--seed", type=int, default=1,
                   help="Graine aléatoire")
    p.add_argument("--n_train", type=int, default=None,
                   help="Nombre d'échantillons d'entraînement")
    p.add_argument("--n_total", type=int, default=None,
                   help="Nombre total d'échantillons (train + test)")
    p.add_argument("--n_modes", type=int, default=None,
                   help="Si fourni, remplace la liste de modes par [n_modes]")
    p.add_argument("--skip_existing", action="store_true",
                   help="Sauter les modèles déjà présents dans le zarr")
    p.add_argument("--no_rc",  action="store_true", help="Exclure les modèles RC")
    p.add_argument("--no_ci",  action="store_true", help="Exclure les modèles CI")
    p.add_argument("--no_fi",  action="store_true", help="Exclure les modèles FI")
    p.add_argument("--no_fm",  action="store_true", help="Exclure les modèles FM")
    p.add_argument("--quiet",  action="store_true", help="Réduire la verbosité")
    p.add_argument("--no_summary", action="store_true",
                   help="Ne pas afficher le résumé final")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_benchmark_config(args) -> dict:
    """Construit le dict de configuration à partir des arguments parsés.

    Compatible avec un objet argparse.Namespace ou tout objet ayant les
    mêmes attributs (utile pour l'appel programmatique depuis le script
    multi-seeds).
    """
    if getattr(args, "quick", False):
        cfg = dict(
            n_train          = getattr(args, "n_train", None) or 30,
            n_total          = getattr(args, "n_total", None) or 80,
            n_modes_list     = [4],
            n_restarts       = 10,
            maxiter          = 50,
            noise_var        = 1e-3,
            # FM (MOGP-LCM)
            n_kernels_lmc    = 1,
            rank_lmc         = [1],
            # RC (Constrained MOGP)
            n_kernels_rc     = 1,
            rank_rc          = [1],
            t_end            = 20.0,
            dt               = 0.1,
            fixed_indices    = [0, 1, 2, 3],
        )
    else:
        cfg = dict(
            n_train          = getattr(args, "n_train", None) or 30,
            n_total          = getattr(args, "n_total", None) or 100,
            n_modes_list     = [10],
            n_restarts       = 30,
            maxiter          = 100,
            noise_var        = 1e-3,
            # FM (MOGP-LCM)
            n_kernels_lmc    = 2,
            rank_lmc         = [2, 2],
            # RC (Constrained MOGP)
            n_kernels_rc     = 2,
            rank_rc          = [2, 2],
            t_end            = 20.0,
            dt               = 0.001,
            fixed_indices    = [0, 1, 2, 3],
        )

    # --n_modes remplace la liste complète par un seul mode
    n_modes = getattr(args, "n_modes", None)
    if n_modes is not None:
        cfg["n_modes_list"] = [n_modes]

    cfg.update(
        include_rc = not getattr(args, "no_rc", False),
        include_ci = not getattr(args, "no_ci", False),
        include_fi = not getattr(args, "no_fi", False),
        include_fm = not getattr(args, "no_fm", False),
    )
    return cfg


# ---------------------------------------------------------------------------
# Fonction principale (appelable programmatiquement)
# ---------------------------------------------------------------------------

def run_benchmark(
    seed: int,
    storage_path: str,
    config: dict,
    skip_existing: bool = False,
    verbose: bool = True,
) -> None:
    """Exécute le benchmark complet pour un seed donné.

    Peut être appelée directement depuis un script multi-seeds ou un
    notebook sans passer par la CLI.

    Parameters
    ----------
    seed         : graine aléatoire
    storage_path : chemin du fichier zarr de sortie
    config       : dict retourné par build_benchmark_config()
    skip_existing: ne pas ré-entraîner les modèles déjà stockés
    verbose      : afficher la progression
    """
    dataset = LotkaVolterraDataset(t_end=config["t_end"], dt=config["dt"])
    u = dataset.constraint_vector
    Q = dataset.n_outputs

    if verbose:
        print(f"[seed={seed}] {type(dataset).__name__}  "
              f"d={dataset.input_dim}  Q={Q}  u={u.tolist()}")
        print(f"[seed={seed}] n_train={config['n_train']}  "
              f"n_total={config['n_total']}  modes={config['n_modes_list']}")

    _base = {
        "n_restarts": config["n_restarts"],
        "maxiter":    config["maxiter"],
        "noise_var":  config.get("noise_var", 1e-3),
        "seed":       seed,
    }
    gp_config_lmc = {
        **_base,
        "n_kernels": config["n_kernels_lmc"],
        "rank":      config["rank_lmc"],
    }
    gp_config_constrained = {
        **_base,
        "n_kernels": config["n_kernels_rc"],
        "rank":      config["rank_rc"],
    }
    gp_config_sogp = _base   # CI and FI (no n_kernels/rank)

    suite = ModelRegistry.create_benchmark_suite(
        n_modes_list          = config["n_modes_list"],
        u                     = u,
        Q                     = Q,
        fixed_indices         = config["fixed_indices"],
        gp_config             = gp_config_sogp,
        gp_config_lmc         = gp_config_lmc,
        gp_config_constrained = gp_config_constrained,
        include_rc            = config.get("include_rc", True),
        include_ci            = config.get("include_ci", True),
        include_fi            = config.get("include_fi", True),
        include_fm            = config.get("include_fm", True),
    )

    if verbose:
        print(f"[seed={seed}] standard={len(suite['standard'])}  "
              f"optimisés={len(suite['optimized'])}")

    runner = BenchmarkRunner(
        dataset       = dataset,
        storage_path  = storage_path,
        n_train       = config["n_train"],
        n_total       = config["n_total"],
        seed          = seed,
        verbose       = verbose,
        skip_existing = skip_existing,
    )
    runner.run_from_suite(suite)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    storage_path = (args.storage
                    or f"{args.storage_prefix}_seed{args.seed}.pkl")
    config  = build_benchmark_config(args)
    verbose = not args.quiet

    if verbose:
        print(f"Storage : {storage_path}")
        print(f"Seed    : {args.seed}")
        print(f"Modes   : {config['n_modes_list']}")
        print()

    run_benchmark(
        seed          = args.seed,
        storage_path  = storage_path,
        config        = config,
        skip_existing = args.skip_existing,
        verbose       = verbose,
    )

    if not args.no_summary:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        ResultsAnalyzer(storage_path).print_summary()


if __name__ == "__main__":
    main()
