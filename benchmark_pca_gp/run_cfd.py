"""Benchmark CFD diffuser — script mono-seed (compatible cluster).

The CFD dataset has a fixed train/test split (50 train, 100 test).
``seed`` only affects GP hyper-parameter optimisation restarts.

Usage
-----
    # Standard run
    python run_cfd.py --seed 42

    # Explicit storage path
    python run_cfd.py --seed 7 --storage results_cfd_seed7.pkl

    # Quick test (few modes, few restarts)
    python run_cfd.py --quick --seed 0

    # Single n_modes
    python run_cfd.py --seed 3 --n_modes 5

    # Skip predictions storage (saves ~250 MB/model for S~141k)
    python run_cfd.py --seed 42 --no_predictions

Results stored in  {storage_prefix}_seed{seed}.pkl  by default.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from benchmark_pca_gp.data.cfd import CFDDataset
from benchmark_pca_gp.models.registry import ModelRegistry
from benchmark_pca_gp.benchmark.runner import BenchmarkRunner
from benchmark_pca_gp.postprocessing.analysis import ResultsAnalyzer


# ---------------------------------------------------------------------------
# Default data root — adjust if your cfd_diffuseur folder is elsewhere
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Constrained_MOGP", "Application", "CFD", "cfd_diffuseur",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark CFD — single seed")
    p.add_argument("--quick", action="store_true",
                   help="Mode rapide : peu de modes et peu de restarts")
    p.add_argument("--storage", default=None,
                   help="Chemin pkl explicite")
    p.add_argument("--storage_prefix", default="results_n=20_cfd",
                   help="Préfixe du nommage auto '{prefix}_seed{N}.zarr'")
    p.add_argument("--seed", type=int, default=22,
                   help="Graine aléatoire (affecte uniquement les restarts GP)")
    p.add_argument("--n_modes", type=int, default=None,
                   help="Si fourni, remplace la liste de modes par [n_modes]")
    p.add_argument("--data_root", default=None,
                   help="Chemin vers cfd_diffuseur/ (remplace le défaut)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Sauter les modèles déjà stockés")
    p.add_argument("--no_rc",  action="store_true", help="Exclure les modèles RC")
    p.add_argument("--no_ci",  action="store_true", help="Exclure les modèles CI")
    p.add_argument("--no_fi",  action="store_true", help="Exclure les modèles FI")
    p.add_argument("--no_fm",  action="store_true", help="Exclure les modèles FM")
    p.add_argument("--no_predictions", action="store_true",
                   help="Ne pas stocker les tableaux de prédictions brutes "
                        "(économise ~250 MB/modèle pour S~141k)")
    p.add_argument("--quiet", action="store_true", help="Réduire la verbosité")
    p.add_argument("--no_summary", action="store_true",
                   help="Ne pas afficher le résumé final")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_benchmark_config(args) -> dict:
    """Construit la configuration à partir des args (Namespace ou objet similaire)."""
    if getattr(args, "quick", False):
        cfg = dict(
            n_train          = 50,
            n_total          = 150,   # 50 train + 100 test (fixed)
            n_modes_list     = [4],
            n_restarts       = 5,
            maxiter          = 50,
            noise_var        = 1e-3,
            # FM (MOGP-LCM)
            n_kernels_lmc    = 1,
            rank_lmc         = [1],
            # RC (Constrained MOGP)
            n_kernels_rc     = 1,
            rank_rc          = [1],
            fixed_indices    = [0, 1, 2],
        )
    else:
        cfg = dict(
            n_train          = 50,
            n_total          = 150,
            n_modes_list     = [8],
            n_restarts       = 50,
            maxiter          = 150,
            noise_var        = 1e-3,
            # FM (MOGP-LCM)
            n_kernels_lmc    = 2,
            rank_lmc         = [1,2],
            # RC (Constrained MOGP)
            n_kernels_rc     = 2,
            rank_rc          = [2, 2],
            fixed_indices    = [0, 1, 2],
        )

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
# Main benchmark function (callable programmatically)
# ---------------------------------------------------------------------------

def run_benchmark(
    seed: int,
    storage_path: str,
    config: dict,
    data_root: str = None,
    skip_existing: bool = False,
    store_predictions: bool = False,
    verbose: bool = True,
    dataset: CFDDataset = None,
) -> None:
    """Run the full CFD benchmark for one seed.

    Parameters
    ----------
    seed             : random seed (GP restarts only)
    storage_path     : path to output zarr
    config           : dict from build_benchmark_config()
    data_root        : path to cfd_diffuseur directory
    skip_existing    : skip models already in zarr
    store_predictions: store raw prediction arrays (large for S~141k)
    verbose          : print progress
    dataset          : pre-loaded CFDDataset (pass to avoid re-loading data)
    """
    if dataset is None:
        if data_root is None:
            data_root = _DEFAULT_DATA_ROOT
        dataset = CFDDataset(data_root=data_root)

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
        n_modes_list         = config["n_modes_list"],
        u                    = u,
        Q                    = Q,
        fixed_indices        = config["fixed_indices"],
        gp_config            = gp_config_sogp,
        gp_config_lmc        = gp_config_lmc,
        gp_config_constrained= gp_config_constrained,
        include_rc           = config.get("include_rc", True),
        include_ci           = config.get("include_ci", True),
        include_fi           = config.get("include_fi", True),
        include_fm           = config.get("include_fm", True),
    )

    if verbose:
        print(f"[seed={seed}] standard={len(suite['standard'])}  "
              f"optimisés={len(suite['optimized'])}")

    runner = BenchmarkRunner(
        dataset           = dataset,
        storage_path      = storage_path,
        n_train           = config["n_train"],
        n_total           = config["n_total"],
        seed              = seed,
        verbose           = verbose,
        skip_existing     = skip_existing,
        store_predictions = store_predictions,
    )
    runner.run_from_suite(suite)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_root = args.data_root or _DEFAULT_DATA_ROOT
    storage_path = args.storage or f"{args.storage_prefix}_seed{args.seed}.pkl"
    config  = build_benchmark_config(args)
    verbose = not args.quiet
    store_predictions = not args.no_predictions

    if verbose:
        print(f"Storage          : {storage_path}")
        print(f"Seed             : {args.seed}")
        print(f"Modes            : {config['n_modes_list']}")
        print(f"Store predictions: {store_predictions}")
        print(f"Data root        : {data_root}")
        print()

    run_benchmark(
        seed              = args.seed,
        storage_path      = storage_path,
        config            = config,
        data_root         = data_root,
        skip_existing     = args.skip_existing,
        store_predictions = store_predictions,
        verbose           = verbose,
    )

    if not args.no_summary:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        ResultsAnalyzer(storage_path).print_summary()


if __name__ == "__main__":
    main()
