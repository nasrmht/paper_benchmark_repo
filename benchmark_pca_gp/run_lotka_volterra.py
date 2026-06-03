"""Benchmark Lotka-Volterra — single-seed script (cluster compatible).

Usage
-----
    # Standard (storage auto-named after seed)
    python run_lotka_volterra.py --seed 42

    # Explicit storage
    python run_lotka_volterra.py --seed 7 --storage my_run_seed7.pkl

    # Quick mode for testing
    python run_lotka_volterra.py --quick --seed 0

    # Single n_modes (for comparative multi-seed benchmarks)
    python run_lotka_volterra.py --seed 3 --n_modes 5

Results stored in {storage_prefix}_seed{seed}.pkl by default.
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
                   help="Quick mode: few samples and few modes")
    p.add_argument("--storage", default=None,
                   help="Explicit pkl path (overrides --storage_prefix)")
    p.add_argument("--storage_prefix", default="results_N_b_=10_lv",
                   help="Prefix for auto-naming '{prefix}_seed{N}.zarr'")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed")
    p.add_argument("--n_train", type=int, default=None,
                   help="Number of training samples")
    p.add_argument("--n_total", type=int, default=None,
                   help="Total number of samples (train + test)")
    p.add_argument("--n_modes", type=int, default=None,
                   help="If provided, replaces the mode list with [n_modes]")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip models already present in zarr")
    p.add_argument("--no_rc",  action="store_true", help="Exclude RC models")
    p.add_argument("--no_ci",  action="store_true", help="Exclude CI models")
    p.add_argument("--no_fi",  action="store_true", help="Exclude FI models")
    p.add_argument("--no_fm",  action="store_true", help="Exclude FM models")
    p.add_argument("--quiet",  action="store_true", help="Reduce verbosity")
    p.add_argument("--no_summary", action="store_true",
                   help="Do not display the final summary")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_benchmark_config(args) -> dict:
    """Constructs the configuration dict from the parsed arguments.

    Compatible with an argparse.Namespace object or any object having the
    same attributes (useful for programmatic calls from the multi-seeds
    script).
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
            latent_dim_lmc         = [1],
            # RC (Constrained MOGP)
            n_kernels_rc     = 1,
            latent_dim_rc          = [1],
            t_end            = 20.0,
            dt               = 0.1,
            fixed_indices    = [0, 1, 2, 3],
        )
    else:
        cfg = dict(
            n_train          = getattr(args, "n_train", None) or 10,
            n_total          = getattr(args, "n_total", None) or 100,
            n_modes_list     = [10],
            n_restarts       = 50,
            maxiter          = 150,
            noise_var        = 1e-4,
            # FM (MOGP-LCM)
            n_kernels_lmc    = 2,
            latent_dim_lmc   = [2, 2],
            # RC (Constrained MOGP)
            n_kernels_rc     = 2,
            latent_dim_rc    = [2, 2],
            t_end            = 20.0,
            dt               = 0.001,
            fixed_indices    = [0, 1, 2, 3],
        )

    # --n_modes replaces the complete list with a single mode
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
    """Runs the complete benchmark for a given seed.

    Can be called directly from a multi-seeds script or a
    notebook without going through the CLI.

    Parameters
    ----------
    seed         : random seed
    storage_path : path to the output zarr file
    config       : dict returned by build_benchmark_config()
    skip_existing: do not retrain models already stored
    verbose      : show progress
    """
    dataset = LotkaVolterraDataset(t_end=config["t_end"], dt=config["dt"], n_train=config["n_train"])
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
        "latent_dim":      config["latent_dim_lmc"],
    }
    gp_config_constrained = {
        **_base,
        "n_kernels": config["n_kernels_rc"],
        "latent_dim":      config["latent_dim_rc"],
    }
    gp_config_sogp = _base   # CI and FI (no n_kernels/latent_dim)

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
              f"optimized={len(suite['optimized'])}")

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
# CLI Entry point
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
