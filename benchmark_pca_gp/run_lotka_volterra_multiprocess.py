"""Benchmark Lotka-Volterra — parallel execution over multiple seeds.

Launches the same benchmark configuration on N seeds in parallel via
concurrent.futures.ProcessPoolExecutor. Each seed writes its own
zarr file: {storage_prefix}_seed{N}.zarr.

Compatibility
-------------
This script complements run_lotka_volterra.py:
- run_lotka_volterra.py      → single seed, cluster compatible (sbatch / qsub)
- run_lotka_volterra_multiprocess.py → multiple seeds in parallel on a single machine

Usage
-----
    # 10 seeds (0..9) in parallel
    python run_lotka_volterra_multiprocess.py --n_seeds 10

    # Explicit seeds
    python run_lotka_volterra_multiprocess.py --seeds 0 1 2 3 4

    # Limit parallelism
    python run_lotka_volterra_multiprocess.py --n_seeds 10 --n_workers 4

    # Comparative benchmark (1 single n_modes, 10 seeds)
    python run_lotka_volterra_multiprocess.py --n_seeds 10 --n_modes 5

    # Quick mode
    python run_lotka_volterra_multiprocess.py --quick --n_seeds 3
"""
import argparse
import sys
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_lotka_volterra import build_benchmark_config, run_benchmark
from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer


# ---------------------------------------------------------------------------
# Worker (must be at module level to be picklable)
# ---------------------------------------------------------------------------

def _worker(args_tuple):
    """Runs run_benchmark for a seed; returns (seed, error|None)."""
    seed, storage_path, config, skip_existing = args_tuple
    try:
        run_benchmark(
            seed          = seed,
            storage_path  = storage_path,
            config        = config,
            skip_existing = skip_existing,
            verbose       = True,
        )
        return seed, None
    except Exception:
        import traceback
        return seed, traceback.format_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark LV — multi-seeds in parallel"
    )

    # --- Seed specification ---
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--seeds", nargs="+", type=int,
                     help="Explicit seed values (e.g. 0 1 2 3 4)")
    grp.add_argument("--n_seeds", type=int, default=5,
                     help="Number of consecutive seeds (default: 10)")

    p.add_argument("--seed_start", type=int, default=1,
                   help="First seed when --n_seeds is used (default: 0)")
    p.add_argument("--n_workers", type=int, default=None,
                   help="Number of parallel processes (default: n_cpu)")

    # --- Benchmark configuration (identical to run_lotka_volterra.py) ---
    p.add_argument("--quick", action="store_true",
                   help="Quick mode (few samples)")
    p.add_argument("--storage_prefix", default="results_N_b=10_lv",
                   help="Prefix of the results files (default: results_lv)")
    p.add_argument("--n_train",  type=int, default=None)
    p.add_argument("--n_total",  type=int, default=None)
    p.add_argument("--n_modes",  type=int, default=None,
                   help="Force a single n_modes (e.g. 5 for comparative benchmark)")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_rc",  action="store_true")
    p.add_argument("--no_ci",  action="store_true")
    p.add_argument("--no_fi",  action="store_true")
    p.add_argument("--no_fm",  action="store_true")
    p.add_argument("--no_summary", action="store_true",
                   help="Do not display the aggregated summary at the end")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve the seed list
    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    n_workers = args.n_workers or min(len(seeds), multiprocessing.cpu_count())
    storage_paths = [f"{args.storage_prefix}_seed{s}.zarr" for s in seeds]

    # Build config by reusing build_benchmark_config
    # (we simulate a minimal argparse namespace)
    class _Cfg:
        quick   = args.quick
        n_train = args.n_train
        n_total = args.n_total
        n_modes = args.n_modes
        no_rc   = args.no_rc
        no_ci   = args.no_ci
        no_fi   = args.no_fi
        no_fm   = args.no_fm

    config = build_benchmark_config(_Cfg())

    print("=" * 60)
    print(f"Seeds    : {seeds}")
    print(f"Workers  : {n_workers}")
    print(f"Storage  : {args.storage_prefix}_seed{{N}}.zarr")
    print(f"Modes    : {config['n_modes_list']}")
    print(f"n_train  : {config['n_train']}  n_total: {config['n_total']}")
    print("=" * 60)
    print()

    tasks = [
        (s, p, config, args.skip_existing)
        for s, p in zip(seeds, storage_paths)
    ]

    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            s, err = fut.result()
            if err:
                print(f"\n[FAILED] seed={s}:\n{err}")
                failed.append(s)
            else:
                print(f"[DONE]   seed={s}")

    print()
    if failed:
        print(f"Failed seeds: {failed}")
        succeeded = [s for s in seeds if s not in failed]
    else:
        print(f"All {len(seeds)} seeds completed successfully.")
        succeeded = seeds

    # Aggregated multi-seeds summary
    if not args.no_summary and succeeded:
        done_paths = [
            f"{args.storage_prefix}_seed{s}.zarr" for s in succeeded
        ]
        print()
        print("=" * 70)
        print("MULTI-SEEDS SUMMARY (mean ± std on successful seeds)")
        print("=" * 70)
        MultiSeedAnalyzer(done_paths).print_summary()


if __name__ == "__main__":
    main()
