"""Benchmark Lotka-Volterra — exécution parallèle sur plusieurs seeds.

Lance la même configuration de benchmark sur N seeds en parallèle via
concurrent.futures.ProcessPoolExecutor.  Chaque seed écrit son propre
fichier zarr : {storage_prefix}_seed{N}.zarr.

Compatibilité
-------------
Ce script est complémentaire de run_lotka_volterra.py :
- run_lotka_volterra.py      → un seed, compatible cluster (sbatch / qsub)
- run_lotka_volterra_multiprocess.py → plusieurs seeds en parallèle sur un PC

Usage
-----
    # 10 seeds (0..9) en parallèle
    python run_lotka_volterra_multiprocess.py --n_seeds 10

    # Seeds explicites
    python run_lotka_volterra_multiprocess.py --seeds 0 1 2 3 4

    # Limiter le parallélisme
    python run_lotka_volterra_multiprocess.py --n_seeds 10 --n_workers 4

    # Benchmark comparatif (1 seul n_modes, 10 seeds)
    python run_lotka_volterra_multiprocess.py --n_seeds 10 --n_modes 5

    # Mode rapide
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
# Worker (doit être au niveau module pour être picklable)
# ---------------------------------------------------------------------------

def _worker(args_tuple):
    """Exécute run_benchmark pour un seed ; renvoie (seed, erreur|None)."""
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
        description="Benchmark LV — multi-seeds en parallèle"
    )

    # --- Spécification des seeds ---
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--seeds", nargs="+", type=int,
                     help="Valeurs de seeds explicites (ex: 0 1 2 3 4)")
    grp.add_argument("--n_seeds", type=int, default=10,
                     help="Nombre de seeds consécutifs (défaut: 10)")

    p.add_argument("--seed_start", type=int, default=1,
                   help="Premier seed quand --n_seeds est utilisé (défaut: 0)")
    p.add_argument("--n_workers", type=int, default=None,
                   help="Nombre de processus parallèles (défaut: n_cpu)")

    # --- Configuration benchmark (identique à run_lotka_volterra.py) ---
    p.add_argument("--quick", action="store_true",
                   help="Mode rapide (peu d'échantillons)")
    p.add_argument("--storage_prefix", default="results_N_=30_lv",
                   help="Préfixe des fichiers de résultats (défaut: results_lv)")
    p.add_argument("--n_train",  type=int, default=None)
    p.add_argument("--n_total",  type=int, default=None)
    p.add_argument("--n_modes",  type=int, default=None,
                   help="Forcer un seul n_modes (ex: 5 pour benchmark comparatif)")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_rc",  action="store_true")
    p.add_argument("--no_ci",  action="store_true")
    p.add_argument("--no_fi",  action="store_true")
    p.add_argument("--no_fm",  action="store_true")
    p.add_argument("--no_summary", action="store_true",
                   help="Ne pas afficher le résumé agrégé à la fin")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Résoudre la liste de seeds
    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    n_workers = args.n_workers or min(len(seeds), multiprocessing.cpu_count())
    storage_paths = [f"{args.storage_prefix}_seed{s}.zarr" for s in seeds]

    # Construire la config en réutilisant build_benchmark_config
    # (on simule un namespace argparse minimal)
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
        print(f"Seeds échoués : {failed}")
        succeeded = [s for s in seeds if s not in failed]
    else:
        print(f"Tous les {len(seeds)} seeds terminés avec succès.")
        succeeded = seeds

    # Résumé agrégé multi-seeds
    if not args.no_summary and succeeded:
        done_paths = [
            f"{args.storage_prefix}_seed{s}.zarr" for s in succeeded
        ]
        print()
        print("=" * 70)
        print("RÉSUMÉ MULTI-SEEDS (mean ± std sur les seeds réussis)")
        print("=" * 70)
        MultiSeedAnalyzer(done_paths).print_summary()


if __name__ == "__main__":
    main()
