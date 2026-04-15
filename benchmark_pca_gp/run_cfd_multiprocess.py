"""Benchmark CFD diffuser — version multi-seeds parallèle.

Optimisation mémoire
--------------------
Les données CFD sont fixes (S ~ 141 039 points, 3 champs, 150 échantillons).
Charger ces données une fois dans le processus principal puis forker
permet aux workers d'hériter une copie-sur-écriture (COW) du dataset,
sans copier les tableaux numpy tant qu'ils ne sont pas modifiés.

  Main process: charge CFDDataset (~420 MB float32)
       ↓  fork
  Worker 0, 1, … : héritent du dataset COW, aucune copie des champs

Usage
-----
    # 10 seeds, 4 workers en parallèle
    python run_cfd_multiprocess.py --seeds 0 1 2 3 4 5 6 7 8 9 --n_workers 4

    # Mode rapide pour tester
    python run_cfd_multiprocess.py --seeds 0 1 --quick --n_workers 2

    # Sans stocker les prédictions (recommandé pour S~141k)
    python run_cfd_multiprocess.py --seeds 0 1 2 --no_predictions

    # Reprendre des seeds manquants (--skip_existing)
    python run_cfd_multiprocess.py --seeds 0 1 2 --skip_existing

Résultats : un fichier zarr par seed,  {prefix}_seed{N}.zarr
"""
import argparse
import os
import sys
import traceback
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_pca_gp.data.cfd import CFDDataset
from benchmark_pca_gp.run_cfd import build_benchmark_config, run_benchmark, _DEFAULT_DATA_ROOT


# ---------------------------------------------------------------------------
# Global shared dataset (inherited by worker processes via fork)
# ---------------------------------------------------------------------------

_SHARED_DATASET: CFDDataset = None


def _worker_init(dataset: CFDDataset) -> None:
    """Initialise le dataset global dans le worker (utilisé avec initializer)."""
    global _SHARED_DATASET
    _SHARED_DATASET = dataset


def _run_one_seed(args_tuple) -> str:
    """Worker function: run benchmark for one seed.

    Returns a short status string.
    """
    seed, storage_path, config, skip_existing, store_predictions, verbose = args_tuple
    try:
        run_benchmark(
            seed              = seed,
            storage_path      = storage_path,
            config            = config,
            data_root         = None,           # ignored — dataset passed directly
            skip_existing     = skip_existing,
            store_predictions = store_predictions,
            verbose           = verbose,
            dataset           = _SHARED_DATASET,
        )
        return f"seed {seed}: OK → {storage_path}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"seed {seed}: ERROR — {e}\n{tb}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark CFD multi-seeds avec données partagées"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(23,30)),
                   help="Liste des seeds à exécuter (défaut: 0 .. 9)")
    p.add_argument("--n_workers", type=int, default=-1,
                   help="Nombre de workers parallèles (défaut: 4)")
    p.add_argument("--storage_prefix", default="results_n=20_cfd",
                   help="Préfixe zarr : '{prefix}_seed{N}.zarr'")
    p.add_argument("--data_root", default=None,
                   help="Chemin vers cfd_diffuseur/")
    p.add_argument("--quick", action="store_true",
                   help="Mode rapide")
    p.add_argument("--n_modes", type=int, default=None,
                   help="Forcer un seul n_modes")
    p.add_argument("--skip_existing", action="store_true",
                   help="Sauter les modèles déjà stockés dans le zarr")
    p.add_argument("--no_rc",  action="store_true")
    p.add_argument("--no_ci",  action="store_true")
    p.add_argument("--no_fi",  action="store_true")
    p.add_argument("--no_fm",  action="store_true")
    p.add_argument("--no_predictions", action="store_true",
                   help="Ne pas stocker les prédictions brutes (recommandé)")
    p.add_argument("--quiet", action="store_true",
                   help="Réduire la verbosité par seed")
    p.add_argument("--sequential", action="store_true",
                   help="Exécuter les seeds séquentiellement (débug)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_root = args.data_root or _DEFAULT_DATA_ROOT
    config    = build_benchmark_config(args)
    store_preds = not args.no_predictions
    verbose_worker = not args.quiet

    print(f"CFD benchmark multi-seeds")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Workers     : {args.n_workers}")
    print(f"  Modes       : {config['n_modes_list']}")
    print(f"  Storage     : {args.storage_prefix}_seed{{N}}.zarr")
    print(f"  Predictions : {store_preds}")
    print(f"  Data root   : {data_root}")
    print()

    # ---- Load dataset ONCE in the main process ----
    print("Loading CFD dataset (shared across workers)…")
    dataset = CFDDataset(data_root=data_root)
    dataset.generate()   # trigger actual load into memory
    print(f"  n_train={dataset.n_train_fixed}  "
          f"n_test={dataset.n_test_fixed}  "
          f"S={dataset.n_spatial}")
    print()

    # ---- Build task list ----
    tasks = [
        (
            seed,
            f"{args.storage_prefix}_seed{seed}.zarr",
            config,
            args.skip_existing,
            store_preds,
            verbose_worker,
        )
        for seed in args.seeds
    ]

    # ---- Run ----
    if args.sequential or args.n_workers <= 1:
        # Sequential fallback (useful for debugging or single-core machines)
        global _SHARED_DATASET
        _SHARED_DATASET = dataset
        results = [_run_one_seed(t) for t in tasks]
    else:
        # Fork-based multiprocessing: workers inherit dataset via COW
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=args.n_workers,
            initializer=_worker_init,
            initargs=(dataset,),
        ) as pool:
            results = pool.map(_run_one_seed, tasks)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("MULTI-SEED RESULTS")
    print("=" * 70)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
