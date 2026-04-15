# Constrained Multi-Output GP Benchmark

Reproduction code and pre-computed results for the paper:

> **Prediction of physical fields under linear constraints** — [Mahamat], [2026]

This repository covers three classes of experiments:

1. **Constraint benchmark** – Direct comparison of Constrained MOGP vs Independent GP vs LCM on synthetic test cases (deductive approach).
2. **Lotka-Volterra benchmark** – RC (Row-wise Constrained PCA+MOGP) vs deductive baselines on the Lotka-Volterra dynamical system.
3. **CFD diffuser benchmark** – RC vs deductive baselines on a CFD Reynolds-stress dataset (requires private data to re-run, results included for plot reproduction).

---

## Repository structure

```
paper_benchmark_repo/
├── LcGP/                        # GP library (mogp, sogp, utils)
├── benchmark_pca_gp/            # Benchmark framework
│   ├── data/                    # Dataset definitions (LotkaVolterra, CFD)
│   ├── models/                  # Model registry (RC, CI, FI, FM)
│   ├── regression/              # GP regressors per mode
│   ├── reduction/               # PCA reducers (rowwise, colwise, fieldwise)
│   ├── metrics/                 # Evaluation metrics
│   ├── benchmark/               # Runner and storage backends
│   ├── postprocessing/          # Analysis and plot scripts
│   │   ├── plot_lv_rrmse.py     # Lotka-Volterra RRMSE/dominance figure
│   │   ├── plot_cfd_rrmse.py    # CFD RRMSE/dominance figure
│   │   ├── ms_analyse_compare_n.py  # Multi-N combined RRMSE figure
│   │   └── ms_plot_cfd_fields.py    # CFD field visualisation (needs zarr)
│   ├── run_lotka_volterra.py    # Run LV benchmark (single seed)
│   ├── run_lotka_volterra_multiprocess.py  # Multi-seed parallel (PC)
│   ├── run_cfd.py               # Run CFD benchmark (single seed)
│   └── run_cfd_multiprocess.py  # Multi-seed parallel (PC)
├── constraint_benchmark/
│   ├── run_benchmark.py         # Run constraint benchmark (single seed)
│   ├── plot_violinplot.py       # Violin-plot figure
│   └── plot_ranking_percentages.py  # Ranking percentage figure
└── results/
    ├── constraint_benchmark/    # 192 pickle files (seeds 1–200)
    ├── lotka_volterra/          # Zarr results, N=10/15/30, seeds 1–9
    └── cfd/                     # Zarr results, N=20, seeds 20–28
```

---

## Requirements

```
python >= 3.9
numpy
scipy
scikit-learn
matplotlib
zarr >= 3.0
pandas          # for dataframe analysis
joblib          # for parallel runs
```

Install dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib zarr pandas joblib
```

Add `LcGP` and the repo root to your Python path (or install editably):

```bash
# From the repo root
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## 1. Reproduce figures — Constraint benchmark

Figures use pre-computed results in `results/constraint_benchmark/`.

```bash
cd constraint_benchmark

# Violin-plot figure (per metric, per output)
python plot_violinplot.py

# Ranking percentage figure
python plot_ranking_percentages.py
```

Output figures are saved in `constraint_benchmark/figures/`.

**To re-run experiments** (one seed per job, parallelisable on a cluster):

```bash
cd constraint_benchmark
python run_benchmark.py --seed 1
python run_benchmark.py --seed 2
# ...
```

Results are saved as `results_constraint_benchmark/benchmark_complet_seed=N.pkl`.
With 200 seeds as in the paper, launch one job per seed on a cluster.

---

## 2. Reproduce figures — Lotka-Volterra

Figures use pre-computed zarr results in `results/lotka_volterra/`.

```bash
cd benchmark_pca_gp/postprocessing

# RRMSE + dominance figure (main paper figure, N=10/15/30)
python plot_lv_rrmse.py

# Combined RRMSE vs modes figure
python ms_analyse_compare_n.py
```

**To re-run experiments:**

Single seed (can be submitted as an array job on a cluster):

```bash
# From the repo root
python benchmark_pca_gp/run_lotka_volterra.py --seed 1 --n_train 10
python benchmark_pca_gp/run_lotka_volterra.py --seed 1 --n_train 15
python benchmark_pca_gp/run_lotka_volterra.py --seed 1 --n_train 30
```

Results are saved as `results_N_=10_lv_seed1.pkl` etc. by default.

Multi-seed in parallel on a single machine:

```bash
python benchmark_pca_gp/run_lotka_volterra_multiprocess.py \
    --seeds 1 2 3 4 5 --n_workers 4 --n_train 10
```

The paper uses seeds 1–9 for N=10, 15, 30.

---

## 3. Reproduce figures — CFD diffuser

Figures use pre-computed zarr results in `results/cfd/`.
**Note:** Re-running experiments requires the private CFD dataset (not distributed).

```bash
cd benchmark_pca_gp/postprocessing

# RRMSE + dominance figure
python plot_cfd_rrmse.py --seeds 20 21 22 23 24 25 26 27 28

# CFD field visualisation (requires zarr with stored predictions)
python ms_plot_cfd_fields.py --prefix results_n=20_cfd --seed 20 \
    --zarr_dir ../../results/cfd
```

**To re-run experiments** (requires CFD data; see `benchmark_pca_gp/data/cfd.py` for data paths):

```bash
python benchmark_pca_gp/run_cfd.py --seed 20
python benchmark_pca_gp/run_cfd_multiprocess.py --seeds 20 21 22 --n_workers 3
```

The paper uses seeds 20–28 with N=20 training samples.

---

## Storage format

Results are stored in two formats:

- **`.pkl`** (new default): lightweight pickle, metrics only (~few KB per seed). Used for LV and CFD when re-running with current code.
- **`.zarr`** (legacy, in `results/`): full data including raw field predictions. Required by `ms_plot_cfd_fields.py` for field visualisation.

The `open_storage(path)` factory in `benchmark_pca_gp/benchmark/storage.py` selects the backend automatically based on file extension.

---

## Notes

- **LcGP** is included directly in this repo. No separate installation needed.
- The **constraint benchmark** data (pickle files) is committed to git (~1.6 MB total).
- The **LV and CFD zarr results** are large (~1.4 GB per LV seed, ~6.5 GB per CFD seed). Consider hosting them on Zenodo/figshare for a public release and providing a download script.
- The CFD training/test field data is **not included** (private dataset). Contact the authors for access.
