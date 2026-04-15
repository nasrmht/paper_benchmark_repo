"""CFD diffuser dataset: 3 constrained output fields.

Physical context
----------------
Reynolds-Averaged Navier-Stokes simulation of a diffuser.
Outputs (spatial fields over S ~ 141 039 mesh points):
    f1 = tau_11   Reynolds stress component
    f2 = tau_22   Reynolds stress component
    f3 = k        turbulent kinetic energy

Conservation constraint (Cauchy-Schwarz / trace relation):
    tau_11 + tau_22 - (4/3) * k = 0
    →  u = [1, 1, -4/3]

GP inputs (5 physical parameters):
    columns 0-4 of design files  (last column dropped — always 1.0)

Fixed train/test split
----------------------
- Train : 50 samples (IDs 1 … 50 in Datasets_train_16072025/)
- Test  : 100 converged samples from
          Datasets_test_15072025/converged_runs_n=100.txt

Because the split is fixed, ``generate()`` ignores ``n_total`` and
``seed``.  ``split_train_test()`` ignores the seed and simply returns
the first ``n_train`` rows as train and the rest as test.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import Dataset


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _load_field_files(
    folder: str,
    prefix: str,
    ids: np.ndarray,
    col: Optional[int],
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Load per-sample npy files and return a (N, S) array.

    Parameters
    ----------
    folder : directory containing the files
    prefix : filename prefix  (e.g. ``"tau1_"``)
    ids    : 1-D int array of run IDs (1-indexed)
    col    : None → file is already 1-D (S,); int → 2-D file, select column
    dtype  : output array dtype (default float64 to preserve constraint precision)
    """
    folder = Path(folder)
    first = np.load(folder / f"{prefix}{ids[0]}.npy", mmap_mode="r")
    n_pts = first.shape[0]
    out = np.empty((len(ids), n_pts), dtype=dtype)
    for i, run_id in enumerate(ids):
        arr = np.load(folder / f"{prefix}{run_id}.npy", mmap_mode="r")
        out[i] = arr if col is None else arr[:, col]
    return out


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CFDDataset(Dataset):
    """Diffuser CFD benchmark: 3 output fields under a trace constraint.

    Parameters
    ----------
    data_root : path to the ``cfd_diffuseur`` directory that contains
                ``Datasets_train_16072025/`` and ``Datasets_test_15072025/``
    dtype     : numpy dtype for field arrays (default float32 to save RAM)
    """

    TRAIN_DIR  = "Datasets_train_16072025"
    TEST_DIR   = "Datasets_test_15072025"
    CONV_FILE  = "converged_runs_n=100.txt"

    # Field specs: (subfolder, prefix, col_index or None)
    _FIELD_SPECS = [
        ("tau_11", "tau1_", None),   # f1 = tau_11  →  (S,) files
        ("tau_22", "tau2_", None),   # f2 = tau_22  →  (S,) files
        ("k",      "k_",    2),      # f3 = k       →  (S,3) files, col 2
    ]

    def __init__(
        self,
        data_root: str,
        dtype: np.dtype = np.float64,
    ):
        self.data_root = Path(data_root)
        self.dtype = dtype

        # Cached data (loaded once on first call to generate())
        self._X_all: Optional[np.ndarray] = None
        self._fields_all: Optional[List[np.ndarray]] = None
        self._n_train_fixed: Optional[int] = None
        self._n_test_fixed: Optional[int] = None

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    def constraint_vector(self) -> np.ndarray:
        return np.array([1.0, 1.0, -4.0 / 3.0])

    @property
    def n_outputs(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return 5   # 6 columns minus the last dummy column

    @property
    def field_names(self) -> List[str]:
        return ["tau11", "tau22", "k"]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load train + test data into memory (called once)."""
        train_dir = self.data_root / self.TRAIN_DIR
        test_dir  = self.data_root / self.TEST_DIR

        # ---- Train ----
        train_ids = np.arange(1, 51, dtype=int)   # IDs 1 … 50
        X_train = np.loadtxt(
            train_dir / "design_n=50.dat"
        )[:, :-1].astype(np.float64)              # (50, 5)

        fields_train = [
            _load_field_files(train_dir / sub, prefix, train_ids, col, self.dtype)
            for sub, prefix, col in self._FIELD_SPECS
        ]                                          # List[3 × (50, S)]

        # ---- Test ----
        test_ids = np.loadtxt(
            test_dir / self.CONV_FILE, dtype=int
        )                                          # (100,)  1-indexed
        design_test = np.loadtxt(
            test_dir / "design_n=100.dat"
        )[:, :-1].astype(np.float64)              # (100, 5)
        X_test = design_test[test_ids - 1]         # rows selected by IDs

        fields_test = [
            _load_field_files(test_dir / sub, prefix, test_ids, col, self.dtype)
            for sub, prefix, col in self._FIELD_SPECS
        ]                                          # List[3 × (100, S)]

        # ---- Combine: train first, then test ----
        self._n_train_fixed = len(X_train)
        self._n_test_fixed  = len(X_test)

        self._X_all = np.vstack([X_train, X_test])             # (150, 5)
        self._fields_all = [
            np.vstack([f_tr, f_te])                            # (150, S)
            for f_tr, f_te in zip(fields_train, fields_test)
        ]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def generate(
        self, n_total: int = None, seed: int = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Load all CFD data.

        ``n_total`` and ``seed`` are ignored — the split is fixed.
        Returns train data first (rows 0…49), test data after (rows 50…149).
        """
        if self._X_all is None:
            self._load_data()
        return self._X_all, self._fields_all

    def split_train_test(
        self,
        X: np.ndarray,
        fields: List[np.ndarray],
        n_train: int,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Return a fixed train/test split (seed is ignored).

        The first ``n_train`` rows are train, the rest are test.
        Since ``generate()`` always puts the 50 training samples first,
        pass ``n_train=50`` (or the loaded ``n_train_fixed``) to get the
        correct fixed partition.
        """
        X_train = X[:n_train]
        X_test  = X[n_train:]

        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_std[X_std < 1e-9] = 1.0

        X_train_normalized = (X_train - X_mean) / X_std
        X_test_normalized = (X_test - X_mean) / X_std
        f_train = [f[:n_train] for f in fields]
        f_test  = [f[n_train:] for f in fields]
        return X_train_normalized, X_test_normalized, f_train, f_test

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_train_fixed(self) -> int:
        if self._n_train_fixed is None:
            self._load_data()
        return self._n_train_fixed

    @property
    def n_test_fixed(self) -> int:
        if self._n_test_fixed is None:
            self._load_data()
        return self._n_test_fixed

    @property
    def n_spatial(self) -> int:
        """Number of spatial points S."""
        if self._fields_all is None:
            self._load_data()
        return self._fields_all[0].shape[1]
