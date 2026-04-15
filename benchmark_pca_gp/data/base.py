"""Abstract base class for benchmark datasets."""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Dataset(ABC):
    """Abstract dataset for benchmarking PCA-GP surrogate models.

    A dataset provides data generation, train/test splitting, and centering
    following Algorithm 1 of the paper (three cases handled transparently):

    Case 1 – Constant α = u, zero RHS (homogeneous):
        No transformation: z = y.
        Adjusted mean: μ^k_adj = (1/N) Σ_i y^k_i − (u_k / ‖u‖²) · r_mean
        where r_mean = (1/N) Σ_i Σ_j u_j y^j_i.
        If training data satisfies constraint exactly: r_mean = 0, μ_adj = raw mean.

    Case 2 – Constant α = u, non-zero constant RHS c:
        Standard centering absorbs c automatically:
            Σ_j u_j μ_j = (1/N) Σ_i Σ_j u_j y^j_i = c,
        so Σ_j u_j (y^j_i − μ_j) = 0.  Identical to Case 1 (no extra code needed).

    Case 3 – Input-dependent α^j(x) with (possibly) non-zero H(x):
        Step 1  [reduce to homogeneous]:
            ỹ^k_i = y^k_i − (α^k_i / ‖α_i‖²) · H_i
        Step 2  [unit-coefficient constraint Σ_j z^j_i = 0]:
            z^k_i = α^k_i · ỹ^k_i
        Adjusted mean in z-space (u_eff = [1,...,1]).
        Override input_weights() and constraint_rhs_per_sample() to enable.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Mandatory abstract interface
    # ──────────────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def constraint_vector(self) -> np.ndarray:
        """u in Σ_j u_j z_j(x) = 0 in the (possibly transformed) field space z."""

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """Number of output fields Q."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of input X."""

    @abstractmethod
    def generate(
        self, n_total: int, seed: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate n_total samples.

        Returns
        -------
        X : (N, d)  input parameters
        fields : List[Q arrays of shape (N, S)]  raw output fields y^k
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Optional overrides for generalised constraints (Cases 2 and 3)
    # ──────────────────────────────────────────────────────────────────────────

    def input_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Per-sample constraint coefficients α^j(x_i), shape (N, Q).

        Returns
        -------
        None (default)
            Constant coefficients; use constraint_vector for all samples.
            center() applies the adjusted mean formula to y directly (Case 1/2).
        (N, Q) ndarray
            α^j(x_i) per sample.  center() applies the full Case 3 pipeline:
                ỹ^k_i = y^k_i − (α^k_i / ‖α_i‖²) · H_i   [if H non-zero]
                z^k_i = α^k_i · ỹ^k_i
            so that Σ_j z^j_i = 0 (unit constraint).

        Important: only return non-None if the fields provided to center() are
        the RAW (un-multiplied) outputs y^k.  If the dataset already outputs
        z^k = α^k · y^k_raw (transformation applied at generation time), leave
        this method returning None and use constraint_vector for the z-space
        constraint.
        """
        return None

    def constraint_rhs_per_sample(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Non-zero RHS H(x_i) in Σ_j α^j(x_i) y^j_i = H(x_i), shape (N,).

        Returns None (default) for homogeneous constraint H = 0.
        H_i is a scalar per sample (constant over the spatial/temporal dimension S),
        broadcast to (N, S) inside center().
        Used only when input_weights() returns non-None.
        """
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Split
    # ──────────────────────────────────────────────────────────────────────────

    def split_train_test(
        self,
        X: np.ndarray,
        fields: List[np.ndarray],
        n_train: int,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Shuffle and split into train / test sets.

        Returns
        -------
        X_train, X_test, fields_train, fields_test
        """
        rng = np.random.RandomState(seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        tr, te = idx[:n_train], idx[n_train:]
        return (
            X[tr], X[te],
            [f[tr] for f in fields],
            [f[te] for f in fields],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Centering  (Algorithm 1)
    # ──────────────────────────────────────────────────────────────────────────

    def center(
        self,
        fields_train: List[np.ndarray],
        fields_test: Optional[List[np.ndarray]] = None,
        X_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], List[np.ndarray]]:
        """Center fields following Algorithm 1 (adjusted mean formula).

        Parameters
        ----------
        fields_train : List[Q of (N_train, S)]
        fields_test  : List[Q of (N_test, S)] or None
        X_train      : (N_train, d) — required when input_weights() is non-None
        X_test       : (N_test, d)  — required when fields_test is given and
                       input_weights() is non-None

        Returns
        -------
        fields_train_c : List[Q of (N_train, S)]  centred z-space fields
        fields_test_c  : List[Q of (N_test, S)] or None
        means          : List[Q of (S,)]
            Adjusted z-space training means.  Adding means to GP predictions
            recovers z-space not-centred predictions.  For Case 3, call
            decode_fields() to convert z predictions to the original y scale.
        """
        Q = len(fields_train)
        u = np.asarray(self.constraint_vector, dtype=float)

        # ── Case 3: input-dependent α + optional non-zero H ──────────────────
        alpha_tr = None
        if X_train is not None:
            _aw = self.input_weights(X_train)
            if _aw is not None:
                alpha_tr = np.asarray(_aw, dtype=float)   # (N_tr, Q)

        if alpha_tr is not None:
            fields_train, fields_test = self._apply_alpha_transform(
                fields_train, fields_test, alpha_tr, X_train, X_test
            )
            u_eff = np.ones(Q)   # after transformation, constraint is Σ_j z^j = 0
        else:
            # Case 1 / 2: constant α = constraint_vector
            u_eff = u

        # ── Adjusted mean (projects training means onto the constraint) ───────
        # μ^k_adj = raw_mean^k − (u_eff_k / ‖u_eff‖²) · r_mean
        # r_mean  = Σ_j u_eff_j · raw_mean^j   (≈ 0 if constraint satisfied)
        raw_means = [f.mean(axis=0) for f in fields_train]        # List[Q × (S,)]
        r_mean    = sum(float(u_eff[k]) * raw_means[k] for k in range(Q))  # (S,)
        u_eff_sq  = float(np.dot(u_eff, u_eff))
        means = [
            raw_means[k] - (float(u_eff[k]) / u_eff_sq) * r_mean
            for k in range(Q)
        ]

        # ── Standard centering ────────────────────────────────────────────────
        fields_train_c = [f - m for f, m in zip(fields_train, means)]
        fields_test_c = (
            [f - m for f, m in zip(fields_test, means)]
            if fields_test is not None else None
        )
        return fields_train_c, fields_test_c, means

    def _apply_alpha_transform(
        self,
        fields_train: List[np.ndarray],
        fields_test: Optional[List[np.ndarray]],
        alpha_tr: np.ndarray,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray],
    ):
        """Apply Case 3 transformation (H-shift + α multiplication) to fields."""
        Q = len(fields_train)
        alpha_sq_tr = np.sum(alpha_tr ** 2, axis=1)   # (N_tr,)

        H_tr_raw = self.constraint_rhs_per_sample(X_train)
        if H_tr_raw is not None:
            H_tr   = np.asarray(H_tr_raw, dtype=float)      # (N_tr,)
            coef_tr = H_tr / alpha_sq_tr                      # (N_tr,) = H/‖α‖²
            # ỹ^k_i = y^k_i − (α^k_i / ‖α_i‖²) · H_i
            fields_train = [
                fields_train[k]
                - coef_tr[:, np.newaxis] * alpha_tr[:, k : k + 1]
                for k in range(Q)
            ]

        # z^k_i = α^k_i · ỹ^k_i
        fields_train = [
            fields_train[k] * alpha_tr[:, k : k + 1]
            for k in range(Q)
        ]

        if fields_test is not None:
            if X_test is None:
                raise ValueError(
                    "X_test must be provided when input_weights() is non-None "
                    "and fields_test is not None."
                )
            alpha_te    = np.asarray(self.input_weights(X_test), dtype=float)
            alpha_sq_te = np.sum(alpha_te ** 2, axis=1)

            H_te_raw = self.constraint_rhs_per_sample(X_test)
            if H_te_raw is not None:
                H_te    = np.asarray(H_te_raw, dtype=float)
                coef_te = H_te / alpha_sq_te
                fields_test = [
                    fields_test[k]
                    - coef_te[:, np.newaxis] * alpha_te[:, k : k + 1]
                    for k in range(Q)
                ]

            fields_test = [
                fields_test[k] * alpha_te[:, k : k + 1]
                for k in range(Q)
            ]

        return fields_train, fields_test

    # ──────────────────────────────────────────────────────────────────────────
    # Inverse transform (z-space → original y-space)
    # ──────────────────────────────────────────────────────────────────────────

    def decode_fields(
        self,
        fields_z_nc: List[np.ndarray],
        X: np.ndarray,
    ) -> List[np.ndarray]:
        """Convert z-space predictions back to the original y-space.

        For Case 1/2 (input_weights=None): identity, returns fields_z_nc.
        For Case 3:
            z^k = α^k · (y^k − (α^k / ‖α‖²) · H)
            ⟹  y^k = z^k / α^k + (α^k / ‖α‖²) · H

        Parameters
        ----------
        fields_z_nc : List[Q of (N_test, S)]
            z-space predictions (means already added, i.e. not-centred).
        X : (N_test, d)

        Returns
        -------
        List[Q of (N_test, S)] in original y-space.
        """
        _aw = self.input_weights(X) if X is not None else None
        if _aw is None:
            return fields_z_nc   # z = y, identity

        Q = len(fields_z_nc)
        alpha       = np.asarray(_aw, dtype=float)          # (N, Q)
        alpha_sq    = np.sum(alpha ** 2, axis=1)             # (N,)
        H_raw       = self.constraint_rhs_per_sample(X)

        fields_y = []
        for k in range(Q):
            a_k   = alpha[:, k : k + 1]                      # (N, 1)
            y_k   = fields_z_nc[k] / a_k                     # z/α
            if H_raw is not None:
                coef  = (alpha[:, k] / alpha_sq)[:, np.newaxis]  # (N, 1)
                y_k   = y_k + coef * np.asarray(H_raw, dtype=float)[:, np.newaxis]
            fields_y.append(y_k)
        return fields_y
